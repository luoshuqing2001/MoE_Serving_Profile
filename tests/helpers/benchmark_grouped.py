"""Benchmark one real GPT-OSS-120B rank-local expert layer; no full model load."""

import json
import time
from pathlib import Path

import torch
from safetensors import safe_open

from moe_serving.efficient_kernels import prepare_mxfp4
from moe_serving.gpt_oss_parallel import (
    ParallelExperts,
    ParallelGroup,
    ensure_transformers_5200,
)

ensure_transformers_5200()
from transformers import GptOssConfig

root = Path("LLMs/gpt-oss-120b")
config = GptOssConfig.from_pretrained(root)
config.num_local_experts = 32
reference = ParallelExperts(config, ParallelGroup(1), numerics="fast")
actual = ParallelExperts(config, ParallelGroup(1), numerics="efficient")
weight_map = json.loads((root / "model.safetensors.index.json").read_text())[
    "weight_map"
]
for proj in ["gate_up_proj", "down_proj"]:
    for suffix in ["blocks", "scales", "bias"]:
        attr = f"{proj}_{suffix}"
        key = f"model.layers.0.mlp.experts.{attr}"
        with safe_open(root / weight_map[key], framework="pt") as f:
            value = f.get_slice(key)[:32].to("cuda")
        reference.register_buffer(
            attr, value.to(torch.bfloat16) if suffix == "bias" else value
        )
        actual.register_buffer(attr, value)
prepare_mxfp4(actual, config)
torch.manual_seed(3)
x = torch.randn(4, config.hidden_size, device="cuda", dtype=torch.bfloat16)
i = torch.stack([torch.randperm(32, device="cuda")[:4] for _ in range(4)])
w = torch.softmax(torch.randn(4, 4, device="cuda"), -1)
with torch.inference_mode():
    expected = reference(x, i, w)
    out = actual(x, i, w)
    relative = ((out - expected).float().norm() / expected.float().norm()).item()
    assert relative < 0.05, relative

    def bench(model, count):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(count):
            model(x, i, w)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / count

    for _ in range(3):
        actual(x, i, w)
    old = bench(reference, 3)
    new = bench(actual, 20)
report = {
    "checkpoint": "gpt-oss-120b",
    "layer": 0,
    "local_experts": 32,
    "tokens": 4,
    "top_k": 4,
    "relative_l2": relative,
    "python_loop_ms": old,
    "grouped_mxfp4_ms": new,
    "speedup": old / new,
    "scope": "single layer, no TP/EP communication; other GPU workloads may be active",
}
print(json.dumps(report, indent=2))
Path("tests/validation_results/efficient_layer_benchmark.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
