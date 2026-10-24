"""Four-GPU kernel checks against dequantized checkpoint arithmetic."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from moe_serving.efficient_kernels import prepare_fp8, prepare_mxfp4
from moe_serving.gpt_oss_parallel import (
    ParallelExperts,
    ParallelGroup,
    ensure_transformers_5200,
)

ensure_transformers_5200()
from moe_serving.qwen3_parallel import ExpertProjection
from moe_serving.qwen3_parallel import ParallelExperts as QwenExperts

dist.init_process_group("nccl")
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)
parallel = ParallelGroup(dist.get_world_size())
results = []
for family in ["gpt_oss", "qwen3_moe"]:
    torch.manual_seed(12)
    config = SimpleNamespace(
        num_local_experts=4,
        num_experts=4,
        hidden_act="silu",
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        hidden_size=256,
        intermediate_size=256,
    )
    if family == "gpt_oss":
        reference = ParallelExperts(config, ParallelGroup(1), numerics="fast")
        for proj, n in [("gate_up_proj", 512), ("down_proj", 256)]:
            reference.register_buffer(
                proj + "_blocks",
                torch.randint(0, 256, (4, n, 8, 16), device=device, dtype=torch.uint8),
            )
            reference.register_buffer(
                proj + "_scales",
                torch.full((4, n, 8), 120, device=device, dtype=torch.uint8),
            )
            reference.register_buffer(
                proj + "_bias", torch.randn(4, n, device=device) * 0.01
            )
        actual = ParallelExperts(config, parallel, numerics="efficient")
        for name, value in reference.named_buffers():
            actual.register_buffer(
                name, value[actual.start : actual.start + actual.num_experts].clone()
            )
        prepare_mxfp4(actual, config)
    else:
        reference = QwenExperts(config, ParallelGroup(1), numerics="fast")
        for ps in reference.local:
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                weight = (torch.randn(256, 256, device=device) * 0.05).to(
                    torch.float8_e4m3fn
                )
                ps[proj] = ExpertProjection(
                    weight, torch.ones(2, 2, device=device), (128, 128)
                )
        actual = QwenExperts(config, parallel, numerics="efficient")
        for e, ps in enumerate(actual.local):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                p = reference.local[actual.start + e][proj]
                ps[proj] = ExpertProjection(
                    p.weight.clone(), p.scales.clone(), p.block_size
                )
        prepare_fp8(actual)
    for tokens in [4, 97, 1]:
        x = torch.randn(tokens, 256, device=device, dtype=torch.bfloat16)
        indices = torch.stack(
            [torch.randperm(4, device=device)[:2] for _ in range(tokens)]
        )
        if tokens == 1:
            indices[:] = torch.tensor([0, 1], device=device)
        weights = torch.softmax(torch.randn(tokens, 2, device=device), -1)
        with torch.inference_mode():
            expected = reference(
                x.float() if family == "gpt_oss" else x, indices, weights
            )
            output = actual(x, indices, weights)
        assert output.isfinite().all()
        rel = ((output - expected).float().norm() / expected.float().norm()).item()
        assert rel < (0.03 if family == "gpt_oss" else 0.08), (family, tokens, rel)
        results.append(
            {
                "family": family,
                "tokens": tokens,
                "relative_l2": rel,
                "ep_size": parallel.size,
            }
        )
if rank == 0:
    print(json.dumps(results, indent=2))
    path = Path("tests/validation_results/efficient_grouped_ep.json")
    path.write_text(json.dumps(results, indent=2) + "\n")
dist.destroy_process_group()
