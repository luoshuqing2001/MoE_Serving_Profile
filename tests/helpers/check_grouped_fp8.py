import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from types import SimpleNamespace

import torch

from moe_serving.gpt_oss_parallel import ParallelGroup, ensure_transformers_5200

ensure_transformers_5200()
from moe_serving.efficient_kernels import grouped_fp8, prepare_fp8
from moe_serving.qwen3_parallel import ExpertProjection, ParallelExperts

c = SimpleNamespace(num_experts=4, hidden_act="silu")
e = ParallelExperts(c, ParallelGroup(1), numerics="fast")
torch.manual_seed(5)
for ps in e.local:
    for p, n, k in [
        ("gate_proj", 256, 256),
        ("up_proj", 256, 256),
        ("down_proj", 256, 256),
    ]:
        w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.float8_e4m3fn)
        ps[p] = ExpertProjection(w, torch.ones(2, 2, device="cuda"), (128, 128))
x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
i = torch.stack([torch.randperm(4, device="cuda")[:2] for _ in range(16)])
w = torch.softmax(torch.randn(16, 2, device="cuda"), -1)
with torch.inference_mode():
    ref = e(x, i, w)
    prepare_fp8(e)
    out = grouped_fp8(e, x, i, w)
    rel = ((out - ref).float().norm() / ref.float().norm()).item()
    print(
        "finite",
        out.isfinite().all().item(),
        "max",
        (out - ref).abs().max().item(),
        "relative",
        rel,
    )
    assert rel < 0.08
