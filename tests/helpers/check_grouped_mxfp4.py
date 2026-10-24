import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from types import SimpleNamespace

import torch

from moe_serving.efficient_kernels import grouped_mxfp4, prepare_mxfp4
from moe_serving.gpt_oss_parallel import (
    ParallelExperts,
    ParallelGroup,
    ensure_transformers_5200,
)

ensure_transformers_5200()
c = SimpleNamespace(
    num_local_experts=4,
    swiglu_alpha=1.702,
    swiglu_limit=7.0,
    hidden_size=256,
    intermediate_size=256,
)
e = ParallelExperts(c, ParallelGroup(1), numerics="fast").cuda()
torch.manual_seed(12)
for p, n, k in [("gate_up_proj", 512, 256), ("down_proj", 256, 256)]:
    e.register_buffer(
        p + "_blocks",
        torch.randint(0, 256, (4, n, k // 32, 16), dtype=torch.uint8, device="cuda"),
    )
    e.register_buffer(
        p + "_scales",
        torch.full((4, n, k // 32), 120, dtype=torch.uint8, device="cuda"),
    )
    e.register_buffer(
        p + "_bias", torch.randn(4, n, device="cuda", dtype=torch.bfloat16) * 0.01
    )
x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
i = torch.stack([torch.randperm(4, device="cuda")[:2] for _ in range(16)])
w = torch.softmax(torch.randn(16, 2, device="cuda"), -1)
with torch.inference_mode():
    ref = e(x, i, w)
    prepare_mxfp4(e, c)
    out = grouped_mxfp4(e, x, i, w)
    print(
        "finite",
        out.isfinite().all().item(),
        "max",
        (out - ref).abs().max().item(),
        "relative",
        ((out - ref).float().norm() / ref.float().norm()).item(),
    )
    torch.testing.assert_close(out, ref, atol=0.25, rtol=0.05)
