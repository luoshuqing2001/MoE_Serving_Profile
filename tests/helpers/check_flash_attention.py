import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch

from moe_serving.efficient_kernels import flash_attention

for qlen, klen in [(7, 7), (1, 9)]:
    torch.manual_seed(1)
    q = torch.randn(2, 4, qlen, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 2, klen, 64, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones(2, klen, device="cuda", dtype=torch.long)
    mask[0, :2] = 0
    sink = torch.randn(4, device="cuda")
    out, _ = flash_attention(None, q, k, v, mask, scaling=0.125, s_aux=sink)
    logits = q.float() @ k.repeat_interleave(2, 1).float().transpose(-1, -2) * 0.125
    valid = (
        torch.arange(klen, device="cuda")[None, :]
        <= torch.arange(klen - qlen, klen, device="cuda")[:, None]
    )[None, None] & mask[:, None, None, :].bool()
    logits = logits.masked_fill(~valid, -float("inf"))
    p = torch.softmax(
        torch.cat([logits, sink[None, :, None, None].expand(2, -1, qlen, 1)], -1), -1
    )[..., :-1]
    ref = (p @ v.repeat_interleave(2, 1).float()).transpose(1, 2)
    diff = (out - ref).abs()[mask[:, -qlen:].bool()].max().item()
    print(qlen, klen, diff)
    assert diff < 0.02
