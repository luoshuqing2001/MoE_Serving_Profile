import torch

from moe_serving.grouped_fp8 import patch_grouped_fp8_kernel


def test_grouped_fp8_matches_the_2d_kernel():
    if not torch.cuda.is_available():
        return
    patch_grouped_fp8_kernel()
    from kernels import get_kernel

    kernel = get_kernel("kernels-community/finegrained-fp8", version=4)
    device = "cuda"
    block = [128, 128]
    experts = torch.randn(2, 256, 256, device=device).to(torch.float8_e4m3fn)
    scales = torch.rand(2, 2, 2, device=device) + 0.1
    hidden = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
    tokens = torch.tensor([3, 1], device=device, dtype=torch.int32)
    offsets = torch.cumsum(tokens, dim=0).to(torch.int32)

    grouped = kernel.matmul_grouped(hidden, experts, scales, offsets, tokens, block)
    reference = kernel.matmul_2d(hidden[:3], experts[0], scales[0], block)
    assert grouped.shape == (4, 256)
    assert torch.allclose(grouped[:3].float(), reference.float(), atol=1e-2, rtol=1e-2)
