"""GPU-only FlashAttention 3 and fused grouped MXFP4 execution.

No eager fallback: an unavailable kernel is a startup error.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def mxfp4_kernel():
    return _local_kernel(
        "gpt-oss-triton-kernels", "28e0fa9b31fdada34afa34d266c897a7c6c60d74"
    )


def _local_kernel(name, revision, patch=False):
    import shutil
    from pathlib import Path

    from filelock import FileLock
    from huggingface_hub import snapshot_download
    from kernels import get_local_kernel

    from moe_serving.grouped_fp8 import patch_grouped_fp8_kernel

    target = Path.home() / ".cache" / "moe_serving" / (name + "-" + revision)
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(target) + ".lock"):
        if not (target / ".ready").exists():
            source = snapshot_download(
                "kernels-community/" + name,
                repo_type="kernel",
                revision=revision,
                allow_patterns=["build/torch-cuda/*"],
            )
            shutil.copytree(source, target, dirs_exist_ok=True)
            if patch:
                patch_grouped_fp8_kernel(target)
            (target / ".ready").touch()
    return get_local_kernel(target)


def flash_attention(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling=None,
    sliding_window=None,
    s_aux=None,
    **kwargs,
):
    from flash_attn_interface import flash_attn_varlen_func

    batch, heads, qlen, dim = query.shape
    klen = key.shape[2]
    if attention_mask is None:
        kmask = torch.ones((batch, klen), device=query.device, dtype=torch.bool)
    else:
        if attention_mask.ndim != 2:
            raise ValueError("FlashAttention requires a 2D padding mask")
        kmask = attention_mask[:, -klen:].bool()
    qmask = kmask[:, -qlen:]
    if qlen == 1:
        from flash_attn_interface import flash_attn_with_kvcache

        positions = torch.arange(klen, device=query.device, dtype=torch.int32) + 1
        lengths = (kmask * positions).amax(-1).int()
        leftpad = kmask.int().argmax(-1).int()
        out, lse, *_ = flash_attn_with_kvcache(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            cache_seqlens=lengths,
            cache_leftpad=leftpad,
            softmax_scale=scaling,
            causal=True,
            window_size=(sliding_window - 1, 0) if sliding_window else (-1, -1),
            return_softmax_lse=True,
        )
        if s_aux is not None:
            lse = lse.transpose(1, 2).float()
            factor = torch.exp(lse - torch.logaddexp(lse, s_aux.float()[None, None]))
            out = (out.float() * factor[..., None]).to(query.dtype)
        out = out.masked_fill(~qmask[:, :, None, None], 0)
        return out, None
    q = query.transpose(1, 2).reshape(-1, heads, dim)
    k = key.transpose(1, 2).reshape(-1, key.shape[1], dim)
    v = value.transpose(1, 2).reshape_as(k)
    qi = qmask.flatten().nonzero().flatten()
    ki = kmask.flatten().nonzero().flatten()

    def offsets(mask):
        lengths = mask.sum(-1, dtype=torch.int32)
        return torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))

    out, lse = flash_attn_varlen_func(
        q[qi],
        k[ki],
        v[ki],
        offsets(qmask),
        offsets(kmask),
        qlen,
        klen,
        softmax_scale=scaling,
        causal=True,
        window_size=(sliding_window - 1, 0) if sliding_window else (-1, -1),
        return_attn_probs=True,
    )
    if s_aux is not None:
        # The sink has zero value. Recover its softmax denominator using FA's LSE.
        lse = lse.transpose(0, 1).float()
        factor = torch.exp(lse - torch.logaddexp(lse, s_aux.float()[None]))
        out = (out.float() * factor[..., None]).to(query.dtype)
    result = torch.zeros_like(q)
    result.index_copy_(0, qi, out)
    return result.view(batch, qlen, heads, dim), None


def install_flash_attention(model):
    import flash_attn_interface  # noqa: F401
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    name = "moe_flash_attention_3"
    ALL_ATTENTION_FUNCTIONS.register(name, flash_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(
        name, ALL_MASK_ATTENTION_FUNCTIONS["flash_attention_3"]
    )
    model.config._attn_implementation = name


def prepare_mxfp4(experts, config):
    from transformers.integrations.mxfp4 import swizzle_mxfp4_convertops

    kernel = mxfp4_kernel()
    experts.hidden_size = config.hidden_size
    experts.intermediate_size = config.intermediate_size
    for proj in ("gate_up_proj", "down_proj"):
        swizzle_mxfp4_convertops(
            getattr(experts, proj + "_blocks"),
            getattr(experts, proj + "_scales"),
            experts,
            proj,
            getattr(experts, proj + "_blocks").device,
            kernel,
        )
        delattr(experts, proj + "_blocks")
        delattr(experts, proj + "_scales")
        setattr(experts, proj + "_bias", getattr(experts, proj + "_bias").float())
    experts._grouped_kernel = kernel


def grouped_mxfp4(experts, hidden, indices, weights):
    kernel = experts._grouped_kernel
    routing = kernel.routing
    ids = indices.reshape(-1).int()
    local = (ids >= experts.start) & (ids < experts.start + experts.num_experts)
    # Remote assignments sort after every local expert; fixed-size metadata avoids
    # CPU expert loops and retains global IDs in the recorder's native router hook.
    keys = torch.where(local, ids - experts.start, experts.num_experts)
    order = torch.argsort(keys, stable=True).int()
    inverse = torch.argsort(order).int()
    inverse = torch.where(local, inverse, -1)
    gather = torch.where(local[order.long()], order, -1)
    hist = torch.histc(
        keys.int(), bins=experts.num_experts + 1, min=0, max=experts.num_experts
    )[:-1].int()
    data = routing.RoutingData(
        weights.reshape(-1)[order.long()],
        hist,
        experts.num_experts,
        indices.shape[1],
        routing.compute_expt_data_torch(hist, experts.num_experts, ids.numel()),
    )
    gi = routing.GatherIndx(src_indx=gather, dst_indx=inverse)
    si = routing.ScatterIndx(src_indx=inverse, dst_indx=gather)
    ops = kernel.matmul_ogs
    activation = ops.FusedActivation(
        ops.FnSpecs("swiglu", kernel.swiglu.swiglu_fn, ("alpha", "limit")),
        (experts.alpha, experts.limit),
        2,
    )
    with torch.cuda.device(hidden.device):
        intermediate = ops.matmul_ogs(
            hidden,
            experts.gate_up_proj,
            experts.gate_up_proj_bias,
            data,
            gather_indx=gi,
            precision_config=experts.gate_up_proj_precision_config,
            fused_activation=activation,
        )
        output = ops.matmul_ogs(
            intermediate,
            experts.down_proj,
            experts.down_proj_bias,
            data,
            scatter_indx=si,
            precision_config=experts.down_proj_precision_config,
            gammas=data.gate_scal,
        )
    return experts.parallel.reduce(output)


@lru_cache(maxsize=1)
def fp8_kernel():
    # Triton captures source at decoration: patch the private copy before import.
    return _local_kernel(
        "finegrained-fp8", "0e5764aa5c27d0f005117a92645156481540950a", patch=True
    )


def prepare_fp8(experts):
    experts._grouped_fp8 = fp8_kernel()
    experts.block_size = experts.local[0]["gate_proj"].block_size
    for proj, sources in [
        ("gate_up_proj", ("gate_proj", "up_proj")),
        ("down_proj", ("down_proj",)),
    ]:
        weights, scales = [], []
        for local in experts.local:
            values = [local[name] for name in sources]
            if any(p.scales is None for p in values):
                raise ValueError(
                    "efficient Qwen mode requires block FP8 expert weights"
                )
            if len(sources) > 1 and values[0].weight.shape[0] % experts.block_size[0]:
                raise ValueError("Gate/up boundary must align with the FP8 scale block")
            weights.append(torch.cat([p.weight for p in values], dim=0))
            scales.append(torch.cat([p.scales for p in values], dim=0))
        experts.register_buffer(proj + "_weight", torch.stack(weights))
        experts.register_buffer(proj + "_scale", torch.stack(scales))
        for local in experts.local:
            for name in sources:
                del local[name]
    del experts.local


def grouped_fp8(experts, hidden, indices, weights):
    topk = indices.shape[-1]
    ids = indices.flatten() - experts.start
    valid = (ids >= 0) & (ids < experts.num_experts)
    ids = torch.where(valid, ids, experts.num_experts)
    sorted_ids, perm = torch.sort(ids)
    x = hidden[perm // topk]
    hist = torch.histc(
        sorted_ids.int(), bins=experts.num_experts + 1, min=0, max=experts.num_experts
    )[:-1]
    offsets = hist.cumsum(0, dtype=torch.int32)

    def project(x, name):
        return experts._grouped_fp8.matmul_grouped(
            x,
            getattr(experts, name + "_weight"),
            getattr(experts, name + "_scale"),
            offsets=offsets,
            tokens_per_expert=hist,
            block_size=experts.block_size,
        )

    gate, up = project(x, "gate_up_proj").chunk(2, dim=-1)
    activated = experts.act_fn(gate) * up
    # Unwritten remote rows must never carry NaNs into activation quantization.
    activated.masked_fill_((sorted_ids == experts.num_experts)[:, None], 0)
    values = project(activated, "down_proj")
    values = values * weights.flatten()[perm, None].to(values.dtype)
    values.masked_fill_((sorted_ids == experts.num_experts)[:, None], 0)
    inverse = torch.argsort(perm)
    output = values[inverse].reshape(hidden.shape[0], topk, hidden.shape[-1]).sum(1)
    return experts.parallel.reduce(output)


def check_runtime(family, device, dtype, storage):
    if torch.device(device).type != "cuda" or dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        raise ValueError("efficient mode requires CUDA and BF16/FP16 compute")
    import flash_attn_interface  # noqa: F401

    expected = "mxfp4" if family == "gpt_oss" else "fp8"
    if storage != expected:
        raise ValueError(f"efficient {family} requires {expected} checkpoint storage")
    mxfp4_kernel() if family == "gpt_oss" else fp8_kernel()


def backend_metadata(family, numerics):
    if numerics != "efficient":
        return {"attention": "eager", "experts": "python_reference"}
    return {
        "attention": "flash_attention_3",
        "experts": "triton_matmul_ogs_mxfp4"
        if family == "gpt_oss"
        else "triton_grouped_fp8",
        "expert_parallel": "local_grouped_gemm_then_all_reduce",
        "gpt_oss_kernel_revision": "28e0fa9b31fdada34afa34d266c897a7c6c60d74",
        "qwen_fp8_kernel_revision": "0e5764aa5c27d0f005117a92645156481540950a",
    }
