"""Native arithmetic order for model-parallel reference inference."""

from __future__ import annotations

import torch


def validate_numerics(numerics):
    if numerics not in {"reference", "fast", "efficient"}:
        raise ValueError("numerics must be reference, fast, or efficient")


def ordered_expert_merge(hidden, indices, weights, parallel, local_count, project):
    """Broadcast each expert contribution, then accumulate in global expert order.

    A BF16 sum of rank-local partial sums differs from the native eager expert
    loop. All ranks have identical router decisions; only the owning rank runs
    each expert. This bounds temporary memory by one expert's selected tokens.
    """
    result = torch.zeros_like(hidden)
    for expert in torch.unique(indices, sorted=True).tolist():
        # Native one_hot(...).permute(E, K, T) enumerates slots before tokens.
        slots, tokens = torch.where(indices.T == expert)
        owner, local_id = divmod(expert, local_count)
        if parallel.rank == owner:
            values = project(local_id, hidden[tokens])
            contribution = (values * weights[tokens, slots, None]).to(hidden.dtype)
        else:
            contribution = torch.empty(
                (tokens.numel(), hidden.shape[-1]),
                device=hidden.device,
                dtype=hidden.dtype,
            )
        parallel.broadcast(contribution, owner)
        result.index_add_(0, tokens, contribution)
    return result


def reference_attention(module, query, key, value, attention_mask, **kwargs):
    """Use native full-head GEMM shapes while retaining head-local KV storage.

    BF16 batched GEMM can round differently when the head batch dimension is
    sharded. Gather for the native eager operation on the TP leader and scatter
    the resulting heads. Q/K/V projections and KV cache remain partitioned.
    """
    from types import SimpleNamespace

    parallel = module._reference_parallel

    def gather_heads(tensor):
        full = parallel.gather(tensor, dim=1)
        # Preserve native B,S,H,D projection storage, including a local H=1.
        # Decoded cache concatenations instead have contiguous B,H,S,D storage.
        if parallel.size > 1 and tensor.stride(1) == tensor.shape[-1]:
            full = full.transpose(1, 2).contiguous().transpose(1, 2)
        return full

    full_query = gather_heads(query)
    full_key = gather_heads(key)
    full_value = gather_heads(value)
    sinks = parallel.gather(module.sinks, dim=0) if hasattr(module, "sinks") else None
    if parallel.rank == 0:
        proxy = SimpleNamespace(
            num_key_value_groups=module.num_key_value_groups,
            training=module.training,
            sinks=sinks,
        )
        context, weights = module._reference_eager(
            proxy, full_query, full_key, full_value, attention_mask, **kwargs
        )
    else:
        context = weights = None
    batch, heads, length, dim = query.shape
    context = parallel.scatter(
        context,
        (batch, length, heads, dim),
        dim=2,
        dtype=query.dtype,
        device=query.device,
    )
    weights = parallel.scatter(
        weights,
        (batch, heads, length, key.shape[-2]),
        dim=1,
        dtype=query.dtype,
        device=query.device,
    )
    return context, weights


def install_reference_attention(model, parallel, eager):
    """Register both the attention implementation and its causal-mask backend."""
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    name = "moe_parallel_reference"
    ALL_ATTENTION_FUNCTIONS.register(name, reference_attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register(name, ALL_MASK_ATTENTION_FUNCTIONS["eager"])
    model.config._attn_implementation = name
    for block in model.model.layers:
        block.self_attn._reference_parallel = parallel
        block.self_attn._reference_eager = eager
