"""Qwen3 MoE TP/EP with rank-local checkpoint reads and block FP8 experts.

Efficient mode uses FlashAttention 3 and grouped FP8 GEMM with dynamic FP8
activations. Reference mode dequantizes selected experts for eager diagnostics.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from moe_serving.gpt_oss_parallel import (
    ColumnParallelLinear,
    HeadParallelNorm,
    ParallelGroup,
    RowParallelLinear,
    _slice_attention,
    ensure_transformers_5200,
)
from moe_serving.gpt_oss_parallel import (
    RoutingRecorder as BaseRoutingRecorder,
)
from moe_serving.parallel_numerics import (
    install_reference_attention,
    ordered_expert_merge,
    validate_numerics,
)


class RoutingRecorder(BaseRoutingRecorder):
    def __init__(self, model):
        self.records, self.handles = [], []
        self._copy_events = []
        self.step, self.context = -1, {}
        for layer, block in enumerate(model.model.layers):
            if hasattr(block.mlp, "gate"):
                self.handles.append(
                    block.mlp.gate.register_forward_hook(self._hook(layer))
                )


def dequantize_blocks(weight, scales, block_size, dtype):
    """Checkpoint weight_scale_inv contains multiplicative dequant scales."""
    rows, cols = weight.shape
    br, bc = block_size
    expanded = scales.float().repeat_interleave(br, 0).repeat_interleave(bc, 1)
    return (weight.float() * expanded[:rows, :cols]).to(dtype)


class ExpertProjection(nn.Module):
    def __init__(self, weight, scales=None, block_size=(128, 128)):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.block_size = block_size

    def materialize(self, dtype):
        weight = self.weight
        if self.scales is not None:
            weight = dequantize_blocks(weight, self.scales, self.block_size, dtype)
        return weight

    def forward(self, hidden):
        return F.linear(hidden, self.materialize(hidden.dtype))


class ParallelExperts(nn.Module):
    def __init__(self, config, parallel, numerics="reference"):
        super().__init__()
        from transformers.activations import ACT2FN

        self.parallel, self.numerics = parallel, numerics
        self.num_experts = config.num_experts // parallel.size
        self.start = self.num_experts * parallel.rank
        self.local = nn.ModuleList([nn.ModuleDict() for _ in range(self.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def project(self, expert, hidden):
        projections = self.local[expert]
        gate_up = torch.cat(
            (
                projections["gate_proj"].materialize(hidden.dtype),
                projections["up_proj"].materialize(hidden.dtype),
            ),
            dim=0,
        )
        gate, up = F.linear(hidden, gate_up).chunk(2, dim=-1)
        return projections["down_proj"](self.act_fn(gate) * up)

    def forward(self, hidden, indices, weights):
        if self.numerics == "efficient":
            from moe_serving.efficient_kernels import grouped_fp8

            return grouped_fp8(self, hidden, indices, weights)
        if self.numerics == "reference":
            return ordered_expert_merge(
                hidden, indices, weights, self.parallel, self.num_experts, self.project
            )
        output = torch.zeros_like(hidden)
        for local_id, projections in enumerate(self.local):
            tokens, slots = torch.where(indices == self.start + local_id)
            if tokens.numel() == 0:
                continue
            selected = hidden[tokens]
            activated = self.act_fn(projections["gate_proj"](selected)) * projections[
                "up_proj"
            ](selected)
            values = projections["down_proj"](activated)
            output.index_add_(
                0, tokens, (values * weights[tokens, slots, None]).to(output.dtype)
            )
        return self.parallel.reduce(output)


def load_parallel_model(
    checkpoint,
    tp_size=1,
    ep_size=1,
    device="cpu",
    dtype=torch.bfloat16,
    expert_storage="fp8",
    numerics="reference",
):
    """Load legacy per-expert FP8 or native fused nonquantized Qwen3 weights.

    TP and EP share ranks, as in the GPT-OSS backend. Each rank reads only its
    attention heads and owned experts. Embeddings, router and norms replicate.
    """
    validate_numerics(numerics)
    ensure_transformers_5200()
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeRotaryEmbedding

    if expert_storage not in {"fp8", "bf16"}:
        raise ValueError("expert_storage must be fp8 or bf16")
    if tp_size < 1 or ep_size < 1:
        raise ValueError("TP and EP sizes must be positive")
    checkpoint, device = Path(checkpoint), torch.device(device)
    if numerics == "efficient":
        from moe_serving.efficient_kernels import check_runtime

        check_runtime("qwen3_moe", device, dtype, expert_storage)
    config = Qwen3MoeConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.model_type != "qwen3_moe":
        raise ValueError("Expected a qwen3_moe checkpoint")
    if config.num_attention_heads % tp_size or config.num_key_value_heads % tp_size:
        raise ValueError(
            "Attention and KV head counts must both be divisible by TP size"
        )
    if config.num_experts % ep_size:
        raise ValueError("Expert count must be divisible by EP size")
    quant = getattr(config, "quantization_config", {}) or {}
    if quant and (
        quant.get("quant_method") != "fp8" or quant.get("fmt", "e4m3") != "e4m3"
    ):
        raise ValueError("Only block-scaled e4m3 FP8 checkpoints are supported")
    block_size = tuple(quant.get("weight_block_size", (128, 128)))
    if len(block_size) != 2 or min(block_size) <= 0:
        raise ValueError("Expected a positive two-dimensional weight_block_size")
    tp, ep = ParallelGroup(tp_size), ParallelGroup(ep_size)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    config._parallel_numerics = numerics
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)
    for block in model.model.layers:
        if hasattr(block.mlp, "experts"):
            block.mlp.experts = ParallelExperts(config, ep, numerics)
        block.self_attn.o_proj = RowParallelLinear(block.self_attn.o_proj, tp, numerics)
        if numerics == "reference":
            block.self_attn.q_norm = HeadParallelNorm(block.self_attn.q_norm, tp)
            block.self_attn.k_norm = HeadParallelNorm(block.self_attn.k_norm, tp)
        if numerics == "reference":
            for projection in ("q_proj", "k_proj", "v_proj"):
                setattr(
                    block.self_attn,
                    projection,
                    ColumnParallelLinear(getattr(block.self_attn, projection), tp),
                )
    with torch.device(device):
        model.model.rotary_emb = Qwen3MoeRotaryEmbedding(config)
    index = checkpoint / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
    else:
        with safe_open(checkpoint / "model.safetensors", framework="pt") as handle:
            weight_map = dict.fromkeys(handle.keys(), "model.safetensors")

    with ExitStack() as stack:
        # Open lazily: a rank need not touch shards containing only remote experts.
        handles = {}

        def view(name):
            if name not in weight_map:
                raise ValueError(f"Missing checkpoint tensor: {name}")
            file = weight_map[name]
            if file not in handles:
                handles[file] = stack.enter_context(
                    safe_open(checkpoint / file, framework="pt")
                )
            return handles[file].get_slice(name)

        def projection(name, slices=None, preserve_fp8=False):
            source = view(name)
            shape = source.get_shape()
            slices = slices or tuple(slice(None) for _ in shape)
            scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
            if scale_name not in weight_map:
                value = source[slices]
                if value.dtype in {torch.float8_e4m3fn, torch.float8_e5m2}:
                    raise ValueError(f"Missing FP8 scales for {name}")
                return ExpertProjection(value.to(device=device, dtype=dtype))
            if len(shape) != 2:
                raise ValueError(f"Expected a 2D FP8 projection: {name}")
            starts, stops = zip(
                *(sl.indices(size)[:2] for sl, size in zip(slices, shape))
            )
            br, bc = block_size
            r0, c0 = starts[0] // br, starts[1] // bc
            r1, c1 = (stops[0] + br - 1) // br, (stops[1] + bc - 1) // bc
            value = source[
                r0 * br : min(r1 * br, shape[0]), c0 * bc : min(c1 * bc, shape[1])
            ].to(device)
            scale_view = view(scale_name)
            expected = [(shape[0] + br - 1) // br, (shape[1] + bc - 1) // bc]
            if scale_view.get_shape() != expected:
                raise ValueError(f"Invalid FP8 scale shape for {name}")
            scales = scale_view[r0:r1, c0:c1].to(device)
            if preserve_fp8:
                return ExpertProjection(value, scales, block_size)
            value = dequantize_blocks(value, scales, block_size, dtype)
            value = value[
                starts[0] - r0 * br : stops[0] - r0 * br,
                starts[1] - c0 * bc : stops[1] - c0 * bc,
            ].contiguous()
            return ExpertProjection(value)

        for name, parameter in list(model.named_parameters()):
            source_name = name
            if (
                source_name not in weight_map
                and name == "lm_head.weight"
                and config.tie_word_embeddings
            ):
                source_name = "model.embed_tokens.weight"
            shape = view(source_name).get_shape()
            slices = _slice_attention(name, shape, config, tp)
            loaded = projection(source_name, slices)
            module_name, attr = name.rsplit(".", 1)
            model.get_submodule(module_name)._parameters[attr] = nn.Parameter(
                loaded.weight, requires_grad=False
            )

        for layer, block in enumerate(model.model.layers):
            if not isinstance(
                block.mlp.experts if hasattr(block.mlp, "experts") else None,
                ParallelExperts,
            ):
                continue
            experts = block.mlp.experts
            prefix = f"model.layers.{layer}.mlp.experts"
            for local_id, projections in enumerate(experts.local):
                global_id = experts.start + local_id
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    name = f"{prefix}.{global_id}.{proj}.weight"
                    if name in weight_map:
                        projections[proj] = projection(
                            name, preserve_fp8=expert_storage == "fp8"
                        )
                    else:
                        # Transformers 5.20 saves unquantized experts fused.
                        fused = "down_proj" if proj == "down_proj" else "gate_up_proj"
                        source = view(f"{prefix}.{fused}")
                        rows = slice(None)
                        if proj != "down_proj":
                            width = config.moe_intermediate_size
                            start = 0 if proj == "gate_proj" else width
                            rows = slice(start, start + width)
                        value = source[global_id, rows, :].to(
                            device=device, dtype=dtype
                        )
                        projections[proj] = ExpertProjection(value)
    for block in model.model.layers:
        for proj in (
            block.self_attn.q_proj,
            block.self_attn.k_proj,
            block.self_attn.v_proj,
        ):
            proj.out_features = proj.weight.shape[0]
    leftovers = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.is_meta
    ]
    if leftovers:
        raise RuntimeError(f"Unmaterialized tensors: {leftovers}")
    if numerics == "reference":
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            eager_attention_forward,
        )

        install_reference_attention(model, tp, eager_attention_forward)
    if numerics == "efficient":
        if device.type != "cuda" or expert_storage != "fp8":
            raise ValueError("efficient Qwen mode requires CUDA and FP8 experts")
        from moe_serving.efficient_kernels import install_flash_attention, prepare_fp8

        for block in model.model.layers:
            if isinstance(getattr(block.mlp, "experts", None), ParallelExperts):
                prepare_fp8(block.mlp.experts)
        install_flash_attention(model)
    return model.eval()
