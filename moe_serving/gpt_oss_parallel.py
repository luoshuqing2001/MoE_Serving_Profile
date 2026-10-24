"""GPT-OSS inference using the local Transformers 5.20.0 implementation.

All ranks run the same token stream. Attention heads and experts are sharded
independently; each sublayer reduces its partial output before the residual.
This replicated-token EP layout needs an all-reduce, rather than token all-to-all.
Packed MXFP4 experts remain packed between calls (reference, not fused, kernels).
"""

from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "transformers-5.20.0" / "src"


def ensure_transformers_5200():
    if not (SOURCE_ROOT / "transformers/models/gpt_oss/modeling_gpt_oss.py").is_file():
        raise FileNotFoundError(f"Missing Transformers source: {SOURCE_ROOT}")
    loaded = sys.modules.get("transformers")
    if (
        loaded is not None
        and SOURCE_ROOT.resolve() not in Path(loaded.__file__).resolve().parents
    ):
        raise RuntimeError(
            "Import gpt_oss_parallel before importing another Transformers installation"
        )
    if str(SOURCE_ROOT) not in sys.path:
        sys.path.insert(0, str(SOURCE_ROOT))
    import transformers

    if transformers.__version__ != "5.20.0":
        raise RuntimeError(
            f"Expected Transformers 5.20.0, got {transformers.__version__}"
        )
    return SOURCE_ROOT


class ParallelGroup:
    """Contiguous subgroups, with identical replicated inputs in every group."""

    def __init__(self, size: int):
        world = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        if size < 1 or world % size:
            raise ValueError(f"Parallel size {size} must divide world size {world}")
        self.size, self.rank, self.group = size, rank % size, None
        if size > 1:
            # Every rank creates groups in the same order, including nonmembers.
            for start in range(0, world, size):
                group = dist.new_group(list(range(start, start + size)))
                if start <= rank < start + size:
                    self.group = group

    def reduce(self, tensor):
        if self.size > 1:
            dist.all_reduce(tensor, group=self.group)
        return tensor


class RowParallelLinear(nn.Module):
    """Local input columns; add the replicated output bias exactly once."""

    def __init__(self, linear, parallel):
        super().__init__()
        self.weight, self.bias = linear.weight, linear.bias
        self.parallel = parallel

    def forward(self, hidden):
        output = F.linear(hidden, self.weight)
        self.parallel.reduce(output)
        return output if self.bias is None else output + self.bias


class ParallelExperts(nn.Module):
    def __init__(self, config, parallel, storage="mxfp4"):
        super().__init__()
        self.parallel, self.storage = parallel, storage
        if config.num_local_experts % parallel.size:
            raise ValueError("Number of experts must be divisible by EP size")
        self.num_experts = config.num_local_experts // parallel.size
        self.start = self.num_experts * parallel.rank
        self.alpha, self.limit = config.swiglu_alpha, config.swiglu_limit

    def projection(self, name, expert, dtype):
        if self.storage == "bf16":
            return getattr(self, name)[expert]
        from transformers.integrations.mxfp4 import _convert_moe_packed_tensors

        return _convert_moe_packed_tensors(
            getattr(self, name + "_blocks")[expert : expert + 1],
            getattr(self, name + "_scales")[expert : expert + 1],
            dtype=dtype,
            rows_per_chunk=65536,
        )[0]

    def forward(self, hidden_states, router_indices, routing_weights):
        output = torch.zeros_like(hidden_states)
        # Router IDs stay global in the trace, even though weights are local.
        for expert in range(self.num_experts):
            tokens, slots = torch.where(router_indices == self.start + expert)
            if tokens.numel() == 0:
                continue
            gate_up = hidden_states[tokens] @ self.projection(
                "gate_up_proj", expert, hidden_states.dtype
            )
            gate_up = gate_up + self.gate_up_proj_bias[expert]
            gate, up = gate_up[..., ::2], gate_up[..., 1::2]
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            glu = gate * torch.sigmoid(gate * self.alpha)
            activated = (up + 1) * glu
            values = activated @ self.projection(
                "down_proj", expert, hidden_states.dtype
            )
            values = values + self.down_proj_bias[expert]
            output.index_add_(
                0,
                tokens,
                (values * routing_weights[tokens, slots, None]).to(output.dtype),
            )
        return self.parallel.reduce(output)


class RoutingRecorder:
    """CPU records per forward and layer; router outputs are captured directly."""

    def __init__(self, model):
        self.records, self.handles = [], []
        self.step = -1
        self.context = {}
        for layer, block in enumerate(model.model.layers):
            self.handles.append(
                block.mlp.router.register_forward_hook(self._hook(layer))
            )

    def begin(self, input_ids, attention_mask, position_ids):
        self.step += 1
        self.context = {
            "step": self.step,
            "phase": "prefill" if self.step == 0 else "decode",
            "input_ids": input_ids.detach().cpu().clone(),
            "position_ids": position_ids.detach().cpu().clone(),
            "valid_tokens": attention_mask[:, -input_ids.shape[1] :]
            .bool()
            .cpu()
            .clone(),
        }

    def _hook(self, layer):
        def capture(module, inputs, output):
            _, scores, indices = output
            shape = (*self.context["input_ids"].shape, module.top_k)
            self.records.append(
                {
                    **self.context,
                    "layer": layer,
                    "expert_indices": indices.detach().reshape(shape).cpu(),
                    "routing_weights": scores.detach().reshape(shape).float().cpu(),
                }
            )

        return capture

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _slice_attention(name, shape, config, tp):
    slices = [slice(None)] * len(shape)
    for proj, heads in (
        ("q_proj", config.num_attention_heads),
        ("k_proj", config.num_key_value_heads),
        ("v_proj", config.num_key_value_heads),
    ):
        if f".self_attn.{proj}." in name:
            width = heads * config.head_dim // tp.size
            slices[0] = slice(tp.rank * width, (tp.rank + 1) * width)
    if name.endswith(".self_attn.o_proj.weight"):
        width = config.num_attention_heads * config.head_dim // tp.size
        slices[1] = slice(tp.rank * width, (tp.rank + 1) * width)
    if name.endswith(".self_attn.sinks"):
        width = config.num_attention_heads // tp.size
        slices[0] = slice(tp.rank * width, (tp.rank + 1) * width)
    return tuple(slices)


def load_parallel_model(
    checkpoint,
    tp_size=1,
    ep_size=1,
    device="cpu",
    dtype=torch.bfloat16,
    expert_storage="mxfp4",
):
    """Read only owned slices; never materialize the full model on any rank."""
    ensure_transformers_5200()
    from transformers import GptOssConfig, GptOssForCausalLM
    from transformers.integrations.mxfp4 import _convert_moe_packed_tensors

    if expert_storage not in {"mxfp4", "bf16"}:
        raise ValueError("expert_storage must be mxfp4 or bf16")
    if tp_size < 1 or ep_size < 1:
        raise ValueError("TP and EP sizes must be positive")
    checkpoint, device = Path(checkpoint), torch.device(device)
    config = GptOssConfig.from_pretrained(checkpoint, local_files_only=True)
    if config.num_attention_heads % tp_size or config.num_key_value_heads % tp_size:
        raise ValueError(
            "Attention and KV head counts must both be divisible by TP size"
        )
    if config.num_local_experts % ep_size:
        raise ValueError("Expert count must be divisible by EP size")
    tp, ep = ParallelGroup(tp_size), ParallelGroup(ep_size)
    # Eager implements attention sinks and sliding windows without extra kernels.
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    with torch.device("meta"):
        model = GptOssForCausalLM(config)
    for block in model.model.layers:
        block.mlp.experts = ParallelExperts(config, ep, expert_storage)
        block.self_attn.o_proj = RowParallelLinear(block.self_attn.o_proj, tp)
    index_file = checkpoint / "model.safetensors.index.json"
    if index_file.exists():
        weight_map = json.loads(index_file.read_text())["weight_map"]
    else:
        with safe_open(checkpoint / "model.safetensors", framework="pt") as handle:
            weight_map = dict.fromkeys(handle.keys(), "model.safetensors")
    # A meta model contains nonpersistent RoPE buffers; rebuild those on device.
    from transformers.models.gpt_oss.modeling_gpt_oss import GptOssRotaryEmbedding

    with torch.device(device):
        model.model.rotary_emb = GptOssRotaryEmbedding(config)
    expected = set(dict(model.named_parameters()))
    # Experts are custom modules with buffers populated below, checked separately.
    expert_prefixes = [
        f"model.layers.{i}.mlp.experts" for i in range(config.num_hidden_layers)
    ]
    expected.update(
        f"{prefix}.{suffix}"
        for prefix in expert_prefixes
        for suffix in (
            "gate_up_proj_bias",
            "down_proj_bias",
            "gate_up_proj_blocks",
            "gate_up_proj_scales",
            "down_proj_blocks",
            "down_proj_scales",
        )
    )
    # Nonquantized checkpoints are also useful for parity tests and debugging.
    for prefix in expert_prefixes:
        if prefix + ".gate_up_proj" in weight_map:
            for proj in ("gate_up_proj", "down_proj"):
                expected.discard(f"{prefix}.{proj}_blocks")
                expected.discard(f"{prefix}.{proj}_scales")
                expected.add(f"{prefix}.{proj}")
            model.get_submodule(prefix).storage = "bf16"
    missing = expected - weight_map.keys()
    if missing:
        raise ValueError(f"Missing checkpoint tensors: {sorted(missing)}")
    with ExitStack() as stack:
        handles = {
            file: stack.enter_context(safe_open(checkpoint / file, framework="pt"))
            for file in set(weight_map.values())
        }
        for name in sorted(expected):
            view = handles[weight_map[name]].get_slice(name)
            slices = _slice_attention(name, view.get_shape(), config, tp)
            if ".mlp.experts." in name:
                count = config.num_local_experts // ep.size
                slices = (slice(ep.rank * count, (ep.rank + 1) * count),) + slices[1:]
            value = view[slices].to(device=device)
            module_name, attr = name.rsplit(".", 1)
            module = model.get_submodule(module_name)
            if ".mlp.experts." in name:
                module.register_buffer(
                    attr, value.to(dtype) if value.is_floating_point() else value
                )
            else:
                module._parameters[attr] = nn.Parameter(
                    value.to(dtype), requires_grad=False
                )
    if expert_storage == "bf16":
        for prefix in expert_prefixes:
            module = model.get_submodule(prefix)
            for proj in ("gate_up_proj", "down_proj"):
                if hasattr(module, proj + "_blocks"):
                    # Dequantize one expert at a time to bound transient memory.
                    weights = torch.empty(
                        module.num_experts,
                        config.hidden_size
                        if proj == "gate_up_proj"
                        else config.intermediate_size,
                        2 * config.intermediate_size
                        if proj == "gate_up_proj"
                        else config.hidden_size,
                        device=device,
                        dtype=dtype,
                    )
                    for i in range(module.num_experts):
                        weights[i] = _convert_moe_packed_tensors(
                            getattr(module, proj + "_blocks")[i : i + 1],
                            getattr(module, proj + "_scales")[i : i + 1],
                            dtype=dtype,
                            rows_per_chunk=65536,
                        )[0]
                    module.register_buffer(proj, weights)
                    delattr(module, proj + "_blocks")
                    delattr(module, proj + "_scales")
    for block in model.model.layers:
        attn = block.self_attn
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            proj.out_features = proj.weight.shape[0]
    leftovers = [
        name
        for name, tensor in list(model.named_parameters()) + list(model.named_buffers())
        if tensor.is_meta
    ]
    if leftovers:
        raise RuntimeError(f"Unmaterialized tensors: {leftovers}")
    return model.eval()


@torch.inference_mode()
def greedy_generate(model, input_ids, attention_mask, max_new_tokens, recorder=None):
    """Synchronized greedy generation, with native per-rank local-head KV caches."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    generated, current, cache = input_ids, input_ids, None
    finished = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )
    eos = model.config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    for _ in range(max_new_tokens):
        positions = attention_mask.long().cumsum(-1) - 1
        positions.masked_fill_(attention_mask == 0, 0)
        positions = positions[:, -current.shape[1] :]
        if recorder is not None:
            recorder.begin(current, attention_mask, positions)
        result = model(
            input_ids=current,
            attention_mask=attention_mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        cache = result.past_key_values
        token = result.logits[:, -1].argmax(-1)
        if dist.is_initialized():
            # Keep all collective participants on the same control-flow path.
            dist.broadcast(token, src=0)
        pad = model.config.pad_token_id
        token = torch.where(finished, pad if pad is not None else 0, token)
        generated = torch.cat((generated, token[:, None]), dim=1)
        attention_mask = torch.cat(
            (attention_mask, (~finished).to(attention_mask.dtype)[:, None]), dim=1
        )
        for stop in eos:
            finished |= token == stop
        if bool(finished.all()):
            break
        current = token[:, None]
    return generated
