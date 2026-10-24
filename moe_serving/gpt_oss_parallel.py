"""GPT-OSS TP/EP inference using local Transformers 5.20.0.

Efficient mode runs head-local FlashAttention 3 and grouped packed MXFP4 GEMM.
Reference mode preserves eager GEMM shapes and expert accumulation order for
numerical diagnostics. TP and EP share ranks and replicate the token stream.
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from torch import nn

from moe_serving.parallel_numerics import (
    install_reference_attention,
    ordered_expert_merge,
    validate_numerics,
)

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
        self.start_rank = rank - self.rank
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

    def broadcast(self, tensor, owner=0):
        if self.size > 1:
            dist.broadcast(tensor, src=self.start_rank + owner, group=self.group)
        return tensor

    def gather(self, tensor, dim):
        if self.size == 1:
            return tensor
        pieces = [
            torch.empty_like(tensor, memory_format=torch.contiguous_format)
            for _ in range(self.size)
        ]
        dist.all_gather(pieces, tensor.contiguous(), group=self.group)
        return torch.cat(pieces, dim=dim)

    def gather_columns(self, tensor):
        return self.gather(tensor, dim=-1)

    def scatter(self, tensor, shape, *, dim, dtype, device):
        if self.size == 1:
            return tensor
        pieces = (
            [piece.contiguous() for piece in tensor.chunk(self.size, dim=dim)]
            if self.rank == 0
            else None
        )
        result = torch.empty(shape, dtype=dtype, device=device)
        dist.scatter(result, scatter_list=pieces, src=self.start_rank, group=self.group)
        return result


class HeadParallelNorm(nn.Module):
    """Keep native head-normalization reduction geometry in reference mode."""

    def __init__(self, norm, parallel):
        super().__init__()
        self.weight = norm.weight
        self.variance_epsilon = norm.variance_epsilon
        self.parallel = parallel
        self._native_forward = type(norm).forward

    def forward(self, hidden):
        full = self.parallel.gather(hidden, dim=2)
        output = self._native_forward(self, full) if self.parallel.rank == 0 else None
        return self.parallel.scatter(
            output, hidden.shape, dim=2, dtype=hidden.dtype, device=hidden.device
        )


class ColumnParallelLinear(nn.Module):
    """Keep weight rows sharded; reproduce the native full-width projection."""

    def __init__(self, linear, parallel):
        super().__init__()
        self.weight, self.bias = linear.weight, linear.bias
        self.parallel = parallel
        self.in_features = linear.in_features
        self.out_features = linear.out_features // parallel.size

    def forward(self, hidden):
        weight = self.parallel.gather(self.weight, dim=0)
        bias = self.parallel.gather(self.bias, dim=0) if self.bias is not None else None
        full = F.linear(hidden, weight, bias) if self.parallel.rank == 0 else None
        return self.parallel.scatter(
            full,
            (*hidden.shape[:-1], self.out_features),
            dim=-1,
            dtype=hidden.dtype,
            device=hidden.device,
        )


class RowParallelLinear(nn.Module):
    """Sharded input columns, with a native full projection or fast partial sums."""

    def __init__(self, linear, parallel, numerics="reference"):
        super().__init__()
        self.weight, self.bias = linear.weight, linear.bias
        self.parallel, self.numerics = parallel, numerics

    def forward(self, hidden):
        if self.numerics == "reference":
            full_hidden = self.parallel.gather_columns(hidden)
            full_weight = self.parallel.gather(self.weight, dim=1)
            if self.parallel.rank == 0:
                output = F.linear(full_hidden, full_weight, self.bias)
            else:
                output = torch.empty(
                    (*hidden.shape[:-1], self.weight.shape[0]),
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            return self.parallel.broadcast(output)
        output = F.linear(hidden, self.weight)
        self.parallel.reduce(output)
        return output if self.bias is None else output + self.bias


class ParallelExperts(nn.Module):
    def __init__(self, config, parallel, storage="mxfp4", numerics="reference"):
        super().__init__()
        self.parallel, self.storage, self.numerics = parallel, storage, numerics
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
        )[0].contiguous()

    def project(self, expert, hidden):
        gate_up = hidden @ self.projection("gate_up_proj", expert, hidden.dtype)
        gate_up = gate_up + self.gate_up_proj_bias[expert]
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        activated = (up + 1) * glu
        values = activated @ self.projection("down_proj", expert, hidden.dtype)
        return values + self.down_proj_bias[expert]

    def forward(self, hidden_states, router_indices, routing_weights):
        if self.numerics == "efficient":
            from moe_serving.efficient_kernels import grouped_mxfp4

            return grouped_mxfp4(self, hidden_states, router_indices, routing_weights)
        if self.numerics == "reference":
            return ordered_expert_merge(
                hidden_states,
                router_indices,
                routing_weights,
                self.parallel,
                self.num_experts,
                self.project,
            )
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
        self._copy_events = []
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
                    "expert_indices": self._copy(indices.detach().reshape(shape)),
                    "routing_weights": self._copy(
                        scores.detach().reshape(shape).float()
                    ),
                }
            )

        return capture

    def _copy(self, value):
        if value.device.type != "cuda":
            return value.cpu()
        # Pinned DMA copies enqueue on the current stream without a CPU barrier
        # at every router. close() waits before any JSON serialization.
        target = torch.empty_like(value, device="cpu", pin_memory=True)
        target.copy_(value, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._copy_events.append(event)
        return target

    def close(self):
        for event in self._copy_events:
            event.synchronize()
        self._copy_events.clear()
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
    numerics="reference",
    progress_callback=None,
):
    """Read only owned slices; never materialize the full model on any rank."""
    validate_numerics(numerics)
    ensure_transformers_5200()
    from transformers import GptOssConfig, GptOssForCausalLM
    from transformers.integrations.mxfp4 import _convert_moe_packed_tensors

    if expert_storage not in {"mxfp4", "bf16"}:
        raise ValueError("expert_storage must be mxfp4 or bf16")
    if tp_size < 1 or ep_size < 1:
        raise ValueError("TP and EP sizes must be positive")
    checkpoint, device = Path(checkpoint), torch.device(device)
    if numerics == "efficient":
        from moe_serving.efficient_kernels import check_runtime

        if progress_callback:
            progress_callback("Checking FlashAttention / MXFP4 kernels")
        check_runtime("gpt_oss", device, dtype, expert_storage)
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
    config._parallel_numerics = numerics
    with torch.device("meta"):
        model = GptOssForCausalLM(config)
    for block in model.model.layers:
        block.mlp.experts = ParallelExperts(config, ep, expert_storage, numerics)
        block.self_attn.o_proj = RowParallelLinear(block.self_attn.o_proj, tp, numerics)
        if numerics == "reference":
            for projection in ("q_proj", "k_proj", "v_proj"):
                setattr(
                    block.self_attn,
                    projection,
                    ColumnParallelLinear(getattr(block.self_attn, projection), tp),
                )
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
        load_started = last_load_log = time.monotonic()
        if progress_callback:
            progress_callback(f"Loading checkpoint tensors: 0/{len(expected)}")
        for tensor_index, name in enumerate(sorted(expected), 1):
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
                    attr,
                    value.to(
                        torch.float32
                        if numerics == "efficient" and attr.endswith("bias")
                        else dtype
                    )
                    if value.is_floating_point()
                    else value,
                )
            else:
                module._parameters[attr] = nn.Parameter(
                    value.to(dtype), requires_grad=False
                )
            now = time.monotonic()
            if progress_callback and (
                now - last_load_log >= 5 or tensor_index == len(expected)
            ):
                progress_callback(
                    f"Checkpoint tensors {tensor_index}/{len(expected)} "
                    f"({100 * tensor_index / len(expected):.1f}%), "
                    f"elapsed={now - load_started:.1f}s"
                )
                last_load_log = now
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
    if numerics == "reference":
        from transformers.models.gpt_oss.modeling_gpt_oss import eager_attention_forward

        install_reference_attention(model, tp, eager_attention_forward)
    if numerics == "efficient":
        if torch.device(device).type != "cuda" or expert_storage != "mxfp4":
            raise ValueError("efficient mode requires CUDA and packed MXFP4 experts")
        from moe_serving.efficient_kernels import install_flash_attention, prepare_mxfp4

        for layer_index, block in enumerate(model.model.layers, 1):
            if progress_callback:
                progress_callback(
                    f"Preparing grouped MXFP4 layer {layer_index}/{len(model.model.layers)}"
                )
            prepare_mxfp4(block.mlp.experts, config)
        install_flash_attention(model)
    return model.eval()


@torch.inference_mode()
def greedy_generate(
    model,
    input_ids,
    attention_mask,
    max_new_tokens,
    recorder=None,
    progress_callback=None,
    progress_interval=10.0,
):
    """Synchronized greedy generation, with native per-rank local-head KV caches."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    generated, current, cache = input_ids, input_ids, None
    finished = torch.zeros(
        input_ids.shape[0], dtype=torch.bool, device=input_ids.device
    )
    eos = model.config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    started = last_progress = time.monotonic()
    initial_tokens = attention_mask.sum() if progress_callback else None
    for step in range(max_new_tokens):
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
        all_finished = bool(finished.all())
        now = time.monotonic()
        if progress_callback and (
            step == 0
            or now - last_progress >= progress_interval
            or all_finished
            or step + 1 == max_new_tokens
        ):
            progress_callback(
                {
                    "step": step + 1,
                    "max_new_tokens": max_new_tokens,
                    "finished": int(finished.sum()),
                    "batch_size": input_ids.shape[0],
                    "generated_tokens": int(attention_mask.sum() - initial_tokens),
                    "elapsed_seconds": now - started,
                }
            )
            last_progress = now
        if all_finished:
            break
        current = token[:, None]
    return generated
