"""Small Qwen3 checkpoints: compare local TP/EP with the full native model."""

import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from moe_serving.gpt_oss_parallel import greedy_generate
from moe_serving.qwen3_parallel import (
    RoutingRecorder,
    dequantize_blocks,
    ensure_transformers_5200,
    load_parallel_model,
)
from moe_serving.record_gpt_oss import write_sample_records

ensure_transformers_5200()
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM


def fixture(path, packed, kv_heads=2):
    torch.manual_seed(23)
    config = Qwen3MoeConfig(
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=24,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=None,
        attention_bias=True,
        tie_word_embeddings=False,
    )
    config._attn_implementation = config._experts_implementation = "eager"
    model = Qwen3MoeForCausalLM(config).eval()
    for name, parameter in model.named_parameters():
        if name.endswith("bias"):
            parameter.data.uniform_(-0.02, 0.02)
    state = dict(model.state_dict())
    if packed:
        # Non-aligned TP cuts (8 vs block width 16) exercise scale offsets.
        block = (16, 16)
        for name in list(state):
            weight = state[name]
            if ".mlp.experts." in name:
                del state[name]
                for expert in range(config.num_experts):
                    if name.endswith("gate_up_proj"):
                        for proj, part in zip(
                            ("gate_proj", "up_proj"), weight[expert].chunk(2, 0)
                        ):
                            state[
                                name.rsplit(".", 1)[0] + f".{expert}.{proj}.weight"
                            ] = part.clone()
                    else:
                        state[
                            name.rsplit(".", 1)[0] + f".{expert}.down_proj.weight"
                        ] = weight[expert].clone()
                continue
        for name, weight in list(state.items()):
            if (
                weight.ndim != 2
                or ".mlp.gate." in name
                or name.startswith(("lm_head", "model.embed_tokens"))
            ):
                continue
            rows, cols = weight.shape
            scales = torch.rand((rows + 15) // 16, (cols + 15) // 16) * 0.03 + 0.01
            expanded = scales.repeat_interleave(16, 0).repeat_interleave(16, 1)[
                :rows, :cols
            ]
            quantized = (weight / expanded).to(torch.float8_e4m3fn)
            state[name] = quantized
            state[name.removesuffix(".weight") + ".weight_scale_inv"] = scales
            restored = dequantize_blocks(quantized, scales, block, torch.float32)
            if ".mlp.experts." in name:
                prefix, expert, proj, _ = name.rsplit(".", 3)
                fused = "down_proj" if proj == "down_proj" else "gate_up_proj"
                target = model.get_parameter(prefix + "." + fused).data[int(expert)]
                if proj != "down_proj":
                    offset = 0 if proj == "gate_proj" else config.moe_intermediate_size
                    target = target[offset : offset + config.moe_intermediate_size]
                target.copy_(restored)
            else:
                model.get_parameter(name).data.copy_(restored)
        config.quantization_config = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": list(block),
        }
    path.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(path)
    save_file(state, path / "model.safetensors")
    return model


class Tokenizer:
    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def main():
    torch.set_num_threads(1)
    path, packed, storage, tp, ep = (
        Path(sys.argv[1]),
        sys.argv[2] == "fp8",
        sys.argv[3],
        int(sys.argv[4]),
        int(sys.argv[5]),
    )
    if "RANK" in os.environ:
        dist.init_process_group("gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    reference = fixture(path / f"rank{rank}", packed, 4 if tp == 4 else 2)
    dtype = getattr(torch, sys.argv[6]) if len(sys.argv) > 6 else torch.float32
    for parameter in reference.parameters():
        parameter.data = parameter.data.to(dtype)
    model = load_parallel_model(
        path / f"rank{rank}",
        tp,
        ep,
        dtype=dtype,
        expert_storage=storage,
        numerics=sys.argv[7] if len(sys.argv) > 7 else "reference",
    )
    for block in model.model.layers:
        assert block.self_attn.q_proj.weight.shape[0] == 32 // tp
        assert block.mlp.experts.num_experts == 4 // ep
        assert len(block.mlp.experts.local) == 4 // ep
        if packed and storage == "fp8":
            assert (
                block.mlp.experts.local[0]["gate_proj"].weight.dtype
                == torch.float8_e4m3fn
            )
    ids = torch.tensor([[0, 0, 1, 2], [3, 4, 5, 6]])
    original_mask = mask = (ids != 0).long()
    positions = (mask.cumsum(-1) - 1).clamp_min(0)
    actual_trace, ref_trace = RoutingRecorder(model), RoutingRecorder(reference)
    with torch.inference_mode():
        actual_trace.begin(ids, mask, positions)
        ref_trace.begin(ids, mask, positions)
        actual = model(ids, attention_mask=mask, position_ids=positions, use_cache=True)
        expected = reference(
            ids, attention_mask=mask, position_ids=positions, use_cache=True
        )
        torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-5)
        if dtype == torch.bfloat16 and (
            len(sys.argv) <= 7 or sys.argv[7] == "reference"
        ):
            assert torch.equal(actual.logits, expected.logits)
        assert (
            actual.past_key_values.layers[0].keys.shape[1]
            == reference.config.num_key_value_heads // tp
        )
        for step in range(3):
            token = torch.tensor([[7 + step], [11 + step]])
            mask = torch.cat((mask, torch.ones(2, 1, dtype=mask.dtype)), -1)
            positions = mask.sum(-1, keepdim=True) - 1
            actual_trace.begin(token, mask, positions)
            ref_trace.begin(token, mask, positions)
            actual = model(
                token,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=actual.past_key_values,
                use_cache=True,
            )
            expected = reference(
                token,
                attention_mask=mask,
                position_ids=positions,
                past_key_values=expected.past_key_values,
                use_cache=True,
            )
            torch.testing.assert_close(
                actual.logits, expected.logits, atol=2e-5, rtol=2e-5
            )
    for a, b in zip(actual_trace.records, ref_trace.records):
        assert torch.equal(a["expert_indices"], b["expert_indices"])
        torch.testing.assert_close(a["routing_weights"], b["routing_weights"])
    actual_trace.close()
    ref_trace.close()
    trace = RoutingRecorder(model)
    out = greedy_generate(model, ids, original_mask, 3, trace)
    trace.close()
    expected = greedy_generate(reference, ids, original_mask, 3)
    assert torch.equal(out, expected)
    assert len(trace.records) == 6
    # Check single-sample generation matches the left-padded batch.
    single = greedy_generate(model, ids[:1, 2:], original_mask[:1, 2:], 3)
    assert torch.equal(single[0, 2:], out[0, 4:])
    if rank == 0:
        write_sample_records(
            path / "json",
            prompts=["a", "b"],
            input_ids=ids,
            attention_mask=original_mask,
            generated=out,
            records=trace.records,
            tokenizer=Tokenizer(),
            eos_token_id=None,
            metadata={"model_type": "qwen3_moe", "tp_size": tp, "ep_size": ep},
        )
        layer = json.loads((path / "json/000/layers/00.json").read_text())
        assert layer["name"].endswith(".gate")
        assert layer["steps"][0]["token_ids"] == [1, 2]
        assert len(layer["steps"]) == 3
    if dist.is_initialized():
        dist.destroy_process_group()
    print(f"PASS rank={rank}")


if __name__ == "__main__":
    main()
