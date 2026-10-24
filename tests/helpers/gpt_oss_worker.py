"""Isolated Transformers 5.20 process; also runnable under torchrun."""

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from moe_serving.gpt_oss_parallel import (
    RoutingRecorder,
    ensure_transformers_5200,
    greedy_generate,
    load_parallel_model,
)

ensure_transformers_5200()
from transformers import GptOssConfig, GptOssForCausalLM
from transformers.integrations.mxfp4 import _convert_moe_packed_tensors


def fixture(path, packed, kv_heads=2):
    torch.manual_seed(17)
    config = GptOssConfig(
        hidden_size=32,
        intermediate_size=32,
        head_dim=8,
        num_attention_heads=4,
        num_key_value_heads=kv_heads,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        vocab_size=64,
        sliding_window=3,
        pad_token_id=0,
        eos_token_id=None,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    config._attn_implementation = "eager"
    model = GptOssForCausalLM(config).eval()
    # Nonzero biases catch incorrectly reduced replicated biases.
    for name, parameter in model.named_parameters():
        if "bias" in name:
            parameter.data.uniform_(-0.05, 0.05)
    state = model.state_dict()
    if packed:
        for layer in range(2):
            for proj, out in (("gate_up_proj", 64), ("down_proj", 32)):
                key = f"model.layers.{layer}.mlp.experts.{proj}"
                blocks = torch.randint(0, 256, (4, out, 1, 16), dtype=torch.uint8)
                scales = torch.full((4, out, 1), 121, dtype=torch.uint8)
                state[key] = _convert_moe_packed_tensors(
                    blocks, scales, dtype=torch.float32
                )
                # Keep the unquantized reference separate.
                model.get_parameter(key).data.copy_(state[key])
                del state[key]
                state[key + "_blocks"], state[key + "_scales"] = blocks, scales
    path.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(path)
    save_file(state, path / "model.safetensors")
    return model


def main():
    torch.set_num_threads(1)
    path = Path(sys.argv[1])
    packed = sys.argv[2] == "packed"
    storage = sys.argv[3]
    if "RANK" in os.environ:
        dist.init_process_group("gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    # Construct the identical reference on every rank; only rank zero writes.
    reference = fixture(path / f"rank{rank}", packed, 4 if world == 4 else 2)
    tp = int(sys.argv[4]) if len(sys.argv) > 4 else world
    ep = int(sys.argv[5]) if len(sys.argv) > 5 else world
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
    ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    mask = (ids != 0).long()
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    trace, ref_trace = RoutingRecorder(model), RoutingRecorder(reference)
    trace.begin(ids, mask, pos)
    ref_trace.begin(ids, mask, pos)
    with torch.inference_mode():
        actual = model(ids, attention_mask=mask, position_ids=pos, use_cache=True)
        expected = reference(ids, attention_mask=mask, position_ids=pos, use_cache=True)
        torch.testing.assert_close(actual.logits, expected.logits, atol=2e-5, rtol=2e-5)
        if dtype == torch.bfloat16 and (
            len(sys.argv) <= 7 or sys.argv[7] == "reference"
        ):
            assert torch.equal(actual.logits, expected.logits)
        for record, other in zip(trace.records, ref_trace.records):
            assert torch.equal(record["expert_indices"], other["expert_indices"])
            torch.testing.assert_close(
                record["routing_weights"], other["routing_weights"]
            )
        trace.close()
        ref_trace.close()
        # Decode beyond the sliding window with head-local KV caches.
        for step in range(4):
            token = torch.tensor([[8 + step], [12 + step]])
            mask = torch.cat((mask, torch.ones(2, 1, dtype=mask.dtype)), -1)
            pos = mask.sum(-1, keepdim=True) - 1
            actual = model(
                token,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=actual.past_key_values,
                use_cache=True,
            )
            expected = reference(
                token,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=expected.past_key_values,
                use_cache=True,
            )
            torch.testing.assert_close(
                actual.logits, expected.logits, atol=2e-5, rtol=2e-5
            )
    trace.close()
    ref_trace.close()
    # Generation hooks have exact batch/sequence metadata for every invocation.
    trace = RoutingRecorder(model)
    out = greedy_generate(model, ids, (ids != 0).long(), 3, trace)
    trace.close()
    expected_ids = greedy_generate(reference, ids, (ids != 0).long(), 3)
    assert torch.equal(out, expected_ids)
    assert out.shape == (2, 7)
    assert len(trace.records) == 6
    assert [r["phase"] for r in trace.records] == ["prefill"] * 2 + ["decode"] * 4
    assert trace.records[0]["expert_indices"].shape == (2, 4, 2)
    assert trace.records[-1]["expert_indices"].shape == (2, 1, 2)
    assert trace.records[0]["valid_tokens"].sum() == 7
    if dist.is_initialized():
        dist.destroy_process_group()
    print(f"PASS rank={rank} packed={packed} storage={storage}")


if __name__ == "__main__":
    main()
