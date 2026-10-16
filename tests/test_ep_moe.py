import json

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from moe_serving.engine import Engine
from moe_serving.moe import ExpertPlan, LocalSwiGLUExperts, SharedSwiGLU, route, simulate_ranks


def _experts(num_experts, hidden, intermediate, generator):
    gate_up = torch.randn(num_experts, 2 * intermediate, hidden, generator=generator)
    down = torch.randn(num_experts, hidden, intermediate, generator=generator)
    return gate_up, down


def _reference(hidden, gate, gate_up, down, top_k):
    index, weight = route(hidden, gate, top_k, renormalize=True)
    out = hidden.new_zeros(hidden.shape)
    for token in range(hidden.shape[0]):
        for slot in range(top_k):
            expert = int(index[token, slot])
            gate_part, up_part = F.linear(hidden[token], gate_up[expert]).chunk(2)
            value = F.linear(F.silu(gate_part) * up_part, down[expert])
            out[token] += value * weight[token, slot]
    return out


def test_two_ranks_match_a_single_expert_loop():
    generator = torch.Generator().manual_seed(4)
    hidden_size, intermediate, num_experts, top_k = 8, 4, 4, 2
    gate_up, down = _experts(num_experts, hidden_size, intermediate, generator)
    gate = torch.randn(num_experts, hidden_size, generator=generator)
    tokens = torch.randn(6, hidden_size, generator=generator)
    banks = [
        LocalSwiGLUExperts(gate_up[0:2], down[0:2]),
        LocalSwiGLUExperts(gate_up[2:4], down[2:4]),
    ]
    left, right = simulate_ranks([tokens[:3], tokens[3:]], gate, banks, top_k)
    whole = torch.cat([left, right], dim=0)
    reference = _reference(tokens, gate, gate_up, down, top_k)
    assert torch.allclose(whole, reference, atol=1e-5, rtol=1e-5)


def test_shared_expert_is_added_on_the_owning_rank():
    generator = torch.Generator().manual_seed(5)
    hidden_size, intermediate = 8, 4
    gate_up, down = _experts(2, hidden_size, intermediate, generator)
    gate = torch.randn(2, hidden_size, generator=generator)
    tokens = torch.randn(4, hidden_size, generator=generator)
    shared_gate = torch.randn(intermediate, hidden_size, generator=generator)
    shared_up = torch.randn(intermediate, hidden_size, generator=generator)
    shared_down = torch.randn(hidden_size, intermediate, generator=generator)
    score = torch.randn(1, hidden_size, generator=generator)
    shared = [
        SharedSwiGLU(shared_gate, shared_up, shared_down, score),
        SharedSwiGLU(shared_gate.clone(), shared_up.clone(), shared_down.clone(), score.clone()),
    ]
    banks = [LocalSwiGLUExperts(gate_up[0:1], down[0:1]), LocalSwiGLUExperts(gate_up[1:2], down[1:2])]
    left, right = simulate_ranks([tokens[:2], tokens[2:]], gate, banks, top_k=1, shared=shared)
    routed = _reference(tokens, gate, gate_up, down, top_k=1)
    extra = shared[0](tokens)
    assert torch.allclose(torch.cat([left, right]), routed + extra, atol=1e-5, rtol=1e-5)


def test_plan_rejects_an_uneven_expert_split():
    try:
        ExpertPlan(num_experts=6, ep_size=4, rank=0)
    except ValueError as exc:
        assert "do not divide" in str(exc)
    else:
        raise AssertionError("expected an uneven expert split to fail")


def _tiny_qwen_checkpoint(directory):
    hidden, intermediate, experts = 8, 4, 4
    generator = torch.Generator().manual_seed(9)
    gate_up, down = _experts(experts, hidden, intermediate, generator)
    tensors = {
        "model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.layers.0.mlp.experts.down_proj": down,
        "model.layers.0.mlp.gate.weight": torch.randn(experts, hidden, generator=generator),
        "model.layers.0.mlp.shared_expert.gate_proj.weight": torch.randn(intermediate, hidden, generator=generator),
        "model.layers.0.mlp.shared_expert.up_proj.weight": torch.randn(intermediate, hidden, generator=generator),
        "model.layers.0.mlp.shared_expert.down_proj.weight": torch.randn(hidden, intermediate, generator=generator),
        "model.layers.0.mlp.shared_expert_gate.weight": torch.randn(1, hidden, generator=generator),
        "model.embed_tokens.weight": torch.randn(32, hidden, generator=generator),
    }
    save_file(tensors, directory / "model.safetensors")
    config = {
        "hidden_size": hidden,
        "moe_intermediate_size": intermediate,
        "num_experts": experts,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 1,
        "vocab_size": 32,
        "norm_topk_prob": True,
        "eos_token_id": 0,
        "model_type": "qwen3_moe",
    }
    (directory / "config.json").write_text(json.dumps(config))


def test_qwen3_moe_checkpoint_loads_a_different_expert_shard_per_rank(tmp_path):
    _tiny_qwen_checkpoint(tmp_path)
    rank0 = Engine.from_pretrained("qwen3_moe", str(tmp_path), rank=0, ep_size=2, device="cpu", load_embeddings=True)
    rank1 = Engine.from_pretrained("qwen3_moe", str(tmp_path), rank=1, ep_size=2, device="cpu", load_embeddings=False)
    assert rank0.model.plan.expert_start == 0 and rank0.model.plan.expert_end == 2
    assert rank1.model.plan.expert_start == 2 and rank1.model.plan.expert_end == 4
    assert rank0.model.ple is None
    tokens = torch.randint(0, 32, (2, 5))
    try:
        rank0.profile(tokens)
    except RuntimeError as exc:
        assert "process group" in str(exc)
    else:
        raise AssertionError("a split expert shard must not run without a process group")


def _tiny_flash_checkpoint(directory):
    hidden, intermediate, experts, hc, head_dim = 8, 4, 4, 2, 4
    generator = torch.Generator().manual_seed(11)
    gate_up, down = _experts(experts, hidden, intermediate, generator)
    embed = head_dim * 2
    tensors = {
        "model.language_model.layers.0.mlp.experts.gate_up_proj": gate_up,
        "model.language_model.layers.0.mlp.experts.down_proj": down,
        "model.language_model.layers.0.mlp.gate.weight": torch.randn(experts, hidden, generator=generator),
        "model.language_model.layers.0.ple.ple_embedding.layer_multipliers": torch.tensor([3, 5, 7]),
        "model.language_model.layers.0.ple.ple_embedding.ngram_heads_vocab_sizes": torch.tensor([32, 32]),
        "model.language_model.layers.0.ple.ple_embedding.ngram_heads_offsets": torch.tensor([0, 32]),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight": torch.randn(32, head_dim, generator=generator),
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight": torch.randn(32, head_dim, generator=generator),
        "model.language_model.layers.0.ple.key_proj.weight": torch.randn(hc * hidden, embed, generator=generator),
        "model.language_model.layers.0.ple.value_proj.weight": torch.randn(hidden, embed, generator=generator),
        "model.language_model.layers.0.ple.norm_query.weight": torch.ones(hc * hidden),
        "model.language_model.layers.0.ple.norm_key.weight": torch.ones(hc * hidden),
        "model.language_model.layers.0.ple.norm_conv.weight": torch.ones(hc * hidden),
        "model.language_model.layers.0.ple.conv1d.weight": torch.randn(hc * hidden, 1, 4, generator=generator),
        "model.language_model.embed_tokens.weight": torch.randn(32, hidden, generator=generator),
    }
    save_file(tensors, directory / "model.safetensors")
    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "hidden_size": hidden,
            "moe_intermediate_size": intermediate,
            "num_experts": experts,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 1,
            "vocab_size": 32,
            "hc_count": hc,
            "heads_per_ngram": 1,
            "ngram_size": 3,
            "eos_token_id": 0,
            "ple_layer_ids": [1],
        },
    }
    (directory / "config.json").write_text(json.dumps(config))


def test_qwen38_flash_runs_expert_parallel_with_engram_in_dram(tmp_path):
    _tiny_flash_checkpoint(tmp_path)
    engine = Engine.from_pretrained(
        "qwen3_8_flash_next",
        str(tmp_path),
        rank=0,
        ep_size=1,
        device="cpu",
        load_embeddings=True,
    )
    assert engine.model.spec.ple_layer == 0
    assert engine.model.ple.table._views == [None, None]
    tokens = torch.tensor([[1, 2, 0, 4, 5]])
    hidden, stats = engine.model(tokens)
    assert hidden.shape == (1, 5, 8)
    assert torch.isfinite(hidden).all()
    assert engine.model.ple.table._views[0] is not None or engine.model.ple.table._views[1] is not None
    assert int(stats[0].counts.sum()) == 10
    rows = engine.profile(tokens)
    assert rows[0]["engram_seconds"] >= 0
    assert "engram=on" in engine.describe()


def test_single_sequence_stays_on_rank_zero_when_the_batch_does_not_divide():
    from moe_serving.model import ModelSpec
    from moe_serving.moe import ExpertPlan

    spec = ModelSpec(
        family="qwen3_moe",
        hidden=4,
        intermediate=2,
        num_experts=4,
        top_k=1,
        num_layers=1,
        vocab_size=8,
        renormalize=True,
        rms_eps=1e-6,
        eos_token_id=0,
        hc_count=1,
        shared_intermediate=None,
        ple_layer=None,
        ngram_size=3,
        heads_per_ngram=1,
        prefix="model.",
    )
    plan = ExpertPlan(4, 4, 0)
    engine = Engine.__new__(Engine)
    engine.model = type("M", (), {"spec": spec, "plan": plan})()
    engine.rank = 0
    engine.ep_size = 4
    engine.device = torch.device("cpu")
    tokens = torch.arange(6).view(1, 6)
    assert engine.shard_tokens(tokens).shape == (1, 6)
    engine.rank = 1
    assert engine.shard_tokens(tokens).shape[0] == 0
