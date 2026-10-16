import json

import torch
from safetensors import safe_open

from moe_serving.record_routing import summarize_layer
from moe_serving.transformers_src import REQUIRED_VERSION, ensure_transformers_5190

FP8 = "/home/sky/MoE_Serving_Profile/LLMs/Qwen3.8-Flash-Next-FP8"


def test_vendored_transformers_is_5190():
    source = ensure_transformers_5190()
    import transformers

    assert transformers.__version__ == REQUIRED_VERSION
    assert source in __import__("pathlib").Path(transformers.__file__).parents


def test_fp8_checkpoint_config_uses_block_scaled_fp8():
    ensure_transformers_5190()
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(FP8, local_files_only=True)
    quant = config.quantization_config
    assert quant["quant_method"] == "fp8"
    assert quant["weight_block_size"] == [128, 128]
    assert config.text_config.num_experts == 512
    assert config.text_config.num_experts_per_tok == 10


def test_official_router_reads_the_fp8_checkpoint_gate():
    ensure_transformers_5190()
    from transformers import AutoConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter

    config = AutoConfig.from_pretrained(FP8, local_files_only=True)
    index = json.load(open(f"{FP8}/model.safetensors.index.json"))
    name = "model.language_model.layers.0.mlp.gate.weight"
    with safe_open(f"{FP8}/{index['weight_map'][name]}", framework="pt") as handle:
        gate = handle.get_tensor(name)
    router = Qwen4ExpTextTopKRouter(config.text_config).to(dtype=gate.dtype)
    assert tuple(gate.shape) == tuple(router.weight.shape)
    router.weight.data.copy_(gate)
    hidden = torch.randn(3, config.text_config.hidden_size, dtype=gate.dtype)
    _logits, scores, indices = router(hidden)
    assert indices.shape == (3, 10)
    assert int(indices.min()) >= 0 and int(indices.max()) < 512
    assert torch.allclose(scores.float().sum(-1), torch.ones(3), atol=1e-2)
    summary = summarize_layer(indices, 512)
    assert summary["assignments"] == 30
