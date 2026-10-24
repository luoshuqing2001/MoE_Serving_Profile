"""Real GPU kernels: TP1/EP1 versus TP4/EP4, padded batch and cached decode."""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from moe_serving.gpt_oss_parallel import ensure_transformers_5200, load_parallel_model

ensure_transformers_5200()
from transformers import (
    GptOssConfig,
    GptOssForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from moe_serving.qwen3_parallel import load_parallel_model as load_qwen

dist.init_process_group("nccl")
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
device = torch.device("cuda", rank)
reports = []
for family in ["gpt_oss", "qwen3_moe"]:
    path = Path("outputs/efficient_tiny_checkpoints") / family
    if rank == 0:
        torch.manual_seed(19)
        common = {
            "hidden_size": 256,
            "head_dim": 64,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 2,
            "vocab_size": 64,
            "eos_token_id": None,
            "pad_token_id": 0,
        }
        if family == "gpt_oss":
            config = GptOssConfig(
                **common,
                intermediate_size=256,
                num_local_experts=4,
                num_experts_per_tok=2,
                sliding_window=3,
                rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
            )
            native = GptOssForCausalLM(config)
            state = native.state_dict()
            for layer in range(2):
                for proj, n in [("gate_up_proj", 512), ("down_proj", 256)]:
                    key = f"model.layers.{layer}.mlp.experts.{proj}"
                    del state[key]
                    state[key + "_blocks"] = torch.randint(
                        0, 256, (4, n, 8, 16), dtype=torch.uint8
                    )
                    state[key + "_scales"] = torch.full(
                        (4, n, 8), 117, dtype=torch.uint8
                    )
        else:
            config = Qwen3MoeConfig(
                **common,
                intermediate_size=256,
                moe_intermediate_size=256,
                num_experts=4,
                num_experts_per_tok=2,
            )
            native = Qwen3MoeForCausalLM(config)
            state = native.state_dict()
            for layer in range(2):
                prefix = f"model.layers.{layer}.mlp.experts"
                gu = state.pop(prefix + ".gate_up_proj")
                down = state.pop(prefix + ".down_proj")
                for e in range(4):
                    for proj, weight in [
                        ("gate_proj", gu[e, :256]),
                        ("up_proj", gu[e, 256:]),
                        ("down_proj", down[e]),
                    ]:
                        key = f"{prefix}.{e}.{proj}"
                        state[key + ".weight"] = weight.to(torch.float8_e4m3fn)
                        state[key + ".weight_scale_inv"] = torch.ones(2, 2)
            config.quantization_config = {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "weight_block_size": [128, 128],
            }
        path.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(path)
        save_file(state, path / "model.safetensors")
        del native, state
    dist.barrier()
    loader = load_parallel_model if family == "gpt_oss" else load_qwen
    kwargs = {"device": device, "dtype": torch.bfloat16, "numerics": "efficient"}
    baseline = loader(path, 1, 1, **kwargs)
    parallel = loader(path, 4, 4, **kwargs)
    ids = torch.tensor([[0, 0, 1, 2, 3], [4, 5, 6, 7, 8]], device=device)
    mask = (ids != 0).long()
    caches = [None, None]
    for step in range(5):
        pos = (mask.cumsum(-1) - 1).clamp_min(0)[:, -ids.shape[1] :]
        with torch.inference_mode():
            a = baseline(
                ids,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=caches[0],
                use_cache=True,
            )
            b = parallel(
                ids,
                attention_mask=mask,
                position_ids=pos,
                past_key_values=caches[1],
                use_cache=True,
            )
        caches = [a.past_key_values, b.past_key_values]
        ref = a.logits[:, -1].float()
        out = b.logits[:, -1].float()
        relative = ((out - ref).norm() / ref.norm()).item()
        assert out.isfinite().all() and relative < 0.03, (family, step, relative)
        reports.append(
            {
                "family": family,
                "step": step,
                "relative_l2": relative,
                "greedy_equal": bool(torch.equal(ref.argmax(-1), out.argmax(-1))),
            }
        )
        ids = torch.tensor([[9 + step], [15 + step]], device=device)
        mask = torch.cat([mask, torch.ones(2, 1, device=device, dtype=torch.long)], -1)
    del baseline, parallel, a, b, caches
    torch.cuda.empty_cache()
if rank == 0:
    print(json.dumps(reports, indent=2))
    Path("tests/validation_results/efficient_model_parallel.json").write_text(
        json.dumps(reports, indent=2) + "\n"
    )
dist.destroy_process_group()
