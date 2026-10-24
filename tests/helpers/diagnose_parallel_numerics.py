"""Replay TP/EP components on identical native inputs to localize rounding drift."""

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from real_parallel_worker import NativeReader, compare, layer_checkpoint, prepare_cases

from moe_serving import gpt_oss_parallel, qwen3_parallel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--inputs", type=Path)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device, dtype = torch.device("cuda", rank), torch.bfloat16
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoint = args.checkpoint.resolve()
    raw = json.loads((checkpoint / "config.json").read_text())
    wm = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    backend = gpt_oss_parallel if raw["model_type"] == "gpt_oss" else qwen3_parallel
    backend.ensure_transformers_5200()
    baseline = torch.load(args.baseline, weights_only=True, map_location="cpu")
    cases = prepare_cases(
        baseline["token_rows"], 2 if args.inputs else 0, raw["vocab_size"], device
    )
    case = cases[0]
    captures, ref_captures = {}, {}

    def hook(bucket, name):
        def capture(module, inputs, output):
            bucket[name] = (
                tuple(v.detach().clone() for v in inputs),
                output.detach().clone(),
            )

        return capture

    with layer_checkpoint(checkpoint, raw, wm, args.layer, backend) as (view, config):
        actual = backend.load_parallel_model(
            view,
            4,
            4,
            device,
            dtype,
            "mxfp4" if raw["model_type"] == "gpt_oss" else "fp8",
        )
    reader = NativeReader(checkpoint, wm, raw, device, dtype) if rank == 0 else None
    expected = (
        reader.model(args.layer, config, raw["model_type"]) if rank == 0 else None
    )
    handles = []
    for name in [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.experts",
    ]:
        handles.append(
            actual.model.layers[0]
            .get_submodule(name)
            .register_forward_hook(hook(captures, name))
        )
        if rank == 0:
            handles.append(
                expected.model.layers[0]
                .get_submodule(name)
                .register_forward_hook(hook(ref_captures, name))
            )
    report = {}

    def metric(name, a, b):
        if rank == 0:
            report[name] = {
                **compare(a, b, 0, 0),
                "equal_fraction": float((a == b).float().mean()),
            }

    def broadcast(value):
        info = [(tuple(value.shape), value.dtype) if rank == 0 else None]
        dist.broadcast_object_list(info, src=0)
        result = (
            value.contiguous()
            if rank == 0
            else torch.empty(info[0][0], device=device, dtype=info[0][1])
        )
        dist.broadcast(result, src=0)
        return result

    with torch.inference_mode():
        if args.inputs:
            saved = torch.load(args.inputs, weights_only=True, map_location="cpu")[
                "batch"
            ]
            cache = ref_cache = None
            for step in range(2):
                out = actual(
                    inputs_embeds=saved["actual"][step].to(device),
                    attention_mask=case["mask"][step],
                    position_ids=case["positions"][step],
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                if rank == 0:
                    out = expected(
                        inputs_embeds=saved["reference"][step].to(device),
                        attention_mask=case["mask"][step],
                        position_ids=case["positions"][step],
                        past_key_values=ref_cache,
                        use_cache=True,
                    )
                    ref_cache = out.past_key_values
        else:
            actual(
                input_ids=case["ids"][0],
                attention_mask=case["mask"][0],
                position_ids=case["positions"][0],
            )
            if rank == 0:
                expected(
                    input_ids=case["ids"][0],
                    attention_mask=case["mask"][0],
                    position_ids=case["positions"][0],
                )
        for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
            local = captures[name][1].contiguous()
            pieces = [torch.empty_like(local) for _ in range(4)]
            dist.all_gather(pieces, local)
            metric(
                name,
                torch.cat(pieces, -1),
                ref_captures[name][1] if rank == 0 else None,
            )
        local_context = captures["self_attn.o_proj"][0][0].contiguous()
        pieces = [torch.empty_like(local_context) for _ in range(4)]
        dist.all_gather(pieces, local_context)
        metric(
            "attention_context",
            torch.cat(pieces, -1),
            ref_captures["self_attn.o_proj"][0][0] if rank == 0 else None,
        )
        full_context = broadcast(
            ref_captures["self_attn.o_proj"][0][0] if rank == 0 else None
        )
        width = full_context.shape[-1] // 4
        replay = actual.model.layers[0].self_attn.o_proj(
            full_context[..., rank * width : (rank + 1) * width]
        )
        metric(
            "o_proj_identical_input",
            replay,
            ref_captures["self_attn.o_proj"][1] if rank == 0 else None,
        )
        inputs = [
            broadcast(ref_captures["mlp.experts"][0][i] if rank == 0 else None)
            for i in range(3)
        ]
        replay = actual.model.layers[0].mlp.experts(*inputs)
        metric(
            "experts_identical_input",
            replay,
            ref_captures["mlp.experts"][1] if rank == 0 else None,
        )
    if rank == 0:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        reader.close()
    for handle in handles:
        handle.remove()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
