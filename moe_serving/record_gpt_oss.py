"""torchrun entry point for GPT-OSS TP/EP inference and routing traces."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from moe_serving.gpt_oss_parallel import (
    RoutingRecorder,
    ensure_transformers_5200,
    greedy_generate,
    load_parallel_model,
)


def write_sample_records(output, *, prompts, input_ids, attention_mask, generated,
                         records, tokenizer, eos_token_id, metadata):
    """Write one directory per sample, excluding padding and tokens after EOS."""
    root = output.with_suffix("") if output.suffix in {".pt", ".json"} else output
    inputs, masks, sequences = input_ids.cpu(), attention_mask.cpu(), generated.cpu()
    eos_ids = {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id or [])

    def write(path, payload):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    requests = []
    for index, prompt in enumerate(prompts):
        sample_id = f"{index:03d}"
        directory = root / sample_id
        prompt_ids = inputs[index][masks[index].bool()].tolist()
        new_ids = []
        for token in sequences[index, inputs.shape[1]:].tolist():
            new_ids.append(token)
            if token in eos_ids:
                break
        write(directory / "input.json", {
            "sample_id": sample_id, "user_text": prompt, "token_ids": prompt_ids,
            "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=False),
        })
        write(directory / "output.json", {
            **metadata, "sample_id": sample_id,
            "token_ids": prompt_ids + new_ids, "new_token_ids": new_ids,
            "generated_text": tokenizer.decode(new_ids, skip_special_tokens=True),
            "text": tokenizer.decode(prompt_ids + new_ids, skip_special_tokens=False),
        })
        layers = {}
        for record in records:
            if record["phase"] == "decode" and record["step"] >= len(new_ids):
                continue
            valid = record["valid_tokens"][index].bool()
            if not bool(valid.any()):
                continue
            layers.setdefault(record["layer"], []).append({
                "step": record["step"], "kind": record["phase"],
                "token_ids": record["input_ids"][index][valid].tolist(),
                "position_ids": record["position_ids"][index][valid].tolist(),
                "indices": record["expert_indices"][index][valid].tolist(),
                "scores": record["routing_weights"][index][valid].float().tolist(),
            })
        for layer, steps in sorted(layers.items()):
            write(directory / "layers" / f"{layer:02d}.json", {
                "layer": layer, "name": f"model.layers.{layer}.mlp.router", "steps": steps,
            })
        requests.append({"id": sample_id, "directory": sample_id})
    write(root / "workload.json", {**metadata, "batch_size": len(prompts), "requests": requests})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=None, help="Default: WORLD_SIZE")
    parser.add_argument(
        "--ep-size",
        type=int,
        default=None,
        help="Default: WORLD_SIZE (same ranks as TP)",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--expert-storage", choices=["mxfp4", "bf16"], default="mxfp4")
    parser.add_argument(
        "--prompt",
        action="append",
        help="Repeat for a batch; uses the checkpoint chat template",
    )
    parser.add_argument(
        "--raw-prompt", action="store_true", help="Skip the chat template"
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/gpt_oss_routing")
    )
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.device == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    tp_size = world if args.tp_size is None else args.tp_size
    ep_size = world if args.ep_size is None else args.ep_size
    try:
        ensure_transformers_5200()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint, local_files_only=True, padding_side="left"
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        prompts = args.prompt or ["Explain mixture-of-experts models in one sentence."]
        texts = (
            prompts
            if args.raw_prompt
            else [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
        )
        inputs = tokenizer(
            texts, padding=True, return_tensors="pt", add_special_tokens=args.raw_prompt
        ).to(device)
        if rank == 0:
            print(
                f"Loading {args.checkpoint}: TP={tp_size}, EP={ep_size}, experts={args.expert_storage}",
                flush=True,
            )
        model = load_parallel_model(
            args.checkpoint,
            tp_size,
            ep_size,
            device,
            getattr(torch, args.dtype),
            args.expert_storage,
        )
        recorder = RoutingRecorder(model) if rank == 0 else None
        try:
            generated = greedy_generate(
                model,
                inputs.input_ids,
                inputs.attention_mask,
                args.max_new_tokens,
                recorder,
            )
        finally:
            if recorder is not None:
                recorder.close()
        if rank == 0:
            decoded = tokenizer.batch_decode(
                generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
            )
            write_sample_records(
                args.output, prompts=prompts, input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask, generated=generated,
                records=recorder.records, tokenizer=tokenizer,
                eos_token_id=model.config.eos_token_id,
                metadata={
                    "format_version": 2,
                    "checkpoint": str(args.checkpoint.resolve()),
                    "transformers_source": str(ensure_transformers_5200()),
                    "tp_size": tp_size, "ep_size": ep_size, "world_size": world,
                    "expert_storage": args.expert_storage, "dtype": args.dtype,
                    "num_experts": model.config.num_local_experts,
                    "num_layers": model.config.num_hidden_layers,
                },
            )
            for text in decoded:
                print(text)
            print(
                f"Saved {len(recorder.records)} layer/step routing records to {args.output}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
