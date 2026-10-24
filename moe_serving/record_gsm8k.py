"""Replay GSM8K with GPT-OSS TP/EP and one routing directory per sample."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import torch
import torch.distributed as dist

from moe_serving.datasets import dataset_root, load_requests
from moe_serving.durable_io import sync_directory, write_json
from moe_serving.efficient_kernels import backend_metadata
from moe_serving.gpt_oss_parallel import (
    RoutingRecorder,
    ensure_transformers_5200,
    greedy_generate,
    load_parallel_model,
)
from moe_serving.record_gpt_oss import write_sample_records

ROOT = Path(__file__).resolve().parents[1]
INSTRUCTION = 'Solve the problem and give your final numeric answer as "#### <answer>".'


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--checkpoint", type=Path, default=ROOT / "LLMs/gpt-oss-120b")
    result.add_argument(
        "--dataset-path",
        type=Path,
        help="Local GSM8K JSON/JSONL; default downloads the official test split into datasets/gsm8k/test.jsonl",
    )
    result.add_argument(
        "--output", type=Path, default=ROOT / "outputs/gpt_oss_120b_gsm8k"
    )
    result.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Default 8; -1 means all test questions",
    )
    result.add_argument("--start-index", type=int, default=0)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Independent generation budget; never capped to reference answer length",
    )
    result.add_argument(
        "--reasoning-effort", choices=["low", "medium", "high"], default="low"
    )
    result.add_argument("--nproc-per-node", type=int, default=4)
    result.add_argument("--tp-size", type=int, default=4)
    result.add_argument("--ep-size", type=int, default=4)
    result.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    result.add_argument("--expert-storage", choices=["mxfp4", "bf16"], default="mxfp4")
    result.add_argument(
        "--parallel-numerics",
        choices=["reference", "fast", "efficient"],
        default="efficient",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the distributed command; no download, model loading or inference",
    )
    result.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare dataset and tokenized requests; no model loading or inference",
    )
    result.add_argument(
        "--log-interval",
        type=float,
        default=10.0,
        help="Generation progress interval in seconds (default: 10)",
    )
    return result


def validate(args):
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive")
    for name in (
        "batch_size",
        "max_new_tokens",
        "nproc_per_node",
        "tp_size",
        "ep_size",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.start_index < 0 or args.num_samples == 0 or args.num_samples < -1:
        raise ValueError(
            "start-index must be nonnegative; num-samples must be positive or -1"
        )
    world = int(os.environ.get("WORLD_SIZE", args.nproc_per_node))
    if world % args.tp_size or world % args.ep_size:
        raise ValueError("TP and EP sizes must both divide the process count")
    config = json.loads((args.checkpoint / "config.json").read_text())
    if config["model_type"] != "gpt_oss":
        raise ValueError("This entry point requires a GPT-OSS checkpoint")
    if (
        config["num_attention_heads"] % args.tp_size
        or config["num_key_value_heads"] % args.tp_size
    ):
        raise ValueError("TP must divide query and KV heads")
    if config["num_local_experts"] % args.ep_size:
        raise ValueError("EP must divide the number of experts")
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())
    missing = [
        file
        for file in set(index["weight_map"].values())
        if not (args.checkpoint / file).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint shards: {missing}")
    if any(
        (args.output / name).exists()
        for name in ("workload.json", "results.jsonl", "run_config.json")
    ):
        raise FileExistsError(
            f"{args.output} already contains a run; choose a new --output"
        )


def distributed_command(args, argv):
    child = [arg for arg in argv if arg not in {"--dry-run", "--prepare-only"}]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={args.nproc_per_node}",
        "-m",
        "moe_serving.record_gsm8k",
        *child,
    ]


def prepare_samples(args, tokenizer):
    requests = load_requests("gsm8k", args.dataset_path)
    selected = requests[args.start_index :]
    if args.num_samples > 0:
        selected = selected[: args.num_samples]
    if not selected:
        raise ValueError("No questions selected; check --start-index and --num-samples")
    samples = []
    for offset, request in enumerate(selected, args.start_index):
        question = request["messages"][-1]["content"]
        prompt = question + "\n\n" + INSTRUCTION
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            reasoning_effort=args.reasoning_effort,
            return_dict=False,
        )
        if isinstance(ids, Mapping):
            ids = ids["input_ids"]
        if torch.is_tensor(ids):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise ValueError("Expected one tokenized request")
            ids = ids[0]
        ids = [int(token) for token in ids]
        if not ids:
            raise ValueError(f"Question {offset} produced an empty prompt")
        samples.append(
            {
                "id": f"{offset:06d}",
                "source_id": request["source_id"],
                "question": question,
                "user_text": prompt,
                "reference": request.get("reference"),
                "input_ids": list(ids),
            }
        )
    return samples


def padded_batch(samples, pad_id, device):
    width = max(len(sample["input_ids"]) for sample in samples)
    ids = [[pad_id] * (width - len(s["input_ids"])) + s["input_ids"] for s in samples]
    mask = [
        [0] * (width - len(s["input_ids"])) + [1] * len(s["input_ids"]) for s in samples
    ]
    return torch.tensor(ids, device=device), torch.tensor(mask, device=device)


def store_batch(
    args, tokenizer, group, ids, mask, generated, records, metadata, completed, seconds
):
    write_sample_records(
        args.output,
        prompts=[s["user_text"] for s in group],
        input_ids=ids,
        attention_mask=mask,
        generated=generated,
        records=records,
        tokenizer=tokenizer,
        eos_token_id=metadata["eos_token_id"],
        metadata={**metadata, "batch_seconds": seconds},
        sample_ids=[s["id"] for s in group],
        request_metadata=[
            {k: s[k] for k in ("source_id", "question", "reference")} for s in group
        ],
        write_manifest=False,
    )
    with (args.output / "results.jsonl").open("a", encoding="utf-8") as handle:
        for sample in group:
            path = args.output / sample["id"] / "output.json"
            output = json.loads(path.read_text())
            raw = tokenizer.decode(output["new_token_ids"], skip_special_tokens=False)
            output["generated_text_raw"] = raw
            output["reached_token_limit"] = len(
                output["new_token_ids"]
            ) >= args.max_new_tokens and (
                not output["new_token_ids"]
                or output["new_token_ids"][-1] not in metadata["eos_token_id"]
            )
            write_json(path, output)
            row = {
                "id": sample["id"],
                "source_id": sample["source_id"],
                "question": sample["question"],
                "reference": sample["reference"],
                "generated_text": output["generated_text"],
                "generated_text_raw": raw,
                "new_tokens": len(output["new_token_ids"]),
                "reached_token_limit": output["reached_token_limit"],
                "directory": sample["id"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed.append(
                {
                    "id": sample["id"],
                    "directory": sample["id"],
                    "source_id": sample["source_id"],
                }
            )
            print(
                json.dumps(
                    {
                        "sample": sample["id"],
                        "new_tokens": row["new_tokens"],
                        "reached_token_limit": row["reached_token_limit"],
                        "directory": str(path.parent),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        handle.flush()
        os.fsync(handle.fileno())
    # Persist sample directory entries as well as their individual JSON files.
    for sample in group:
        directory = args.output / sample["id"]
        if (directory / "layers").is_dir():
            sync_directory(directory / "layers")
        sync_directory(directory)
    sync_directory(args.output)
    write_json(
        args.output / "workload.json",
        {
            **metadata,
            "status": "running",
            "completed_samples": len(completed),
            "requests": completed,
        },
    )

    # Written last: its presence means all files and the result index were synced.
    write_json(
        args.output / "batches" / f"{group[0]['id']}-{group[-1]['id']}.json",
        {
            "status": "complete",
            "sample_ids": [sample["id"] for sample in group],
            "completed_samples": len(completed),
            "batch_seconds": seconds,
            "contents": ["input.json", "output.json", "layers/*.json"],
        },
    )
    sync_directory(args.output)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    validate(args)
    command = distributed_command(args, argv)
    if args.dry_run:
        print(shlex.join(command))
        print(
            f"Output: {args.output}\nNo dataset download, model loading or inference performed."
        )
        return
    ensure_transformers_5200()
    from transformers import AutoTokenizer

    # Prepare once before torchrun; ranks then read the same cached/local file.
    if "RANK" not in os.environ:
        load_requests("gsm8k", args.dataset_path)
        if args.dataset_path is None:
            args.dataset_path = dataset_root() / "gsm8k/test.jsonl"
            argv += ["--dataset-path", str(args.dataset_path.resolve())]
        if not args.prepare_only:
            command = distributed_command(args, argv)
            print("Launching: " + shlex.join(command), flush=True)
            os.execv(sys.executable, command)
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, local_files_only=True, padding_side="left"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    samples = prepare_samples(args, tokenizer)
    if args.prepare_only:
        if int(os.environ.get("RANK", "0")) != 0:
            return
        write_json(
            args.output / "prepared.json",
            {
                "checkpoint": str(args.checkpoint),
                "dataset": "gsm8k",
                "dataset_path": str(args.dataset_path),
                "samples": samples,
                "tp_size": args.tp_size,
                "ep_size": args.ep_size,
                "max_new_tokens": args.max_new_tokens,
                "reasoning_effort": args.reasoning_effort,
            },
        )
        print(
            f"Prepared {len(samples)} GSM8K requests: {args.output / 'prepared.json'}",
            flush=True,
        )
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        config = json.loads((args.checkpoint / "config.json").read_text())
        context = config.get("max_position_embeddings", 0)
        if (
            context
            and max(len(s["input_ids"]) for s in samples) + args.max_new_tokens
            > context
        ):
            raise ValueError(
                "Prompt plus generation budget exceeds checkpoint context length"
            )
        eos = config.get("eos_token_id") or []
        eos = [eos] if isinstance(eos, int) else list(eos)
        metadata = {
            "format_version": 2,
            "dataset": "gsm8k",
            "split": "test",
            "dataset_path": str(args.dataset_path),
            "checkpoint": str(args.checkpoint),
            "transformers_source": str(ensure_transformers_5200()),
            "model_type": "gpt_oss",
            "tp_size": args.tp_size,
            "ep_size": args.ep_size,
            "world_size": world,
            "dtype": args.dtype,
            "expert_storage": args.expert_storage,
            "parallel_numerics": args.parallel_numerics,
            "backends": backend_metadata("gpt_oss", args.parallel_numerics),
            "reasoning_effort": args.reasoning_effort,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "selected_samples": len(samples),
            "start_index": args.start_index,
            "num_experts": config["num_local_experts"],
            "num_layers": config["num_hidden_layers"],
            "eos_token_id": eos,
        }
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(args.output / "run_config.json", metadata)
            print(json.dumps(metadata), flush=True)

        def log(message):
            line = (
                f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {message}"
            )
            print(line, flush=True)
            with (args.output / "progress.log").open("a") as handle:
                handle.write(line + "\n")

        load_began = time.monotonic()
        if rank == 0:
            log(
                "Loading model; initial kernel preparation can take longer on the first run"
            )
        model = load_parallel_model(
            args.checkpoint,
            args.tp_size,
            args.ep_size,
            device,
            getattr(torch, args.dtype),
            args.expert_storage,
            numerics=args.parallel_numerics,
            progress_callback=log if rank == 0 else None,
        )
        dist.barrier()
        if rank == 0:
            log(
                f"Model ready on all {world} ranks; load time={time.monotonic() - load_began:.1f}s"
            )
        run_began = time.monotonic()
        completed = []
        for start in range(0, len(samples), args.batch_size):
            group = samples[start : start + args.batch_size]
            ids, mask = padded_batch(group, tokenizer.pad_token_id, device)
            recorder = RoutingRecorder(model) if rank == 0 else None
            torch.cuda.synchronize()
            began = time.perf_counter()
            batch_number = start // args.batch_size + 1
            batch_total = (len(samples) + args.batch_size - 1) // args.batch_size
            if rank == 0:
                log(
                    f"Batch {batch_number}/{batch_total} start; samples={[s['id'] for s in group]}; "
                    f"prompt_tokens={[len(s['input_ids']) for s in group]}; "
                    f"completed={len(completed)}/{len(samples)}; starting prefill"
                )

            def generation_progress(
                info, batch_number=batch_number, batch_total=batch_total
            ):
                elapsed = info["elapsed_seconds"]
                rate = info["generated_tokens"] / max(elapsed, 1e-9)
                log(
                    f"Batch {batch_number}/{batch_total}: step={info['step']}/{info['max_new_tokens']}, "
                    f"finished={info['finished']}/{info['batch_size']}, "
                    f"generated_tokens={info['generated_tokens']}, elapsed={elapsed:.1f}s, "
                    f"throughput={rate:.2f} tokens/s (batch aggregate)"
                )

            try:
                generated = greedy_generate(
                    model,
                    ids,
                    mask,
                    args.max_new_tokens,
                    recorder,
                    progress_callback=generation_progress if rank == 0 else None,
                    progress_interval=args.log_interval,
                )
            finally:
                if recorder is not None:
                    recorder.close()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - began
            error = [None]
            if rank == 0:
                try:
                    log(f"Batch {batch_number}/{batch_total}: saving routing JSON")
                    store_batch(
                        args,
                        tokenizer,
                        group,
                        ids,
                        mask,
                        generated,
                        recorder.records,
                        metadata,
                        completed,
                        seconds,
                    )
                    elapsed = time.monotonic() - run_began
                    eta = elapsed / len(completed) * (len(samples) - len(completed))
                    log(
                        f"Batch {batch_number}/{batch_total} saved; completed={len(completed)}/{len(samples)} "
                        f"({100 * len(completed) / len(samples):.1f}%), run_elapsed={elapsed:.1f}s, "
                        f"estimated_remaining={eta:.1f}s"
                    )
                except Exception as exc:  # noqa: BLE001 - propagate any writer failure to all collective participants
                    error[0] = repr(exc)
            dist.broadcast_object_list(error, src=0)
            if error[0]:
                raise RuntimeError("Writing the batch failed: " + error[0])
            del recorder, generated
        if rank == 0:
            write_json(
                args.output / "workload.json",
                {
                    **metadata,
                    "status": "complete",
                    "completed_samples": len(completed),
                    "requests": completed,
                },
            )
            print(
                f"Completed {len(completed)} GSM8K samples: {args.output}", flush=True
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
