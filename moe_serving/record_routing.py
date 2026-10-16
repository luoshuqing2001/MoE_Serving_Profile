"""Record MoE routing from the local FP8 checkpoint via Transformers 5.19.0.

The router is ``Qwen4ExpTextTopKRouter`` in the vendored source. Experts run
with ``experts_implementation="grouped_mm"``. ``model.generate`` does one
prefill forward and then one decode forward per new token. A hook on each
router keeps the expert indices for every forward. A hook on each Engram
n-gram table keeps the row indices that lookup activates.

    source /home/sky/llm-venv/bin/activate
    python -m moe_serving.record_routing \\
        --checkpoint /home/sky/MoE_Serving_Profile/LLMs/Qwen3.8-Flash-Next-FP8 \\
        --workload gsm8k \\
        --num-samples 8 \\
        --output outputs/routing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

from moe_serving.grouped_fp8 import patch_grouped_fp8_kernel
from moe_serving.transformers_src import REQUIRED_VERSION, ensure_transformers_5190

DEFAULT_CHECKPOINT = Path("/home/sky/MoE_Serving_Profile/LLMs/Qwen3.8-Flash-Next-FP8")


class RoutingRecorder:
    """Capture router indices and Engram row indices for each model forward.

    The first forward is prefill. Later forwards from ``generate`` are decode
    steps of one token. Engram rows are the indices passed to
    ``ngram_embedding``, which are the rows of that table the token activates.
    """

    def __init__(self):
        self.steps: list[dict] = []
        self._pending: dict[str, dict[str, torch.Tensor]] = {}
        self._engram: dict[str, dict] = {}
        self._handles = []
        self._started = 0.0
        self._seq_len = 0
        self._token_ids: torch.Tensor | None = None
        self._token_rows: list[list[int]] | None = None

    def attach(self, model) -> None:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextTopKRouter

        for name, module in model.named_modules():
            if isinstance(module, Qwen4ExpTextTopKRouter):
                self._handles.append(module.register_forward_hook(self._router_hook(name)))
            embedding = getattr(module, "ngram_embedding", None)
            if embedding is not None and hasattr(module, "ngram_size") and hasattr(module, "heads_per_ngram"):
                self._handles.append(
                    embedding.register_forward_pre_hook(self._engram_hook(name, module), with_kwargs=True)
                )
        self._handles.append(model.register_forward_pre_hook(self._begin, with_kwargs=True))
        self._handles.append(model.register_forward_hook(self._end, with_kwargs=True))

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _router_hook(self, name: str):
        def hook(_module, _inputs, output):
            _logits, scores, indices = output
            self._pending[name] = {
                "indices": indices.detach().to(device="cpu", dtype=torch.int16),
                "scores": scores.detach().to(device="cpu", dtype=torch.float16),
            }

        return hook

    def _engram_hook(self, name: str, module):
        meta = {
            "ngram_size": int(module.ngram_size),
            "heads_per_ngram": int(module.heads_per_ngram),
        }

        def hook(_embedding, args, kwargs):
            indices = args[0] if args else kwargs.get("input")
            if indices is None:
                return
            self._engram[name] = {
                "indices": _cpu_indices(indices),
                "ngram_size": meta["ngram_size"],
                "heads_per_ngram": meta["heads_per_ngram"],
            }

        return hook

    def _begin(self, _module, args, kwargs) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._started = time.perf_counter()
        self._pending = {}
        self._engram = {}
        self._seq_len = _sequence_length(args, kwargs)
        self._token_ids = _token_ids(args, kwargs)
        self._token_rows = _token_rows(args, kwargs)

    def _end(self, _module, _args, _kwargs, _output) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.steps.append(
            {
                "kind": "prefill" if not self.steps else "decode",
                "seq_len": self._seq_len,
                "seconds": time.perf_counter() - self._started,
                "experts_implementation": _experts_implementation(_module),
                "token_ids": self._token_ids,
                "token_rows": self._token_rows,
                "layers": self._pending,
                "engram": self._engram,
            }
        )
        self._pending = {}
        self._engram = {}


def _experts_implementation(module) -> str | None:
    config = getattr(module, "config", None)
    if config is None:
        return None
    text = getattr(config, "text_config", None)
    source = text if text is not None else config
    return getattr(source, "_experts_implementation", None)


def _forward_input_ids(args, kwargs) -> torch.Tensor | None:
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0]
    if not torch.is_tensor(input_ids):
        return None
    return input_ids.detach().to(device="cpu", dtype=torch.int64)


def _sequence_length(args, kwargs) -> int:
    input_ids = _forward_input_ids(args, kwargs)
    if input_ids is None:
        embeds = kwargs.get("inputs_embeds")
        return 0 if embeds is None else int(embeds.shape[1])
    return int(input_ids.shape[-1] if input_ids.ndim >= 1 else 1)


def _token_rows(args, kwargs) -> list[list[int]] | None:
    input_ids = _forward_input_ids(args, kwargs)
    if input_ids is None:
        return None
    if input_ids.ndim == 1:
        return [input_ids.tolist()]
    return input_ids.tolist()


def _token_ids(args, kwargs) -> list[int] | None:
    input_ids = _forward_input_ids(args, kwargs)
    if input_ids is None:
        return None
    if input_ids.ndim >= 2 and input_ids.shape[0] == 1:
        input_ids = input_ids[0]
    return input_ids.reshape(-1).tolist()


def summarize_layer(indices: torch.Tensor, num_experts: int) -> dict:
    counts = torch.bincount(indices.reshape(-1).to(torch.int64), minlength=num_experts)
    mean = counts.float().mean()
    cv = 0.0 if float(mean) == 0 else float(counts.float().std(unbiased=False) / mean)
    return {
        "assignments": int(counts.sum()),
        "expert_load_cv": cv,
        "expert_dead_fraction": float((counts == 0).float().mean()),
        "counts": counts,
    }


def _relaunch_under_torchrun(ep_size: int) -> None:
    """Expert parallel needs one process per GPU. A plain ``python -m`` call starts that."""
    if ep_size <= 1 or os.environ.get("WORLD_SIZE"):
        return
    cmd = [
        "torchrun",
        "--nproc_per_node",
        str(ep_size),
        "--master_port",
        os.environ.get("MASTER_PORT", "29531"),
        "-m",
        "moe_serving.record_routing",
        *sys.argv[1:],
    ]
    os.execvp(cmd[0], cmd)


def load_fp8_model(
    checkpoint: Path,
    device_map: str | None = None,
    experts_implementation: str = "grouped_mm",
    ep_size: int = 1,
):
    ensure_transformers_5190()
    patch_grouped_fp8_kernel()
    from transformers import DistributedConfig, Qwen4ExpForConditionalGeneration

    kwargs = {}
    if ep_size > 1:
        # This Transformers build requires ep_size == tp_size. Expert weights are
        # removed from the tensor-parallel plan and sharded by the expert plan:
        # grouped_gemm on the expert dimension, then all-to-all token dispatch.
        kwargs["distributed_config"] = DistributedConfig(tp_size=ep_size, ep_size=ep_size)
    elif device_map is None:
        if torch.cuda.is_available():
            device_map = "balanced"
            kwargs["max_memory"] = {index: "80GiB" for index in range(torch.cuda.device_count())}
            kwargs["max_memory"]["cpu"] = "800GiB"
        else:
            device_map = "cpu"
    return Qwen4ExpForConditionalGeneration.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map=device_map,
        local_files_only=True,
        experts_implementation=experts_implementation,
        **kwargs,
    )


def record_routing(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 0,
    attention_mask: torch.Tensor | None = None,
) -> tuple[list[dict], torch.Tensor | None]:
    """Run prefill, then ``max_new_tokens`` decode steps.

    Returns the per-forward routing records and the generated token ids. The
    generated tensor includes the prompt. Transformers switches experts to
    ``batched_mm`` inside ``generate`` because that is faster for a single token.
    This benchmark keeps the implementation requested at load time, which is
    ``grouped_mm``.
    """
    recorder = RoutingRecorder()
    recorder.attach(model)
    sequences = None
    try:
        with torch.no_grad():
            device = next(model.parameters()).device
            input_ids = input_ids.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            if max_new_tokens > 0:
                from contextlib import contextmanager

                @contextmanager
                def _keep_loaded_experts(_self):
                    yield

                model._optimize_model_for_decode = _keep_loaded_experts.__get__(model, model.__class__)
                generated = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                sequences = _generated_token_ids(generated)
            else:
                model(input_ids=input_ids)
    finally:
        recorder.close()
    return recorder.steps, sequences


def _generated_token_ids(generated) -> torch.Tensor:
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    if hasattr(sequences, "to_local"):
        sequences = sequences.to_local()
    return sequences.detach().to(device="cpu", dtype=torch.int64)


def left_pad_batch(samples: list[dict], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Pad prompts on the left so one forward can prefill every sample."""
    width = max(len(sample["input_ids"]) for sample in samples)
    input_rows = []
    mask_rows = []
    pad_counts = []
    for sample in samples:
        token_ids = sample["input_ids"]
        pad = width - len(token_ids)
        input_rows.append([pad_id] * pad + list(token_ids))
        mask_rows.append([0] * pad + [1] * len(token_ids))
        pad_counts.append(pad)
    return (
        torch.tensor(input_rows, dtype=torch.long),
        torch.tensor(mask_rows, dtype=torch.long),
        pad_counts,
    )


def _cpu_indices(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    return tensor.detach().to(device="cpu", dtype=torch.int32)


def _routing_rows(tensor: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
    """Router outputs are flattened to ``(batch * seq, top_k)``."""
    if tensor.ndim == 3 and tensor.shape[0] == batch_size:
        return tensor
    if tensor.ndim != 2:
        raise ValueError(f"routing tensor must be (tokens, top_k), got {tuple(tensor.shape)}")
    top_k = tensor.shape[-1]
    expected = batch_size * seq_len
    if tensor.shape[0] != expected:
        raise ValueError(f"routing rows {tensor.shape[0]} != batch {batch_size} * seq {seq_len}")
    return tensor.view(batch_size, seq_len, top_k)


def _finished_new_tokens(generated: list[int], max_new_tokens: int, eos_ids: set[int], pad_id: int) -> list[int]:
    kept = []
    for token in generated:
        if token == pad_id:
            break
        kept.append(int(token))
        if token in eos_ids or len(kept) >= max_new_tokens:
            break
    return kept


def split_batched_steps(
    steps: list[dict],
    samples: list[dict],
    sequences: torch.Tensor,
    pad_counts: list[int],
    eos_ids: set[int],
    pad_id: int,
) -> list[dict]:
    """Turn one batched generate into one single-sample record per request.

    Padding positions are dropped. Each sample keeps its own prompt tokens and
    the decode tokens that belong to it, in the same layout as a solo run.
    """
    batch_size = len(samples)
    prefill = next(step for step in steps if step["kind"] == "prefill")
    decode_steps = [step for step in steps if step["kind"] == "decode"]
    padded_len = prefill["seq_len"]
    records = []
    for index, sample in enumerate(samples):
        prompt_ids = list(sample["input_ids"])
        left_pad = pad_counts[index]
        row = sequences[index].tolist()
        prompt_at = left_pad
        if row[prompt_at : prompt_at + len(prompt_ids)] != prompt_ids:
            found = _find_subsequence(row, prompt_ids)
            if found is None:
                raise ValueError(f"sample {sample.get('id')} prompt was not found in the generated row")
            prompt_at = found
        new_ids = _finished_new_tokens(
            row[prompt_at + len(prompt_ids) :],
            int(sample["max_new_tokens"]),
            eos_ids,
            pad_id,
        )
        sample_steps = [_slice_prefill_step(prefill, index, batch_size, padded_len, left_pad, len(prompt_ids), prompt_ids)]
        routed = max(len(new_ids) - 1, 0)
        for step_index in range(routed):
            sample_steps.append(
                _slice_decode_step(decode_steps[step_index], index, batch_size, new_ids[step_index])
            )
        records.append(
            {
                "input_ids": prompt_ids,
                "output_ids": prompt_ids + new_ids,
                "steps": sample_steps,
            }
        )
    return records


def _find_subsequence(row: list[int], needle: list[int]) -> int | None:
    if not needle:
        return 0
    last = len(row) - len(needle)
    for start in range(last + 1):
        if row[start : start + len(needle)] == needle:
            return start
    return None


def _slice_prefill_step(step, index, batch_size, padded_len, left_pad, prompt_len, prompt_ids) -> dict:
    layers = {}
    for name, captured in step["layers"].items():
        rows = _routing_rows(captured["indices"], batch_size, padded_len)
        scores = _routing_rows(captured["scores"], batch_size, padded_len)
        start = left_pad
        layers[name] = {
            "indices": rows[index, start : start + prompt_len],
            "scores": scores[index, start : start + prompt_len],
        }
    return {
        "kind": "prefill",
        "seq_len": prompt_len,
        "seconds": step["seconds"],
        "experts_implementation": step["experts_implementation"],
        "token_ids": prompt_ids,
        "layers": layers,
        "engram": _slice_engram(step, index, batch_size, padded_len, left_pad, prompt_len),
    }


def _slice_decode_step(step, index, batch_size, token_id: int) -> dict:
    layers = {}
    for name, captured in step["layers"].items():
        rows = _routing_rows(captured["indices"], batch_size, 1)
        scores = _routing_rows(captured["scores"], batch_size, 1)
        layers[name] = {
            "indices": rows[index],
            "scores": scores[index],
        }
    return {
        "kind": "decode",
        "seq_len": 1,
        "seconds": step["seconds"],
        "experts_implementation": step["experts_implementation"],
        "token_ids": [token_id],
        "layers": layers,
        "engram": _slice_engram(step, index, batch_size, 1, 0, 1),
    }


def _slice_engram(step, index, batch_size, seq_len, start, length) -> dict:
    """Keep one sample's Engram row ids and drop padding positions."""
    captured = {}
    for name, record in step.get("engram", {}).items():
        rows = _routing_rows(record["indices"], batch_size, seq_len)
        sliced = rows[index] if length == 1 and seq_len == 1 else rows[index, start : start + length]
        captured[name] = {
            "indices": sliced,
            "ngram_size": record["ngram_size"],
            "heads_per_ngram": record["heads_per_ngram"],
        }
    return captured


def _engram_heads(ngram_size: int, heads_per_ngram: int) -> list[dict]:
    heads = []
    for ngram in range(2, ngram_size + 1):
        for head in range(heads_per_ngram):
            heads.append({"ngram": ngram, "head": head})
    return heads


def _nested_numbers(tensor: torch.Tensor):
    while tensor.ndim >= 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.dtype.is_floating_point:
        return tensor.float().tolist()
    return tensor.to(torch.int64).tolist()


def _layer_sort_key(name: str) -> tuple[int, str]:
    import re

    match = re.search(r"layers\.(\d+)", name)
    index = int(match.group(1)) if match else -1
    return index, name


def _decode_text(tokenizer, token_ids: list[int]) -> str | None:
    if tokenizer is None:
        return None
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception:
        return None


def decode_trace(tokenizer, new_ids: list[int], steps: list[dict]) -> list[dict]:
    """One row per generated token, with the text produced up to that token."""
    routed = [step for step in steps if step.get("kind") == "decode"]
    rows = []
    for index, token_id in enumerate(new_ids):
        rows.append(
            {
                "index": index,
                "token_id": token_id,
                "piece": _decode_text(tokenizer, [token_id]),
                "text_so_far": _decode_text(tokenizer, new_ids[: index + 1]),
                "routing_step": routed[index]["step"] if index < len(routed) else None,
            }
        )
    return rows


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sample_directory(output: Path, sample_id: str | None, sample_count: int) -> Path:
    """Where one sample is written.

    A single request is written directly into ``output``. A serving run writes
    ``output/000``, ``output/001``, and so on, with no extra folder named after
    the token file.
    """
    root = output.with_suffix("") if output.suffix in {".json", ".pt"} else output
    if sample_count == 1 or sample_id is None:
        return root
    return root / sample_id


def write_routing_dir(
    directory: Path,
    *,
    checkpoint: Path,
    experts_implementation: str,
    ep_size: int,
    num_experts: int,
    input_ids: list[int],
    output_ids: list[int] | None,
    steps: list[dict],
    prefill_seconds: float,
    decode_seconds: float,
    decode_tokens: int,
    decode_rate: float | None,
    tokenizer=None,
    request: dict | None = None,
) -> None:
    """Write input, output, one JSON per MoE layer, and one JSON per Engram table."""
    prompt_len = len(input_ids)
    new_ids = [] if output_ids is None else output_ids[prompt_len:]
    prompt_text = _decode_text(tokenizer, input_ids)
    by_layer: dict[tuple[int, str], list[dict]] = {}
    by_engram: dict[tuple[int, str], dict] = {}
    saved_steps = []
    for step_index, step in enumerate(steps):
        token_ids = step.get("token_ids")
        text = None if not token_ids else _decode_text(tokenizer, token_ids)
        saved_steps.append({"step": step_index, "kind": step["kind"]})
        for name, captured in step["layers"].items():
            key = _layer_sort_key(name)
            by_layer.setdefault(key, []).append(
                {
                    "step": step_index,
                    "kind": step["kind"],
                    "token_ids": token_ids,
                    "text": text,
                    "indices": _nested_numbers(captured["indices"]),
                    "scores": _nested_numbers(captured["scores"]),
                }
            )
        for name, captured in step.get("engram", {}).items():
            key = _layer_sort_key(name)
            bucket = by_engram.setdefault(key, {"meta": captured, "steps": []})
            bucket["steps"].append(
                {
                    "step": step_index,
                    "kind": step["kind"],
                    "token_ids": token_ids,
                    "text": text,
                    "indices": _nested_numbers(captured["indices"]),
                }
            )
    _write_json(
        directory / "input.json",
        {
            "prompt": prompt_text,
            "user_text": None if request is None else request.get("user_text"),
            "category": None if request is None else request.get("category"),
            "source_id": None if request is None else request.get("source_id"),
            "reference": None if request is None else request.get("reference"),
            "token_ids": input_ids,
        },
    )
    _write_json(
        directory / "output.json",
        {
            "generated_text": _decode_text(tokenizer, new_ids),
            "text": None if output_ids is None else _decode_text(tokenizer, output_ids),
            "token_ids": output_ids,
            "new_token_ids": new_ids,
            "decode": decode_trace(tokenizer, new_ids, saved_steps),
            "prefill_seconds": prefill_seconds,
            "decode_seconds": decode_seconds,
            "decode_tokens": decode_tokens,
            "decode_tokens_per_second": decode_rate,
            "ep_size": ep_size,
            "experts_implementation": experts_implementation,
            "num_experts": num_experts,
            "engram_layers": [layer_index for (layer_index, _name) in sorted(by_engram)],
            "checkpoint": str(checkpoint),
            "transformers": REQUIRED_VERSION,
        },
    )
    layers_dir = directory / "layers"
    for (layer_index, name), layer_steps in sorted(by_layer.items()):
        filename = f"{layer_index:02d}.json" if layer_index >= 0 else "unknown.json"
        _write_json(
            layers_dir / filename,
            {
                "layer": layer_index,
                "name": name,
                "steps": layer_steps,
            },
        )
    engram_dir = directory / "engram"
    for (layer_index, name), bucket in sorted(by_engram.items()):
        meta = bucket["meta"]
        filename = f"{layer_index:02d}.json" if layer_index >= 0 else "unknown.json"
        _write_json(
            engram_dir / filename,
            {
                "layer": layer_index,
                "name": name,
                "parameter": "ngram_embedding.weight",
                "ngram_size": meta["ngram_size"],
                "heads_per_ngram": meta["heads_per_ngram"],
                "heads": _engram_heads(meta["ngram_size"], meta["heads_per_ngram"]),
                "steps": bucket["steps"],
            },
        )


def _eos_ids(tokenizer) -> set[int]:
    eos = tokenizer.eos_token_id
    if isinstance(eos, int):
        values = {eos}
    elif eos is None:
        values = set()
    else:
        values = {int(token_id) for token_id in eos}
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if isinstance(im_end, int) and im_end >= 0:
        values.add(im_end)
    return values


def _timing(steps: list[dict]) -> tuple[float, float, int, float | None]:
    prefill_seconds = 0.0
    decode_seconds = 0.0
    decode_tokens = 0
    for step in steps:
        if step["kind"] == "prefill":
            prefill_seconds += step["seconds"]
        else:
            decode_seconds += step["seconds"]
            decode_tokens += step["seq_len"]
    decode_rate = None if decode_seconds == 0 else decode_tokens / decode_seconds
    return prefill_seconds, decode_seconds, decode_tokens, decode_rate


def _run_single(model, sample: dict) -> dict:
    input_ids = torch.tensor(sample["input_ids"], dtype=torch.long).unsqueeze(0)
    steps, sequences = record_routing(model, input_ids, sample["max_new_tokens"])
    if sequences is None:
        output_ids = None
    elif sequences.ndim == 1:
        output_ids = sequences.tolist()
    else:
        output_ids = sequences[0].tolist()
    return {"input_ids": list(sample["input_ids"]), "output_ids": output_ids, "steps": steps}


def _run_batch(model, samples: list[dict], eos_ids: set[int], pad_id: int) -> list[dict]:
    input_ids, attention_mask, pad_counts = left_pad_batch(samples, pad_id)
    max_new_tokens = max(int(sample["max_new_tokens"]) for sample in samples)
    steps, sequences = record_routing(model, input_ids, max_new_tokens, attention_mask)
    if sequences is None:
        raise RuntimeError("batched generate did not return token ids")
    return split_batched_steps(steps, samples, sequences, pad_counts, eos_ids - {pad_id}, pad_id)


def _store_sample(args, rank, tokenizer, num_experts, sample_count, sample, record, recorded) -> None:
    prefill_seconds, decode_seconds, decode_tokens, decode_rate = _timing(record["steps"])
    sample_dir = sample_directory(args.output, sample["id"], sample_count)
    if rank == 0:
        write_routing_dir(
            sample_dir,
            checkpoint=args.checkpoint,
            experts_implementation=args.experts_implementation,
            ep_size=args.ep_size,
            num_experts=num_experts,
            input_ids=record["input_ids"],
            output_ids=record["output_ids"],
            steps=record["steps"],
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            decode_tokens=decode_tokens,
            decode_rate=decode_rate,
            tokenizer=tokenizer,
            request=sample,
        )
        generated = json.loads((sample_dir / "output.json").read_text())["generated_text"]
        print(
            json.dumps(
                {
                    "sample": sample["id"],
                    "category": sample.get("category"),
                    "generated_text": generated,
                    "prefill_seconds": prefill_seconds,
                    "decode_seconds": decode_seconds,
                    "decode_tokens": decode_tokens,
                    "decode_tokens_per_second": decode_rate,
                    "output": str(sample_dir),
                },
                ensure_ascii=False,
            )
        )
    recorded.append(
        {
            "id": sample["id"],
            "category": sample.get("category"),
            "user_text": sample.get("user_text"),
            "prompt_tokens": len(record["input_ids"]),
            "max_new_tokens": sample["max_new_tokens"],
            "decode_tokens": decode_tokens,
            "directory": sample_dir.name if sample_count > 1 else ".",
        }
    )


def _collect_samples(args, tokenizer) -> list[dict]:
    if args.workload or args.dataset_path:
        from moe_serving.datasets import load_samples

        seq_len = 2048 if args.seq_len == 32 else args.seq_len
        name, samples = load_samples(
            args.workload,
            tokenizer,
            seq_len=seq_len,
            max_new_tokens=args.max_new_tokens,
            num_samples=args.num_samples,
            dataset_path=args.dataset_path,
        )
        args.workload = name
        return samples
    if args.token_ids:
        token_ids = json.loads(args.token_ids.read_text())
    else:
        token_ids = tokenizer("Record the MoE routing choices.", return_tensors="pt").input_ids[0].tolist()
    if args.seq_len and len(token_ids) > args.seq_len:
        token_ids = token_ids[: args.seq_len]
    return [
        {
            "id": None,
            "category": None,
            "max_new_tokens": 16 if args.max_new_tokens is None else args.max_new_tokens,
            "user_text": None,
            "input_ids": [int(token_id) for token_id in token_ids],
        }
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Record Qwen3.8-Flash-Next FP8 MoE routing choices")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Decode cap. Default 16 for one prompt. For a dataset, the per-dataset cap, shortened to the reference answer when one exists. 0 records prefill only.",
    )
    parser.add_argument("--token-ids", type=Path, default=None, help="JSON list of token ids for one request.")
    parser.add_argument(
        "--workload",
        choices=("serving", "sharegpt", "mt-bench", "gsm8k", "humaneval", "alpaca", "messages"),
        default=None,
        help="Dataset to replay. serving is the built-in mix. The others download into datasets/ unless --dataset-path is set.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Local JSON or JSONL dataset. Format is detected when --workload is omitted.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="How many requests to run. Default: all of serving, mt-bench, and humaneval; 128 of sharegpt, gsm8k, and alpaca. Negative means all.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/routing"),
        help="Directory. One request is written here. A serving workload writes 000, 001, ... inside it.",
    )
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--experts-implementation", default="grouped_mm")
    parser.add_argument("--ep-size", type=int, default=4, help="Expert-parallel ranks. 1 keeps the single-process device map.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Requests per forward. Default: the whole serving workload in one batch, otherwise 1.",
    )
    args = parser.parse_args(argv)
    _relaunch_under_torchrun(args.ep_size)

    ensure_transformers_5190()
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    quant = config.quantization_config
    quant_method = quant["quant_method"] if isinstance(quant, dict) else quant.quant_method
    if quant_method != "fp8":
        raise ValueError(f"{args.checkpoint} is {quant_method}, expected fp8")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    samples = _collect_samples(args, tokenizer)
    rank = int(os.environ.get("RANK", "0"))
    print(
        f"transformers {REQUIRED_VERSION}  checkpoint {args.checkpoint}  "
        f"experts {args.experts_implementation}  ep_size {args.ep_size}  "
        f"workload {args.workload}  samples {len(samples)}  rank {rank}"
    )
    model = load_fp8_model(args.checkpoint, args.device_map, args.experts_implementation, args.ep_size)
    model.eval()
    if rank == 0:
        experts = model.get_submodule("model.language_model.layers.0.mlp.experts")
        weight = experts.gate_up_proj if hasattr(experts, "gate_up_proj") else experts.up_proj
        tensor = weight.data if isinstance(weight, torch.nn.Parameter) else weight
        local = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        print(f"local_experts {local.shape[0]}  weight {tuple(tensor.shape)}  device {local.device}")
    device_map_used = getattr(model, "hf_device_map", None)
    if rank == 0 and device_map_used:
        used = sorted({str(device) for device in device_map_used.values()})
        print("devices", " ".join(used))

    num_experts = config.text_config.num_experts
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = len(samples) if args.workload == "serving" and len(samples) > 1 else 1
    eos_ids = _eos_ids(tokenizer)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, int) else next(iter(eos_ids))
    recorded = []
    groups = [samples[start : start + batch_size] for start in range(0, len(samples), batch_size)]
    for group in groups:
        if len(group) == 1:
            group_records = [_run_single(model, group[0])]
        else:
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "batch": [sample["id"] for sample in group],
                            "max_new_tokens": max(sample["max_new_tokens"] for sample in group),
                        },
                        ensure_ascii=False,
                    )
                )
            group_records = _run_batch(model, group, eos_ids, pad_id)
        for sample, record in zip(group, group_records):
            _store_sample(
                args,
                rank,
                tokenizer,
                num_experts,
                len(samples),
                sample,
                record,
                recorded,
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
    if rank == 0 and len(samples) > 1:
        _write_json(
            (args.output if args.output.suffix not in {".json", ".pt"} else args.output.with_suffix("")) / "workload.json",
            {
                "workload": args.workload,
                "dataset_path": None if args.dataset_path is None else str(args.dataset_path),
                "num_samples": args.num_samples,
                "batch_size": batch_size,
                "requests": recorded,
            },
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
