"""Prompt datasets for a recorded serving run.

``serving`` is the built-in mix. The others are the usual public sets:

- ``sharegpt``: first turn of ShareGPT Vicuna conversations, the trace vLLM and SGLang replay
- ``mt-bench``: MT-Bench first-turn questions
- ``gsm8k``: GSM8K test questions
- ``humaneval``: HumanEval function prompts
- ``alpaca``: Stanford Alpaca instructions
- ``messages``: a local JSON or JSONL file of ``{"messages": [{"role", "content"}]}``

Files are cached under ``datasets/`` (or ``MOE_DATASET_DIR``). A local file
passed with ``--dataset-path`` is used as-is.
"""

from __future__ import annotations

import gzip
import json
import os
import urllib.request
from pathlib import Path

from moe_serving.serving_workload import render_messages, serving_samples

WORKLOAD_NAMES = ("serving", "sharegpt", "mt-bench", "gsm8k", "humaneval", "alpaca", "messages")

# Used when the dataset has no reference completion, and as the cap when
# ``--max-new-tokens`` is omitted. A reference completion can only shorten this.
DEFAULT_DECODE_CAP = {
    "sharegpt": 128,
    "mt-bench": 256,
    "gsm8k": 256,
    "humaneval": 256,
    "alpaca": 128,
    "messages": 128,
}

# ``None`` keeps every request. The large sets stay bounded unless asked.
DEFAULT_NUM_SAMPLES = {
    "serving": None,
    "sharegpt": 128,
    "mt-bench": None,
    "gsm8k": 128,
    "humaneval": None,
    "alpaca": 128,
    "messages": 128,
}

_REMOTE_FILES = {
    "mt-bench": (
        "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl",
        "mt-bench/question.jsonl",
    ),
    "gsm8k": (
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "gsm8k/test.jsonl",
    ),
    "humaneval": (
        "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
        "humaneval/HumanEval.jsonl",
    ),
    "alpaca": (
        "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json",
        "alpaca/alpaca_data.json",
    ),
}

_SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
_SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"


def dataset_root() -> Path:
    override = os.environ.get("MOE_DATASET_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "datasets"


def load_samples(
    name: str | None,
    tokenizer,
    *,
    seq_len: int,
    max_new_tokens: int | None,
    num_samples: int | None,
    dataset_path: Path | None = None,
) -> tuple[str, list[dict]]:
    """Return ``(dataset name, samples)`` ready for ``record_routing``."""
    path = None if dataset_path is None else Path(dataset_path)
    if name is None:
        if path is None:
            raise ValueError("pass --workload or --dataset-path")
        name = detect_format(path)
    if name not in WORKLOAD_NAMES:
        known = ", ".join(WORKLOAD_NAMES)
        raise ValueError(f"unknown dataset {name}. known datasets: {known}")
    if name == "serving":
        samples = serving_samples(tokenizer, seq_len)
        return name, _take(samples, num_samples if num_samples is not None else DEFAULT_NUM_SAMPLES["serving"])

    requests = load_requests(name, path)
    cap = DEFAULT_DECODE_CAP[name] if max_new_tokens is None else max_new_tokens
    limit = num_samples if num_samples is not None else DEFAULT_NUM_SAMPLES[name]
    samples = []
    for request in requests:
        if limit is not None and limit >= 0 and len(samples) >= limit:
            break
        token_ids = render_messages(tokenizer, request["messages"], seq_len, truncate=False)
        if not token_ids or (seq_len and len(token_ids) > seq_len):
            continue
        decode_cap = _decode_cap(tokenizer, request, cap)
        samples.append(
            {
                "id": f"{len(samples):03d}",
                "source_id": request.get("source_id"),
                "category": request.get("category") or name,
                "max_new_tokens": decode_cap,
                "user_text": request["messages"][-1]["content"],
                "reference": request.get("reference"),
                "messages": request["messages"],
                "input_ids": token_ids,
            }
        )
    if not samples:
        raise ValueError(f"{name} produced no prompts that fit in {seq_len} tokens")
    return name, samples


def load_requests(name: str, path: Path | None = None) -> list[dict]:
    """Read one dataset into chat requests, downloading the public file when needed."""
    source = path if path is not None else _ensure_cached(name)
    records = _read_records(source)
    parser = {
        "sharegpt": _parse_sharegpt,
        "mt-bench": _parse_mt_bench,
        "gsm8k": _parse_gsm8k,
        "humaneval": _parse_humaneval,
        "alpaca": _parse_alpaca,
        "messages": _parse_messages,
    }[name]
    requests = []
    for index, record in enumerate(records):
        parsed = parser(record, index)
        if parsed is not None:
            requests.append(parsed)
    if not requests:
        raise ValueError(f"{source} has no usable {name} requests")
    return requests


def detect_format(path: Path) -> str:
    records = _read_records(path)
    if not records:
        raise ValueError(f"{path} is empty")
    row = records[0]
    if not isinstance(row, dict):
        raise ValueError(f"{path} does not look like a prompt dataset")
    if "conversations" in row:
        return "sharegpt"
    if "instruction" in row and "output" in row:
        return "alpaca"
    if "turns" in row and "question_id" in row:
        return "mt-bench"
    if "task_id" in row and "canonical_solution" in row:
        return "humaneval"
    if "question" in row and "answer" in row:
        return "gsm8k"
    if "messages" in row:
        return "messages"
    raise ValueError(
        f"{path} is not sharegpt, alpaca, mt-bench, humaneval, gsm8k, or a messages file"
    )


def _take(samples: list[dict], limit: int | None) -> list[dict]:
    if limit is None or limit < 0:
        return samples
    return samples[:limit]


def _decode_cap(tokenizer, request: dict, cap: int) -> int:
    explicit = request.get("max_new_tokens")
    if explicit is not None:
        length = int(explicit)
    else:
        reference = (request.get("reference") or "").strip()
        counted = _count_tokens(tokenizer, reference) if reference else 0
        length = counted if counted > 0 else cap
    if cap >= 0:
        length = min(length, cap)
    return max(int(length), 0)


def _count_tokens(tokenizer, text: str) -> int:
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return len(token_ids)


def _ensure_cached(name: str) -> Path:
    if name == "messages":
        raise ValueError("messages datasets need --dataset-path")
    if name == "sharegpt":
        return _cache_sharegpt()
    url, relative = _REMOTE_FILES[name]
    dest = dataset_root() / relative
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + f".partial.{os.getpid()}")
    try:
        if url.endswith(".gz"):
            gz_path = partial.with_suffix(".gz")
            _urlretrieve(url, gz_path)
            with gzip.open(gz_path, "rb") as source, partial.open("wb") as target:
                target.write(source.read())
            gz_path.unlink(missing_ok=True)
        else:
            _urlretrieve(url, partial)
        os.replace(partial, dest)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return dest


def _cache_sharegpt() -> Path:
    dest = dataset_root() / "sharegpt" / _SHAREGPT_FILE
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    offline = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(_SHAREGPT_REPO, _SHAREGPT_FILE, repo_type="dataset")
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not download ShareGPT ({_SHAREGPT_REPO} {_SHAREGPT_FILE}). "
            f"Place that JSON at {dest} or pass --dataset-path. ({exc})"
        ) from exc
    finally:
        if offline is not None:
            os.environ["HF_HUB_OFFLINE"] = offline
    if not dest.exists():
        os.symlink(cached, dest)
    return dest


def _urlretrieve(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "moe-serving"})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def _read_records(path: Path) -> list:
    if path.suffix == ".gz":
        text = gzip.open(path, "rt", encoding="utf-8").read()
    else:
        text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped[0] == "[":
        data = json.loads(stripped)
        if not isinstance(data, list):
            raise ValueError(f"{path} must contain a list of records")
        return data
    records = []
    for line in stripped.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def _role(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in {"human", "user"}:
        return "user"
    if text in {"gpt", "assistant", "chatgpt", "bard"}:
        return "assistant"
    if text == "system":
        return "system"
    return text


def _text(record: dict, *keys: str) -> str:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _parse_sharegpt(record: dict, index: int) -> dict | None:
    conversations = record.get("conversations") or record.get("messages") or []
    for turn_index, turn in enumerate(conversations):
        if _role(turn.get("from") or turn.get("role")) != "user":
            continue
        prompt = _text(turn, "value", "content")
        if not prompt:
            continue
        reference = None
        if turn_index + 1 < len(conversations):
            reply = conversations[turn_index + 1]
            if _role(reply.get("from") or reply.get("role")) == "assistant":
                reference = _text(reply, "value", "content") or None
        return {
            "source_id": str(record.get("id", index)),
            "category": "sharegpt",
            "messages": [{"role": "user", "content": prompt}],
            "reference": reference,
        }
    return None


def _parse_mt_bench(record: dict, index: int) -> dict | None:
    turns = record.get("turns") or []
    if not turns or not str(turns[0]).strip():
        return None
    return {
        "source_id": str(record.get("question_id", index)),
        "category": record.get("category") or "mt-bench",
        "messages": [{"role": "user", "content": str(turns[0]).strip()}],
        "reference": None,
    }


def _parse_gsm8k(record: dict, index: int) -> dict | None:
    question = _text(record, "question")
    if not question:
        return None
    return {
        "source_id": str(record.get("id", index)),
        "category": "gsm8k",
        "messages": [{"role": "user", "content": question}],
        "reference": _text(record, "answer") or None,
    }


def _parse_humaneval(record: dict, index: int) -> dict | None:
    prompt = _text(record, "prompt")
    if not prompt:
        return None
    return {
        "source_id": str(record.get("task_id", index)),
        "category": "humaneval",
        "messages": [{"role": "user", "content": prompt}],
        "reference": _text(record, "canonical_solution") or None,
    }


def _parse_alpaca(record: dict, index: int) -> dict | None:
    instruction = _text(record, "instruction")
    if not instruction:
        return None
    extra = _text(record, "input")
    content = f"{instruction}\n\n{extra}" if extra else instruction
    return {
        "source_id": str(record.get("id", index)),
        "category": "alpaca",
        "messages": [{"role": "user", "content": content}],
        "reference": _text(record, "output") or None,
    }


def _parse_messages(record: dict, index: int) -> dict | None:
    messages = []
    for turn in record.get("messages") or []:
        role = _role(turn.get("role") or turn.get("from"))
        content = _text(turn, "content", "value")
        if role in {"system", "user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    last_user = None
    for turn_index, turn in enumerate(messages):
        if turn["role"] == "user":
            last_user = turn_index
    if last_user is None:
        return None
    reference = record.get("reference") or record.get("output")
    if isinstance(reference, str):
        reference = reference.strip() or None
    else:
        reference = None
    if reference is None and last_user + 1 < len(messages) and messages[last_user + 1]["role"] == "assistant":
        reference = messages[last_user + 1]["content"]
    return {
        "source_id": str(record.get("id", index)),
        "category": record.get("category") or "messages",
        "messages": messages[: last_user + 1],
        "reference": reference,
        "max_new_tokens": record.get("max_new_tokens"),
    }
