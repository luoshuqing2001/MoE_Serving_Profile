import json

import torch

from moe_serving.record_gsm8k import (
    distributed_command,
    padded_batch,
    parser,
    prepare_samples,
    store_batch,
)


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return list(range(10, 10 + len(messages[0]["content"].split())))

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def questions(path):
    rows = [
        {
            "question": f"Question number {i}: what is 2 + 3?",
            "answer": f"REFERENCE-ONLY-{i} #### 5",
        }
        for i in range(3)
    ]
    path.write_text("\n".join(map(json.dumps, rows)) + "\n")


def test_preparation_uses_chat_template_without_gold_or_reference_cap(tmp_path):
    dataset = tmp_path / "gsm.jsonl"
    questions(dataset)
    args = parser().parse_args(
        [
            "--dataset-path",
            str(dataset),
            "--start-index",
            "1",
            "--num-samples",
            "-1",
            "--max-new-tokens",
            "1024",
        ]
    )
    tokenizer = Tokenizer()
    samples = prepare_samples(args, tokenizer)
    assert [s["id"] for s in samples] == ["000001", "000002"]
    assert samples[0]["reference"] == "REFERENCE-ONLY-1 #### 5"
    assert args.max_new_tokens == 1024
    assert all(
        "REFERENCE-ONLY" not in calls[0][0]["content"] for calls in tokenizer.calls
    )
    assert all(call[1]["reasoning_effort"] == "low" for call in tokenizer.calls)
    assert all(call[1]["add_generation_prompt"] for call in tokenizer.calls)


def test_distributed_command_keeps_tp_ep_and_avoids_preflight_recursion():
    argv = ["--tp-size", "4", "--ep-size", "4", "--dry-run", "--prepare-only"]
    args = parser().parse_args(argv)
    command = distributed_command(args, argv)
    assert "--nproc-per-node=4" in command
    assert command[-4:] == ["--tp-size", "4", "--ep-size", "4"]
    assert "--dry-run" not in command and "--prepare-only" not in command


def test_multiple_batches_keep_unique_directories_and_progress(tmp_path):
    args = parser().parse_args(["--output", str(tmp_path), "--max-new-tokens", "2"])
    tokenizer = Tokenizer()
    completed = []
    metadata = {
        "model_type": "gpt_oss",
        "eos_token_id": [9],
        "parallel_numerics": "reference",
    }
    for start in [0, 2]:
        group = [
            {
                "id": f"{index:06d}",
                "source_id": str(index),
                "question": "2+3?",
                "reference": "#### 5",
                "user_text": "2+3?",
                "input_ids": [1] * (index % 2 + 1),
            }
            for index in range(start, start + 2)
        ]
        ids, mask = padded_batch(group, 0, "cpu")
        generated = torch.cat((ids, torch.tensor([[5, 9], [6, 7]])), -1)
        records = [
            {
                "layer": 0,
                "step": 0,
                "phase": "prefill",
                "valid_tokens": mask.bool(),
                "input_ids": ids,
                "position_ids": (mask.cumsum(-1) - 1).clamp_min(0),
                "expert_indices": torch.zeros(*ids.shape, 1, dtype=torch.long),
                "routing_weights": torch.ones(*ids.shape, 1),
            }
        ]
        store_batch(
            args,
            tokenizer,
            group,
            ids,
            mask,
            generated,
            records,
            metadata,
            completed,
            1.0,
        )
        marker = tmp_path / "batches" / f"{start:06d}-{start + 1:06d}.json"
        assert json.loads(marker.read_text())["status"] == "complete"
        for sample in group:
            directory = tmp_path / sample["id"]
            assert (directory / "input.json").is_file()
            assert (directory / "output.json").is_file()
            assert (directory / "layers/00.json").is_file()
        assert len((tmp_path / "results.jsonl").read_text().splitlines()) == start + 2
    assert len(completed) == 4
    for i in range(4):
        directory = tmp_path / f"{i:06d}"
        assert (directory / "layers/00.json").exists()
        source = json.loads((directory / "input.json").read_text())
        result = json.loads((directory / "output.json").read_text())
        assert source["source_id"] == str(i)
        assert result["reached_token_limit"] == (i % 2 == 1)
        assert result["new_token_ids"] == ([5, 9] if i % 2 == 0 else [6, 7])
    manifest = json.loads((tmp_path / "workload.json").read_text())
    assert manifest["completed_samples"] == 4
    assert [s["directory"] for s in manifest["requests"]] == [
        f"{i:06d}" for i in range(4)
    ]
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 4


def test_chat_template_mapping_returns_token_ids_not_mapping_keys(tmp_path):
    dataset = tmp_path / "gsm.jsonl"
    questions(dataset)
    args = parser().parse_args(["--dataset-path", str(dataset), "--num-samples", "1"])

    class MappingTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["return_dict"] is False
            return {"input_ids": [101, 102, 103], "attention_mask": [1, 1, 1]}

    samples = prepare_samples(args, MappingTokenizer())
    assert samples[0]["input_ids"] == [101, 102, 103]


def test_dry_run_does_not_prepare_or_load_model(monkeypatch, capsys):
    from moe_serving import record_gsm8k

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not prepare data, launch, or load a model")

    monkeypatch.setattr(record_gsm8k, "validate", lambda args: None)
    monkeypatch.setattr(record_gsm8k, "load_requests", forbidden)
    monkeypatch.setattr(record_gsm8k, "ensure_transformers_5200", forbidden)
    monkeypatch.setattr(record_gsm8k, "load_parallel_model", forbidden)
    monkeypatch.setattr(record_gsm8k.os, "execv", forbidden)
    record_gsm8k.main(["--dry-run"])
    assert "--nproc-per-node=4" in capsys.readouterr().out


def test_atomic_json_sync_failure_preserves_previous_file(tmp_path, monkeypatch):
    import pytest

    from moe_serving import durable_io

    path = tmp_path / "result.json"
    durable_io.write_json(path, {"status": "previous"})

    def fail_sync(descriptor):
        raise OSError("disk sync failed")

    monkeypatch.setattr(durable_io.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="disk sync failed"):
        durable_io.write_json(path, {"status": "new"})
    assert json.loads(path.read_text()) == {"status": "previous"}
    assert list(tmp_path.iterdir()) == [path]
