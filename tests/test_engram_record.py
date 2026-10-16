import json

import torch

from moe_serving.record_routing import RoutingRecorder, split_batched_steps, write_routing_dir
from moe_serving.transformers_src import ensure_transformers_5190


class _Ngram(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ngram_embedding = torch.nn.Embedding(8, 2)
        self.ngram_size = 3
        self.heads_per_ngram = 2

    def forward(self, input_ids):
        return self.ngram_embedding(input_ids)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = _Ngram()

    def forward(self, input_ids=None, **kwargs):
        return self.table(input_ids)


def test_engram_hook_records_activated_rows():
    ensure_transformers_5190()
    model = _Model()
    recorder = RoutingRecorder()
    recorder.attach(model)
    rows = torch.tensor([[[1, 2], [3, 4]]])
    model(input_ids=rows)
    recorder.close()
    saved = recorder.steps[0]["engram"]["table"]
    assert saved["ngram_size"] == 3
    assert saved["heads_per_ngram"] == 2
    assert torch.equal(saved["indices"], rows.to(torch.int32))


def test_engram_rows_follow_each_sample(tmp_path):
    heads = 4
    table = "model.layers.2.ple.ple_embedding"
    prefill_rows = torch.arange(2 * 4 * heads, dtype=torch.int32).view(2, 4, heads)
    gate = "model.layers.0.mlp.gate"
    prefill = {
        "kind": "prefill",
        "seq_len": 4,
        "seconds": 0.1,
        "experts_implementation": "grouped_mm",
        "layers": {
            gate: {
                "indices": torch.arange(16, dtype=torch.int16).view(8, 2),
                "scores": torch.ones(8, 2, dtype=torch.float16),
            }
        },
        "engram": {
            table: {"indices": prefill_rows, "ngram_size": 3, "heads_per_ngram": 2},
        },
    }
    decode_rows = torch.tensor([[[10, 11, 12, 13]], [[20, 21, 22, 23]]], dtype=torch.int32)
    decode = {
        "kind": "decode",
        "seq_len": 1,
        "seconds": 0.01,
        "experts_implementation": "grouped_mm",
        "layers": {
            gate: {
                "indices": torch.tensor([[1, 2], [3, 4]], dtype=torch.int16),
                "scores": torch.ones(2, 2, dtype=torch.float16),
            }
        },
        "engram": {
            table: {"indices": decode_rows, "ngram_size": 3, "heads_per_ngram": 2},
        },
    }
    samples = [
        {"id": "000", "input_ids": [5, 6, 7], "max_new_tokens": 2, "category": "qa", "user_text": "q"},
        {"id": "001", "input_ids": [8, 9, 10, 11], "max_new_tokens": 2, "category": "qa", "user_text": "q"},
    ]
    sequences = torch.tensor(
        [
            [0, 5, 6, 7, 100, 101],
            [8, 9, 10, 11, 200, 201],
        ]
    )
    records = split_batched_steps([prefill, decode], samples, sequences, [1, 0], set(), 0)
    assert torch.equal(records[0]["steps"][0]["engram"][table]["indices"], prefill_rows[0, 1:4])
    assert torch.equal(records[1]["steps"][0]["engram"][table]["indices"], prefill_rows[1])
    write_routing_dir(
        tmp_path,
        checkpoint=tmp_path,
        experts_implementation="grouped_mm",
        ep_size=4,
        num_experts=512,
        input_ids=records[0]["input_ids"],
        output_ids=records[0]["output_ids"],
        steps=records[0]["steps"],
        prefill_seconds=0.1,
        decode_seconds=0.01,
        decode_tokens=1,
        decode_rate=100.0,
        request=samples[0],
    )
    payload = json.loads((tmp_path / "engram" / "02.json").read_text())
    assert payload["parameter"] == "ngram_embedding.weight"
    assert payload["heads"] == [
        {"ngram": 2, "head": 0},
        {"ngram": 2, "head": 1},
        {"ngram": 3, "head": 0},
        {"ngram": 3, "head": 1},
    ]
    assert payload["steps"][0]["indices"] == prefill_rows[0, 1:4].tolist()
    assert payload["steps"][1]["indices"] == [10, 11, 12, 13]
    summary = json.loads((tmp_path / "output.json").read_text())
    assert summary["engram_layers"] == [2]
