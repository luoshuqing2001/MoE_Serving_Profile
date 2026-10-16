import torch
from safetensors.torch import save_file

from moe_serving.checkpoint import MappedCheckpoint
from moe_serving.engram.dram import DramEngramTable
from moe_serving.engram.ple import NgramParameters, PleLayer


def _write_shards(directory, shards):
    weight_map = {}
    for index, rows in enumerate(shards):
        name = f"shard_{index}.weight"
        filename = f"shard_{index}.safetensors"
        save_file({name: rows}, directory / filename)
        weight_map[name] = filename
    import json

    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return [f"shard_{index}.weight" for index in range(len(shards))]


def test_dram_gather_reads_across_shards_without_copying_the_table(tmp_path):
    first = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    second = torch.arange(8, 16, dtype=torch.bfloat16).reshape(4, 2)
    names = _write_shards(tmp_path, [first, second])
    table = DramEngramTable(MappedCheckpoint(str(tmp_path)), names)
    assert table._views == [None, None]
    gathered = table.gather(torch.tensor([[0, 5], [3, 7]]))
    assert table._views[0] is not None and table._views[1] is not None
    expected = torch.stack(
        [
            torch.stack([first[0], second[1]]),
            torch.stack([first[3], second[3]]),
        ]
    )
    assert torch.equal(gathered, expected)


def test_ple_uses_gathered_rows_and_keeps_a_convolution_state(tmp_path):
    torch.manual_seed(0)
    rows = torch.randn(6, 4)
    names = _write_shards(tmp_path, [rows[:3], rows[3:]])
    table = DramEngramTable(MappedCheckpoint(str(tmp_path)), names)
    parameters = NgramParameters(
        multipliers=torch.tensor([1, 1, 1]),
        vocab_sizes=torch.tensor([3, 3]),
        offsets=torch.tensor([0, 3]),
        heads_per_ngram=1,
        eos_token_id=0,
    )
    hidden_size = 4
    hc = 2
    embed = table.dim * parameters.num_heads
    ple = PleLayer(
        table=table,
        parameters=parameters,
        key_proj=torch.randn(hc * hidden_size, embed),
        value_proj=torch.randn(hidden_size, embed),
        norm_query=torch.ones(hc * hidden_size),
        norm_key=torch.ones(hc * hidden_size),
        norm_conv=torch.ones(hc * hidden_size),
        conv_weight=torch.randn(hc * hidden_size, 1, 3),
        hc_count=hc,
    )
    tokens = torch.tensor([[1, 2, 1]])
    embeddings = ple.lookup(tokens)
    assert embeddings.shape == (1, 3, embed)
    hidden = torch.randn(1, 3, hidden_size)
    updated, state = ple(hidden, embeddings)
    assert updated.shape == hidden.shape
    assert state.shape == (1, 2, hc * hidden_size)
    assert torch.isfinite(updated).all()
