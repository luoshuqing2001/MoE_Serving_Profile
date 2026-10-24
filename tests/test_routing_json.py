import json

import torch

from moe_serving.record_gpt_oss import write_sample_records
from moe_serving.record_routing import sample_directory


class Tokenizer:
    def decode(self, ids, **kwargs):
        return ' '.join(map(str, ids))


def test_batch_json_separates_samples_and_trims_eos(tmp_path):
    inputs = torch.tensor([[0, 11], [21, 22]])
    mask = torch.tensor([[0, 1], [1, 1]])
    records = []
    for step, ids, valid in [
        (0, inputs, mask),
        (1, torch.tensor([[9], [31]]), torch.ones(2, 1)),
        (2, torch.tensor([[0], [32]]), torch.tensor([[0], [1]])),
    ]:
        for layer in range(2):
            records.append(dict(
                step=step, phase='prefill' if step == 0 else 'decode', layer=layer,
                input_ids=ids, valid_tokens=valid,
                position_ids=torch.zeros_like(ids),
                expert_indices=ids.unsqueeze(-1),
                routing_weights=torch.ones(*ids.shape, 1),
            ))
    write_sample_records(
        tmp_path, prompts=['a', 'b'], input_ids=inputs, attention_mask=mask,
        generated=torch.tensor([[0, 11, 9, 0, 0], [21, 22, 31, 32, 9]]),
        records=records, tokenizer=Tokenizer(), eos_token_id=9, metadata={'tp_size': 2},
    )
    first = json.loads((tmp_path / '000/layers/00.json').read_text())
    second = json.loads((tmp_path / '001/layers/00.json').read_text())
    assert len(first['steps']) == 1
    assert first['steps'][0]['indices'] == [[11]]
    assert [s['token_ids'] for s in second['steps']] == [[21, 22], [31], [32]]
    assert json.loads((tmp_path / '000/output.json').read_text())['new_token_ids'] == [9]
    assert (tmp_path / '001/layers/01.json').is_file()
    assert json.loads((tmp_path / 'workload.json').read_text())['batch_size'] == 2


def test_single_sample_has_own_directory(tmp_path):
    assert sample_directory(tmp_path / 'trace.pt', None, 1) == tmp_path / 'trace/000'
    assert sample_directory(tmp_path, '007', 1) == tmp_path / '007'
