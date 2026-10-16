import torch

from moe_serving.engram.hash import (
    NgramContext,
    build_layer_multipliers,
    derive_head_layout,
    hash_context,
    hash_token_ids,
)

# Buffers shipped in Qwen3.8-Flash-Next.
MULTIPLIERS = [23703573157769, 20109073645365, 8052911324071]
VOCAB_SIZES = [
    20000003,
    20000023,
    20000033,
    20000047,
    20000059,
    20000063,
    20000069,
    20000077,
    20000081,
    20000093,
    20000107,
    20000147,
    20000153,
    20000159,
    20000161,
    20000171,
]
OFFSETS = [
    0,
    20000003,
    40000026,
    60000059,
    80000106,
    100000165,
    120000228,
    140000297,
    160000374,
    180000455,
    200000548,
    220000655,
    240000802,
    260000955,
    280001114,
    300001275,
]


def test_checkpoint_multipliers_match_the_hash_derivation():
    derived = build_layer_multipliers(248320, 3, ple_layer_index=0, seed=1234)
    assert derived == MULTIPLIERS


def test_checkpoint_head_layout_matches_the_prime_derivation():
    sizes, offsets = derive_head_layout(20_000_000, ngram_size=3, heads_per_ngram=8)
    assert sizes == VOCAB_SIZES
    assert offsets == OFFSETS


def test_prefill_hash_matches_the_decode_shift_register():
    eos = 7
    multipliers = torch.tensor([3, 5, 9], dtype=torch.int64)
    vocab_sizes = torch.tensor([11, 13, 17, 19], dtype=torch.int64)
    offsets = torch.tensor([0, 11, 24, 41], dtype=torch.int64)
    tokens = torch.tensor([[4, 5, eos, 8, 1]], dtype=torch.int64)
    prefill = hash_token_ids(tokens, multipliers, vocab_sizes, offsets, heads_per_ngram=2, eos_token_id=eos)

    context = NgramContext(ngram_size=3, eos_token_id=eos, batch=1)
    stepped = []
    for step in range(tokens.shape[1]):
        window = context.step(tokens[:, step])
        stepped.append(hash_context(window, multipliers, vocab_sizes, offsets, heads_per_ngram=2))
    assert torch.equal(prefill, torch.stack(stepped, dim=1))


def test_eos_blocks_older_tokens_and_bigram_heads_come_first():
    eos = 7
    multipliers = torch.tensor([3, 5, 9], dtype=torch.int64)
    vocab_sizes = torch.tensor([11, 13, 17, 19], dtype=torch.int64)
    offsets = torch.tensor([0, 11, 24, 41], dtype=torch.int64)
    # After EOS, the next token only sees EOS in its history.
    tokens = torch.tensor([4, eos, 8], dtype=torch.int64)
    rows = hash_token_ids(tokens, multipliers, vocab_sizes, offsets, heads_per_ngram=2, eos_token_id=eos)
    current, previous, older = 8, eos, eos
    mixed_bi = (current * 3) ^ (previous * 5)
    mixed_tri = mixed_bi ^ (older * 9)
    expected_last = [
        mixed_bi % 11 + 0,
        mixed_bi % 13 + 11,
        mixed_tri % 17 + 24,
        mixed_tri % 19 + 41,
    ]
    assert rows[-1].tolist() == expected_last
    # The token immediately after a real token still sees that token.
    current, previous, older = eos, 4, eos
    mixed_bi = (current * 3) ^ (previous * 5)
    assert int(rows[1, 0]) == mixed_bi % 11
