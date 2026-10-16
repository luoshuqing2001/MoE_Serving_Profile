"""Qwen n-gram hash used by the PLE / Engram table.

The multiplier at shift ``s`` multiplies the token ``s`` positions back from
the current one. Shift 0 is the current token. Bigram heads are emitted
first, then trigram heads. A missing context slot, including everything
across an EOS boundary, uses ``eos_token_id``. Products stay inside int64 for
the checkpoint's multiplier bound; the row id is the non-negative remainder
plus that head's packed-table offset.
"""

from __future__ import annotations

import torch

SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
HASH_SEED_PRIME = 10007
_MASK64 = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & _MASK64
    return value ^ (value >> 31)


def build_layer_multipliers(
    unigram_vocab_size: int,
    ngram_size: int,
    ple_layer_index: int,
    seed: int,
) -> list[int]:
    """Cross-check for the ``layer_multipliers`` buffer shipped in a checkpoint."""
    max_long = (1 << 63) - 1
    half_bound = max((max_long // max(unigram_vocab_size, 1)) // 2, 1)
    base_seed = (seed + HASH_SEED_PRIME * ple_layer_index) & _MASK64
    multipliers = []
    for index in range(ngram_size):
        step = (SPLITMIX_GAMMA * (index + 1)) & _MASK64
        hashed = _splitmix64((base_seed + step) & _MASK64) % half_bound
        multipliers.append(2 * hashed + 1)
    return multipliers


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def derive_head_layout(
    ngram_vocab_size_base: int,
    ngram_size: int,
    heads_per_ngram: int,
) -> tuple[list[int], list[int]]:
    """Prime modulus and packed-table offset of every hash head."""
    sizes: list[int] = []
    offsets: list[int] = []
    running = 0
    total_heads = (ngram_size - 1) * heads_per_ngram
    for head in range(total_heads):
        size = find_nth_prime_after(ngram_vocab_size_base - 1, head + 1)
        sizes.append(size)
        offsets.append(running)
        running += size
    return sizes, offsets


def context_at(token_ids: torch.Tensor, shift: int, eos_token_id: int) -> torch.Tensor:
    """Token ``shift`` steps back, or EOS when that slot crosses a boundary.

    ``token_ids`` has shape ``[batch, time]``. An EOS token is still visible
    as the immediately preceding token. It blocks anything older than itself.
    """
    if shift == 0:
        return token_ids
    out = torch.full_like(token_ids, eos_token_id)
    time = token_ids.shape[1]
    if shift >= time:
        return out
    source = token_ids[:, : time - shift]
    blocked = torch.zeros(token_ids.shape[0], time - shift, dtype=torch.bool, device=token_ids.device)
    for gap in range(1, shift):
        blocked = blocked | (token_ids[:, shift - gap : time - gap] == eos_token_id)
    out[:, shift:] = torch.where(blocked, torch.full_like(source, eos_token_id), source)
    return out


def hash_token_ids(
    token_ids: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    heads_per_ngram: int,
    eos_token_id: int,
) -> torch.Tensor:
    """Packed Engram row ids of shape ``[batch, time, ngram_heads]``."""
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    token_ids = token_ids.to(torch.int64)
    ngram_size = int(multipliers.numel())
    orders = ngram_size - 1
    terms = [
        context_at(token_ids, shift, eos_token_id) * int(multipliers[shift])
        for shift in range(ngram_size)
    ]
    rows = []
    for order_idx in range(orders):
        mixed = terms[0]
        for shift in range(1, order_idx + 2):
            mixed = torch.bitwise_xor(mixed, terms[shift])
        for local in range(heads_per_ngram):
            head = order_idx * heads_per_ngram + local
            rows.append(torch.remainder(mixed, int(vocab_sizes[head])) + int(offsets[head]))
    stacked = torch.stack(rows, dim=-1)
    return stacked.squeeze(0) if squeeze else stacked


class NgramContext:
    """Decode-time shift register. Slot 0 is one token back."""

    def __init__(self, ngram_size: int, eos_token_id: int, batch: int = 1):
        self.eos_token_id = eos_token_id
        self.history = torch.full((batch, ngram_size - 1), eos_token_id, dtype=torch.int64)

    def step(self, token: torch.Tensor) -> torch.Tensor:
        """Return ``[batch, ngram_size]`` context, current token first, then advance."""
        token = token.to(torch.int64).view(-1)
        context = torch.cat([token.unsqueeze(1), self.history], dim=1)
        reset = token == self.eos_token_id
        shifted = torch.cat([token.unsqueeze(1), self.history[:, :-1]], dim=1)
        eos_fill = torch.full_like(self.history, self.eos_token_id)
        self.history = torch.where(reset.unsqueeze(1), eos_fill, shifted)
        return context


def hash_context(
    context: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    heads_per_ngram: int,
) -> torch.Tensor:
    """Hash an explicit context ``[batch, ngram_size]`` (current token at index 0)."""
    ngram_size = int(multipliers.numel())
    terms = [context[:, shift] * int(multipliers[shift]) for shift in range(ngram_size)]
    rows = []
    for order_idx in range(ngram_size - 1):
        mixed = terms[0]
        for shift in range(1, order_idx + 2):
            mixed = torch.bitwise_xor(mixed, terms[shift])
        for local in range(heads_per_ngram):
            head = order_idx * heads_per_ngram + local
            rows.append(torch.remainder(mixed, int(vocab_sizes[head])) + int(offsets[head]))
    return torch.stack(rows, dim=-1)
