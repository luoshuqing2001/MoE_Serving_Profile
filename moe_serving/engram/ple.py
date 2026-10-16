"""Mix DRAM Engram rows into a hidden stream.

The checkpoint stores four small GPU tensors next to the table:

- ``key_proj`` maps the concatenated n-gram vector to one key per
  hyper-connection branch
- ``value_proj`` maps it to a single value vector
- RMSNorm weights for the query (the hidden state), the key, and the
  depthwise-convolution input
- a causal depthwise convolution over the key stream

The branch gate is a sigmoid of the dot product between the normalized
query and key. The value is broadcast across branches, added to the
convolved key, and added back onto the hidden state. A rank only looks up
the tokens it already holds.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from moe_serving.engram.dram import DramEngramTable
from moe_serving.engram.hash import hash_context, hash_token_ids


@dataclass
class NgramParameters:
    multipliers: torch.Tensor
    vocab_sizes: torch.Tensor
    offsets: torch.Tensor
    heads_per_ngram: int
    eos_token_id: int

    @property
    def ngram_size(self) -> int:
        return int(self.multipliers.numel())

    @property
    def num_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram


class PleLayer(torch.nn.Module):
    def __init__(
        self,
        table: DramEngramTable,
        parameters: NgramParameters,
        key_proj: torch.Tensor,
        value_proj: torch.Tensor,
        norm_query: torch.Tensor,
        norm_key: torch.Tensor,
        norm_conv: torch.Tensor,
        conv_weight: torch.Tensor,
        hc_count: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.table = table
        self.ngram = parameters
        self.hc_count = hc_count
        self.eps = eps
        self.key_proj = torch.nn.Parameter(key_proj, requires_grad=False)
        self.value_proj = torch.nn.Parameter(value_proj, requires_grad=False)
        self.norm_query = torch.nn.Parameter(norm_query, requires_grad=False)
        self.norm_key = torch.nn.Parameter(norm_key, requires_grad=False)
        self.norm_conv = torch.nn.Parameter(norm_conv, requires_grad=False)
        self.conv_weight = torch.nn.Parameter(conv_weight, requires_grad=False)
        self.head_dim = table.dim
        self.embed_dim = self.head_dim * parameters.num_heads

    @torch.no_grad()
    def lookup(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Return concatenated Engram rows on the token-id device, shape ``[..., embed_dim]``."""
        row_ids = hash_token_ids(
            token_ids,
            self.ngram.multipliers,
            self.ngram.vocab_sizes,
            self.ngram.offsets,
            self.ngram.heads_per_ngram,
            self.ngram.eos_token_id,
        )
        rows = self.table.gather(row_ids)
        flat = rows.reshape(*row_ids.shape[:-1], self.embed_dim)
        return self.table.to_device(flat, token_ids.device)

    @torch.no_grad()
    def lookup_context(self, context: torch.Tensor) -> torch.Tensor:
        row_ids = hash_context(
            context,
            self.ngram.multipliers,
            self.ngram.vocab_sizes,
            self.ngram.offsets,
            self.ngram.heads_per_ngram,
        )
        rows = self.table.gather(row_ids)
        flat = rows.reshape(*row_ids.shape[:-1], self.embed_dim)
        return self.table.to_device(flat, context.device)

    def forward(
        self,
        hidden: torch.Tensor,
        embeddings: torch.Tensor,
        conv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the Engram delta. ``hidden`` is ``[batch, time, hidden]`` or with an HC axis.

        Returns the updated hidden state and the new convolution state
        ``[batch, kernel - 1, hc * hidden]``.
        """
        hidden, embeddings, restore = _pack(hidden, embeddings, self.hc_count)
        batch, time, hc, width = hidden.shape
        channels = hc * width
        query = _rms(hidden.reshape(batch, time, channels), self.norm_query, self.eps).view(batch, time, hc, width)
        key = F.linear(embeddings, self.key_proj.to(embeddings.dtype))
        value = F.linear(embeddings, self.value_proj.to(embeddings.dtype))
        key_n = _rms(key, self.norm_key, self.eps).view(batch, time, hc, width)
        gate = torch.sigmoid((query * key_n).sum(dim=-1, keepdim=True))
        conv_in = _rms(key, self.norm_conv, self.eps)
        conv_out, new_state = _causal_depthwise(conv_in, self.conv_weight.to(conv_in.dtype), conv_state)
        delta = gate * value.unsqueeze(2) + conv_out.view(batch, time, hc, width)
        updated = hidden + delta.to(hidden.dtype)
        return restore(updated), new_state


def _pack(hidden: torch.Tensor, embeddings: torch.Tensor, hc_count: int):
    if embeddings.ndim == hidden.ndim - 1 and hidden.shape[-2] == hc_count:
        # [..., hc, hidden] against [..., embed]
        if hidden.ndim == 3:
            hidden = hidden.unsqueeze(0)
            embeddings = embeddings.unsqueeze(0)
            return hidden, embeddings, lambda tensor: tensor.squeeze(0)
        return hidden, embeddings, lambda tensor: tensor
    if hidden.ndim == 2:
        hidden = hidden.unsqueeze(0).unsqueeze(2).expand(-1, -1, hc_count, -1).contiguous()
        embeddings = embeddings.unsqueeze(0)
        return hidden, embeddings, lambda tensor: tensor.mean(dim=2).squeeze(0)
    if hidden.ndim == 3:
        hidden = hidden.unsqueeze(2).expand(-1, -1, hc_count, -1).contiguous()
        return hidden, embeddings, lambda tensor: tensor.mean(dim=2)
    raise ValueError(f"unsupported hidden shape {tuple(hidden.shape)}")


def _rms(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    normalized = hidden.float() * torch.rsqrt(variance + eps)
    return (normalized * weight.float()).to(hidden.dtype)


def _causal_depthwise(
    sequence: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Depthwise causal convolution. ``sequence`` is ``[batch, time, channels]``."""
    kernel = weight.shape[-1]
    channels = sequence.shape[-1]
    if state is None:
        padded = F.pad(sequence.transpose(1, 2), (kernel - 1, 0))
    else:
        padded = torch.cat([state, sequence], dim=1).transpose(1, 2)
    output = F.conv1d(padded, weight, groups=channels).transpose(1, 2)
    history = torch.cat([state, sequence], dim=1) if state is not None else sequence
    new_state = history[:, -(kernel - 1) :, :]
    if new_state.shape[1] < kernel - 1:
        new_state = F.pad(new_state, (0, 0, kernel - 1 - new_state.shape[1], 0))
    return output, new_state
