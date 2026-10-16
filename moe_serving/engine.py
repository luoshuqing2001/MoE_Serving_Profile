"""Run expert-parallel prefill and collect per-layer routing stats."""

from __future__ import annotations

import torch
import torch.distributed as dist

from moe_serving.distributed import Collective, SingleRank, collective_for, init_process_group
from moe_serving.model import QwenExpertParallelModel, expert_shard_bytes, load_model
from moe_serving.moe import RoutingStats


class Engine:
    def __init__(self, model: QwenExpertParallelModel, rank: int, ep_size: int, device: torch.device):
        self.model = model
        self.rank = rank
        self.ep_size = ep_size
        self.device = device

    @classmethod
    def from_pretrained(cls, family: str, checkpoint: str, **kwargs) -> "Engine":
        rank = kwargs.pop("rank", None)
        ep_size = kwargs.pop("ep_size", None)
        device = kwargs.pop("device", None)
        if rank is None or ep_size is None:
            rank, ep_size = init_process_group()
        if device is None:
            if torch.cuda.is_available() and ep_size > 0:
                local = int(__import__("os").environ.get("LOCAL_RANK", rank))
                device = torch.device(f"cuda:{local}")
            else:
                device = torch.device("cpu")
        device = torch.device(device)
        collective: Collective = kwargs.pop("collective", None) or collective_for(ep_size)
        model = load_model(family, checkpoint, rank, ep_size, device, collective=collective, **kwargs)
        return cls(model, rank, ep_size, device)

    @torch.no_grad()
    def profile(self, token_ids: torch.Tensor) -> list[dict]:
        local = self.shard_tokens(token_ids).to(self.device)
        _hidden, stats = self.model(local)
        self._reduce(stats)
        rows = []
        for item in stats:
            rows.append(
                {
                    "layer": item.layer,
                    "expert_load_cv": item.expert_load_cv,
                    "expert_dead_fraction": item.expert_dead_fraction,
                    "assignments": int(item.counts.sum()),
                    "moe_seconds": item.moe_seconds,
                    "engram_seconds": item.engram_seconds,
                    "local_tokens": int(local.numel()),
                }
            )
        return rows

    def shard_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Give each rank whole sequences when the batch divides across ranks.

        A single sequence stays on rank 0. The other ranks enter the expert
        collective with an empty token list so Engram context is not split.
        """
        if token_ids.ndim == 1:
            token_ids = token_ids.unsqueeze(0)
        if self.ep_size == 1:
            return token_ids
        if token_ids.shape[0] >= self.ep_size and token_ids.shape[0] % self.ep_size == 0:
            return token_ids.chunk(self.ep_size, dim=0)[self.rank]
        if self.rank == 0:
            return token_ids
        return token_ids[:0]

    def _reduce(self, stats: list[RoutingStats]) -> None:
        if not dist.is_initialized() or self.ep_size == 1:
            return
        for item in stats:
            counts = item.counts.to(self.device)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            item.counts = counts.cpu()
            timing = torch.tensor([item.moe_seconds, item.engram_seconds], device=self.device)
            dist.all_reduce(timing, op=dist.ReduceOp.MAX)
            item.moe_seconds = float(timing[0])
            item.engram_seconds = float(timing[1])

    def describe(self) -> str:
        spec = self.model.spec
        selected = self.model.layer_ids
        expert_bytes = expert_shard_bytes(spec, self.ep_size, len(selected))
        engram = "on" if self.model.ple is not None else "off"
        return (
            f"{spec.family} ep={self.ep_size} rank={self.rank} "
            f"experts {self.model.plan.expert_start}:{self.model.plan.expert_end} "
            f"layers={selected} engram={engram} "
            f"expert_shard={expert_bytes / 1024**3:.2f} GiB"
        )
