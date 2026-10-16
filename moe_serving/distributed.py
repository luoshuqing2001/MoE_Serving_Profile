"""Expert-parallel collectives.

``exchange`` moves a row tensor that is already grouped by destination rank.
The reverse direction uses the counts this rank received: result rows go back
to the rank that sent them, still grouped by that rank.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


class Collective:
    world_size: int

    def exchange(self, rows: torch.Tensor, send_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SingleRank(Collective):
    world_size = 1

    def exchange(self, rows: torch.Tensor, send_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return rows, send_counts


class ProcessGroupCollective(Collective):
    def __init__(self, group=None):
        self.group = group
        self.world_size = dist.get_world_size(group)

    def exchange(self, rows: torch.Tensor, send_counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        device = rows.device
        send_counts = send_counts.to(device=device, dtype=torch.int64)
        recv_counts = torch.empty(self.world_size, device=device, dtype=torch.int64)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        recv_rows = rows.new_empty(int(recv_counts.sum()), *rows.shape[1:])
        dist.all_to_all_single(
            recv_rows,
            rows.contiguous(),
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
            group=self.group,
        )
        return recv_rows, recv_counts


def exchange_inplace(payloads: list[torch.Tensor], splits: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One-process stand-in for an expert-parallel all-to-all.

    ``payloads[rank]`` is grouped by destination. The returned tensor for a
    rank is grouped by source, and the count vector is indexed by source.
    """
    world = len(payloads)
    received: list[tuple[torch.Tensor, torch.Tensor]] = []
    for dest in range(world):
        chunks = []
        counts = []
        for src in range(world):
            count = int(splits[src][dest])
            start = int(splits[src][:dest].sum()) if dest else 0
            chunks.append(payloads[src][start : start + count])
            counts.append(count)
        rows = torch.cat(chunks, dim=0) if chunks else payloads[dest][:0]
        received.append((rows, torch.tensor(counts, dtype=torch.int64)))
    return received


def init_process_group(backend: str | None = None):
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    if "RANK" not in os_environ():
        return 0, 1
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    return dist.get_rank(), dist.get_world_size()


def os_environ() -> dict[str, str]:
    import os

    return os.environ


def collective_for(world_size: int) -> Collective:
    if world_size == 1 or not dist.is_initialized():
        return SingleRank()
    return ProcessGroupCollective()
