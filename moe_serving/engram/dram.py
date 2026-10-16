"""Engram n-gram table stored in host DRAM.

The table is a concatenation of equal-sized embedding shards. Each shard is
a read-only mapping of the checkpoint file, so expert-parallel processes on
one machine share the physical pages. A forward gathers the handful of rows
selected by the hash and copies only those rows to the GPU.
"""

from __future__ import annotations

import torch

from moe_serving.checkpoint import MappedCheckpoint


class DramEngramTable:
    def __init__(self, checkpoint: MappedCheckpoint, shard_names: list[str]):
        if not shard_names:
            raise ValueError("Engram table has no shards")
        self.checkpoint = checkpoint
        self.shard_names = shard_names
        first = checkpoint.ref(shard_names[0])
        self.rows_per_shard = first.shape[0]
        self.dim = first.shape[1]
        self.dtype = first.dtype
        self.num_shards = len(shard_names)
        self.num_rows = self.rows_per_shard * self.num_shards
        self._views: list[torch.Tensor | None] = [None] * self.num_shards
        for name in shard_names[1:]:
            shape = checkpoint.ref(name).shape
            if shape != first.shape or checkpoint.ref(name).dtype != first.dtype:
                raise ValueError(f"{name} has shape {shape}, expected {first.shape}")

    def shard(self, index: int) -> torch.Tensor:
        view = self._views[index]
        if view is None:
            view = self.checkpoint.mmap(self.shard_names[index])
            self._views[index] = view
        return view

    @torch.no_grad()
    def gather(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Gather rows. ``row_ids`` may live on any device; the result is CPU."""
        flat = row_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
        if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= self.num_rows):
            raise IndexError(f"Engram row id outside [0, {self.num_rows})")
        out = torch.empty(flat.shape[0], self.dim, dtype=self.dtype)
        if flat.numel() == 0:
            return out.view(*row_ids.shape, self.dim)
        shard_index = torch.div(flat, self.rows_per_shard, rounding_mode="floor")
        local_row = flat - shard_index * self.rows_per_shard
        for shard in shard_index.unique().tolist():
            mask = shard_index == shard
            gathered = self.shard(int(shard)).index_select(0, local_row[mask])
            out[mask] = gathered
        return out.view(*row_ids.shape, self.dim)

    def to_device(self, rows: torch.Tensor, device: torch.device | str) -> torch.Tensor:
        if str(device) == "cpu" or device == torch.device("cpu"):
            return rows
        pinned = rows.pin_memory() if torch.cuda.is_available() else rows
        return pinned.to(device, non_blocking=True)
