"""Read a Hugging Face safetensors checkpoint without pulling it onto the GPU.

Expert slices are copied out of the file. Engram rows stay as read-only
file mappings so several expert-parallel ranks share one DRAM-resident table
through the page cache.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch

_DTYPES = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


@dataclass
class TensorRef:
    path: str
    dtype: torch.dtype
    shape: tuple[int, ...]
    absolute_offset: int


class MappedCheckpoint:
    """Header index plus lazy whole-file mappings."""

    def __init__(self, root: str):
        self.root = root
        index_path = os.path.join(root, "model.safetensors.index.json")
        if os.path.isfile(index_path):
            with open(index_path) as handle:
                self.weight_map: dict[str, str] = json.load(handle)["weight_map"]
        else:
            single = os.path.join(root, "model.safetensors")
            if not os.path.isfile(single):
                raise FileNotFoundError(f"no safetensors checkpoint under {root}")
            header, _data_offset = _header(single)
            self.weight_map = {
                name: "model.safetensors" for name in header if name != "__metadata__"
            }
        self._headers: dict[str, tuple[dict, int]] = {}
        self._storage: dict[str, torch.UntypedStorage] = {}

    def names(self) -> list[str]:
        return list(self.weight_map)

    def ref(self, name: str) -> TensorRef:
        filename = self.weight_map[name]
        header, data_offset = self._header(filename)
        meta = header[name]
        start, _end = meta["data_offsets"]
        return TensorRef(
            path=os.path.join(self.root, filename),
            dtype=_DTYPES[meta["dtype"]],
            shape=tuple(meta["shape"]),
            absolute_offset=data_offset + start,
        )

    def mmap(self, name: str) -> torch.Tensor:
        """Read-only view of one tensor. The mapping is shared across ranks."""
        ref = self.ref(name)
        itemsize = torch.empty((), dtype=ref.dtype).element_size()
        if ref.absolute_offset % itemsize:
            raise ValueError(f"{name} is not aligned for {ref.dtype}")
        storage = self._file_storage(ref.path)
        tensor = torch.empty(ref.shape, dtype=ref.dtype)
        tensor.set_(storage, ref.absolute_offset // itemsize, ref.shape)
        return tensor

    def slice_experts(self, name: str, start: int, end: int) -> torch.Tensor:
        """Copy expert rows [start, end) of a fused expert tensor."""
        from safetensors import safe_open

        with safe_open(self.ref(name).path, framework="pt", device="cpu") as handle:
            return handle.get_slice(name)[start:end]

    def load(self, name: str) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(self.ref(name).path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(name)

    def _header(self, filename: str) -> tuple[dict, int]:
        cached = self._headers.get(filename)
        if cached is None:
            cached = _header(os.path.join(self.root, filename))
            self._headers[filename] = cached
        return cached

    def _file_storage(self, path: str) -> torch.UntypedStorage:
        cached = self._storage.get(path)
        if cached is None:
            cached = torch.UntypedStorage.from_file(path, shared=True, nbytes=os.path.getsize(path))
            self._storage[path] = cached
        return cached


def _header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_len))
    return header, 8 + header_len
