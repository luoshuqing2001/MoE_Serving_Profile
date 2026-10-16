"""Make the hub finegrained-fp8 grouped and batched kernels compile on this Triton.

Both kernels load one weight scale per N tile through a scalar pointer (``pid_n``),
then index that rank-0 value as ``b_s[None, :]``. This Triton rejects that index.
The 2D kernel already builds a length ``BLOCK_SIZE_N`` pointer vector from
``offs_bn // BLOCK_SIZE_N``. With the tile width equal to the scale block, every
lane still reads the same scale.
"""

from __future__ import annotations

from pathlib import Path

_PATCHES = (
    (
        "grouped.py",
        "bs_ptrs = Bs + expert_id * stride_bs_e + pid_n * stride_bs_n",
        "offs_bsn = offs_bn // BLOCK_SIZE_N\n"
        "    bs_ptrs = Bs + expert_id * stride_bs_e + offs_bsn * stride_bs_n",
    ),
    (
        "batched.py",
        "bs_ptrs = Bs + pid_n * stride_bs_n",
        "offs_bsn = offs_bn // BLOCK_SIZE_N\n    bs_ptrs = Bs + offs_bsn * stride_bs_n",
    ),
)


def patch_grouped_fp8_kernel() -> list[Path]:
    """Patch cached kernel files. Already-patched files are left as they are."""
    root = Path.home() / ".cache" / "huggingface" / "hub"
    if not root.is_dir():
        return []
    patched: list[Path] = []
    for path in root.glob("kernels--kernels-community--finegrained-fp8/**/*.py"):
        text = path.read_text()
        updated = text
        for filename, old, new in _PATCHES:
            if path.name != filename or old not in updated:
                continue
            updated = updated.replace(old, new, 1)
        if updated != text:
            path.write_text(updated)
            patched.append(path)
    return patched
