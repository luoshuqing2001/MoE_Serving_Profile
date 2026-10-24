"""Exercise actual FP8 files, TP/EP collectives, batch generation and JSON."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "world,packed,storage,tp,ep",
    [
        (1, "plain", "bf16", 1, 1),
        (2, "plain", "bf16", 2, 2),
        (2, "fp8", "fp8", 2, 2),
        (2, "fp8", "bf16", 2, 2),
        (2, "fp8", "fp8", 2, 1),
        (2, "fp8", "fp8", 1, 2),
        (4, "fp8", "fp8", 4, 4),
        (4, "fp8", "bf16", 4, 4),
    ],
)
def test_qwen3_parallel_parity(
    tmp_path, world, packed, storage, tp, ep, dtype="float32", numerics="reference"
):
    cmd = [sys.executable]
    if world > 1:
        cmd += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={world}",
        ]
    cmd += [
        str(ROOT / "tests/helpers/qwen3_worker.py"),
        str(tmp_path),
        packed,
        storage,
        str(tp),
        str(ep),
        dtype,
        numerics,
    ]
    result = subprocess.run(
        cmd,
        env=dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1"),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS rank=") == world


@pytest.mark.parametrize("storage", ["fp8", "bf16"])
def test_qwen3_bf16_reference_order(tmp_path, storage):
    test_qwen3_parallel_parity(tmp_path, 4, "fp8", storage, 4, 4, dtype="bfloat16")


def test_qwen3_fast_fp32_parity(tmp_path):
    test_qwen3_parallel_parity(tmp_path, 2, "fp8", "fp8", 2, 2, numerics="fast")
