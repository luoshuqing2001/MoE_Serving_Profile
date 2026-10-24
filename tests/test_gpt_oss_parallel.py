"""Use subprocesses to isolate distributed inference and model imports."""

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
        (2, "packed", "mxfp4", 2, 2),
        (2, "packed", "bf16", 2, 2),
        (2, "plain", "bf16", 2, 1),
        (2, "plain", "bf16", 1, 2),
        (4, "packed", "mxfp4", 4, 4),
        (4, "packed", "bf16", 4, 4),
    ],
)
def test_gpt_oss_parallel_parity(
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
        str(ROOT / "tests/helpers/gpt_oss_worker.py"),
        str(tmp_path),
        packed,
        storage,
        str(tp),
        str(ep),
        dtype,
        numerics,
    ]
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS rank=") == world


@pytest.mark.parametrize("storage", ["mxfp4", "bf16"])
def test_gpt_oss_bf16_reference_order(tmp_path, storage):
    test_gpt_oss_parallel_parity(tmp_path, 4, "packed", storage, 4, 4, dtype="bfloat16")


def test_gpt_oss_fast_fp32_parity(tmp_path):
    test_gpt_oss_parallel_parity(tmp_path, 2, "packed", "mxfp4", 2, 2, numerics="fast")
