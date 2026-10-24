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
    ],
)
def test_gpt_oss_parallel_parity(tmp_path, world, packed, storage, tp, ep):
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
    ]
    env = dict(os.environ, PYTHONPATH=str(ROOT), OMP_NUM_THREADS="1")
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("PASS rank=") == world
