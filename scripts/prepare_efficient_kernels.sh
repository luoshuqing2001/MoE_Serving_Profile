#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/sky/titan-venv/bin/python}"
cd -- "$repo_root"
"$python_bin" - <<'PY'
import torch
from moe_serving.efficient_kernels import check_runtime
for family, storage in [('gpt_oss', 'mxfp4'), ('qwen3_moe', 'fp8')]:
    check_runtime(family, 'cuda', torch.bfloat16, storage)
    print(f'{family}: FlashAttention 3 + grouped {storage} ready')
PY
