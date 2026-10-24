#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/home/sky/titan-venv/bin/python}"
run_output="${GSM8K_OUTPUT:-${repo_root}/outputs/gpt_oss_120b_gsm8k_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
cd -- "$repo_root"

exec "$python_bin" -m moe_serving.record_gsm8k \
  --checkpoint "$repo_root/LLMs/gpt-oss-120b" \
  --nproc-per-node 4 --tp-size 4 --ep-size 4 \
  --dtype bfloat16 --expert-storage mxfp4 --parallel-numerics efficient \
  --num-samples -1 --batch-size 4 --max-new-tokens 1024 --reasoning-effort low \
  --output "$run_output" \
  "$@"
