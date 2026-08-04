#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  ./scripts/env/bootstrap_sr.sh --phase gpu --server-profile server1_rtx3090 --hardware-profile sm86 --vllm-profile patched --build-vllm no --dry-run
echo "SM86 build/import smoke only. Real GLM sparse-MLA runtime is expected to be unsupported unless a compatible backend is added."
