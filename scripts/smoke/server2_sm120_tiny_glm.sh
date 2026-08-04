#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  ./scripts/env/bootstrap_sr.sh --phase gpu --server-profile server2_rtxpro6000_blackwell --hardware-profile sm120 --vllm-profile patched --build-vllm no --dry-run
PYTHONPATH=src python -m unittest discover -s tests -p 'test_glm_backend_routing.py' -v
echo "SM120 routing smoke complete. This does not prove SM90/Hopper compatibility."
