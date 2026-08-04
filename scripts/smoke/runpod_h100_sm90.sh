#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
./scripts/env/bootstrap_sr.sh --phase gpu --server-profile runpod_hopper --hardware-profile sm90 --vllm-profile patched --build-vllm no
SR_ASSERT_NO_TINY_GLM_KERNEL=1 PYTHONPATH=src python -m unittest discover -s tests -p 'test_glm_backend_routing.py' -v
echo "One-H100/SM90 smoke complete. Full GLM loading is intentionally not attempted here."
