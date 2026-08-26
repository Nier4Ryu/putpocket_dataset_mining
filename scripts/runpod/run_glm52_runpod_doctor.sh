#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# != 5 ]]; then
  echo "usage: run_glm52_runpod_doctor.sh VLLM_ROOT MODEL_ROOT EXPECTED_PROJECT_COMMIT OUTPUT_JSON PYTHON_BIN" >&2
  exit 2
fi

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
VLLM_ROOT=$(realpath "$1")
MODEL_ROOT=$(realpath "$2")
EXPECTED_PROJECT_COMMIT=$3
OUTPUT_JSON=$4
PYTHON_BIN=$5
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli doctor \
  --project-root "$PROJECT_ROOT" \
  --vllm-root "$VLLM_ROOT" \
  --model-root "$MODEL_ROOT" \
  --expected-project-commit "$EXPECTED_PROJECT_COMMIT" \
  --output "$OUTPUT_JSON"
