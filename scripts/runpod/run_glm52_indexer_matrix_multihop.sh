#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 10 ]]; then
  echo "usage: run_glm52_indexer_matrix_multihop.sh MODEL_ROOT DOCTOR_JSON FROZEN_MATRIX_EPISODE ARTIFACT_ROOT Q1_START:END Q2_START:END WINDOW_START:END LAYERS MAX_LEVEL PYTHON_BIN" >&2
  exit 64
fi

MODEL_ROOT=$1
DOCTOR_JSON=$2
EPISODE=$3
ARTIFACT_ROOT=$4
Q1_RANGE=$5
Q2_RANGE=$6
WINDOW=$7
LAYERS=$8
MAX_LEVEL=$9
PYTHON_BIN=${10}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MATRIX_ROOT=$ARTIFACT_ROOT/matrix
REPORT_ROOT=$ARTIFACT_ROOT/multihop

test -f "$DOCTOR_JSON"
test -f "$EPISODE"
mkdir -p "$MATRIX_ROOT" "$REPORT_ROOT"

"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli capture-matrix \
  --doctor-report "$DOCTOR_JSON" --episode-manifest "$EPISODE" \
  --model-root "$MODEL_ROOT" --output-root "$MATRIX_ROOT"

"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli score-matrix \
  --episode-manifest "$EPISODE" --capture-root "$MATRIX_ROOT" \
  --output-root "$REPORT_ROOT" --q1-range "$Q1_RANGE" --q2-range "$Q2_RANGE" \
  --window "$WINDOW" --layers "$LAYERS" --max-level "$MAX_LEVEL"

(
  cd "$PROJECT_ROOT"
  sha256sum \
    "$EPISODE" \
    "$MATRIX_ROOT"/matrix-rank-*-chunk-*.jsonl \
    "$REPORT_ROOT"/indexer-multihop-report.json \
    "$REPORT_ROOT"/indexer-multihop-token-scores.jsonl
) > "$ARTIFACT_ROOT/multihop-SHA256SUMS"
