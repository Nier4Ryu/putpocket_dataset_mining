#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# != 5 ]]; then
  echo "usage: run_glm52_attention_indexer_score_test.sh MODEL_ROOT HARNESS_ROOT DOCTOR_JSON ARTIFACT_ROOT PYTHON_BIN" >&2
  exit 2
fi

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
MODEL_ROOT=$(realpath "$1")
HARNESS_ROOT=$(realpath "$2")
DOCTOR_JSON=$(realpath "$3")
ARTIFACT_ROOT=$4
PYTHON_BIN=$5
LOCK=$PROJECT_ROOT/configs/runpod/glm52_attention_indexer_package.lock.json
PROBE=$ARTIFACT_ROOT/probe.json
CAPTURE=$ARTIFACT_ROOT/capture
REPORT=$ARTIFACT_ROOT/report

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CAPTURE" "$REPORT"

"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli prepare-probe \
  --lock "$LOCK" --model-root "$MODEL_ROOT" --harness-root "$HARNESS_ROOT" \
  --output "$PROBE"
"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli capture \
  --lock "$LOCK" --doctor-report "$DOCTOR_JSON" --probe "$PROBE" \
  --model-root "$MODEL_ROOT" --output-root "$CAPTURE"
"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli analyze \
  --lock "$LOCK" --doctor-report "$DOCTOR_JSON" --capture-root "$CAPTURE" \
  --output-root "$REPORT"

sha256sum "$DOCTOR_JSON" "$PROBE" "$CAPTURE"/capture-rank-*.jsonl \
  "$REPORT/attention-indexer-report.json" "$REPORT/aligned-token-scores.jsonl" \
  > "$ARTIFACT_ROOT/SHA256SUMS"
