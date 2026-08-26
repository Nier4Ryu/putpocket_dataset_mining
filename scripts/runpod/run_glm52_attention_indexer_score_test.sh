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
PROBE=$ARTIFACT_ROOT/frozen-two-query-probe.json
MATRIX_EPISODE=$ARTIFACT_ROOT/frozen-matrix-episode.json
CAPTURE=$ARTIFACT_ROOT/query-sum-capture
REPORT=$ARTIFACT_ROOT/query-sum-report
MATRIX_CAPTURE=$ARTIFACT_ROOT/matrix
MULTIHOP=$ARTIFACT_ROOT/multihop

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$CAPTURE" "$REPORT" "$MATRIX_CAPTURE" "$MULTIHOP"

"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli prepare-final-probe \
  --lock "$LOCK" --model-root "$MODEL_ROOT" --harness-root "$HARNESS_ROOT" \
  --output "$PROBE" --matrix-output "$MATRIX_EPISODE"
"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli capture-query-sum \
  --lock "$LOCK" --doctor-report "$DOCTOR_JSON" --probe "$PROBE" \
  --model-root "$MODEL_ROOT" --output-root "$CAPTURE"
"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli analyze-query-sum \
  --lock "$LOCK" --doctor-report "$DOCTOR_JSON" --probe "$PROBE" \
  --capture-root "$CAPTURE" \
  --output-root "$REPORT"
"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli capture-matrix \
  --lock "$LOCK" --doctor-report "$DOCTOR_JSON" \
  --episode-manifest "$MATRIX_EPISODE" --model-root "$MODEL_ROOT" \
  --output-root "$MATRIX_CAPTURE"
mapfile -t SCORE_ARGS < <("$PYTHON_BIN" - "$MATRIX_EPISODE" <<'PY'
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
for label in ("q1_ranges", "q2_ranges"):
    option="--q1-range" if label == "q1_ranges" else "--q2-range"
    for start,end in value["segments"][label]:
        print(option); print(f"{start}:{end}")
start,end=value["propagation_window"]
print("--window"); print(f"{start}:{end}")
print("--layers"); print(",".join(map(str,value["capture"]["layers"])))
PY
)
"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli score-matrix \
  --episode-manifest "$MATRIX_EPISODE" --capture-root "$MATRIX_CAPTURE" \
  --output-root "$MULTIHOP" "${SCORE_ARGS[@]}" --max-level 6

sha256sum "$DOCTOR_JSON" "$PROBE" "$MATRIX_EPISODE" \
  "$CAPTURE"/capture-rank-*.jsonl \
  "$REPORT/query-sum-attention-indexer-report.json" \
  "$REPORT/query-summed-token-scores.jsonl" \
  "$REPORT/per-query-comparisons.json" \
  "$MATRIX_CAPTURE"/matrix-rank-*-chunk-*.jsonl \
  "$MULTIHOP/indexer-multihop-report.json" \
  "$MULTIHOP/indexer-multihop-token-scores.jsonl" \
  > "$ARTIFACT_ROOT/SHA256SUMS"
