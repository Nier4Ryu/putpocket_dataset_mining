#!/usr/bin/env bash
set -euo pipefail
umask 077

PROJECT_COMMIT=${1:-}
PROJECT_URL=https://github.com/Nier4Ryu/putpocket_dataset_mining.git
STORAGE_ROOT=/local-data/jslee202403/putpocket-glm52-smoke
RUN_ROOT="$STORAGE_ROOT/artifacts/glm52-swepro-smoke/${SLURM_JOB_ID:-unknown}"
BOOTSTRAP_PYTHON=/home2/jslee202403/miniconda3/bin/python3
UV_ROOT="$STORAGE_ROOT/tools/uv-0.11.31"
UV_EXECUTABLE="$UV_ROOT/bin/uv"
RENDER_ENV="$STORAGE_ROOT/tools/render-python"
RENDER_PYTHON="$RENDER_ENV/bin/python"

[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || { echo E_SLURM_ALLOCATION_REQUIRED >&2; exit 20; }
[[ -n ${SLURM_JOB_NODELIST:-} && ${SLURM_JOB_NUM_NODES:-0} == 1 ]] || { echo E_SLURM_ALLOCATION_REQUIRED >&2; exit 20; }
[[ $PROJECT_COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo E_PROJECT_COMMIT_REQUIRED >&2; exit 21; }
mkdir -p "$RUN_ROOT"

# The official per-row evaluator requires a usable Docker daemon. This check
# intentionally precedes every install or download performed by this entrypoint.
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  printf '{"schema_version":1,"status":"failed","failure_class":"OFFICIAL_EVALUATION_DOCKER_REQUIRED","official_evaluation_supported":false}\n' \
    > "$RUN_ROOT/container_preflight.json"
  printf '%s\n' NON_SCORE_ELIGIBLE_SMOKE_ONLY > "$RUN_ROOT/claim_boundary.txt"
  exit 42
fi
printf '{"schema_version":1,"status":"passed","runtime":"docker","official_evaluation_supported":true}\n' \
  > "$RUN_ROOT/container_preflight.json"

SCRIPT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
[[ $(/usr/bin/git -C "$SCRIPT_ROOT" rev-parse HEAD) == "$PROJECT_COMMIT" ]] || {
  echo E_PROJECT_COMMIT_MISMATCH >&2
  exit 22
}
[[ -x $BOOTSTRAP_PYTHON ]] || { echo E_COMPUTE_TOOL_MISSING:bootstrap-python >&2; exit 43; }

# Login has no uv. Bootstrap the version pinned by the H200 environment lock,
# and the tiny renderer dependency, only here inside the compute allocation.
if [[ ! -x $UV_EXECUTABLE ]]; then
  "$BOOTSTRAP_PYTHON" -m venv "$UV_ROOT"
  "$UV_ROOT/bin/python" -m pip install --disable-pip-version-check --no-input 'uv==0.11.31'
fi
if ! "$RENDER_PYTHON" -c 'import yaml' >/dev/null 2>&1; then
  "$UV_EXECUTABLE" venv --allow-existing --python "$BOOTSTRAP_PYTHON" "$RENDER_ENV"
  "$UV_EXECUTABLE" pip install --python "$RENDER_PYTHON" --no-input 'PyYAML==6.0.3'
fi

RENDERED="$RUN_ROOT/rendered-smoke.sbatch"
PYTHONPATH="$SCRIPT_ROOT/src" "$RENDER_PYTHON" -m putpocket_dataset_mining.swebench_pro_cli render \
  --site "$SCRIPT_ROOT/configs/cluster/sites/herdr_h200_smoke.yaml" \
  --project-url "$PROJECT_URL" --project-commit "$PROJECT_COMMIT" --smoke-only \
  > "$RENDERED.partial"
mv "$RENDERED.partial" "$RENDERED"
/bin/bash -n "$RENDERED"
exec /bin/bash "$RENDERED"
