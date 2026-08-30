#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_ID="T20260830-001__glm53-montblanc-deployment"
RUNTIME_ROOT="${GLM53_RUNTIME_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/putpocket-runtime/${TASK_ID}}"
MODEL_DIR="${GLM53_MODEL_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/putpocket-models/${TASK_ID}/RedHatAI--GLM-5.3-Flash-NVFP4--36c184c6}"
PORT="${GLM53_PORT:-8137}"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi
RUN_DIR="$(realpath "$1")"
CID="$(tr -d '[:space:]' < "${RUN_DIR}/container.cid")"
if [[ "$(docker inspect --format '{{ index .Config.Labels "putpocket.task_id" }}' "${CID}")" != "${TASK_ID}" ]]; then
  echo "Refusing smoke against a container not owned by ${TASK_ID}." >&2
  exit 2
fi

CLIENT_ENV="${RUNTIME_ROOT}/client-env-uv"
UV_BIN="${PUTPOCKET_UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" && -x "${HOME}/putpocket_dataset_mining/.local_python/bin/uv" ]]; then
  UV_BIN="${HOME}/putpocket_dataset_mining/.local_python/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  echo "A uv executable is required for the task-local smoke client environment." >&2
  exit 2
fi
if [[ ! -x "${CLIENT_ENV}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.13 "${CLIENT_ENV}"
fi
"${UV_BIN}" pip install --python "${CLIENT_ENV}/bin/python" \
  --no-deps "${REPO_ROOT}"
"${UV_BIN}" pip install --python "${CLIENT_ENV}/bin/python" \
  "transformers==5.15.0"

nvidia-smi -q -x > "${RUN_DIR}/nvidia_smi_before_smoke.xml"
"${CLIENT_ENV}/bin/python" "${REPO_ROOT}/scripts/glm53/smoke_client.py" \
  --base-url "http://127.0.0.1:${PORT}" \
  --model glm-5.3-flash-nvfp4 \
  --model-dir "${MODEL_DIR}" \
  --output "${RUN_DIR}/smoke_report.json" \
  2>&1 | tee "${RUN_DIR}/smoke_client.log"
nvidia-smi -q -x > "${RUN_DIR}/nvidia_smi_after_smoke.xml"
docker logs --timestamps "${CID}" > "${RUN_DIR}/server_running.log" 2>&1
sha256sum \
  "${RUN_DIR}/smoke_report.json" \
  "${RUN_DIR}/smoke_client.log" \
  "${RUN_DIR}/server_running.log" \
  "${RUN_DIR}/nvidia_smi_before_smoke.xml" \
  "${RUN_DIR}/nvidia_smi_after_smoke.xml" \
  > "${RUN_DIR}/smoke_sha256sums.txt"
