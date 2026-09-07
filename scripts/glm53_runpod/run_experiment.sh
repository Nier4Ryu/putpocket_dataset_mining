#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/putpocket
RUN_ID="${PUTPOCKET_RUN_ID:?set a unique PUTPOCKET_RUN_ID}"
RUN_DIR="/results/${RUN_ID}"
HARNESS=/data/run_putpocket_harness.sh
[[ ! -e "${RUN_DIR}" ]] || { echo "refusing to reuse ${RUN_DIR}" >&2; exit 2; }
[[ -x "${HARNESS}" ]] || { echo "mount the frozen benchmark harness at ${HARNESS}" >&2; exit 2; }
mkdir -p "${RUN_DIR}"

export PUTPOCKET_PHASE_FILE="${PUTPOCKET_PHASE_FILE:-/data/phase.json}"
export PUTPOCKET_EPISODE_PATH="${PUTPOCKET_EPISODE_PATH:-/data/frozen-episode.json}"
export PUTPOCKET_CAPTURE_PATH="${RUN_DIR}/turn1-capture.json"
export PUTPOCKET_PROXY_LOG="${RUN_DIR}/proxy.jsonl"
[[ -f "${PUTPOCKET_PHASE_FILE}" && -f "${PUTPOCKET_EPISODE_PATH}" ]] || { echo "frozen phase/episode inputs are required" >&2; exit 2; }

server_pid=
proxy_pid=
cleanup() {
  if [[ -n "${proxy_pid}" ]] && kill -0 "${proxy_pid}" 2>/dev/null; then kill "${proxy_pid}"; wait "${proxy_pid}" || true; fi
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then kill "${server_pid}"; wait "${server_pid}" || true; fi
}
trap cleanup EXIT INT TERM

"${ROOT}/scripts/glm53_runpod/launch_server.sh" >"${RUN_DIR}/server.log" 2>&1 &
server_pid=$!
for _ in $(seq 1 360); do
  if curl --fail --silent "http://127.0.0.1:${VLLM_PORT:-8000}/health" >/dev/null; then break; fi
  kill -0 "${server_pid}" 2>/dev/null || { echo "model server exited during startup" >&2; exit 3; }
  sleep 5
done
curl --fail --silent "http://127.0.0.1:${VLLM_PORT:-8000}/health" >/dev/null || { echo "model server health timeout" >&2; exit 3; }

"${ROOT}/scripts/glm53_runpod/launch_proxy.sh" >"${RUN_DIR}/proxy.log" 2>&1 &
proxy_pid=$!
sleep 1
kill -0 "${proxy_pid}" 2>/dev/null || { echo "proxy failed to start" >&2; exit 3; }

export OPENAI_BASE_URL="http://127.0.0.1:${PUTPOCKET_PROXY_PORT:-18000}/v1"
export PUTPOCKET_RUN_DIR="${RUN_DIR}"
"${HARNESS}"
