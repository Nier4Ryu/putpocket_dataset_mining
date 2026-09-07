#!/usr/bin/env bash
set -euo pipefail

: "${PUTPOCKET_PHASE_FILE:?set PUTPOCKET_PHASE_FILE}"
: "${PUTPOCKET_CAPTURE_PATH:?set PUTPOCKET_CAPTURE_PATH}"
: "${PUTPOCKET_PROXY_LOG:?set PUTPOCKET_PROXY_LOG}"

args=(
  --listen-port "${PUTPOCKET_PROXY_PORT:-18000}"
  --backend-port "${VLLM_PORT:-8000}"
  --phase-file "${PUTPOCKET_PHASE_FILE}"
  --log "${PUTPOCKET_PROXY_LOG}"
  --capture "${PUTPOCKET_CAPTURE_PATH}"
)
if [[ -n "${PUTPOCKET_EPISODE_PATH:-}" ]]; then
  [[ -f "${PUTPOCKET_EPISODE_PATH}" ]] || { echo "PUTPOCKET_EPISODE_PATH is not a file" >&2; exit 2; }
  args+=(--episode "${PUTPOCKET_EPISODE_PATH}")
fi
exec python3 -m putpocket_dataset_mining.glm53_runpod_stateful_proxy "${args[@]}" "$@"
