#!/usr/bin/env bash
set -euo pipefail

TASK_ID="T20260903-001__glm53-retarget-all"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 RUN_DIR" >&2
  exit 2
fi
RUN_DIR="$(realpath "$1")"
CID_FILE="${RUN_DIR}/container.cid"
if [[ ! -f "${CID_FILE}" ]]; then
  echo "Missing task container CID file: ${CID_FILE}" >&2
  exit 2
fi
CID="$(tr -d '[:space:]' < "${CID_FILE}")"
if [[ ! "${CID}" =~ ^[0-9a-f]{12,64}$ ]]; then
  echo "Invalid container ID in ${CID_FILE}" >&2
  exit 2
fi
if ! docker inspect "${CID}" > "${RUN_DIR}/container_pre_stop_inspect.json"; then
  echo "Task container ${CID} is not running or no longer exists." >&2
  exit 2
fi
LABEL_TASK="$(docker inspect --format '{{ index .Config.Labels "putpocket.task_id" }}' "${CID}")"
LABEL_RUN="$(docker inspect --format '{{ index .Config.Labels "putpocket.run_id" }}' "${CID}")"
if [[ "${LABEL_TASK}" != "${TASK_ID}" || -z "${LABEL_RUN}" ]]; then
  echo "Refusing to stop container without exact task/run ownership labels." >&2
  exit 2
fi
if [[ "$(basename "${RUN_DIR}")" != "${LABEL_RUN}" ]]; then
  echo "Refusing to stop: run directory does not match container run label." >&2
  exit 2
fi

docker logs --timestamps "${CID}" > "${RUN_DIR}/server.log" 2>&1 || true
nvidia-smi -q -x > "${RUN_DIR}/nvidia_smi_pre_stop.xml"
# An infinite Docker stop timeout sends SIGTERM and never escalates to SIGKILL.
docker stop --signal SIGTERM --timeout -1 "${CID}" \
  > "${RUN_DIR}/docker_stop_stdout.txt"
printf '%s\n' "${CID}"
