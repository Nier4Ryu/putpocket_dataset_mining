#!/usr/bin/env bash
set -euo pipefail

# CPU-only packaging step. Docker receives no NVIDIA devices. The resulting
# image remains inert unless a launch explicitly supplies a control path.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPECTED_BASE_ID="sha256:b79bacf76a107fc9ddd67fcc84851e3f5ae222fe6fd6a2f187723297e1400b1e"
BASE_IMAGE="${GLM53_BASE_IMAGE:-putpocket/glm53-flash-sm120:878631b-c2eec1}"
OUTPUT_IMAGE="${GLM53_STATEFUL_IMAGE:-putpocket/glm53-flash-sm120-stateful-edit-v3:878631b}"
OVERLAY_ROOT="${1:?usage: $0 PREPARED_OVERLAY_ROOT}"

[[ -f "${OVERLAY_ROOT}/PUTPOCKET_GLM53_STATEFUL_OVERLAY.json" ]] || {
  echo "Prepared overlay manifest is missing" >&2
  exit 2
}
actual_base="$(docker image inspect "${BASE_IMAGE}" --format '{{.Id}}')"
[[ "${actual_base}" == "${EXPECTED_BASE_ID}" ]] || {
  echo "Base runtime image ID mismatch: ${actual_base}" >&2
  exit 2
}

docker build \
  --build-arg "GLM53_BASE_IMAGE=${BASE_IMAGE}" \
  --file "${REPO_ROOT}/docker/glm53_sm120/Dockerfile.stateful-edit-v3" \
  --tag "${OUTPUT_IMAGE}" \
  "${OVERLAY_ROOT}"

docker image inspect "${OUTPUT_IMAGE}" --format '{{json .Id}}'
