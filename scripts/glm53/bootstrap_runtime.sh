#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_ID="T20260830-001__glm53-montblanc-deployment"
RUNTIME_ROOT="${GLM53_RUNTIME_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/putpocket-runtime/${TASK_ID}}"
LOCK_PATH="${REPO_ROOT}/configs/models/glm53_flash_nvfp4_montblanc.lock.json"
VLLM_SOURCE_DIR="${RUNTIME_ROOT}/build-sources/vllm"
VLLM_IMAGE="${GLM53_VLLM_IMAGE:-putpocket/glm53-flash-sm120-vllm:878631b}"
FINAL_IMAGE="${GLM53_IMAGE:-putpocket/glm53-flash-sm120:878631b-c2eec1}"

if [[ "${PUTPOCKET_GLM53_BUILD_LOCK_HELD:-0}" != "1" ]]; then
  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  exec python3 -m putpocket_dataset_mining.glm53_deployment \
    --lock "${LOCK_PATH}" lock-run -- "$0" "$@"
fi

if [[ "$(git -C "${REPO_ROOT}" branch --show-current)" != "agent/${TASK_ID}" ]]; then
  echo "Refusing bootstrap outside agent/${TASK_ID}" >&2
  exit 2
fi
if [[ "${PUTPOCKET_ALLOW_TASK_PRODUCTION:-0}" != "1" ]]; then
  echo "Set PUTPOCKET_ALLOW_TASK_PRODUCTION=1 for this user-authorized task-local build." >&2
  exit 2
fi
if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked worktree changes must be committed or explicitly staged before runtime build." >&2
  exit 2
fi

mkdir -p "${RUNTIME_ROOT}/build-sources" "${RUNTIME_ROOT}/logs" "${RUNTIME_ROOT}/evidence"
export GLM53_BOOTSTRAP_LOCK_PATH="${LOCK_PATH}"
lock_value() {
  local expression="$1"
  python3 - "${expression}" <<'PY'
import json, os, sys
value=json.load(open(os.environ['GLM53_BOOTSTRAP_LOCK_PATH'], encoding='utf-8'))
for key in sys.argv[1].split('.'):
    value=value[key]
print(value)
PY
}

VLLM_COMMIT="$(lock_value runtime.vllm_commit)"
VLLM_PULL_REF="$(lock_value runtime.vllm_pull_ref)"
FLASHINFER_COMMIT="$(lock_value runtime.flashinfer_commit)"
BUILD_BASE_IMAGE="$(lock_value runtime.build_base_image)"
FINAL_BASE_IMAGE="$(lock_value runtime.final_base_image)"

if [[ ! -d "${VLLM_SOURCE_DIR}/.git" ]]; then
  if [[ -e "${VLLM_SOURCE_DIR}" ]]; then
    echo "Refusing to overwrite non-Git source path ${VLLM_SOURCE_DIR}" >&2
    exit 2
  fi
  git init "${VLLM_SOURCE_DIR}"
  git -C "${VLLM_SOURCE_DIR}" remote add origin https://github.com/vllm-project/vllm.git
  git -C "${VLLM_SOURCE_DIR}" fetch --depth=1 origin "${VLLM_PULL_REF}"
  git -C "${VLLM_SOURCE_DIR}" checkout --detach FETCH_HEAD
fi
test "$(git -C "${VLLM_SOURCE_DIR}" rev-parse HEAD)" = "${VLLM_COMMIT}"
test -z "$(git -C "${VLLM_SOURCE_DIR}" status --porcelain)"
test "$(sha256sum "${VLLM_SOURCE_DIR}/docker/Dockerfile" | awk '{print $1}')" = \
  "$(lock_value runtime.vllm_dockerfile_sha256)"
test "$(sha256sum "${VLLM_SOURCE_DIR}/docker/versions.json" | awk '{print $1}')" = \
  "$(lock_value runtime.vllm_versions_sha256)"

{
  echo "vllm_commit=${VLLM_COMMIT}"
  echo "vllm_pull_ref=${VLLM_PULL_REF}"
  echo "flashinfer_commit=${FLASHINFER_COMMIT}"
  echo "build_base_image=${BUILD_BASE_IMAGE}"
  echo "final_base_image=${FINAL_BASE_IMAGE}"
  echo "vllm_image=${VLLM_IMAGE}"
  echo "final_image=${FINAL_IMAGE}"
} > "${RUNTIME_ROOT}/evidence/runtime_build_inputs.txt"

docker build --progress=plain --target vllm-openai \
  --build-arg "BUILD_BASE_IMAGE=${BUILD_BASE_IMAGE}" \
  --build-arg "FINAL_BASE_IMAGE=${FINAL_BASE_IMAGE}" \
  --build-arg max_jobs=2 \
  --build-arg nvcc_threads=1 \
  --tag "${VLLM_IMAGE}" \
  "${VLLM_SOURCE_DIR}" \
  2>&1 | tee "${RUNTIME_ROOT}/logs/vllm_image_build.log"

docker build --progress=plain \
  --file "${REPO_ROOT}/docker/glm53_sm120/Dockerfile.flashinfer-overlay" \
  --build-arg "VLLM_BASE_IMAGE=${VLLM_IMAGE}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  --build-arg "FLASHINFER_COMMIT=${FLASHINFER_COMMIT}" \
  --tag "${FINAL_IMAGE}" \
  "${REPO_ROOT}" \
  2>&1 | tee "${RUNTIME_ROOT}/logs/flashinfer_overlay_build.log"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m putpocket_dataset_mining.glm53_deployment --lock "${LOCK_PATH}" \
  inspect-image --image "${FINAL_IMAGE}" \
  --output "${RUNTIME_ROOT}/evidence/runtime_image_inspection.json"
docker image inspect "${FINAL_IMAGE}" \
  > "${RUNTIME_ROOT}/evidence/runtime_image_docker_inspect.json"
sha256sum \
  "${RUNTIME_ROOT}/logs/vllm_image_build.log" \
  "${RUNTIME_ROOT}/logs/flashinfer_overlay_build.log" \
  "${RUNTIME_ROOT}/evidence/runtime_image_inspection.json" \
  > "${RUNTIME_ROOT}/evidence/runtime_build_sha256sums.txt"

printf '%s\n' "${FINAL_IMAGE}"
