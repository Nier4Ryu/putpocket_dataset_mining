#!/usr/bin/env bash
set -euo pipefail

# CPU-hosted Docker build: no NVIDIA devices or CUDA runtime probes are used.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${1:?usage: $0 PREPARED_VLLM_SOURCE}"
VLLM_TAG="${VLLM_DUAL_BASE_TAG:-putpocket/vllm-glm53-sm90-sm120:9cd956c7-fa-v2}"
FINAL_TAG="${GLM53_RUNPOD_IMAGE_TAG:-putpocket/glm53-runpod-sm90-sm120-w4a16:9cd956c7-v2}"
MANIFEST="${SOURCE}/PUTPOCKET_GLM53_SM90_SM120_BUILD.json"
PUTPOCKET_SOURCE_COMMIT="${PUTPOCKET_SOURCE_COMMIT:-$(git -C "${REPO_ROOT}" rev-parse HEAD)}"

[[ "${PUTPOCKET_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  echo "PUTPOCKET_SOURCE_COMMIT must be a full Git commit" >&2
  exit 2
}

[[ -f "${MANIFEST}" ]] || { echo "prepared source manifest missing" >&2; exit 2; }
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["torch_cuda_arch_list"])' "${MANIFEST}")" = "9.0a 12.0a"
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cmake_cuda_architectures"])' "${MANIFEST}")" = "90a;120a"

docker build \
  --file "${SOURCE}/docker/Dockerfile" \
  --target vllm-openai \
  --build-arg 'torch_cuda_arch_list=9.0a 12.0a' \
  --build-arg putpocket_glm53_dual_arch=1 \
  --build-arg max_jobs="${MAX_JOBS:-8}" \
  --build-arg nvcc_threads="${NVCC_THREADS:-2}" \
  --build-arg VLLM_BUILD_COMMIT=9cd956c7e6cf54efa366b803cafa15ec6c2df827 \
  --build-arg VLLM_BUILD_PIPELINE=putpocket-glm53-sm90-sm120-fa-fix-cpu-build \
  --tag "${VLLM_TAG}" \
  "${SOURCE}"

docker build \
  --file "${REPO_ROOT}/docker/glm53_sm90_sm120/Dockerfile" \
  --build-arg "VLLM_BASE_IMAGE=${VLLM_TAG}" \
  --build-arg "PUTPOCKET_SOURCE_COMMIT=${PUTPOCKET_SOURCE_COMMIT}" \
  --tag "${FINAL_TAG}" \
  "${REPO_ROOT}"

docker image inspect "${FINAL_TAG}" --format '{{json .Id}}'
