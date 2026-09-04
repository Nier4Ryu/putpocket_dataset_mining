#!/usr/bin/env bash
set -euo pipefail

# CPU-hosted Docker build. This script intentionally contains no NVIDIA
# runtime/device flags and never imports torch or probes CUDA devices.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE="${1:?usage: $0 PREPARED_VLLM_SOURCE}"
VLLM_TAG="${VLLM_SM86_BASE_TAG:-putpocket/vllm-glm53-sm86:9cd956c7}"
FINAL_TAG="${GLM53_A6000_IMAGE_TAG:-putpocket/glm53-a6000-gated:9cd956c7}"

[[ -f "${SOURCE}/PUTPOCKET_SM86_BUILD.json" ]] || { echo "prepared source manifest missing" >&2; exit 2; }
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["torch_cuda_arch_list"])' "${SOURCE}/PUTPOCKET_SM86_BUILD.json")" = "8.6"
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cmake_cuda_architectures"])' "${SOURCE}/PUTPOCKET_SM86_BUILD.json")" = "86"

docker build \
  --file "${SOURCE}/docker/Dockerfile" \
  --target vllm-openai \
  --build-arg torch_cuda_arch_list=8.6 \
  --build-arg putpocket_sm86_only=1 \
  --build-arg max_jobs="${MAX_JOBS:-8}" \
  --build-arg nvcc_threads="${NVCC_THREADS:-2}" \
  --build-arg VLLM_BUILD_COMMIT=9cd956c7e6cf54efa366b803cafa15ec6c2df827 \
  --build-arg VLLM_BUILD_PIPELINE=putpocket-sm86-cpu-build \
  --tag "${VLLM_TAG}" \
  "${SOURCE}"

docker build \
  --file "${REPO_ROOT}/docker/glm53_sm86/Dockerfile" \
  --build-arg "VLLM_BASE_IMAGE=${VLLM_TAG}" \
  --tag "${FINAL_TAG}" \
  "${REPO_ROOT}"

docker image inspect "${FINAL_TAG}" --format '{{json .Id}}'
