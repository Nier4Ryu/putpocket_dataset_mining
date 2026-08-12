#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Usage: source scripts/env/env_activate_glm52_v025.sh" >&2
  exit 2
fi

if [[ "${PUTPOCKET_ALLOW_LEGACY_GLM_ENV:-}" != "1" ]]; then
  echo "env_activate_glm52_v025.sh is retired from the active Server-2 path." >&2
  echo "Set PUTPOCKET_ALLOW_LEGACY_GLM_ENV=1 only for historical GLM inspection." >&2
  return 2
fi

_putpocket_glm52_v025_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PUTPOCKET_DATASET_MINING_ROOT="$(cd "${_putpocket_glm52_v025_script_dir}/../.." && pwd)"
_putpocket_glm52_v025_venv="${PUTPOCKET_DATASET_MINING_ROOT}/Putpocket_env_glm52_v025"

if [[ ! -f "${_putpocket_glm52_v025_venv}/bin/activate" ]]; then
  echo "Putpocket_env_glm52_v025 was not found at ${_putpocket_glm52_v025_venv}." >&2
  echo "Run ./scripts/env/bootstrap_glm52_v025_env.sh first." >&2
  return 2
fi

# shellcheck disable=SC1091
source "${_putpocket_glm52_v025_venv}/bin/activate"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
if [[ -d "${CUDA_HOME}/bin" ]]; then
  case ":${PATH}:" in
    *":${CUDA_HOME}/bin:"*) ;;
    *) export PATH="${CUDA_HOME}/bin:${PATH}" ;;
  esac
fi
if [[ -d "${CUDA_HOME}/lib64" ]]; then
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${CUDA_HOME}/lib64:"*) ;;
    *) export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}" ;;
  esac
fi

case ":${PATH}:" in
  *":${_putpocket_glm52_v025_venv}/bin:"*) ;;
  *) export PATH="${_putpocket_glm52_v025_venv}/bin:${PATH}" ;;
esac

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${_putpocket_glm52_v025_venv}}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TZ="${TZ:-Asia/Seoul}"
export PUTPOCKET_GLM52_V025_VLLM_SOURCE="${PUTPOCKET_DATASET_MINING_ROOT}/externals/vllm_glm52_v025"
if [[ -d "/data/shared/hf_cache/hub" ]]; then
  export PUTPOCKET_HF_HUB_CACHE_DIR="${PUTPOCKET_HF_HUB_CACHE_DIR:-/data/shared/hf_cache/hub}"
else
  export PUTPOCKET_HF_HUB_CACHE_DIR="${PUTPOCKET_HF_HUB_CACHE_DIR:-${HOME}/.cache/huggingface/hub}"
fi

export PUTPOCKET_BUILD_THREADS="${PUTPOCKET_BUILD_THREADS:-8}"
export MAX_JOBS="${MAX_JOBS:-8}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-8}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

echo "Putpocket GLM-5.2 vLLM 0.25 env activated"
echo "  repo: ${PUTPOCKET_DATASET_MINING_ROOT}"
echo "  env: ${VIRTUAL_ENV}"
echo "  python: $(command -v python)"
echo "  cuda_home: ${CUDA_HOME}"
echo "  vllm_source: ${PUTPOCKET_GLM52_V025_VLLM_SOURCE}"
echo "  build_threads: ${PUTPOCKET_BUILD_THREADS}"
