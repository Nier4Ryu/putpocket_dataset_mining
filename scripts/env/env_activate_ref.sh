#!/usr/bin/env bash

_putpocket_env_script="${BASH_SOURCE[0]}"
_putpocket_env_dir="$(cd "$(dirname "${_putpocket_env_script}")" && pwd)"
export PUTPOCKET_DATASET_MINING_ROOT="$(cd "${_putpocket_env_dir}/../.." && pwd)"

export CUDA_HOME="/usr/local/cuda-12.9"
export PATH="${CUDA_HOME}/bin:${PUTPOCKET_DATASET_MINING_ROOT}/Putpocket_env/bin:${PUTPOCKET_DATASET_MINING_ROOT}/.local_python/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PUTPOCKET_DATASET_MINING_ROOT}/src:${PYTHONPATH:-}"
if [[ -d "/data/shared/hf_cache/hub" ]]; then
  export PUTPOCKET_HF_HUB_CACHE_DIR="${PUTPOCKET_HF_HUB_CACHE_DIR:-/data/shared/hf_cache/hub}"
else
  export PUTPOCKET_HF_HUB_CACHE_DIR="${PUTPOCKET_HF_HUB_CACHE_DIR:-${HOME}/.cache/huggingface/hub}"
fi

export RANDOM_SEED="${RANDOM_SEED:-42}"
export PUTPOCKET_BUILD_THREADS="${PUTPOCKET_BUILD_THREADS:-16}"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-16}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

if [[ -f "${PUTPOCKET_DATASET_MINING_ROOT}/Putpocket_env/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PUTPOCKET_DATASET_MINING_ROOT}/Putpocket_env/bin/activate"
fi
