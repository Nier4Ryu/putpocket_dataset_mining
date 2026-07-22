#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source this file instead of executing it." >&2
  echo "Usage: source scripts/env/env_activate.sh" >&2
  exit 2
fi

_putpocket_env_script="${BASH_SOURCE[0]}"
_putpocket_env_dir="$(cd "$(dirname "${_putpocket_env_script}")" && pwd)"
export PUTPOCKET_DATASET_MINING_ROOT="$(cd "${_putpocket_env_dir}/../.." && pwd)"

_putpocket_venv="${PUTPOCKET_DATASET_MINING_ROOT}/Putpocket_env"
if [[ ! -f "${_putpocket_venv}/bin/activate" ]]; then
  echo "Putpocket_env was not found at ${_putpocket_venv}." >&2
  echo "Run ./scripts/env/bootstrap_env.sh first." >&2
  return 2
fi

_putpocket_prepend_path() {
  local path_entry="$1"
  [[ -d "${path_entry}" ]] || return 0
  case ":${PATH}:" in
    *":${path_entry}:"*) ;;
    *) export PATH="${path_entry}:${PATH}" ;;
  esac
}

_putpocket_prepend_var_path() {
  local var_name="$1"
  local path_entry="$2"
  [[ -d "${path_entry}" ]] || return 0
  local current_value="${!var_name:-}"
  case ":${current_value}:" in
    *":${path_entry}:"*) ;;
    *)
      if [[ -n "${current_value}" ]]; then
        export "${var_name}=${path_entry}:${current_value}"
      else
        export "${var_name}=${path_entry}"
      fi
      ;;
  esac
}

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${_putpocket_venv}}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TZ="${TZ:-Asia/Seoul}"

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

_putpocket_prepend_path "${CUDA_HOME}/bin"
_putpocket_prepend_path "${PUTPOCKET_DATASET_MINING_ROOT}/.local_python/bin"
_putpocket_prepend_var_path LD_LIBRARY_PATH "${CUDA_HOME}/lib64"
_putpocket_prepend_var_path PYTHONPATH "${PUTPOCKET_DATASET_MINING_ROOT}/src"

# shellcheck disable=SC1091
source "${_putpocket_venv}/bin/activate"

echo "Putpocket env activated"
echo "  repo root: ${PUTPOCKET_DATASET_MINING_ROOT}"
echo "  venv: ${_putpocket_venv}"
echo "  python: $(command -v python)"
echo "  CUDA_HOME: ${CUDA_HOME}"
echo "  build threads: ${PUTPOCKET_BUILD_THREADS}"

unset _putpocket_env_script
unset _putpocket_env_dir
unset _putpocket_venv
unset -f _putpocket_prepend_path
unset -f _putpocket_prepend_var_path
