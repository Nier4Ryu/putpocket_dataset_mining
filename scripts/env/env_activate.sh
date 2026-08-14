#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source this file instead of executing it." >&2
  echo "Usage: source scripts/env/env_activate.sh" >&2
  exit 2
fi

_putpocket_env_script="${BASH_SOURCE[0]}"
_putpocket_env_dir="$(cd "$(dirname "${_putpocket_env_script}")" && pwd)"
_putpocket_script_root="$(cd "${_putpocket_env_dir}/../.." && pwd)"
_putpocket_default_base="${HOME}"
if [[ "${PUTPOCKET_STORAGE_KIND:-}" == "network-volume" && -d "/workspace" ]]; then
  _putpocket_default_base="/workspace"
fi

if [[ -n "${PUTPOCKET_REPO_ROOT:-}" ]]; then
  _putpocket_repo_root="$(cd "${PUTPOCKET_REPO_ROOT}" && pwd)"
elif [[ -d "${_putpocket_script_root}" ]]; then
  _putpocket_repo_root="${_putpocket_script_root}"
elif command -v git >/dev/null 2>&1 && git rev-parse --show-toplevel >/dev/null 2>&1; then
  _putpocket_repo_root="$(git rev-parse --show-toplevel)"
else
  echo "Unable to resolve Putpocket repository root. Set PUTPOCKET_REPO_ROOT." >&2
  return 2
fi

_putpocket_canonical_root="${PUTPOCKET_CANONICAL_ROOT:-${PUTPOCKET_REPO_ROOT:-${_putpocket_default_base}/putpocket_dataset_mining}}"
_putpocket_worktree_root="${PUTPOCKET_WORKTREE_ROOT:-${_putpocket_default_base}/putpocket_dataset_mining_worktrees}"
export PUTPOCKET_CANONICAL_ROOT="${_putpocket_canonical_root}"
export PUTPOCKET_WORKTREE_ROOT="${_putpocket_worktree_root}"

case "$(cd "${_putpocket_repo_root}" && pwd)" in
  "$(cd "${_putpocket_canonical_root}" 2>/dev/null && pwd 2>/dev/null)")
    export PUTPOCKET_EXECUTION_CONTEXT="canonical-runtime"
    export PUTPOCKET_DATASET_MINING_ROOT="${_putpocket_canonical_root}"
    _putpocket_overlay_root=""
    ;;
  "${_putpocket_worktree_root}"/*)
    export PUTPOCKET_EXECUTION_CONTEXT="task-worktree"
    export PUTPOCKET_DATASET_MINING_ROOT="${_putpocket_canonical_root}"
    _putpocket_overlay_root="${_putpocket_script_root}"
    ;;
  *)
    export PUTPOCKET_EXECUTION_CONTEXT="unknown"
    export PUTPOCKET_DATASET_MINING_ROOT="${_putpocket_script_root}"
    _putpocket_overlay_root=""
    ;;
esac
export PUTPOCKET_REPO_ROOT="${PUTPOCKET_DATASET_MINING_ROOT}"

_putpocket_venv="${PUTPOCKET_ENV_PATH:-${PUTPOCKET_CANONICAL_SERVER2_ENV:-${PUTPOCKET_DATASET_MINING_ROOT}/Putpocket_env}}"
case "${VIRTUAL_ENV:-}" in
  *Putpocket_env_glm52*|*Putpocket_env_glm52_v025*)
    echo "Refusing to activate Server-2 env from legacy GLM environment: ${VIRTUAL_ENV}" >&2
    return 2
    ;;
esac

if [[ ! -f "${_putpocket_venv}/bin/activate" ]]; then
  echo "Putpocket_env was not found at ${_putpocket_venv}." >&2
  echo "Run ./scripts/env/bootstrap_sr.sh --preset server2 first." >&2
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

export PUTPOCKET_ENV_PATH="${_putpocket_venv}"
export UV_PROJECT_ENVIRONMENT="${_putpocket_venv}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TZ="${TZ:-Asia/Seoul}"
export PUTPOCKET_STORAGE_KIND="${PUTPOCKET_STORAGE_KIND:-local}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${PUTPOCKET_DATASET_MINING_ROOT}/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${PUTPOCKET_DATASET_MINING_ROOT}/.cache/uv/python}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${PUTPOCKET_DATASET_MINING_ROOT}/.cache/vllm}"
export TORCH_HOME="${TORCH_HOME:-${PUTPOCKET_DATASET_MINING_ROOT}/.cache/torch}"
export HF_HOME="${HF_HOME:-${PUTPOCKET_DATASET_MINING_ROOT}/models/hf}"
export PUTPOCKET_PRODUCTION_ALLOWED="0"
if [[ "${PUTPOCKET_EXECUTION_CONTEXT}" == "canonical-runtime" ]]; then
  export PUTPOCKET_PRODUCTION_ALLOWED="1"
fi

if [[ -d "/data/shared/hf_cache/hub" ]]; then
  export PUTPOCKET_HF_HUB_CACHE_DIR="${PUTPOCKET_HF_HUB_CACHE_DIR:-/data/shared/hf_cache/hub}"
else
  export PUTPOCKET_HF_HUB_CACHE_DIR="${PUTPOCKET_HF_HUB_CACHE_DIR:-${HF_HOME}/hub}"
fi

export RANDOM_SEED="${RANDOM_SEED:-42}"
export PUTPOCKET_BUILD_THREADS="${PUTPOCKET_BUILD_THREADS:-8}"
export PUTPOCKET_BUILD_JOBS="${PUTPOCKET_BUILD_JOBS:-${PUTPOCKET_BUILD_THREADS}}"
export MAX_JOBS="${MAX_JOBS:-${PUTPOCKET_BUILD_JOBS}}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${PUTPOCKET_BUILD_JOBS}}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-${PUTPOCKET_BUILD_JOBS}}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

if [[ -z "${CUDA_HOME:-}" && -d "/usr/local/cuda-12.9" ]]; then
  export CUDA_HOME="/usr/local/cuda-12.9"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
  _putpocket_prepend_path "${CUDA_HOME}/bin"
  _putpocket_prepend_var_path LD_LIBRARY_PATH "${CUDA_HOME}/lib64"
fi
_putpocket_prepend_path "${PUTPOCKET_DATASET_MINING_ROOT}/.local_python/bin"
_putpocket_prepend_var_path PYTHONPATH "${PUTPOCKET_DATASET_MINING_ROOT}/src"
if [[ -n "${_putpocket_overlay_root}" ]]; then
  _putpocket_prepend_var_path PYTHONPATH "${_putpocket_overlay_root}/src"
fi

# shellcheck disable=SC1091
source "${_putpocket_venv}/bin/activate"

case ":${PATH}:${PYTHONPATH:-}:" in
  *Putpocket_env_glm52*|*Putpocket_env_glm52_v025*)
    echo "Legacy GLM environment leaked into PATH/PYTHONPATH." >&2
    return 2
    ;;
esac

echo "Putpocket env activated"
echo "  context: ${PUTPOCKET_EXECUTION_CONTEXT}"
echo "  profile: ${PUTPOCKET_ACTIVE_PROFILE:-server2}"
echo "  repo root: ${PUTPOCKET_DATASET_MINING_ROOT}"
if [[ -n "${_putpocket_overlay_root}" ]]; then
  echo "  task overlay: ${_putpocket_overlay_root}"
  echo "  production commands: blocked by default"
fi
echo "  venv: ${_putpocket_venv}"
echo "  python: $(command -v python)"
echo "  CUDA_HOME: ${CUDA_HOME:-unset}"
echo "  build threads: ${PUTPOCKET_BUILD_THREADS}"
echo "  build jobs: ${PUTPOCKET_BUILD_JOBS}"

unset _putpocket_env_script
unset _putpocket_env_dir
unset _putpocket_script_root
unset _putpocket_repo_root
unset _putpocket_default_base
unset _putpocket_canonical_root
unset _putpocket_worktree_root
unset _putpocket_venv
unset _putpocket_overlay_root
unset -f _putpocket_prepend_path
unset -f _putpocket_prepend_var_path
