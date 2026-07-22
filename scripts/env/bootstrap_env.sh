#!/usr/bin/env bash
set -Eeuo pipefail

print_help() {
  cat <<'EOF'
Usage: ./scripts/env/bootstrap_env.sh [options]

Create or repair the repo-local Putpocket_env environment, ensure external
repositories and the default Docker image, then run setup smoke checks.

Options:
  --doctor-only           Do not install/build/checkout; only activate and validate the existing env.
  --skip-vllm-build       Skip editable vLLM build/install.
  --skip-deepgemm-build   Skip DeepGEMM build/install.
  --skip-lmcache-build    Skip editable LMCache build/install.
  --skip-externals        Skip external repository checkout/update.
  --skip-docker           Skip Dockerfile/image checks.
  --force-vllm-build      Run editable vLLM install even if import already works.
  --force-deepgemm-build  Run DeepGEMM install even if import already works.
  --force-lmcache-build   Run editable LMCache install even if import already works.
  --force-docker-build    Rebuild the default Docker image even if it already exists.
  --help                  Show this help.

Environment overrides:
  PYTHON_BIN                    Python 3.13 executable to use.
  CUDA_HOME                     Default: /usr/local/cuda-12.9
  TORCH_VERSION                 Default: 2.10.0
  TORCH_CUDA_TAG                Default: cu129
  TORCH_SPEC                    Default: torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}
  TORCH_INDEX_URL               Default: https://download.pytorch.org/whl/${TORCH_CUDA_TAG}
  RAY_VERSION                   Default: 2.55.1
  DEEPGEMM_CUDA_VERSION         Default: 12.9
  PUTPOCKET_BUILD_THREADS       Default: 16
  MAX_JOBS                      Default: 16
  CMAKE_BUILD_PARALLEL_LEVEL    Default: 16
  CARGO_BUILD_JOBS              Default: 16
  NVCC_THREADS                  Default: 1
EOF
}

DOCTOR_ONLY=0
SKIP_VLLM_BUILD=0
SKIP_DEEPGEMM_BUILD=0
SKIP_LMCACHE_BUILD=0
SKIP_EXTERNALS=0
SKIP_DOCKER=0
FORCE_VLLM_BUILD=0
FORCE_DEEPGEMM_BUILD=0
FORCE_LMCACHE_BUILD=0
FORCE_DOCKER_BUILD=0

while (($#)); do
  case "$1" in
    --doctor-only) DOCTOR_ONLY=1 ;;
    --skip-vllm-build) SKIP_VLLM_BUILD=1 ;;
    --skip-deepgemm-build) SKIP_DEEPGEMM_BUILD=1 ;;
    --skip-lmcache-build) SKIP_LMCACHE_BUILD=1 ;;
    --skip-externals) SKIP_EXTERNALS=1 ;;
    --skip-docker) SKIP_DOCKER=1 ;;
    --force-vllm-build) FORCE_VLLM_BUILD=1 ;;
    --force-deepgemm-build) FORCE_DEEPGEMM_BUILD=1 ;;
    --force-lmcache-build) FORCE_LMCACHE_BUILD=1 ;;
    --force-docker-build) FORCE_DOCKER_BUILD=1 ;;
    --help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Run ./scripts/env/bootstrap_env.sh --help for usage." >&2
      exit 2
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT
VENV_DIR="${REPO_ROOT}/Putpocket_env"
DOCKERFILE="${REPO_ROOT}/docker/default_python/Dockerfile"
DOCKER_CONTEXT="${REPO_ROOT}/docker"
DOCKER_IMAGE="putpocket-default-python:ubuntu22.04-py313-v1"

LOG_ROOT="${REPO_ROOT}/logs/env_setup"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_ROOT}/${RUN_TIMESTAMP}"
COMMAND_LOG="${LOG_DIR}/commands.log"
VLLM_RETRY_SUMMARY="${LOG_DIR}/vllm_build_retry_summary.tsv"
TORCH_CONSTRAINT_FILE="${LOG_DIR}/torch_cuda_constraints.txt"
mkdir -p "${LOG_DIR}"
ln -sfn "${RUN_TIMESTAMP}" "${LOG_ROOT}/latest"
: >"${COMMAND_LOG}"
: >"${VLLM_RETRY_SUMMARY}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export TORCH_VERSION="${TORCH_VERSION:-2.10.0}"
export TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu129}"
export TORCH_SPEC="${TORCH_SPEC:-torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}}"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA_TAG}}"
export RAY_VERSION="${RAY_VERSION:-2.55.1}"
export DEEPGEMM_CUDA_VERSION="${DEEPGEMM_CUDA_VERSION:-12.9}"
export PUTPOCKET_PIP_INDEX_URL="${PUTPOCKET_PIP_INDEX_URL:-${TORCH_INDEX_URL}}"
export PUTPOCKET_PIP_EXTRA_INDEX_URL="${PUTPOCKET_PIP_EXTRA_INDEX_URL:-https://pypi.org/simple}"
export PUTPOCKET_TORCH_CONSTRAINT_FILE="${PUTPOCKET_TORCH_CONSTRAINT_FILE:-${TORCH_CONSTRAINT_FILE}}"
export PUTPOCKET_BUILD_THREADS="${PUTPOCKET_BUILD_THREADS:-16}"
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-16}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TZ="${TZ:-Asia/Seoul}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${VENV_DIR}}"
if [[ -d "/data/shared/hf_cache/hub" ]]; then
  export PUTPOCKET_HF_HUB_CACHE_DIR="${PUTPOCKET_HF_HUB_CACHE_DIR:-/data/shared/hf_cache/hub}"
else
  export PUTPOCKET_HF_HUB_CACHE_DIR="${PUTPOCKET_HF_HUB_CACHE_DIR:-${HOME}/.cache/huggingface/hub}"
fi
if [[ -d "${CUDA_HOME}/bin" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if [[ -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ -d "${REPO_ROOT}/.local_python/bin" ]]; then
  export PATH="${REPO_ROOT}/.local_python/bin:${PATH}"
fi
printf "%s\n" "${TORCH_SPEC}" >"${TORCH_CONSTRAINT_FILE}"

PYTHON13_BIN=""
UV_BIN=""
VENV_PYTHON="${VENV_DIR}/bin/python"
VENV_CLI="${VENV_DIR}/bin/putpocket-dataset-mining"

quote_command() {
  local rendered=""
  local arg
  local quoted
  for arg in "$@"; do
    printf -v quoted "%q" "${arg}"
    rendered+="${quoted} "
  done
  echo "${rendered% }"
}

stage_log_name() {
  printf "%s" "$1" | tr -c "A-Za-z0-9_.-" "_"
}

run_stage() {
  local stage="$1"
  shift
  local log_file="${LOG_DIR}/$(stage_log_name "${stage}").log"
  local cmd_display
  cmd_display="$(quote_command "$@")"

  echo "[stage] ${stage}"
  echo "  command: ${cmd_display}"
  echo "  log: ${log_file}"
  printf "%s\t%s\t%s\n" "${stage}" "${cmd_display}" "${log_file}" >>"${COMMAND_LOG}"

  set +e
  (
    set -Eeuo pipefail
    printf "$ %s\n" "${cmd_display}"
    "$@"
  ) >"${log_file}" 2>&1
  local status=$?
  set -e

  if [[ "${status}" -eq 0 ]]; then
    echo "  success"
  else
    echo "  failed (exit ${status})" >&2
    echo "Failed stage: ${stage}" >&2
    echo "Failing command: ${cmd_display}" >&2
    echo "Relevant log: ${log_file}" >&2
    echo "Suggested rerun: ./scripts/env/bootstrap_env.sh" >&2
    exit "${status}"
  fi
}

run_stage_current_shell() {
  local stage="$1"
  shift
  local log_file="${LOG_DIR}/$(stage_log_name "${stage}").log"
  local cmd_display
  cmd_display="$(quote_command "$@")"

  echo "[stage] ${stage}"
  echo "  command: ${cmd_display}"
  echo "  log: ${log_file}"
  printf "%s\t%s\t%s\n" "${stage}" "${cmd_display}" "${log_file}" >>"${COMMAND_LOG}"

  set +e
  {
    printf "$ %s\n" "${cmd_display}"
    "$@"
  } >"${log_file}" 2>&1
  local status=$?
  set -e

  if [[ "${status}" -eq 0 ]]; then
    echo "  success"
  else
    echo "  failed (exit ${status})" >&2
    echo "Failed stage: ${stage}" >&2
    echo "Failing command: ${cmd_display}" >&2
    echo "Relevant log: ${log_file}" >&2
    echo "Suggested rerun: ./scripts/env/bootstrap_env.sh" >&2
    exit "${status}"
  fi
}

validate_python13() {
  local candidate="$1"
  "${candidate}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)
PY
}

resolve_uv() {
  local candidate=""
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    echo "Using uv from PATH: ${UV_BIN}"
    return 0
  fi

  candidate="${REPO_ROOT}/.local_python/bin/uv"
  if [[ -x "${candidate}" ]]; then
    UV_BIN="${candidate}"
    export PATH="$(dirname "${candidate}"):${PATH}"
    echo "Using repo-local uv: ${UV_BIN}"
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "uv is missing and curl is unavailable; cannot install repo-local uv." >&2
    return 2
  fi

  echo "Installing repo-local uv under ${REPO_ROOT}/.local_python/bin."
  mkdir -p "${REPO_ROOT}/.local_python/bin"
  export UV_INSTALL_DIR="${REPO_ROOT}/.local_python/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [[ -x "${candidate}" ]]; then
    UV_BIN="${candidate}"
    export PATH="$(dirname "${candidate}"):${PATH}"
    echo "Using newly installed repo-local uv: ${UV_BIN}"
    return 0
  fi

  echo "uv installation finished but ${candidate} was not found." >&2
  return 2
}

resolve_python13() {
  local candidate="${PYTHON_BIN:-}"
  if [[ -n "${candidate}" ]]; then
    if command -v "${candidate}" >/dev/null 2>&1 && validate_python13 "${candidate}"; then
      PYTHON13_BIN="$(command -v "${candidate}")"
      echo "Using PYTHON_BIN=${PYTHON13_BIN}"
      return 0
    fi
    echo "PYTHON_BIN is not a Python 3.13 executable: ${candidate}" >&2
    return 2
  fi

  if command -v python3.13 >/dev/null 2>&1 && validate_python13 python3.13; then
    PYTHON13_BIN="$(command -v python3.13)"
    echo "Using python3.13 from PATH: ${PYTHON13_BIN}"
    return 0
  fi

  candidate="${REPO_ROOT}/.local_python/bin/python3.13"
  if [[ -x "${candidate}" ]] && validate_python13 "${candidate}"; then
    PYTHON13_BIN="${candidate}"
    echo "Using repo-local Python: ${PYTHON13_BIN}"
    return 0
  fi

  resolve_uv
  if [[ -n "${UV_BIN}" ]]; then
    if candidate="$("${UV_BIN}" python find 3.13 2>/dev/null)" && [[ -n "${candidate}" ]] && validate_python13 "${candidate}"; then
      PYTHON13_BIN="${candidate}"
      echo "Using uv-managed Python: ${PYTHON13_BIN}"
      return 0
    fi
    echo "Python 3.13 not found; asking uv to install Python 3.13."
    "${UV_BIN}" python install 3.13
    candidate="$("${UV_BIN}" python find 3.13)"
    if [[ -n "${candidate}" ]] && validate_python13 "${candidate}"; then
      PYTHON13_BIN="${candidate}"
      echo "Using newly installed uv Python: ${PYTHON13_BIN}"
      return 0
    fi
  fi

  cat >&2 <<'EOF'
Python 3.13 was not found.
Install Python 3.13, or install uv and rerun:
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ./scripts/env/bootstrap_env.sh
Alternatively set PYTHON_BIN to an existing Python 3.13 executable.
EOF
  return 2
}

ensure_venv() {
  if [[ -x "${VENV_PYTHON}" ]]; then
    if validate_python13 "${VENV_PYTHON}"; then
      echo "Reusing existing env: ${VENV_DIR}"
      "${VENV_PYTHON}" -V
      return 0
    fi
    cat >&2 <<EOF
Existing Putpocket_env does not use Python 3.13: ${VENV_DIR}
Move it aside or remove it, then rerun:
  mv Putpocket_env Putpocket_env.backup
  ./scripts/env/bootstrap_env.sh
EOF
    return 2
  fi

  echo "Creating ${VENV_DIR} with ${PYTHON13_BIN}"
  "${PYTHON13_BIN}" -m venv "${VENV_DIR}"
  "${VENV_PYTHON}" -V
}

activate_for_bootstrap() {
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/env/env_activate.sh" || return
  VENV_PYTHON="$(command -v python)"
  VENV_CLI="${VENV_DIR}/bin/putpocket-dataset-mining"
  echo "Bootstrap shell activated ${VENV_PYTHON}"
}

install_dependencies() {
  "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel setuptools_scm packaging cmake ninja
  "${VENV_PYTHON}" -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "${TORCH_SPEC}"
  "${VENV_PYTHON}" -m pip install "ray==${RAY_VERSION}" datasets transformers pyyaml pytest
  "${VENV_PYTHON}" -m pip install -e "${REPO_ROOT}[dev]"
}

verify_torch_cuda() {
  "${VENV_PYTHON}" - <<'PY'
import sys
import torch

print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("cuda count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false; refusing to continue with a CPU-only runtime")
if torch.version.cuda != "12.9":
    raise SystemExit(f"expected torch.version.cuda == 12.9, got {torch.version.cuda!r}")
PY
}

checkout_externals() {
  "${VENV_CLI}" externals checkout all
}

module_import_ok() {
  local module="$1"
  "${VENV_PYTHON}" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${module}")
PY
}

install_vllm_if_needed() {
  if [[ ! -d "${REPO_ROOT}/externals/vllm" ]]; then
    echo "Missing externals/vllm. Run without --skip-externals first." >&2
    return 2
  fi
  if [[ "${FORCE_VLLM_BUILD}" -eq 0 ]] && module_import_ok vllm; then
    echo "vllm import already works; skipping editable install."
    return 0
  fi
  "${VENV_CLI}" externals install-editable vllm --python "${VENV_PYTHON}"
}

set_build_threads() {
  local threads="$1"
  export PUTPOCKET_BUILD_THREADS="${threads}"
  export MAX_JOBS="${threads}"
  export CMAKE_BUILD_PARALLEL_LEVEL="${threads}"
  export CARGO_BUILD_JOBS="${threads}"
  export NVCC_THREADS=1
}

is_oom_like_log() {
  local log_file="$1"
  grep -Eiq \
    "Killed|Out of memory|out of memory|OOM|Cannot allocate memory|exit code 137|ninja.*killed|compiler process killed|process killed during build" \
    "${log_file}"
}

clean_vllm_build_artifacts() {
  local log_file="$1"
  {
    echo "[cleanup] removing local vLLM build artifacts before retry"
    echo "rm -rf ${REPO_ROOT}/externals/vllm/build ${REPO_ROOT}/externals/vllm/*.egg-info"
  } >>"${log_file}"
  rm -rf "${REPO_ROOT}/externals/vllm/build" "${REPO_ROOT}"/externals/vllm/*.egg-info
}

install_vllm_with_retry() {
  if [[ ! -d "${REPO_ROOT}/externals/vllm" ]]; then
    echo "Missing externals/vllm. Run without --skip-externals first." >&2
    return 2
  fi
  if [[ "${FORCE_VLLM_BUILD}" -eq 0 ]] && module_import_ok vllm; then
    echo "[stage] install-vllm-editable"
    echo "  vllm import already works; skipping editable install."
    printf "skipped\t0\tpass\t%s\t%s\n" "already importable" "" >>"${VLLM_RETRY_SUMMARY}"
    return 0
  fi

  echo "[stage] install-vllm-editable"
  local attempts=(16 12 8)
  local attempt_index=0
  local threads
  for threads in "${attempts[@]}"; do
    attempt_index=$((attempt_index + 1))
    local log_file="${LOG_DIR}/vllm_build_threads_${threads}.log"
    local cmd_display
    cmd_display="$(quote_command env PUTPOCKET_BUILD_THREADS="${threads}" MAX_JOBS="${threads}" CMAKE_BUILD_PARALLEL_LEVEL="${threads}" CARGO_BUILD_JOBS="${threads}" NVCC_THREADS=1 "${VENV_PYTHON}" -m pip install --no-build-isolation --index-url "${TORCH_INDEX_URL}" --extra-index-url https://pypi.org/simple -c "${TORCH_CONSTRAINT_FILE}" -e "${REPO_ROOT}/externals/vllm")"

    echo "  attempt ${attempt_index}/${#attempts[@]} with ${threads} build threads"
    echo "  command: ${cmd_display}"
    echo "  log: ${log_file}"
    printf "%s\t%s\t%s\n" "install-vllm-editable-threads-${threads}" "${cmd_display}" "${log_file}" >>"${COMMAND_LOG}"

    if [[ "${attempt_index}" -gt 1 ]]; then
      clean_vllm_build_artifacts "${log_file}"
    fi

    set +e
    (
      set -Eeuo pipefail
      set_build_threads "${threads}"
      printf "$ %s\n" "${cmd_display}"
      "${VENV_PYTHON}" -m pip install \
        --no-build-isolation \
        --index-url "${TORCH_INDEX_URL}" \
        --extra-index-url https://pypi.org/simple \
        -c "${TORCH_CONSTRAINT_FILE}" \
        -e "${REPO_ROOT}/externals/vllm"
    ) >>"${log_file}" 2>&1
    local status=$?
    set -e

    if [[ "${status}" -eq 0 ]]; then
      echo "  success with ${threads} build threads"
      printf "%s\t%s\tpass\t%s\t%s\n" "attempt" "${threads}" "${log_file}" "" >>"${VLLM_RETRY_SUMMARY}"
      set_build_threads 16
      return 0
    fi

    if is_oom_like_log "${log_file}"; then
      printf "%s\t%s\tfail_oom\t%s\t%s\n" "attempt" "${threads}" "${log_file}" "OOM-like failure; retry lower if available" >>"${VLLM_RETRY_SUMMARY}"
      echo "  failed with OOM-like signature at ${threads} threads"
      if [[ "${threads}" -ne 8 ]]; then
        echo "  retrying with lower build parallelism"
        continue
      fi
      echo "vLLM editable build failed at 8 threads with OOM-like failure." >&2
    else
      printf "%s\t%s\tfail_non_oom\t%s\t%s\n" "attempt" "${threads}" "${log_file}" "non-OOM failure; not retrying" >>"${VLLM_RETRY_SUMMARY}"
      echo "vLLM editable build failed with a non-OOM error; not retrying blindly." >&2
    fi

    echo "Failing command: ${cmd_display}" >&2
    echo "Thread count: ${threads}" >&2
    echo "Relevant log: ${log_file}" >&2
    echo "Last relevant error lines:" >&2
    tail -80 "${log_file}" >&2 || true
    echo "Suggested next action: inspect ${log_file} and rerun ./scripts/env/bootstrap_env.sh --force-vllm-build after resolving the build failure." >&2
    set_build_threads 16
    return "${status}"
  done
}

install_deepgemm_if_needed() {
  local installer="${REPO_ROOT}/externals/vllm/tools/install_deepgemm.sh"
  if [[ ! -x "${installer}" ]]; then
    echo "Missing executable DeepGEMM installer: ${installer}" >&2
    return 2
  fi
  if [[ "${FORCE_DEEPGEMM_BUILD}" -eq 0 ]] && module_import_ok deep_gemm; then
    echo "deep_gemm import already works; skipping DeepGEMM install."
    return 0
  fi

  set_build_threads "${PUTPOCKET_BUILD_THREADS:-16}"
  "${installer}" --cuda-version "${DEEPGEMM_CUDA_VERSION}"
  "${VENV_PYTHON}" - <<'PY'
import deep_gemm

print("deep_gemm", getattr(deep_gemm, "__version__", "unknown"))
PY
}

install_lmcache_if_needed() {
  if [[ ! -d "${REPO_ROOT}/externals/lmcache" ]]; then
    echo "Missing externals/lmcache. Run without --skip-externals first." >&2
    return 2
  fi
  if [[ "${FORCE_LMCACHE_BUILD}" -eq 0 ]] && module_import_ok lmcache; then
    echo "lmcache import already works; skipping editable install."
    return 0
  fi
  "${VENV_CLI}" externals install-editable lmcache --python "${VENV_PYTHON}"
}

ensure_default_dockerfile() {
  if [[ -f "${DOCKERFILE}" ]]; then
    echo "Dockerfile exists: ${DOCKERFILE}"
    return 0
  fi

  mkdir -p "$(dirname "${DOCKERFILE}")"
  cat >"${DOCKERFILE}" <<'EOF'
FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.13.1
ARG PYTHON_BUILD_JOBS=16

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    coreutils \
    curl \
    git \
    jq \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    patch \
    ripgrep \
    tk-dev \
    tree \
    uuid-dev \
    wget \
    xz-utils \
    zlib1g-dev \
  && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/Python-${PYTHON_VERSION}.tgz https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
  && tar -C /tmp -xzf /tmp/Python-${PYTHON_VERSION}.tgz \
  && cd /tmp/Python-${PYTHON_VERSION} \
  && ./configure --prefix=/opt/python/3.13 --enable-optimizations --with-ensurepip=install \
  && make -j "${PYTHON_BUILD_JOBS}" \
  && make install \
  && ln -sf /opt/python/3.13/bin/python3.13 /usr/local/bin/python3.13 \
  && ln -sf /opt/python/3.13/bin/python3.13 /usr/local/bin/python \
  && ln -sf /opt/python/3.13/bin/pip3.13 /usr/local/bin/pip \
  && rm -rf /tmp/Python-${PYTHON_VERSION} /tmp/Python-${PYTHON_VERSION}.tgz

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel pytest

ENV PATH="/opt/python/3.13/bin:${PATH}"
ENV HOME=/tmp/putpocket-home
WORKDIR /workspace
EOF
  echo "Created Dockerfile: ${DOCKERFILE}"
}

ensure_docker_image() {
  ensure_default_dockerfile
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker command is missing. Install Docker or rerun with --skip-docker." >&2
    return 2
  fi
  if [[ "${FORCE_DOCKER_BUILD}" -eq 0 ]] && docker image inspect "${DOCKER_IMAGE}" >/dev/null 2>&1; then
    echo "Docker image already exists: ${DOCKER_IMAGE}"
    return 0
  fi
  docker build \
    --build-arg "PYTHON_BUILD_JOBS=${PUTPOCKET_BUILD_THREADS}" \
    -t "${DOCKER_IMAGE}" \
    -f "${DOCKERFILE}" \
    "${DOCKER_CONTEXT}"
}

write_externals_lock() {
  local lock_file="${LOG_DIR}/externals.lock"
  {
    echo "# Generated by scripts/env/bootstrap_env.sh at ${RUN_TIMESTAMP}"
    for name in vllm lmcache cline; do
      local path="${REPO_ROOT}/externals/${name}"
      if [[ -d "${path}/.git" ]]; then
        local branch
        local commit
        local remote
        branch="$(git -C "${path}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
        commit="$(git -C "${path}" rev-parse HEAD 2>/dev/null || true)"
        remote="$(git -C "${path}" remote get-url origin 2>/dev/null || true)"
        printf "%s path=%s branch=%s commit=%s remote=%s\n" "${name}" "${path}" "${branch}" "${commit}" "${remote}"
      elif [[ -e "${path}" ]]; then
        printf "%s path=%s branch= commit= remote= non_git=true\n" "${name}" "${path}"
      else
        printf "%s path=%s missing=true\n" "${name}" "${path}"
      fi
    done
  } >"${lock_file}"
  cat "${lock_file}"
}

report_visible_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L || true
  else
    echo "nvidia-smi is not available."
  fi
  echo "bootstrap_env.sh does not set CUDA_VISIBLE_DEVICES."
}

run_smoke_checks() {
  "${VENV_PYTHON}" -V
  "${VENV_PYTHON}" - <<'PY'
import importlib

for name in ("torch", "ray", "datasets", "transformers"):
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name} {version}")

import torch
print("torch cuda", torch.version.cuda, "available", torch.cuda.is_available(), "count", torch.cuda.device_count())
PY

  if [[ "${SKIP_VLLM_BUILD}" -eq 1 ]]; then
    module_import_ok vllm && echo "vllm import ok" || echo "warning: vllm import failed after --skip-vllm-build"
  else
    "${VENV_PYTHON}" -c "import vllm; print('vllm import ok')"
  fi

  if [[ "${SKIP_DEEPGEMM_BUILD}" -eq 1 ]]; then
    module_import_ok deep_gemm && echo "deep_gemm import ok" || echo "warning: deep_gemm import failed after --skip-deepgemm-build"
  else
    "${VENV_PYTHON}" -c "import deep_gemm; print('deep_gemm', getattr(deep_gemm, '__version__', 'unknown'))"
  fi

  if [[ "${SKIP_LMCACHE_BUILD}" -eq 1 ]]; then
    module_import_ok lmcache && echo "lmcache import ok" || echo "warning: lmcache import failed after --skip-lmcache-build"
  else
    "${VENV_PYTHON}" -c "import lmcache; print('lmcache import ok')"
  fi

  "${VENV_PYTHON}" -c "import putpocket_dataset_mining; print('putpocket_dataset_mining import ok')"

  if [[ -x "${VENV_CLI}" ]]; then
    "${VENV_CLI}" doctor --json
  else
    echo "warning: ${VENV_CLI} is missing; skipping CLI doctor"
  fi

  if [[ -d "${REPO_ROOT}/src" && -d "${REPO_ROOT}/tests" ]]; then
    "${VENV_PYTHON}" -m compileall -q "${REPO_ROOT}/src" "${REPO_ROOT}/tests"
  elif [[ -d "${REPO_ROOT}/src" ]]; then
    "${VENV_PYTHON}" -m compileall -q "${REPO_ROOT}/src"
  fi

  if [[ -d "${REPO_ROOT}/tests" ]]; then
    "${VENV_PYTHON}" -m unittest discover -s "${REPO_ROOT}/tests" -v
  else
    echo "No tests directory found; skipping unittest discovery."
  fi

  if [[ "${SKIP_DOCKER}" -eq 0 ]]; then
    docker image inspect "${DOCKER_IMAGE}" >/dev/null
    echo "docker image exists: ${DOCKER_IMAGE}"
  else
    echo "Docker validation skipped by --skip-docker."
  fi
}

write_summary() {
  local summary_file="${LOG_DIR}/summary.txt"
  local summary_json="${LOG_DIR}/setup_summary.json"
  {
    echo "Putpocket env bootstrap completed"
    echo "repo_root=${REPO_ROOT}"
    echo "venv=${VENV_DIR}"
    echo "python=${VENV_PYTHON}"
    echo "cuda_home=${CUDA_HOME}"
    echo "torch_spec=${TORCH_SPEC}"
    echo "torch_index_url=${TORCH_INDEX_URL}"
    echo "docker_image=${DOCKER_IMAGE}"
    echo "log_dir=${LOG_DIR}"
    echo "latest_log=${LOG_ROOT}/latest"
  } >"${summary_file}"
  "${VENV_PYTHON}" - "${summary_json}" "${VLLM_RETRY_SUMMARY}" <<'PY'
import importlib
import json
import os
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
retry_path = Path(sys.argv[2])
attempts = []
if retry_path.exists():
    for line in retry_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 5:
            attempts.append(
                {
                    "kind": parts[0],
                    "threads": None if parts[1] == "" else int(parts[1]),
                    "status": parts[2],
                    "log_path": parts[3],
                    "note": parts[4],
                }
            )

payload = {
    "repo_root": os.environ.get("REPO_ROOT", ""),
    "venv": os.environ.get("VIRTUAL_ENV", ""),
    "python": sys.executable,
    "cuda_home": os.environ.get("CUDA_HOME", ""),
    "torch_spec": os.environ.get("TORCH_SPEC", ""),
    "torch_index_url": os.environ.get("TORCH_INDEX_URL", ""),
    "ray_version": os.environ.get("RAY_VERSION", ""),
    "build_env": {
        "PUTPOCKET_BUILD_THREADS": os.environ.get("PUTPOCKET_BUILD_THREADS", ""),
        "MAX_JOBS": os.environ.get("MAX_JOBS", ""),
        "CMAKE_BUILD_PARALLEL_LEVEL": os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", ""),
        "CARGO_BUILD_JOBS": os.environ.get("CARGO_BUILD_JOBS", ""),
        "NVCC_THREADS": os.environ.get("NVCC_THREADS", ""),
    },
    "vllm_build_retry_summary": attempts,
}
try:
    deep_gemm = importlib.import_module("deep_gemm")
except Exception as exc:
    payload["deep_gemm"] = {
        "available": False,
        "error": f"{type(exc).__name__}: {exc}",
    }
else:
    payload["deep_gemm"] = {
        "available": True,
        "version": getattr(deep_gemm, "__version__", "unknown"),
        "file": getattr(deep_gemm, "__file__", ""),
    }
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  cat "${summary_file}"
  echo "setup_summary_json=${summary_json}"
}

echo "Putpocket env bootstrap"
echo "  repo root: ${REPO_ROOT}"
echo "  log dir: ${LOG_DIR}"
echo "  doctor only: ${DOCTOR_ONLY}"

run_stage "gpu-report" report_visible_gpus

if [[ "${DOCTOR_ONLY}" -eq 1 ]]; then
  run_stage_current_shell "activate-env" activate_for_bootstrap
  run_stage "record-externals" write_externals_lock
  run_stage "doctor-smoke" run_smoke_checks
  run_stage "write-summary" write_summary
  exit 0
fi

run_stage_current_shell "resolve-python" resolve_python13
run_stage "ensure-venv" ensure_venv
run_stage_current_shell "activate-env" activate_for_bootstrap
run_stage "install-core-dependencies" install_dependencies
run_stage "verify-torch-cuda" verify_torch_cuda

if [[ "${SKIP_EXTERNALS}" -eq 0 ]]; then
  run_stage "checkout-externals" checkout_externals
else
  echo "[stage] checkout-externals"
  echo "  skipped by --skip-externals"
fi

if [[ "${SKIP_VLLM_BUILD}" -eq 0 ]]; then
  install_vllm_with_retry
else
  echo "[stage] install-vllm-editable"
  echo "  skipped by --skip-vllm-build"
fi

if [[ "${SKIP_DEEPGEMM_BUILD}" -eq 0 ]]; then
  run_stage "install-deepgemm" install_deepgemm_if_needed
else
  echo "[stage] install-deepgemm"
  echo "  skipped by --skip-deepgemm-build"
fi

if [[ "${SKIP_LMCACHE_BUILD}" -eq 0 ]]; then
  run_stage "install-lmcache-editable" install_lmcache_if_needed
else
  echo "[stage] install-lmcache-editable"
  echo "  skipped by --skip-lmcache-build"
fi

run_stage "record-externals" write_externals_lock

if [[ "${SKIP_DOCKER}" -eq 0 ]]; then
  run_stage "ensure-docker-image" ensure_docker_image
else
  echo "[stage] ensure-docker-image"
  echo "  skipped by --skip-docker"
fi

run_stage "doctor-smoke" run_smoke_checks
run_stage "write-summary" write_summary
