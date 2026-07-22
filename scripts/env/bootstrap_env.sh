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
  --skip-lmcache-build    Skip editable LMCache build/install.
  --skip-externals        Skip external repository checkout/update.
  --skip-docker           Skip Dockerfile/image checks.
  --force-vllm-build      Run editable vLLM install even if import already works.
  --force-lmcache-build   Run editable LMCache install even if import already works.
  --force-docker-build    Rebuild the default Docker image even if it already exists.
  --help                  Show this help.

Environment overrides:
  PYTHON_BIN                    Python 3.13 executable to use.
  CUDA_HOME                     Default: /usr/local/cuda-12.8
  PUTPOCKET_BUILD_THREADS       Default: 32
  MAX_JOBS                      Default: 32
  CMAKE_BUILD_PARALLEL_LEVEL    Default: 32
  CARGO_BUILD_JOBS              Default: 32
  NVCC_THREADS                  Default: 1
EOF
}

DOCTOR_ONLY=0
SKIP_VLLM_BUILD=0
SKIP_LMCACHE_BUILD=0
SKIP_EXTERNALS=0
SKIP_DOCKER=0
FORCE_VLLM_BUILD=0
FORCE_LMCACHE_BUILD=0
FORCE_DOCKER_BUILD=0

while (($#)); do
  case "$1" in
    --doctor-only) DOCTOR_ONLY=1 ;;
    --skip-vllm-build) SKIP_VLLM_BUILD=1 ;;
    --skip-lmcache-build) SKIP_LMCACHE_BUILD=1 ;;
    --skip-externals) SKIP_EXTERNALS=1 ;;
    --skip-docker) SKIP_DOCKER=1 ;;
    --force-vllm-build) FORCE_VLLM_BUILD=1 ;;
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
VENV_DIR="${REPO_ROOT}/Putpocket_env"
DOCKERFILE="${REPO_ROOT}/docker/default_python/Dockerfile"
DOCKER_CONTEXT="${REPO_ROOT}/docker"
DOCKER_IMAGE="putpocket-default-python:ubuntu22.04-py313-v1"

LOG_ROOT="${REPO_ROOT}/logs/env_setup"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_ROOT}/${RUN_TIMESTAMP}"
COMMAND_LOG="${LOG_DIR}/commands.log"
mkdir -p "${LOG_DIR}"
ln -sfn "${RUN_TIMESTAMP}" "${LOG_ROOT}/latest"
: >"${COMMAND_LOG}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PUTPOCKET_BUILD_THREADS="${PUTPOCKET_BUILD_THREADS:-32}"
export MAX_JOBS="${MAX_JOBS:-32}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-32}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-32}"
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

PYTHON13_BIN=""
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

  if command -v uv >/dev/null 2>&1; then
    if candidate="$(uv python find 3.13 2>/dev/null)" && [[ -n "${candidate}" ]] && validate_python13 "${candidate}"; then
      PYTHON13_BIN="${candidate}"
      echo "Using uv-managed Python: ${PYTHON13_BIN}"
      return 0
    fi
    echo "Python 3.13 not found; asking uv to install Python 3.13."
    uv python install 3.13
    candidate="$(uv python find 3.13)"
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
    --index-url https://download.pytorch.org/whl/cu128 \
    --extra-index-url https://pypi.org/simple \
    "torch==2.10.0+cu128"
  "${VENV_PYTHON}" -m pip install "ray==2.55.1" datasets transformers pyyaml pytest
  "${VENV_PYTHON}" -m pip install -e "${REPO_ROOT}[dev]"
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
ARG PYTHON_BUILD_JOBS=32

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
  {
    echo "Putpocket env bootstrap completed"
    echo "repo_root=${REPO_ROOT}"
    echo "venv=${VENV_DIR}"
    echo "python=${VENV_PYTHON}"
    echo "docker_image=${DOCKER_IMAGE}"
    echo "log_dir=${LOG_DIR}"
    echo "latest_log=${LOG_ROOT}/latest"
  } >"${summary_file}"
  cat "${summary_file}"
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

if [[ "${SKIP_EXTERNALS}" -eq 0 ]]; then
  run_stage "checkout-externals" checkout_externals
else
  echo "[stage] checkout-externals"
  echo "  skipped by --skip-externals"
fi

if [[ "${SKIP_VLLM_BUILD}" -eq 0 ]]; then
  run_stage "install-vllm-editable" install_vllm_if_needed
else
  echo "[stage] install-vllm-editable"
  echo "  skipped by --skip-vllm-build"
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
