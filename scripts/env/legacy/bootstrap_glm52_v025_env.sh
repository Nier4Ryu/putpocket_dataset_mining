#!/usr/bin/env bash
set -Eeuo pipefail

print_help() {
  cat <<'EOF'
Usage: ./scripts/env/bootstrap_glm52_v025_env.sh [options]

Create or repair the separate GLM-5.2 vLLM 0.25.x environment without
touching Putpocket_env, Putpocket_env_glm52, externals/vllm, or
externals/vllm_glm52.

Options:
  --doctor-only          Do not install/build/checkout; only activate and validate the existing env.
  --skip-vllm-build      Skip editable vLLM 0.25 source build/install.
  --force-vllm-build     Run editable vLLM 0.25 install even if import already works.
  --skip-deepgemm-build  Skip DeepGEMM build/install.
  --force-deepgemm-build Run DeepGEMM install even if import already works.
  --help                 Show this help.

Environment overrides:
  PYTHON_BIN             Python 3.13 executable to use.
  CUDA_HOME              Default: /usr/local/cuda-12.9
  TORCH_CUDA_TAG         Default: cu129
  TORCH_VERSION          Default: 2.11.0
  VLLM_GLM52_V025_REF    Default: v0.25.1
  VLLM_GLM52_V025_URL    Default: https://github.com/vllm-project/vllm.git

The GLM vLLM 0.25 build cap is fixed at 8 jobs by default. This bootstrap
never tries 12 or 16 build threads.
EOF
}

DOCTOR_ONLY=0
SKIP_VLLM_BUILD=0
FORCE_VLLM_BUILD=0
SKIP_DEEPGEMM_BUILD=0
FORCE_DEEPGEMM_BUILD=0

while (($#)); do
  case "$1" in
    --doctor-only) DOCTOR_ONLY=1 ;;
    --skip-vllm-build) SKIP_VLLM_BUILD=1 ;;
    --force-vllm-build) FORCE_VLLM_BUILD=1 ;;
    --skip-deepgemm-build) SKIP_DEEPGEMM_BUILD=1 ;;
    --force-deepgemm-build) FORCE_DEEPGEMM_BUILD=1 ;;
    --help)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Run ./scripts/env/bootstrap_glm52_v025_env.sh --help for usage." >&2
      exit 2
      ;;
  esac
  shift
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export REPO_ROOT

VENV_DIR="${REPO_ROOT}/Putpocket_env_glm52_v025"
VENV_PYTHON="${VENV_DIR}/bin/python"
VLLM_GLM52_V025_DIR="${REPO_ROOT}/externals/vllm_glm52_v025"
LOG_ROOT="${REPO_ROOT}/logs/env_setup_glm52_v025"
RUN_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="${LOG_ROOT}/${RUN_TIMESTAMP}"
COMMAND_LOG="${LOG_DIR}/commands.log"
SUMMARY_JSON="${LOG_DIR}/setup_summary.json"
mkdir -p "${LOG_DIR}"
ln -sfn "${RUN_TIMESTAMP}" "${LOG_ROOT}/latest"
: >"${COMMAND_LOG}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
export TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
export TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu129}"
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA_TAG}}"
export VLLM_GLM52_V025_URL="${VLLM_GLM52_V025_URL:-https://github.com/vllm-project/vllm.git}"
export VLLM_GLM52_V025_REF="${VLLM_GLM52_V025_REF:-v0.25.1}"
export VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-cuda}"
export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-0}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PUTPOCKET_BUILD_THREADS=8
export MAX_JOBS=8
export CMAKE_BUILD_PARALLEL_LEVEL=8
export CARGO_BUILD_JOBS=8
export NVCC_THREADS=1
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TZ="${TZ:-Asia/Seoul}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${VENV_DIR}}"
if [[ -d "${CUDA_HOME}/bin" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if [[ -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi
if [[ -d "${REPO_ROOT}/.local_python/bin" ]]; then
  export PATH="${REPO_ROOT}/.local_python/bin:${PATH}"
fi

PYTHON13_BIN=""
UV_BIN=""

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
    tail -80 "${log_file}" >&2 || true
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
    tail -80 "${log_file}" >&2 || true
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
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
    return 0
  fi
  local candidate="${REPO_ROOT}/.local_python/bin/uv"
  if [[ -x "${candidate}" ]]; then
    UV_BIN="${candidate}"
    export PATH="$(dirname "${candidate}"):${PATH}"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "uv is missing and curl is unavailable." >&2
    return 2
  fi
  mkdir -p "${REPO_ROOT}/.local_python/bin"
  export UV_INSTALL_DIR="${REPO_ROOT}/.local_python/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [[ -x "${candidate}" ]]; then
    UV_BIN="${candidate}"
    export PATH="$(dirname "${candidate}"):${PATH}"
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
      return 0
    fi
    echo "PYTHON_BIN is not a Python 3.13 executable: ${candidate}" >&2
    return 2
  fi
  if command -v python3.13 >/dev/null 2>&1 && validate_python13 python3.13; then
    PYTHON13_BIN="$(command -v python3.13)"
    return 0
  fi
  resolve_uv
  candidate="$("${UV_BIN}" python find 3.13 2>/dev/null || true)"
  if [[ -n "${candidate}" ]] && validate_python13 "${candidate}"; then
    PYTHON13_BIN="${candidate}"
    return 0
  fi
  "${UV_BIN}" python install 3.13
  candidate="$("${UV_BIN}" python find 3.13)"
  if [[ -n "${candidate}" ]] && validate_python13 "${candidate}"; then
    PYTHON13_BIN="${candidate}"
    return 0
  fi
  echo "Python 3.13 was not found and uv could not provide it." >&2
  return 2
}

ensure_venv() {
  if [[ -x "${VENV_PYTHON}" ]]; then
    validate_python13 "${VENV_PYTHON}"
    "${VENV_PYTHON}" -V
    return 0
  fi
  "${PYTHON13_BIN}" -m venv "${VENV_DIR}"
  "${VENV_PYTHON}" -V
}

activate_for_bootstrap() {
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/env/env_activate_glm52_v025.sh"
  VENV_PYTHON="$(command -v python)"
}

module_import_ok() {
  local module="$1"
  "${VENV_PYTHON}" - <<PY >/dev/null 2>&1
import importlib
importlib.import_module("${module}")
PY
}

checkout_vllm_glm52_v025() {
  mkdir -p "${VLLM_GLM52_V025_DIR%/*}"
  if [[ -d "${VLLM_GLM52_V025_DIR}/.git" ]]; then
    if ! git -C "${VLLM_GLM52_V025_DIR}" diff --quiet || ! git -C "${VLLM_GLM52_V025_DIR}" diff --cached --quiet; then
      echo "externals/vllm_glm52_v025 has tracked local changes; refusing to update it." >&2
      return 2
    fi
    git -C "${VLLM_GLM52_V025_DIR}" fetch --tags origin "${VLLM_GLM52_V025_REF}"
    git -C "${VLLM_GLM52_V025_DIR}" checkout --detach "${VLLM_GLM52_V025_REF}"
  elif [[ -e "${VLLM_GLM52_V025_DIR}" ]]; then
    echo "externals/vllm_glm52_v025 exists but is not a git checkout." >&2
    return 2
  else
    git clone --branch "${VLLM_GLM52_V025_REF}" --depth 1 "${VLLM_GLM52_V025_URL}" "${VLLM_GLM52_V025_DIR}"
  fi
  git -C "${VLLM_GLM52_V025_DIR}" rev-parse HEAD
  git -C "${VLLM_GLM52_V025_DIR}" describe --tags --always
}

install_python_deps() {
  "${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
  "${VENV_PYTHON}" -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    "torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}" \
    "torchvision==0.26.0+${TORCH_CUDA_TAG}" \
    "torchaudio==${TORCH_VERSION}+${TORCH_CUDA_TAG}"
  "${VENV_PYTHON}" -m pip install \
    --index-url "${TORCH_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    -r "${VLLM_GLM52_V025_DIR}/requirements/build/cuda.txt" \
    -r "${VLLM_GLM52_V025_DIR}/requirements/cuda.txt"
  "${VENV_PYTHON}" -m pip install -e "${REPO_ROOT}[dev]"
}

build_vllm_glm52_v025() {
  if [[ "${FORCE_VLLM_BUILD}" -eq 0 ]] && module_import_ok vllm; then
    echo "vllm import already works in Putpocket_env_glm52_v025; skipping source build."
    return 0
  fi
  rm -rf "${VLLM_GLM52_V025_DIR}/build" "${VLLM_GLM52_V025_DIR}"/*.egg-info
  "${VENV_PYTHON}" -m pip install \
    --no-build-isolation \
    --index-url "${TORCH_INDEX_URL}" \
    --extra-index-url https://pypi.org/simple \
    -e "${VLLM_GLM52_V025_DIR}"
}

install_deepgemm_glm52_v025() {
  local installer="${VLLM_GLM52_V025_DIR}/tools/install_deepgemm.sh"
  if [[ ! -x "${installer}" ]]; then
    echo "DeepGEMM installer missing: ${installer}" >&2
    return 2
  fi
  if [[ "${FORCE_DEEPGEMM_BUILD}" -eq 0 ]] && module_import_ok deep_gemm; then
    echo "deep_gemm import already works in Putpocket_env_glm52_v025; skipping install."
    return 0
  fi
  "${installer}" --cuda-version 12.9
}

verify_torch_cuda() {
  "${VENV_PYTHON}" - <<'PY'
import torch
print("torch", torch.__version__)
print("torch.cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("cuda count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
if torch.version.cuda != "12.9":
    raise SystemExit(f"expected torch.version.cuda == 12.9, got {torch.version.cuda!r}")
PY
}

doctor_smoke() {
  "${VENV_PYTHON}" -V
  "${VENV_PYTHON}" - <<'PY'
import importlib
for name in ("torch", "transformers", "vllm", "putpocket_dataset_mining"):
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "unknown"), getattr(module, "__file__", ""))
try:
    import deep_gemm
except Exception as exc:
    print("deep_gemm unavailable", type(exc).__name__, exc)
else:
    print("deep_gemm", getattr(deep_gemm, "__version__", "unknown"), getattr(deep_gemm, "__file__", ""))
PY
}

write_manifest() {
  "${VENV_PYTHON}" - "${SUMMARY_JSON}" "${VLLM_GLM52_V025_DIR}" <<'PY'
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
vllm_dir = Path(sys.argv[2])

def run_git(args):
    return subprocess.check_output(["git", "-C", str(vllm_dir), *args], text=True).strip()

payload = {
    "env": os.environ.get("VIRTUAL_ENV", ""),
    "python": sys.executable,
    "cuda_home": os.environ.get("CUDA_HOME", ""),
    "build_env": {
        "PUTPOCKET_BUILD_THREADS": os.environ.get("PUTPOCKET_BUILD_THREADS", ""),
        "MAX_JOBS": os.environ.get("MAX_JOBS", ""),
        "CMAKE_BUILD_PARALLEL_LEVEL": os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", ""),
        "CARGO_BUILD_JOBS": os.environ.get("CARGO_BUILD_JOBS", ""),
        "NVCC_THREADS": os.environ.get("NVCC_THREADS", ""),
        "TORCH_CUDA_ARCH_LIST": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
    },
    "vllm_source": {
        "path": str(vllm_dir),
        "remote": run_git(["remote", "get-url", "origin"]),
        "commit": run_git(["rev-parse", "HEAD"]),
        "describe": run_git(["describe", "--tags", "--always"]),
    },
}
for name in ("torch", "transformers", "vllm", "deep_gemm"):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        payload[name] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    else:
        payload[name] = {
            "available": True,
            "version": getattr(module, "__version__", "unknown"),
            "file": getattr(module, "__file__", ""),
        }
summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(summary_path)
PY
}

echo "Putpocket GLM-5.2 vLLM 0.25 env bootstrap"
echo "  repo root: ${REPO_ROOT}"
echo "  log dir: ${LOG_DIR}"
echo "  doctor only: ${DOCTOR_ONLY}"

if [[ "${DOCTOR_ONLY}" -eq 1 ]]; then
  run_stage_current_shell "activate-env" activate_for_bootstrap
  run_stage "doctor-smoke" doctor_smoke
  run_stage "write-manifest" write_manifest
  exit 0
fi

run_stage_current_shell "resolve-python" resolve_python13
run_stage "ensure-venv" ensure_venv
run_stage_current_shell "activate-env" activate_for_bootstrap
run_stage "checkout-vllm-glm52-v025" checkout_vllm_glm52_v025
run_stage "install-python-deps" install_python_deps
run_stage "verify-torch-cuda" verify_torch_cuda

if [[ "${SKIP_VLLM_BUILD}" -eq 0 ]]; then
  run_stage "build-vllm-glm52-v025" build_vllm_glm52_v025
else
  echo "[stage] build-vllm-glm52-v025"
  echo "  skipped by --skip-vllm-build"
fi

if [[ "${SKIP_DEEPGEMM_BUILD}" -eq 0 ]]; then
  run_stage "install-deepgemm-glm52-v025" install_deepgemm_glm52_v025
else
  echo "[stage] install-deepgemm-glm52-v025"
  echo "  skipped by --skip-deepgemm-build"
fi

run_stage "doctor-smoke" doctor_smoke
run_stage "write-manifest" write_manifest
