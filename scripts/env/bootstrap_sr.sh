#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

find_python() {
  local env_python="${PUTPOCKET_ENV_PATH:-${REPO_ROOT}/Putpocket_env}/bin/python"
  if [[ -x "${env_python}" ]]; then
    printf "%s\n" "${env_python}"
    return 0
  fi
  if command -v python3.13 >/dev/null 2>&1; then
    command -v python3.13
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  echo "No Python interpreter found. Install Python 3.13 or uv, then rerun." >&2
  return 2
}

PYTHON_BIN="${PYTHON_BIN:-$(find_python)}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PUTPOCKET_REPO_ROOT="${PUTPOCKET_REPO_ROOT:-${REPO_ROOT}}"
export PUTPOCKET_ENV_PATH="${PUTPOCKET_ENV_PATH:-${PUTPOCKET_REPO_ROOT}/Putpocket_env}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-${PUTPOCKET_ENV_PATH}}"

exec "${PYTHON_BIN}" -m putpocket_dataset_mining.bootstrap_sr "$@"
