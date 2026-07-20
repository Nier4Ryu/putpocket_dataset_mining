#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
venv_dir="${repo_root}/Putpocket_env"

python_bin="${PYTHON_BIN:-}"
if [[ -z "${python_bin}" ]]; then
  if command -v python3.13 >/dev/null 2>&1; then
    python_bin="python3.13"
  elif [[ -x "${repo_root}/.local_python/bin/python3.13" ]]; then
    python_bin="${repo_root}/.local_python/bin/python3.13"
  elif command -v uv >/dev/null 2>&1; then
    python_bin="$(uv python find 3.13)"
  else
    python_bin="python3.13"
  fi
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Missing ${python_bin}. Install Python 3.13 before bootstrapping the repo env." >&2
  exit 2
fi

"${python_bin}" -m venv "${venv_dir}"
# shellcheck disable=SC1091
source "${repo_root}/scripts/env/env_activate.sh"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${repo_root}[dev]"
