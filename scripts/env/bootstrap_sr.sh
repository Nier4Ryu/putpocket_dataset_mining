#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f "${REPO_ROOT}/scripts/env/env_activate.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/env/env_activate.sh" >/dev/null 2>&1 || true
fi

python -m putpocket_dataset_mining.bootstrap_sr "$@"
