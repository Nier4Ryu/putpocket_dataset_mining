#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if (($#)); then
  for arg in "$@"; do
    case "${arg}" in
      --doctor-only|--dry-run|--force-vllm-build|--force-docker-build|--skip-docker|--skip-gpu-smoke|--help) ;;
      --skip-vllm-build|--skip-lmcache-build|--skip-deepgemm-build|--skip-externals)
        echo "bootstrap_env.sh is deprecated; ${arg} is no longer an active Server-2 setup flag." >&2
        echo "Use ./scripts/env/bootstrap_sr.sh --preset server2 --doctor-only for validation." >&2
        exit 2
        ;;
      --force-lmcache-build|--force-deepgemm-build)
        echo "bootstrap_env.sh is deprecated; ${arg} is not supported by the canonical Server-2 preset." >&2
        exit 2
        ;;
      *)
        echo "Unknown compatibility flag for bootstrap_env.sh: ${arg}" >&2
        echo "Use ./scripts/env/bootstrap_sr.sh --preset server2 --help." >&2
        exit 2
        ;;
    esac
  done
fi

echo "bootstrap_env.sh is deprecated; delegating to bootstrap_sr.sh --preset server2." >&2
exec "${REPO_ROOT}/scripts/env/bootstrap_sr.sh" --preset server2 "$@"
