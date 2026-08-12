#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cat >&2 <<'EOF'
bootstrap_glm52_env.sh is a retired local GLM vLLM 0.23 bootstrap.

The active Server-2 environment is Putpocket_env and is provisioned with:
  ./scripts/env/bootstrap_sr.sh --preset server2

Future full GLM-5.2 serving is expected to run on RunPod Hopper runtimes, not
inside the active Server-2 Qwen environment.
EOF

if [[ "${PUTPOCKET_ALLOW_LEGACY_GLM_ENV:-}" != "1" ]]; then
  echo "Refusing legacy GLM environment setup. Set PUTPOCKET_ALLOW_LEGACY_GLM_ENV=1 to run the archived script." >&2
  exit 2
fi

exec "${REPO_ROOT}/scripts/env/legacy/bootstrap_glm52_env.sh" "$@"
