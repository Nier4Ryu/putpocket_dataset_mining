#!/usr/bin/env bash
set -euo pipefail

CODEX_HOME="${CODEX_HOME:-/workspace/.private/codex}"

if [ -d /workspace ]; then
  mkdir -p "${CODEX_HOME}"
  chmod 700 "${CODEX_HOME}"
  if [ ! -e "${CODEX_HOME}/config.toml" ]; then
    umask 077
    printf 'cli_auth_credentials_store = "file"\n' > "${CODEX_HOME}/config.toml"
  fi
fi

echo "Putpocket RunPod development image"
echo "CODEX_HOME=${CODEX_HOME}"
echo "node: $(node --version 2>/dev/null || true)"
echo "npm: $(npm --version 2>/dev/null || true)"
echo "zellij: $(zellij --version 2>/dev/null || true)"
echo "codex: $(codex --version 2>/dev/null || true)"
echo "uv: $(uv --version 2>/dev/null || true)"
echo "git: $(git --version 2>/dev/null || true)"
echo "nvcc: $(nvcc --version 2>/dev/null | tail -n 1 || true)"

exec sleep infinity
