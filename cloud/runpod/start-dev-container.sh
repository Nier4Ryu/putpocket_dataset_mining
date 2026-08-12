#!/usr/bin/env bash
set -euo pipefail

CODEX_HOME="${CODEX_HOME:-/workspace/.private/codex}"

if [ -d /workspace ] || [ "${PUTPOCKET_RUNPOD_CONFIG_ONLY:-0}" = "1" ]; then
  mkdir -p "${CODEX_HOME}"
  chmod 700 "${CODEX_HOME}"
  umask 077
  python3 - "${CODEX_HOME}/config.toml" <<'PY'
from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
from pathlib import Path


config_path = Path(sys.argv[1])
required = {
    "cli_auth_credentials_store": 'cli_auth_credentials_store = "file"',
    "sandbox_mode": 'sandbox_mode = "danger-full-access"',
    "approval_policy": 'approval_policy = "on-request"',
}
original = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
lines = original.splitlines(keepends=True)
seen: set[str] = set()
reconciled: list[str] = []
assignment = re.compile(r"^(?P<indent>\s*)(?P<key>cli_auth_credentials_store|sandbox_mode|approval_policy)\s*=.*$")
in_top_level = True

for line in lines:
    if line.lstrip().startswith("["):
        in_top_level = False
    match = assignment.match(line.rstrip("\r\n")) if in_top_level else None
    if not match:
        reconciled.append(line)
        continue
    key = match.group("key")
    if key not in seen:
        reconciled.append(f'{match.group("indent")}{required[key]}\n')
        seen.add(key)

missing = [key for key in required if key not in seen]
if missing:
    insertion = next((index for index, line in enumerate(reconciled) if line.lstrip().startswith("[")), len(reconciled))
    reconciled[insertion:insertion] = [f"{required[key]}\n" for key in missing]

updated = "".join(reconciled)
if updated != original:
    if config_path.exists():
        backup = config_path.with_name("config.toml.pre-runpod-policy.bak")
        if not backup.exists():
            shutil.copyfile(config_path, backup)
            os.chmod(backup, 0o600)
    fd, temporary_name = tempfile.mkstemp(prefix=".config.toml.", dir=config_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(updated)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, config_path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
else:
    os.chmod(config_path, 0o600)
PY
fi

if [ "${PUTPOCKET_RUNPOD_CONFIG_ONLY:-0}" = "1" ]; then
  exit 0
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
