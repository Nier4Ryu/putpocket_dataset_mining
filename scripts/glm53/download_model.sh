#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_ID="T20260830-001__glm53-montblanc-deployment"
RUNTIME_ROOT="${GLM53_RUNTIME_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/putpocket-runtime/${TASK_ID}}"
MODEL_DIR="${GLM53_MODEL_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/putpocket-models/${TASK_ID}/RedHatAI--GLM-5.3-Flash-NVFP4--36c184c6}"
LOCK_PATH="${REPO_ROOT}/configs/models/glm53_flash_nvfp4_montblanc.lock.json"

if [[ "${PUTPOCKET_GLM53_BUILD_LOCK_HELD:-0}" != "1" ]]; then
  export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  exec python3 -m putpocket_dataset_mining.glm53_deployment \
    --lock "${LOCK_PATH}" lock-run -- "$0" "$@"
fi

if [[ "$(git -C "${REPO_ROOT}" branch --show-current)" != "agent/${TASK_ID}" ]]; then
  echo "Refusing model download outside agent/${TASK_ID}" >&2
  exit 2
fi
if [[ "${PUTPOCKET_ALLOW_TASK_PRODUCTION:-0}" != "1" ]]; then
  echo "Set PUTPOCKET_ALLOW_TASK_PRODUCTION=1 for this user-authorized task-local download." >&2
  exit 2
fi

DOWNLOAD_ENV="${RUNTIME_ROOT}/download-env-uv"
UV_BIN="${PUTPOCKET_UV_BIN:-$(command -v uv || true)}"
if [[ -z "${UV_BIN}" && -x "${HOME}/putpocket_dataset_mining/.local_python/bin/uv" ]]; then
  UV_BIN="${HOME}/putpocket_dataset_mining/.local_python/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  echo "A uv executable is required for the isolated download environment." >&2
  exit 2
fi
mkdir -p "${RUNTIME_ROOT}/evidence" "$(dirname "${MODEL_DIR}")"
if [[ ! -x "${DOWNLOAD_ENV}/bin/python" ]]; then
  "${UV_BIN}" venv --python 3.13 "${DOWNLOAD_ENV}"
fi
"${UV_BIN}" pip install --python "${DOWNLOAD_ENV}/bin/python" \
  "huggingface_hub==0.36.0" "hf_xet==1.1.10"

export GLM53_DOWNLOAD_LOCK_PATH="${LOCK_PATH}"
export GLM53_DOWNLOAD_MODEL_DIR="${MODEL_DIR}"
"${DOWNLOAD_ENV}/bin/python" - <<'PY'
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

lock = json.loads(Path(os.environ["GLM53_DOWNLOAD_LOCK_PATH"]).read_text(encoding="utf-8"))
target = Path(os.environ["GLM53_DOWNLOAD_MODEL_DIR"])
target.parent.mkdir(parents=True, exist_ok=True)
remaining = sum(
    item["size"]
    for item in lock["files"]
    if not (target / item["path"]).is_file()
    or (target / item["path"]).stat().st_size != item["size"]
)
free = shutil.disk_usage(target.parent).free
reserve = 40 * 1024**3
if free < remaining + reserve:
    raise SystemExit(
        f"fail closed: {free} free bytes cannot hold {remaining} remaining bytes "
        f"plus {reserve} reserve bytes"
    )
snapshot_download(
    repo_id=lock["selection"]["repository"],
    revision=lock["selection"]["revision"],
    local_dir=target,
    allow_patterns=[item["path"] for item in lock["files"]],
    max_workers=2,
)
PY

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m putpocket_dataset_mining.glm53_deployment --lock "${LOCK_PATH}" \
  verify-model --model-dir "${MODEL_DIR}" \
  --output "${RUNTIME_ROOT}/evidence/model_verification.json"

printf '%s\n' "${MODEL_DIR}"
