from __future__ import annotations

import os
import re
from pathlib import Path

from putpocket_dataset_mining.errors import ConfigError

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def remote_job_root() -> Path:
    root = Path(os.environ.get("SR_REMOTE_JOB_ROOT", "data/remote_verifier")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def safe_id(value: str, field: str = "id") -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise ConfigError(f"Invalid {field}: {value!r}")
    return value


def state_dir(state: str) -> Path:
    state = safe_id(state, "state")
    path = remote_job_root() / state
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_dir(state: str, job_id: str) -> Path:
    root = state_dir(state)
    path = (root / safe_id(job_id, "job_id")).resolve()
    if root not in path.parents:
        raise ConfigError(f"Job path escapes state directory: {path}")
    return path


def assert_no_symlink_escape(root: Path) -> None:
    root = root.resolve()
    for item in root.rglob("*"):
        if item.is_symlink():
            target = item.resolve()
            if root not in target.parents and target != root:
                raise ConfigError(f"Symlink escapes job root: {item} -> {target}")
