from __future__ import annotations

import os
import re
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence
from urllib.parse import urlsplit

from .errors import ConfigError


E_SLURM_ALLOCATION_REQUIRED = "E_SLURM_ALLOCATION_REQUIRED"
E_SECRET_BEARING_COMMAND = "E_SECRET_BEARING_COMMAND"
E_UNSAFE_CLUSTER_PATH = "E_UNSAFE_CLUSTER_PATH"

_SENSITIVE_NAMES = {
    "api_key",
    "apikey",
    "auth_token",
    "credential",
    "credentials",
    "hf_token",
    "identity_file",
    "password",
    "private_key",
    "secret",
    "secrets",
    "token",
}
_SENSITIVE_PATH_PARTS = {".aws", ".gnupg", ".private", ".ssh", "credentials", "secrets", "tokens"}
_SECRET_FLAG = re.compile(
    r"^--?(?:api[-_]?key|auth[-_]?token|credential|hf[-_]?token|password|private[-_]?key|secret|token)(?:=|$)",
    re.IGNORECASE,
)
_ENV_ASSIGNMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_OUTPUT_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def require_slurm_allocation(env: Mapping[str, str] | None = None) -> dict[str, str | int]:
    values = dict(os.environ if env is None else env)
    job_id = values.get("SLURM_JOB_ID", "").strip()
    nodelist = values.get("SLURM_JOB_NODELIST", "").strip()
    node_count = values.get("SLURM_JOB_NUM_NODES", "").strip()
    context = (values.get("SLURM_STEP_ID") or values.get("SLURM_JOB_NAME") or "").strip()
    missing: list[str] = []
    if not job_id or not job_id.isdigit():
        missing.append("numeric SLURM_JOB_ID")
    if not nodelist or nodelist.lower() in {"none", "(null)", "unknown"}:
        missing.append("SLURM_JOB_NODELIST")
    if not node_count.isdigit() or int(node_count) < 1:
        missing.append("positive SLURM_JOB_NUM_NODES")
    if not context:
        missing.append("SLURM_STEP_ID or SLURM_JOB_NAME")
    if missing:
        raise ConfigError(
            f"{E_SLURM_ALLOCATION_REQUIRED}: heavy Cluster actions require an active "
            f"Slurm compute allocation; missing/invalid {', '.join(missing)}."
        )
    return {
        "job_id": job_id,
        "nodelist": nodelist,
        "node_count": int(node_count),
        "partition": values.get("SLURM_JOB_PARTITION", ""),
        "step_id": values.get("SLURM_STEP_ID", ""),
        "job_name": values.get("SLURM_JOB_NAME", ""),
        "job_gpus": values.get("SLURM_JOB_GPUS", ""),
        "gpus_on_node": values.get("SLURM_GPUS_ON_NODE", ""),
    }


def allocated_gpu_selector(env: Mapping[str, str] | None = None) -> str | None:
    values = dict(os.environ if env is None else env)
    value = (values.get("SLURM_JOB_GPUS") or values.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not value or value.lower() in {"none", "nodevfiles", "(null)"}:
        return None
    if _CONTROL.search(value) or any(ch.isspace() for ch in value):
        raise ConfigError("Invalid Slurm GPU selector")
    return value


def secret_field_name(name: str) -> bool:
    normalized = name.strip().lower().replace("-", "_")
    return normalized in _SENSITIVE_NAMES or normalized.endswith("_password") or normalized.endswith("_token")


def reject_secret_fields(value: object, *, path: str = "config") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            if secret_field_name(key_text):
                raise ConfigError(f"Secret-bearing field is forbidden in Cluster package config: {path}.{key_text}")
            reject_secret_fields(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_secret_fields(child, path=f"{path}[{index}]")


def safe_absolute_path(value: str | Path, field: str, *, slurm_directive: bool = False) -> Path:
    text = str(value)
    if not text or _CONTROL.search(text):
        raise ConfigError(f"{E_UNSAFE_CLUSTER_PATH}: {field} is empty or contains a control character")
    if slurm_directive and any(ch.isspace() for ch in text):
        raise ConfigError(f"{E_UNSAFE_CLUSTER_PATH}: {field} cannot contain whitespace in an #SBATCH directive")
    parsed = urlsplit(text)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise ConfigError(f"{E_UNSAFE_CLUSTER_PATH}: {field} must be an absolute filesystem path without URI data")
    path = PurePosixPath(text)
    if not path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"{E_UNSAFE_CLUSTER_PATH}: {field} must be absolute and traversal-free")
    if any(part.lower() in _SENSITIVE_PATH_PARTS for part in path.parts):
        raise ConfigError(f"{E_UNSAFE_CLUSTER_PATH}: {field} points through a credential/private directory")
    return Path(path)


def validate_secret_free_command(command: Sequence[str]) -> list[str]:
    if not command:
        raise ConfigError("Guarded Cluster command cannot be empty")
    safe: list[str] = []
    previous_secret_flag = False
    for raw in command:
        arg = str(raw)
        if not arg or _CONTROL.search(arg):
            raise ConfigError(f"{E_SECRET_BEARING_COMMAND}: empty/control-bearing command argument")
        if previous_secret_flag:
            raise ConfigError(f"{E_SECRET_BEARING_COMMAND}: secret values cannot be passed on the command line")
        match = _ENV_ASSIGNMENT.match(arg)
        if match and secret_field_name(match.group(1)):
            raise ConfigError(f"{E_SECRET_BEARING_COMMAND}: secret environment assignment is forbidden")
        if _SECRET_FLAG.match(arg):
            if "=" in arg:
                raise ConfigError(f"{E_SECRET_BEARING_COMMAND}: secret flag value is forbidden")
            previous_secret_flag = True
        parsed = urlsplit(arg)
        if parsed.scheme and (parsed.username is not None or parsed.password is not None):
            raise ConfigError(f"{E_SECRET_BEARING_COMMAND}: credential-bearing URL is forbidden")
        if "sk-" in arg.lower():
            raise ConfigError(f"{E_SECRET_BEARING_COMMAND}: token-like command content is forbidden")
        safe.append(arg)
    if previous_secret_flag:
        raise ConfigError(f"{E_SECRET_BEARING_COMMAND}: secret flag is forbidden")
    return safe


def bounded_text(value: str, *, limit: int = 32768) -> str:
    text = _OUTPUT_CONTROL.sub("?", str(value))
    if len(text) > limit:
        return text[:limit] + "\n[truncated]"
    return text
