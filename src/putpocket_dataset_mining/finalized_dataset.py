from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_yaml
from .constants import REPO_ROOT
from .errors import ConfigError


@dataclass(frozen=True)
class FinalizedDatasetLock:
    path: Path
    dataset_version: str
    accepted_file: Path
    accepted_sha256: str
    final_accepted_count: int
    canonical_source_task_ids: list[str]
    allow_mining: bool


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else REPO_ROOT / path


def load_finalized_lock(path: str | Path) -> FinalizedDatasetLock:
    lock_path = resolve_repo_path(path)
    if not lock_path.exists():
        raise ConfigError(f"Finalized dataset lock file is missing: {lock_path}")
    data = load_yaml(lock_path)
    if not data.get("finalized", False):
        raise ConfigError(f"Finalized dataset lock must declare finalized: true: {lock_path}")
    ids = data.get("canonical_source_task_ids")
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ConfigError(f"Finalized dataset lock has invalid canonical_source_task_ids: {lock_path}")
    accepted_file = data.get("accepted_file")
    if not isinstance(accepted_file, str) or not accepted_file:
        raise ConfigError(f"Finalized dataset lock has invalid accepted_file: {lock_path}")
    dataset_version = data.get("dataset_version")
    if not isinstance(dataset_version, str) or not dataset_version:
        raise ConfigError(f"Finalized dataset lock has invalid dataset_version: {lock_path}")
    accepted_sha256 = data.get("accepted_sha256")
    if not isinstance(accepted_sha256, str) or not accepted_sha256:
        raise ConfigError(f"Finalized dataset lock has invalid accepted_sha256: {lock_path}")
    final_count = int(data.get("final_accepted_count", -1))
    return FinalizedDatasetLock(
        path=lock_path,
        dataset_version=dataset_version,
        accepted_file=resolve_repo_path(accepted_file),
        accepted_sha256=accepted_sha256,
        final_accepted_count=final_count,
        canonical_source_task_ids=ids,
        allow_mining=bool(data.get("allow_mining", True)),
    )


def find_lock_for_dataset_version(dataset_version: str) -> FinalizedDatasetLock | None:
    config_dir = REPO_ROOT / "configs" / "dataset_mining"
    if not config_dir.exists():
        return None
    for path in sorted(config_dir.glob("*.lock.yaml")):
        data = load_yaml(path)
        if data.get("dataset_version") == dataset_version and data.get("finalized", False):
            return load_finalized_lock(path)
    return None


def validate_finalized_dataset(lock: FinalizedDatasetLock) -> dict[str, Any]:
    if not lock.accepted_file.exists():
        raise ConfigError(f"Finalized accepted file is missing: {lock.accepted_file}")
    raw = lock.accepted_file.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"Invalid JSONL in finalized accepted file at line {line_no}: {exc}") from exc
        if not isinstance(item, dict):
            raise ConfigError(f"Finalized accepted file line {line_no} is not a JSON object.")
        rows.append(item)

    ids = [str(row.get("source_task_id") or row.get("sample_id") or "") for row in rows]
    if len(rows) != lock.final_accepted_count:
        raise ConfigError(
            f"Finalized dataset integrity error for {lock.dataset_version}: expected "
            f"{lock.final_accepted_count} accepted rows, found {len(rows)}."
        )
    if len(set(ids)) != len(ids):
        raise ConfigError(f"Finalized dataset integrity error for {lock.dataset_version}: duplicate sample IDs found.")
    if ids != lock.canonical_source_task_ids:
        raise ConfigError(
            f"Finalized dataset integrity error for {lock.dataset_version}: accepted ID order/set does not match lock."
        )
    bad_statuses = sorted({str(row.get("final_status")) for row in rows if row.get("final_status") != "accepted"})
    if bad_statuses:
        raise ConfigError(
            f"Finalized dataset integrity error for {lock.dataset_version}: non-accepted rows found: {bad_statuses}."
        )
    if actual_sha256 != lock.accepted_sha256:
        raise ConfigError(
            f"Finalized dataset integrity error for {lock.dataset_version}: SHA-256 mismatch; "
            f"expected {lock.accepted_sha256}, found {actual_sha256}."
        )
    return {
        "dataset_version": lock.dataset_version,
        "accepted_count": len(rows),
        "accepted_sha256": actual_sha256,
        "canonical_source_task_ids": ids,
        "accepted_file": str(lock.accepted_file),
        "lock_file": str(lock.path),
    }
