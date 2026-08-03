from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .constants import DATASETS_ROOT, FINAL_STATUSES, INDEX_DB, ensure_data_dirs
from .errors import ConfigError
from .finalized_dataset import find_lock_for_dataset_version, validate_finalized_dataset
from .jsonl import write_jsonl


@dataclass(frozen=True)
class AttemptRecord:
    run_id: str
    dataset_version: str
    sample_id: str
    split: str
    row_index: int
    task_id: str
    attempt_id: str
    final_status: str
    failure_class: str | None
    artifact_path: str
    summary: dict[str, Any]


class MiningIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ensure_schema()

    @classmethod
    def default(cls) -> "MiningIndex":
        ensure_data_dirs()
        return cls(INDEX_DB)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    dataset_version TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    split TEXT NOT NULL,
                    row_index INTEGER NOT NULL,
                    task_id TEXT NOT NULL,
                    attempt_id TEXT NOT NULL,
                    final_status TEXT NOT NULL,
                    failure_class TEXT,
                    artifact_path TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(run_id, sample_id, attempt_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_sample_status ON attempts(sample_id, final_status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_dataset_status ON attempts(dataset_version, final_status)")

    def record_attempt(self, record: AttemptRecord) -> None:
        if record.final_status not in FINAL_STATUSES:
            raise ValueError(f"Invalid final_status: {record.final_status}")
        now = time.time()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO attempts (
                    run_id, dataset_version, sample_id, split, row_index, task_id, attempt_id,
                    final_status, failure_class, artifact_path, summary_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, sample_id, attempt_id) DO UPDATE SET
                    final_status=excluded.final_status,
                    failure_class=excluded.failure_class,
                    artifact_path=excluded.artifact_path,
                    summary_json=excluded.summary_json,
                    updated_at=excluded.updated_at
                """,
                (
                    record.run_id,
                    record.dataset_version,
                    record.sample_id,
                    record.split,
                    record.row_index,
                    record.task_id,
                    record.attempt_id,
                    record.final_status,
                    record.failure_class,
                    record.artifact_path,
                    json.dumps(record.summary, sort_keys=True),
                    now,
                    now,
                ),
            )

    def has_prior_status(self, sample_id: str, statuses: set[str]) -> bool:
        placeholders = ",".join("?" for _ in statuses)
        with self.connect() as conn:
            row = conn.execute(
                f"SELECT 1 FROM attempts WHERE sample_id=? AND final_status IN ({placeholders}) LIMIT 1",
                (sample_id, *sorted(statuses)),
            ).fetchone()
        return row is not None

    def count_status(self, dataset_version: str, status: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM attempts WHERE dataset_version=? AND final_status=?",
                (dataset_version, status),
            ).fetchone()
        return int(row["n"])

    def rows_for_dataset(self, dataset_version: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM attempts WHERE dataset_version=? ORDER BY id",
                (dataset_version,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["summary"] = json.loads(item.pop("summary_json"))
            result.append(item)
        return result


class DatasetMaterializer:
    def __init__(self, index: MiningIndex, datasets_root: Path = DATASETS_ROOT) -> None:
        self.index = index
        self.datasets_root = datasets_root

    def materialize_dataset(self, dataset_version: str) -> Path:
        finalized_lock = find_lock_for_dataset_version(dataset_version)
        if finalized_lock is not None:
            validate_finalized_dataset(finalized_lock)
            raise ConfigError(
                f"`{dataset_version}` is immutable and already finalized at "
                f"{finalized_lock.final_accepted_count} accepted samples; materialization would rewrite the locked dataset."
            )
        dataset_root = self.datasets_root / dataset_version
        dataset_root.mkdir(parents=True, exist_ok=True)
        rows = self.index.rows_for_dataset(dataset_version)

        accepted = [self._dataset_row(row) for row in rows if row["final_status"] == "accepted"]
        rejected = [self._dataset_row(row) for row in rows if row["final_status"] == "rejected"]
        uncertain = [self._dataset_row(row) for row in rows if row["final_status"] == "uncertain"]
        artifact_index = [self._artifact_row(row) for row in rows]

        write_jsonl(dataset_root / "accepted.jsonl", accepted)
        write_jsonl(dataset_root / "rejected.jsonl", rejected)
        write_jsonl(dataset_root / "uncertain.jsonl", uncertain)
        write_jsonl(dataset_root / "artifact_index.jsonl", artifact_index)

        manifest = {
            "dataset_version": dataset_version,
            "mode": "local_materialized_view",
            "package_for_other_repo": False,
            "counts": {
                "accepted": len(accepted),
                "rejected": len(rejected),
                "uncertain": len(uncertain),
                "failed_infra": len([row for row in rows if row["final_status"] == "failed_infra"]),
                "total_attempts": len(rows),
            },
            "files": [
                "accepted.jsonl",
                "rejected.jsonl",
                "uncertain.jsonl",
                "artifact_index.jsonl",
            ],
        }
        (dataset_root / "dataset_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
        return dataset_root

    def _dataset_row(self, row: dict[str, Any]) -> dict[str, Any]:
        summary = row["summary"]
        return {
            "sample_id": row["sample_id"],
            "task_id": row["task_id"],
            "split": row["split"],
            "row_index": row["row_index"],
            "attempt_id": row["attempt_id"],
            "final_status": row["final_status"],
            "failure_class": row["failure_class"],
            "artifact_path": row["artifact_path"],
            "query1": summary.get("query1"),
            "query2": summary.get("query2"),
            "policy_delta": summary.get("policy_delta"),
        }

    def _artifact_row(self, row: dict[str, Any]) -> dict[str, Any]:
        return {
            "sample_id": row["sample_id"],
            "attempt_id": row["attempt_id"],
            "final_status": row["final_status"],
            "artifact_path": row["artifact_path"],
            "episode_summary": str(Path(row["artifact_path"]) / "episode_summary.json"),
        }
