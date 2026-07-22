from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from putpocket_dataset_mining.constants import DATASETS_ROOT
from putpocket_dataset_mining.dataset import SourceTask
from putpocket_dataset_mining.errors import ConfigError


SOURCE_ARTIFACT_COMPLETENESS_PATHS = [
    "source_task.json",
    "episode_summary.json",
    "prepared/messages_history1.json",
    "prepared/messages_history2.json",
    "prepared/query1.txt",
    "prepared/query2.txt",
    "prepared/query2_metadata.json",
    "prepared/cline_rules_v1.md",
    "prepared/cline_rules_v2.md",
    "trajectories/history1_trajectory.jsonl",
    "trajectories/history2_trajectory.jsonl",
    "workspace_snapshots/initial",
    "workspace_snapshots/after_history1",
    "workspace_snapshots/after_history2",
    "verification/history1/checklist.json",
    "verification/history2/checklist.json",
    "judge/judge_decision.json",
]

REQUIRED_ROW_FIELDS = [
    "sample_id",
    "task_id",
    "split",
    "row_index",
    "attempt_id",
    "final_status",
    "artifact_path",
    "query1",
    "query2",
    "policy_delta",
]


@dataclass(frozen=True)
class AcceptedDatasetSample:
    dataset_version: str
    dataset_root: Path
    accepted_path: Path
    row_number: int
    row: dict[str, Any]
    source_artifact_path: Path
    source_task: SourceTask
    missing_artifacts: list[str]

    @property
    def sample_id(self) -> str:
        return str(self.row["sample_id"])

    @property
    def task_id(self) -> str:
        return str(self.row["task_id"])


def load_accepted_samples(dataset_version: str, datasets_root: Path = DATASETS_ROOT) -> list[AcceptedDatasetSample]:
    dataset_root = datasets_root / dataset_version
    accepted_path = dataset_root / "accepted.jsonl"
    if not accepted_path.exists():
        raise ConfigError(f"accepted.jsonl does not exist: {accepted_path}")

    samples: list[AcceptedDatasetSample] = []
    with accepted_path.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing_fields = [field for field in REQUIRED_ROW_FIELDS if field not in row]
            if missing_fields:
                raise ConfigError(f"accepted.jsonl row {row_number} is missing fields: {missing_fields}")
            source_artifact_path = Path(str(row["artifact_path"]))
            source_task_path = source_artifact_path / "source_task.json"
            if not source_task_path.exists():
                raise ConfigError(f"accepted row {row_number} is missing semantic source task: {source_task_path}")
            source_task = SourceTask(**json.loads(source_task_path.read_text(encoding="utf-8")))
            missing_artifacts = [
                rel_path
                for rel_path in SOURCE_ARTIFACT_COMPLETENESS_PATHS
                if not (source_artifact_path / rel_path).exists()
            ]
            samples.append(
                AcceptedDatasetSample(
                    dataset_version=dataset_version,
                    dataset_root=dataset_root,
                    accepted_path=accepted_path,
                    row_number=row_number,
                    row=row,
                    source_artifact_path=source_artifact_path,
                    source_task=source_task,
                    missing_artifacts=missing_artifacts,
                )
            )
    return samples
