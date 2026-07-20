from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import MINING_SEED_DEFAULT, SHARED_HF_HUB_CACHE_DIR
from .errors import DependencyError, InfraError


@dataclass(frozen=True)
class SourceTask:
    adapter: str
    dataset_id: str
    split: str
    row_index: int
    task_id: str
    prompt: str
    reference_solution: str
    tests: list[str]
    test_setup: str
    raw: dict[str, Any]

    @property
    def sample_id(self) -> str:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(self.task_id))
        return f"{self.split}_{safe}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


class MBPPDatasetAdapter:
    def __init__(
        self,
        preferred_dataset_id: str,
        fallback_dataset_id: str,
        split_order: list[str],
        field_mapping: dict[str, str],
        cache_dir: Path = SHARED_HF_HUB_CACHE_DIR,
    ) -> None:
        self.preferred_dataset_id = preferred_dataset_id
        self.fallback_dataset_id = fallback_dataset_id
        self.split_order = split_order
        self.field_mapping = field_mapping
        self.cache_dir = cache_dir
        self._dataset: Any | None = None
        self._dataset_id: str | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MBPPDatasetAdapter":
        dataset = config.get("dataset", {})
        return cls(
            preferred_dataset_id=dataset.get("preferred_dataset_id", "google-research-datasets/mbpp"),
            fallback_dataset_id=dataset.get("fallback_dataset_id", "Muennighoff/mbpp"),
            split_order=list(dataset.get("split_order", ["train", "test"])),
            field_mapping=dict(
                dataset.get(
                    "field_mapping",
                    {
                        "task_id": "task_id",
                        "prompt": "text",
                        "reference_solution": "code",
                        "tests": "test_list",
                        "test_setup": "test_setup_code",
                    },
                )
            ),
        )

    def load(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise DependencyError("The datasets package is required for MBPP loading.") from exc

        errors: list[str] = []
        for dataset_id in (self.preferred_dataset_id, self.fallback_dataset_id):
            try:
                self._dataset = load_dataset(dataset_id, cache_dir=str(self.cache_dir))
                self._dataset_id = dataset_id
                return
            except Exception as exc:  # noqa: BLE001 - keep fallback reason for artifact/debugging.
                errors.append(f"{dataset_id}: {exc}")
        raise InfraError("Unable to load MBPP dataset. Attempts: " + " | ".join(errors))

    @property
    def dataset(self) -> Any:
        if self._dataset is None:
            self.load()
        return self._dataset

    @property
    def dataset_id(self) -> str:
        if self._dataset_id is None:
            self.load()
        assert self._dataset_id is not None
        return self._dataset_id

    def iter_indices(self, mining_seed: int = MINING_SEED_DEFAULT) -> list[tuple[str, int]]:
        rows: list[tuple[str, int]] = []
        ds = self.dataset
        for split in self.split_order:
            if split not in ds:
                continue
            rows.extend((split, idx) for idx in range(len(ds[split])))
        random.Random(mining_seed).shuffle(rows)
        return rows

    def get_by_flat_index(self, sample_index: int, split: str | None = None) -> SourceTask:
        if split is not None:
            ds = self.dataset
            if split not in ds:
                raise IndexError(f"Dataset split does not exist: {split}")
            row = ds[split][sample_index]
            return self._normalize(row, split=split, row_index=sample_index)

        offset = sample_index
        ds = self.dataset
        for split_name in self.split_order:
            if split_name not in ds:
                continue
            split_len = len(ds[split_name])
            if offset < split_len:
                return self._normalize(ds[split_name][offset], split=split_name, row_index=offset)
            offset -= split_len
        raise IndexError(f"Sample index out of range: {sample_index}")

    def get_by_split_index(self, split: str, row_index: int) -> SourceTask:
        ds = self.dataset
        if split not in ds:
            raise IndexError(f"Dataset split does not exist: {split}")
        return self._normalize(ds[split][row_index], split=split, row_index=row_index)

    def _normalize(self, row: dict[str, Any], split: str, row_index: int) -> SourceTask:
        def mapped(name: str, default: Any = "") -> Any:
            return row.get(self.field_mapping.get(name, name), default)

        tests = mapped("tests", [])
        if isinstance(tests, str):
            tests = [line for line in tests.splitlines() if line.strip()]
        if not isinstance(tests, list):
            tests = list(tests)

        return SourceTask(
            adapter="mbpp_huggingface",
            dataset_id=self.dataset_id,
            split=split,
            row_index=row_index,
            task_id=str(mapped("task_id", row_index)),
            prompt=str(mapped("prompt", "")),
            reference_solution=str(mapped("reference_solution", "")),
            tests=[str(test) for test in tests],
            test_setup=str(mapped("test_setup", "") or ""),
            raw=dict(row),
        )


class MBPPTestListToPytest:
    def render(self, task: SourceTask) -> str:
        body: list[str] = [
            "import pytest",
            "from solution import *",
            "",
        ]
        if task.test_setup.strip():
            body.append(task.test_setup.rstrip())
            body.append("")
        body.append("def test_mbpp_hidden_contract():")
        if not task.tests:
            body.append("    pytest.fail('MBPP task has no hidden tests')")
        for test in task.tests:
            stripped = test.strip()
            if not stripped:
                continue
            for line in stripped.splitlines():
                body.append(f"    {line}")
        body.append("")
        return "\n".join(body)

    def write(self, task: SourceTask, workspace: Path, injection_path: str = "tests/test_solution.py") -> Path:
        target = workspace / injection_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(task), encoding="utf-8")
        return target


def task_refs_for_jobs(adapter: MBPPDatasetAdapter, mining_seed: int) -> Iterable[tuple[str, int]]:
    return adapter.iter_indices(mining_seed=mining_seed)
