from __future__ import annotations

import ast
import json
import random
import sys
import sysconfig
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import MINING_SEED_DEFAULT, SHARED_HF_HUB_CACHE_DIR
from .errors import ConfigError, DependencyError, InfraError


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
    class_name: str | None = None
    class_description: str | None = None
    class_constructor: str | None = None
    fields: list[Any] | None = None
    methods_info: list[dict[str, Any]] | None = None
    import_statement: str | list[str] | None = None
    skeleton: str | None = None
    verifier_test_code: str | None = None
    method_test_codes: list[str] | None = None
    dependency_metadata: dict[str, Any] | None = None

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


class ClassEvalDatasetAdapter:
    def __init__(
        self,
        dataset_id: str = "FudanSELab/ClassEval",
        split_order: list[str] | None = None,
        cache_dir: Path = SHARED_HF_HUB_CACHE_DIR,
    ) -> None:
        self.hf_dataset_id = dataset_id
        self.split_order = split_order or ["test"]
        self.cache_dir = cache_dir
        self._dataset: Any | None = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ClassEvalDatasetAdapter":
        dataset = config.get("dataset", {})
        return cls(
            dataset_id=dataset.get("hf_dataset_id", dataset.get("preferred_dataset_id", "FudanSELab/ClassEval")),
            split_order=list(dataset.get("split_order", [dataset.get("split", "test")])),
        )

    def load(self) -> None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise DependencyError("The datasets package is required for ClassEval loading.") from exc
        try:
            self._dataset = load_dataset(self.hf_dataset_id, cache_dir=str(self.cache_dir))
        except Exception as exc:  # noqa: BLE001 - include HF failure details in artifacts.
            raise InfraError(f"Unable to load ClassEval dataset {self.hf_dataset_id}: {exc}") from exc

    @property
    def dataset(self) -> Any:
        if self._dataset is None:
            self.load()
        return self._dataset

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
            return self.get_by_split_index(split, sample_index)
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

    @staticmethod
    def _normalize_imports(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "\n".join(str(item).strip() for item in value if str(item).strip())
        return str(value).strip()

    def _normalize(self, row: dict[str, Any], split: str, row_index: int) -> SourceTask:
        raw = dict(row)
        methods_info = [dict(item) for item in (row.get("methods_info") or [])]
        method_test_codes = [
            str(item.get("test_code", "")).strip()
            for item in methods_info
            if str(item.get("test_code", "")).strip()
        ]
        import_statement = self._normalize_imports(row.get("import_statement"))
        skeleton = str(row.get("skeleton", "") or "")
        class_name = str(row.get("class_name", "") or "")
        dependency_metadata = {
            "import_statement": row.get("import_statement"),
            "method_dependencies": [item.get("dependencies") for item in methods_info if item.get("dependencies")],
        }
        return SourceTask(
            adapter="classeval_huggingface",
            dataset_id=self.hf_dataset_id,
            split=split,
            row_index=row_index,
            task_id=str(row.get("task_id", row_index)),
            prompt=str(row.get("class_description", "") or skeleton),
            reference_solution=str(row.get("solution_code", "") or ""),
            tests=[str(row.get("test", "") or ""), *method_test_codes],
            test_setup="",
            raw=raw,
            class_name=class_name or None,
            class_description=str(row.get("class_description", "") or "") or None,
            class_constructor=str(row.get("class_constructor", "") or "") or None,
            fields=list(row.get("fields") or []),
            methods_info=methods_info,
            import_statement=import_statement,
            skeleton=skeleton,
            verifier_test_code=str(row.get("test", "") or ""),
            method_test_codes=method_test_codes,
            dependency_metadata=dependency_metadata,
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


class ClassEvalTestsToPytest:
    def render(self, task: SourceTask) -> str:
        blocks: list[str] = []
        for code in [task.verifier_test_code or "", *(task.method_test_codes or [])]:
            stripped = code.strip()
            if stripped and stripped not in blocks:
                blocks.append(stripped)
        if not blocks:
            raise InfraError(f"ClassEval task has no verifier tests: {task.sample_id}")

        body: list[str] = [
            "import unittest",
            "import pytest",
            "from solution import *",
            "",
        ]
        for block in blocks:
            for line in block.splitlines():
                if line.strip() in {"import unittest", "from solution import *"}:
                    continue
                body.append(line)
            body.append("")
        body.extend(
            [
                "",
                "if __name__ == '__main__':",
                "    unittest.main()",
                "",
            ]
        )
        rendered = "\n".join(body)
        try:
            ast.parse(rendered)
        except SyntaxError as exc:
            raise InfraError(f"ClassEval test materialization failed for {task.sample_id}: {exc}") from exc
        return rendered

    def write(self, task: SourceTask, workspace: Path, injection_path: str = "tests/test_solution.py") -> Path:
        target = workspace / injection_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(task), encoding="utf-8")
        return target


def dataset_adapter_from_config(config: dict[str, Any]) -> MBPPDatasetAdapter | ClassEvalDatasetAdapter:
    adapter_name = str(config.get("dataset", {}).get("adapter", "mbpp_huggingface"))
    if adapter_name == "mbpp_huggingface":
        return MBPPDatasetAdapter.from_config(config)
    if adapter_name == "classeval_huggingface":
        return ClassEvalDatasetAdapter.from_config(config)
    raise ConfigError(f"Unknown dataset adapter: {adapter_name}")


def initial_workspace_files_for_task(task: SourceTask, config: dict[str, Any]) -> dict[str, str]:
    if task.adapter == "classeval_huggingface":
        target_file = str(config.get("workspace", {}).get("target_file", "solution.py"))
        imports = ClassEvalDatasetAdapter._normalize_imports(task.import_statement)
        skeleton = task.skeleton or ""
        content = f"{imports}\n\n{skeleton}" if imports and imports not in skeleton else skeleton
        return {target_file: content.rstrip() + "\n"}
    return {
        str(path): str(content)
        for path, content in config.get("workspace", {})
        .get("initial_files", {"solution.py": "# TODO: implement the required function.\n"})
        .items()
    }


def verifier_materializer_for_task(task: SourceTask) -> MBPPTestListToPytest | ClassEvalTestsToPytest:
    if task.adapter == "classeval_huggingface":
        return ClassEvalTestsToPytest()
    return MBPPTestListToPytest()


def task_refs_for_jobs(adapter: MBPPDatasetAdapter | ClassEvalDatasetAdapter, mining_seed: int) -> Iterable[tuple[str, int]]:
    return adapter.iter_indices(mining_seed=mining_seed)


def _stdlib_names() -> set[str]:
    names = set(getattr(sys, "stdlib_module_names", set()))
    stdlib = sysconfig.get_paths().get("stdlib")
    if stdlib:
        root = Path(stdlib)
        for child in root.iterdir():
            if child.name.startswith("_"):
                continue
            if child.is_dir() and (child / "__init__.py").exists():
                names.add(child.name)
            elif child.suffix == ".py":
                names.add(child.stem)
    return names


def imported_modules_from_code(code: str) -> set[str]:
    if not code.strip():
        return set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def scan_classeval_dependencies(adapter: ClassEvalDatasetAdapter) -> dict[str, Any]:
    stdlib = _stdlib_names()
    all_imports: set[str] = set()
    unresolved_code_blocks = 0
    for split in adapter.split_order:
        ds = adapter.dataset
        if split not in ds:
            continue
        for row in ds[split]:
            methods_info = row.get("methods_info") or []
            code_blocks: list[str] = [
                ClassEvalDatasetAdapter._normalize_imports(row.get("import_statement")),
                str(row.get("skeleton", "") or ""),
                str(row.get("test", "") or ""),
            ]
            code_blocks.extend(str(item.get("test_code", "") or "") for item in methods_info)
            for item in methods_info:
                deps = item.get("dependencies")
                if isinstance(deps, str):
                    code_blocks.append(deps)
                elif isinstance(deps, list):
                    code_blocks.extend(str(dep) for dep in deps)
            for block in code_blocks:
                before = len(all_imports)
                all_imports.update(imported_modules_from_code(block))
                if block.strip() and len(all_imports) == before:
                    try:
                        ast.parse(block)
                    except SyntaxError:
                        unresolved_code_blocks += 1
    local = {"solution"}
    third_party = sorted(name for name in all_imports if name not in stdlib and name not in local)
    return {
        "dataset_id": adapter.hf_dataset_id,
        "splits": adapter.split_order,
        "imports": sorted(all_imports),
        "standard_library": sorted(name for name in all_imports if name in stdlib),
        "third_party": third_party,
        "local_or_dataset": sorted(name for name in all_imports if name in local),
        "unresolved": [],
        "unparsed_code_blocks": unresolved_code_blocks,
    }
