from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from putpocket_dataset_mining.config import load_yaml
from putpocket_dataset_mining.dataset import (
    ClassEvalDatasetAdapter,
    ClassEvalTestsToPytest,
    imported_modules_from_code,
    initial_workspace_files_for_task,
)
from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.finalized_dataset import load_finalized_lock, validate_finalized_dataset
from putpocket_dataset_mining.multi import MultiSampleMaster
from putpocket_dataset_mining.prompts import QueryBuilder
from putpocket_dataset_mining.storage import DatasetMaterializer, MiningIndex


def synthetic_classeval_row() -> dict:
    return {
        "task_id": "ClassEval_X",
        "class_name": "Counter",
        "class_description": '"""A small counter."""',
        "class_constructor": "class Counter:\n    def __init__(self):\n        pass\n",
        "fields": [],
        "import_statement": ["import math"],
        "skeleton": "class Counter:\n    def __init__(self):\n        pass\n\n    def inc(self):\n        pass\n",
        "solution_code": "class Counter:\n    def __init__(self):\n        self.n = 0\n    def inc(self):\n        self.n += 1\n        return self.n\n",
        "test": "import unittest\n\nclass CounterTest(unittest.TestCase):\n    def test_inc(self):\n        c = Counter()\n        self.assertEqual(c.inc(), 1)\n",
        "methods_info": [
            {
                "method_name": "inc",
                "test_code": "class CounterMethodTest(unittest.TestCase):\n    def test_inc_again(self):\n        c = Counter()\n        c.inc()\n        self.assertEqual(c.inc(), 2)\n",
                "solution_code": "hidden",
                "dependencies": ["import decimal"],
            }
        ],
        "test_classes": ["CounterTest"],
    }


class ClassEvalSupportTests(unittest.TestCase):
    def test_row_normalization_preserves_hidden_fields(self) -> None:
        adapter = ClassEvalDatasetAdapter()
        task = adapter._normalize(synthetic_classeval_row(), split="test", row_index=0)

        self.assertEqual(task.adapter, "classeval_huggingface")
        self.assertEqual(task.sample_id, "test_ClassEval_X")
        self.assertEqual(task.class_name, "Counter")
        self.assertIn("solution_code", task.raw)
        self.assertEqual(len(task.method_test_codes or []), 1)

    def test_workspace_materialization_uses_imports_and_skeleton_only(self) -> None:
        adapter = ClassEvalDatasetAdapter()
        task = adapter._normalize(synthetic_classeval_row(), split="test", row_index=0)

        files = initial_workspace_files_for_task(task, {"workspace": {"target_file": "solution.py"}})

        self.assertEqual(list(files), ["solution.py"])
        self.assertIn("import math", files["solution.py"])
        self.assertIn("class Counter", files["solution.py"])
        self.assertNotIn("self.n += 1", files["solution.py"])
        self.assertNotIn("CounterTest", files["solution.py"])

    def test_verifier_materialization_combines_tests(self) -> None:
        adapter = ClassEvalDatasetAdapter()
        task = adapter._normalize(synthetic_classeval_row(), split="test", row_index=0)

        with tempfile.TemporaryDirectory() as tmp:
            target = ClassEvalTestsToPytest().write(task, Path(tmp))
            rendered = target.read_text(encoding="utf-8")

        self.assertIn("from solution import *", rendered)
        self.assertIn("CounterTest", rendered)
        self.assertIn("CounterMethodTest", rendered)

    def test_query1_does_not_leak_tests_or_solution(self) -> None:
        adapter = ClassEvalDatasetAdapter()
        task = adapter._normalize(synthetic_classeval_row(), split="test", row_index=0)

        query = QueryBuilder().build_query1(task)

        self.assertIn("Complete the Python class implementation", query)
        self.assertNotIn("CounterTest", query)
        self.assertNotIn("self.n += 1", query)

    def test_dependency_scanner_extracts_imports(self) -> None:
        imports = imported_modules_from_code("import pandas as pd\nfrom bs4 import BeautifulSoup\n")

        self.assertEqual(imports, {"pandas", "bs4"})

    def test_classeval_configs_load(self) -> None:
        single = load_yaml("configs/dataset_mining/classeval_stateful_single.yaml")
        multi = load_yaml("configs/dataset_mining/classeval_stateful_multi.yaml")

        self.assertEqual(single["dataset"]["adapter"], "classeval_huggingface")
        self.assertEqual(multi["profiles"]["full_server"]["target_accepted"], 18)
        self.assertTrue(multi["profiles"]["full_server"]["finalized"])
        self.assertEqual(multi["worker"]["single_config"], "configs/dataset_mining/classeval_stateful_single.yaml")

    def test_lock_manifest_declares_exactly_18_unique_samples(self) -> None:
        lock = load_finalized_lock("configs/dataset_mining/classeval_stateful_working_v0.lock.yaml")

        self.assertEqual(lock.dataset_version, "classeval_stateful_working_v0")
        self.assertEqual(lock.final_accepted_count, 18)
        self.assertFalse(lock.allow_mining)
        self.assertEqual(len(lock.canonical_source_task_ids), 18)
        self.assertEqual(len(set(lock.canonical_source_task_ids)), 18)

    def test_real_working_accepted_file_matches_lock(self) -> None:
        lock = load_finalized_lock("configs/dataset_mining/classeval_stateful_working_v0.lock.yaml")
        if not lock.accepted_file.exists():
            self.skipTest("Ignored canonical accepted dataset is not present in this clean checkout.")
        status = validate_finalized_dataset(lock)

        self.assertEqual(status["accepted_count"], 18)
        self.assertEqual(status["accepted_sha256"], lock.accepted_sha256)

    def test_finalized_exact_match_schedules_zero_attempts_before_model_init(self) -> None:
        if not load_finalized_lock("configs/dataset_mining/classeval_stateful_working_v0.lock.yaml").accepted_file.exists():
            self.skipTest("Ignored canonical accepted dataset is not present in this clean checkout.")
        config = load_yaml("configs/dataset_mining/classeval_stateful_multi.yaml")
        result = MultiSampleMaster(config).run("full_server", run_id="unit_finalized_noop")

        self.assertTrue(result["finalized"])
        self.assertEqual(result["attempts_assigned"], 0)
        self.assertEqual(result["attempts_finished"], 0)
        self.assertEqual(result["canonical_accepted_count"], 18)

    def test_finalized_rerun_failed_infra_is_blocked(self) -> None:
        if not load_finalized_lock("configs/dataset_mining/classeval_stateful_working_v0.lock.yaml").accepted_file.exists():
            self.skipTest("Ignored canonical accepted dataset is not present in this clean checkout.")
        config = load_yaml("configs/dataset_mining/classeval_stateful_multi.yaml")

        with self.assertRaisesRegex(ConfigError, "immutable"):
            MultiSampleMaster(config).run("full_server", run_id="unit_finalized_retry", rerun_failed_infra=True)

    def test_finalized_append_mode_is_blocked(self) -> None:
        if not load_finalized_lock("configs/dataset_mining/classeval_stateful_working_v0.lock.yaml").accepted_file.exists():
            self.skipTest("Ignored canonical accepted dataset is not present in this clean checkout.")
        config = load_yaml("configs/dataset_mining/classeval_stateful_multi.yaml")
        config["profiles"]["full_server"]["allow_new_attempts"] = True

        with self.assertRaisesRegex(ConfigError, "immutable"):
            MultiSampleMaster(config).run("full_server", run_id="unit_finalized_append")

    def test_finalized_materialization_is_blocked(self) -> None:
        if not load_finalized_lock("configs/dataset_mining/classeval_stateful_working_v0.lock.yaml").accepted_file.exists():
            self.skipTest("Ignored canonical accepted dataset is not present in this clean checkout.")
        with tempfile.TemporaryDirectory() as tmp:
            index = MiningIndex(Path(tmp) / "index.sqlite")
            materializer = DatasetMaterializer(index, datasets_root=Path(tmp) / "datasets")

            with self.assertRaisesRegex(ConfigError, "immutable"):
                materializer.materialize_dataset("classeval_stateful_working_v0")

    def test_read_only_loading_of_18_accepted_samples_succeeds(self) -> None:
        import json

        lock = load_finalized_lock("configs/dataset_mining/classeval_stateful_working_v0.lock.yaml")
        if not lock.accepted_file.exists():
            self.skipTest("Ignored canonical accepted dataset is not present in this clean checkout.")
        rows = [json.loads(line) for line in lock.accepted_file.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(len(rows), 18)
        self.assertEqual([row["sample_id"] for row in rows], lock.canonical_source_task_ids)

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(__import__("json").dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")

    def _temp_lock(self, tmp: Path, rows: list[dict], canonical_ids: list[str] | None = None, sha_override: str | None = None):
        import hashlib
        import yaml

        accepted = tmp / "accepted.jsonl"
        self._write_jsonl(accepted, rows)
        raw = accepted.read_bytes()
        lock_path = tmp / "lock.yaml"
        ids = canonical_ids if canonical_ids is not None else [row["sample_id"] for row in rows]
        lock_path.write_text(
            yaml.safe_dump(
                {
                    "dataset_version": "tmp_finalized",
                    "finalized": True,
                    "final_accepted_count": 18,
                    "accepted_file": str(accepted),
                    "accepted_sha256": sha_override or hashlib.sha256(raw).hexdigest(),
                    "allow_mining": False,
                    "canonical_source_task_ids": ids,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        return load_finalized_lock(lock_path)

    def test_finalized_fewer_than_18_rows_fails_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            rows = [{"sample_id": f"s{i}", "final_status": "accepted"} for i in range(17)]
            lock = self._temp_lock(tmp, rows)
            with self.assertRaisesRegex(ConfigError, "expected 18 accepted rows"):
                validate_finalized_dataset(lock)

    def test_finalized_more_than_18_rows_fails_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            rows = [{"sample_id": f"s{i}", "final_status": "accepted"} for i in range(19)]
            lock = self._temp_lock(tmp, rows)
            with self.assertRaisesRegex(ConfigError, "expected 18 accepted rows"):
                validate_finalized_dataset(lock)

    def test_finalized_id_mismatch_fails_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            rows = [{"sample_id": f"s{i}", "final_status": "accepted"} for i in range(18)]
            ids = [row["sample_id"] for row in rows]
            ids[-1] = "different"
            lock = self._temp_lock(tmp, rows, canonical_ids=ids)
            with self.assertRaisesRegex(ConfigError, "ID order/set"):
                validate_finalized_dataset(lock)

    def test_finalized_hash_mismatch_fails_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            rows = [{"sample_id": f"s{i}", "final_status": "accepted"} for i in range(18)]
            lock = self._temp_lock(tmp, rows, sha_override="0" * 64)
            with self.assertRaisesRegex(ConfigError, "SHA-256 mismatch"):
                validate_finalized_dataset(lock)

    def test_non_finalized_profile_retains_existing_behavior_shape(self) -> None:
        config = load_yaml("configs/dataset_mining/classeval_stateful_multi.yaml")

        self.assertFalse(config["profiles"]["debug"].get("finalized", False))
        self.assertEqual(config["profiles"]["debug"]["target_accepted"], 1)


if __name__ == "__main__":
    unittest.main()
