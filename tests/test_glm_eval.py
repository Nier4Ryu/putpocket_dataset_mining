from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from putpocket_dataset_mining.dataset import SourceTask
from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.model_evaluation.dataset_loader import load_accepted_samples
from putpocket_dataset_mining.model_evaluation.glm_eval import (
    parse_gpu_slots,
    validate_eval_gpu_slots,
    write_summary,
)


def make_source_task() -> SourceTask:
    return SourceTask(
        adapter="mbpp_huggingface",
        dataset_id="google-research-datasets/mbpp",
        split="train",
        row_index=0,
        task_id="1",
        prompt="Write a function to add two numbers.",
        reference_solution="def add(a, b):\n    return a + b\n",
        tests=["assert add(1, 2) == 3"],
        test_setup="",
        raw={"task_id": 1},
    )


class GLMEvalTests(unittest.TestCase):
    def test_loads_accepted_sample_with_source_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_root = root / "datasets" / "ds_v0"
            artifact = root / "runs" / "sample" / "attempt"
            artifact.mkdir(parents=True)
            make_source_task().write_json(artifact / "source_task.json")
            dataset_root.mkdir(parents=True)
            row = {
                "sample_id": "train_1",
                "task_id": "1",
                "split": "train",
                "row_index": 0,
                "attempt_id": "attempt1",
                "final_status": "accepted",
                "artifact_path": str(artifact),
                "query1": "implement",
                "query2": "refactor",
                "policy_delta": "type_hints_required_v1",
            }
            (dataset_root / "accepted.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            samples = load_accepted_samples("ds_v0", datasets_root=root / "datasets")

            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0].sample_id, "train_1")
            self.assertEqual(samples[0].source_task.task_id, "1")
            self.assertIn("episode_summary.json", samples[0].missing_artifacts)

    def test_gpu_slot_validation_rejects_disallowed_gpu(self) -> None:
        with self.assertRaises(ConfigError):
            validate_eval_gpu_slots([[3]], workers=1)

    def test_parse_gpu_slots_assigns_one_gpu_per_worker(self) -> None:
        self.assertEqual(parse_gpu_slots("0,1,2", workers=2, profile="full"), [[0], [1]])

    def test_write_summary_counts_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            run_config = {
                "run_id": "eval1",
                "eval_name": "eval",
                "dataset_version": "ds_v0",
                "accepted_count": 2,
                "model_id": "glm",
                "backend": "local_vllm_python_engine",
            }
            results = [
                {
                    "sample_id": "train_1",
                    "final_status": "succeeded",
                    "history1_status": "verification_passed",
                    "history2_status": "verification_passed",
                    "judge_decision": "pass",
                    "failure_stage": None,
                },
                {
                    "sample_id": "train_2",
                    "final_status": "failed",
                    "history1_status": "verification_failed",
                    "history2_status": "skipped",
                    "judge_decision": "skipped",
                    "failure_stage": "history1_verification",
                    "history1_failure_class": "history1.unit_test.failed",
                },
            ]

            summary = write_summary(run_root, run_config, results)

            self.assertEqual(summary["counts"]["final_status"]["succeeded"], 1)
            self.assertEqual(summary["counts"]["final_status"]["failed"], 1)
            self.assertTrue((run_root / "summary.json").exists())
            self.assertTrue((run_root / "summary.md").exists())


if __name__ == "__main__":
    unittest.main()
