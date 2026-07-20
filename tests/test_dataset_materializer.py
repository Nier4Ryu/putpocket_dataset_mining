from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from putpocket_dataset_mining.storage import AttemptRecord, DatasetMaterializer, MiningIndex


class DatasetMaterializerTests(unittest.TestCase):
    def test_materializer_writes_local_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            index = MiningIndex(tmp_path / "mining_index.sqlite")
            index.record_attempt(
                AttemptRecord(
                    run_id="run1",
                    dataset_version="ds_v0",
                    sample_id="train_1",
                    split="train",
                    row_index=0,
                    task_id="1",
                    attempt_id="attempt1",
                    final_status="accepted",
                    failure_class=None,
                    artifact_path=str(tmp_path / "runs" / "run1"),
                    summary={"query1": "q1", "query2": "q2", "policy_delta": "type_hints_required_v1"},
                )
            )
            root = DatasetMaterializer(index, datasets_root=tmp_path / "datasets").materialize_dataset("ds_v0")
            self.assertTrue((root / "dataset_manifest.yaml").exists())
            accepted = (root / "accepted.jsonl").read_text(encoding="utf-8").strip()
            row = json.loads(accepted)
            self.assertEqual(row["sample_id"], "train_1")
            self.assertEqual(row["final_status"], "accepted")


if __name__ == "__main__":
    unittest.main()
