from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from putpocket_dataset_mining.artifact_sync import build_sync_manifest, copy_from_manifest


class ArtifactSyncTests(unittest.TestCase):
    def test_minimal_profile_excludes_hidden_tests_and_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            (root / "summary.json").write_text("{}", encoding="utf-8")
            (root / "verification/history1").mkdir(parents=True)
            (root / "verification/history1/checklist.json").write_text("{}", encoding="utf-8")
            (root / "verification/history1/workspace/tests").mkdir(parents=True)
            (root / "verification/history1/workspace/tests/test_solution.py").write_text("hidden", encoding="utf-8")
            (root / "token_secret.txt").write_text("secret", encoding="utf-8")
            manifest = build_sync_manifest(root, "analysis_minimal")
        paths = {item["relative_path"] for item in manifest["items"]}
        self.assertIn("summary.json", paths)
        self.assertIn("verification/history1/checklist.json", paths)
        self.assertNotIn("verification/history1/workspace/tests/test_solution.py", paths)
        self.assertNotIn("token_secret.txt", paths)

    def test_copy_is_no_delete_and_atomic_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name) / "src"
            dst = Path(tmp_name) / "dst"
            root.mkdir()
            (root / "summary.json").write_text("{}", encoding="utf-8")
            dst.mkdir()
            (dst / "keep.txt").write_text("keep", encoding="utf-8")
            manifest = build_sync_manifest(root, "analysis_minimal")
            result = copy_from_manifest(root, dst, manifest)
            self.assertEqual(result["item_count"], 1)
            self.assertTrue((dst / "keep.txt").exists())
            self.assertTrue((dst / "SYNC_COMPLETE.json").exists())

    def test_dry_run_copies_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name) / "src"
            dst = Path(tmp_name) / "dst"
            root.mkdir()
            (root / "summary.json").write_text("{}", encoding="utf-8")
            manifest = build_sync_manifest(root, "analysis_minimal")
            copy_from_manifest(root, dst, manifest, dry_run=True)
            self.assertFalse(dst.exists())

    def test_cluster_handoff_excludes_weights_logs_raw_outputs_and_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name)
            (root / "run_manifest.json").write_text("{}", encoding="utf-8")
            (root / "readiness_report.json").write_text("{}", encoding="utf-8")
            (root / "rendered_jobs").mkdir()
            (root / "rendered_jobs/readiness.sbatch").write_text("#!/bin/bash\n", encoding="utf-8")
            (root / "checkpoints").mkdir()
            (root / "checkpoints/model.safetensors").write_bytes(b"never hash or sync checkpoint tensors")
            (root / "raw").mkdir()
            (root / "raw/trace.json").write_text("{}", encoding="utf-8")
            (root / "slurm-readiness-1.out").write_text("runtime", encoding="utf-8")
            (root / "token_secret.txt").write_text("secret", encoding="utf-8")
            manifest = build_sync_manifest(root, "cluster_phase1_handoff")
        paths = {item["relative_path"] for item in manifest["items"]}
        self.assertEqual(paths, {"run_manifest.json", "readiness_report.json", "rendered_jobs/readiness.sbatch"})


if __name__ == "__main__":
    unittest.main()
