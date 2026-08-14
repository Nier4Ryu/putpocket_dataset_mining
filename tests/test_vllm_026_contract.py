from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from putpocket_dataset_mining.bootstrap_sr import run_bootstrap
from putpocket_dataset_mining.externals import EXTERNALS


class Vllm026ContractTests(unittest.TestCase):
    repo_root = Path(__file__).resolve().parents[1]
    target_sha = "568afb3a13806beb53bb2e6bd518269357b237c0"

    def test_vllm_external_is_clean_upstream_026_sha(self) -> None:
        repo = EXTERNALS["vllm"]
        self.assertEqual(repo.url, "https://github.com/vllm-project/vllm.git")
        self.assertEqual(repo.ref, "releases/v0.26.0")
        self.assertEqual(repo.sha, self.target_sha)
        self.assertIsNone(repo.branch)

    def test_server2_lock_pins_vllm_026_and_torch_211(self) -> None:
        lock = yaml.safe_load((self.repo_root / "configs/env/server2_blackwell.lock.yaml").read_text(encoding="utf-8"))
        self.assertEqual(lock["externals"]["vllm"]["tag"], "v0.26.0")
        self.assertEqual(lock["externals"]["vllm"]["head"], self.target_sha)
        self.assertEqual(lock["externals"]["vllm"]["remote"], "https://github.com/vllm-project/vllm.git")
        self.assertEqual(lock["python_packages"]["torch"], "2.11.0")
        self.assertEqual(lock["python_packages"]["torch_wheel"]["sha256"], "cc89b9b173d9adfab59fd227f0ab5e5516d9a52b658ae41d64e59d2e55a418db")

    def test_runpod_lock_uses_same_vllm_026_source(self) -> None:
        lock = json.loads((self.repo_root / "configs/env/runpod_dev.lock.yaml").read_text(encoding="utf-8"))
        self.assertEqual(lock["vllm"]["url"], "https://github.com/vllm-project/vllm.git")
        self.assertEqual(lock["vllm"]["branch"], "releases/v0.26.0")
        self.assertEqual(lock["vllm"]["tag"], "v0.26.0")
        self.assertEqual(lock["vllm"]["sha"], self.target_sha)
        self.assertEqual(lock["torch"]["version"], "2.11.0")

    def test_server2_build_jobs_override_reports_8(self) -> None:
        with patch("builtins.print") as output:
            self.assertEqual(run_bootstrap(["--preset", "server2", "--build-jobs", "8", "--dry-run"]), 0)
        payload = json.loads(output.call_args.args[0])
        self.assertEqual(payload["build_jobs_effective"], 8)
        self.assertEqual(payload["max_jobs"], 8)
        self.assertEqual(payload["cmake_build_parallel_level"], 8)
        self.assertEqual(payload["nvcc_threads"], 1)

    def test_abandoned_sm120_candidate_is_not_active_reference(self) -> None:
        active_files = [
            self.repo_root / "src/putpocket_dataset_mining/externals.py",
            self.repo_root / "src/putpocket_dataset_mining/bootstrap_sr.py",
            self.repo_root / "configs/env/server2_blackwell.lock.yaml",
            self.repo_root / "configs/env/runpod_dev.lock.yaml",
        ]
        text = "\n".join(path.read_text(encoding="utf-8") for path in active_files)
        forbidden = [
            "T20260813-001__glm52-awq-sm120-kernel-runtime",
            "Putpocket-v0.26.0-sm120-glm52",
            "/workspace/runtime_candidates/glm52_awq_sm120",
        ]
        for needle in forbidden:
            self.assertNotIn(needle, text)


if __name__ == "__main__":
    unittest.main()
