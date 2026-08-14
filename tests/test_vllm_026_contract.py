from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from putpocket_dataset_mining.bootstrap_sr import (
    BootstrapRun,
    _parse_nvcc_cuda_version,
    _server2_torch_requirement,
    _validate_server2_torch_cuda_contract,
    run_bootstrap,
)
from putpocket_dataset_mining.errors import ConfigError
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

    def test_server2_lock_pins_vllm_026_and_torch_211_cu129(self) -> None:
        lock = yaml.safe_load((self.repo_root / "configs/env/server2_blackwell.lock.yaml").read_text(encoding="utf-8"))
        self.assertEqual(lock["externals"]["vllm"]["tag"], "v0.26.0")
        self.assertEqual(lock["externals"]["vllm"]["head"], self.target_sha)
        self.assertEqual(lock["externals"]["vllm"]["remote"], "https://github.com/vllm-project/vllm.git")
        self.assertEqual(lock["python_packages"]["torch"], "2.11.0+cu129")
        self.assertEqual(lock["python_packages"]["torch_wheel"]["index_url"], "https://download.pytorch.org/whl/cu129")
        self.assertIn("download-r2.pytorch.org/whl/cu129", lock["python_packages"]["torch_wheel"]["url"])
        self.assertEqual(lock["python_packages"]["torch_wheel"]["sha256"], "fde1830d7f79641680865759dc57780e94a9de7e68a82ed61973e9bc7af29423")
        self.assertEqual(lock["python_packages"]["torch_wheel"]["torch_cuda"], "12.9")
        self.assertNotIn("files.pythonhosted.org", lock["python_packages"]["torch_wheel"]["url"])
        self.assertNotEqual(lock["python_packages"]["torch_wheel"].get("torch_cuda"), "13.0")

    def test_runpod_lock_uses_same_vllm_026_source(self) -> None:
        lock = json.loads((self.repo_root / "configs/env/runpod_dev.lock.yaml").read_text(encoding="utf-8"))
        self.assertEqual(lock["vllm"]["url"], "https://github.com/vllm-project/vllm.git")
        self.assertEqual(lock["vllm"]["branch"], "releases/v0.26.0")
        self.assertEqual(lock["vllm"]["tag"], "v0.26.0")
        self.assertEqual(lock["vllm"]["sha"], self.target_sha)
        self.assertEqual(lock["torch"]["version"], "2.11.0+cu129")
        self.assertEqual(lock["torch"]["source_index"], "https://download.pytorch.org/whl/cu129")
        self.assertEqual(lock["torch"]["torch_cuda"], "12.9")

    def test_server2_torch_requirement_is_exact_wheel_not_generic_pypi(self) -> None:
        requirement = _server2_torch_requirement()
        self.assertIsNotNone(requirement)
        assert requirement is not None
        self.assertIn("download-r2.pytorch.org/whl/cu129", requirement)
        self.assertIn("torch-2.11.0%2Bcu129-cp313-cp313-manylinux_2_28_x86_64.whl", requirement)
        self.assertIn("sha256=fde1830d7f79641680865759dc57780e94a9de7e68a82ed61973e9bc7af29423", requirement)
        self.assertNotIn("torch==2.11.0", requirement)
        self.assertNotIn("cu130", requirement)

    def test_server2_build_jobs_override_reports_8(self) -> None:
        with patch("builtins.print") as output:
            self.assertEqual(run_bootstrap(["--preset", "server2", "--build-jobs", "8", "--dry-run"]), 0)
        payload = json.loads(output.call_args.args[0])
        self.assertEqual(payload["build_jobs_effective"], 8)
        self.assertEqual(payload["max_jobs"], 8)
        self.assertEqual(payload["cmake_build_parallel_level"], 8)
        self.assertEqual(payload["nvcc_threads"], 1)

    def test_nvcc_version_parser_uses_major_minor(self) -> None:
        self.assertEqual(_parse_nvcc_cuda_version("Cuda compilation tools, release 12.9, V12.9.86"), "12.9")

    def test_torch_cuda_mismatch_is_rejected_before_native_build(self) -> None:
        run = BootstrapRun(repo_root=self.repo_root, log_dir=self.repo_root / "tmp-test-log", dry_run=False, doctor_only=False)
        torch_payload = {
            "python": "/fake/python",
            "torch_version": "2.11.0+cu130",
            "torch_cuda": "13.0",
            "direct_url": {"url": "https://files.pythonhosted.org/torch-2.11.0.whl"},
        }

        def fake_command(cmd, *, check, log=None, env=None):  # type: ignore[no-untyped-def]
            if cmd[0].endswith("python"):
                return {"cmd": cmd, "returncode": 0, "stdout": json.dumps(torch_payload), "stderr": ""}
            if cmd[0] == "nvcc":
                return {"cmd": cmd, "returncode": 0, "stdout": "Cuda compilation tools, release 12.9, V12.9.86", "stderr": ""}
            raise AssertionError(cmd)

        with patch("putpocket_dataset_mining.bootstrap_sr._command", side_effect=fake_command), patch(
            "putpocket_dataset_mining.bootstrap_sr._write_json"
        ):
            with self.assertRaisesRegex(ConfigError, "TORCH_SYSTEM_CUDA_CONTRACT_MISMATCH"):
                _validate_server2_torch_cuda_contract(run, env={}, phase="before_vllm_native_build")

    def test_torch_cuda_contract_accepts_cu129(self) -> None:
        run = BootstrapRun(repo_root=self.repo_root, log_dir=self.repo_root / "tmp-test-log", dry_run=False, doctor_only=False)
        torch_payload = {
            "python": "/fake/python",
            "torch_version": "2.11.0+cu129",
            "torch_cuda": "12.9",
            "direct_url": {"url": "https://download-r2.pytorch.org/whl/cu129/torch.whl"},
        }

        def fake_command(cmd, *, check, log=None, env=None):  # type: ignore[no-untyped-def]
            if cmd[0].endswith("python"):
                return {"cmd": cmd, "returncode": 0, "stdout": json.dumps(torch_payload), "stderr": ""}
            if cmd[0] == "nvcc":
                return {"cmd": cmd, "returncode": 0, "stdout": "Cuda compilation tools, release 12.9, V12.9.86", "stderr": ""}
            raise AssertionError(cmd)

        with patch("putpocket_dataset_mining.bootstrap_sr._command", side_effect=fake_command), patch(
            "putpocket_dataset_mining.bootstrap_sr._write_json"
        ):
            _validate_server2_torch_cuda_contract(run, env={}, phase="before_vllm_native_build")

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
