from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.bootstrap_sr import run_bootstrap
from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.runpod_runtime import build_manifest, build_runpod_plan, resolve_build_jobs


class BuildParallelismContractTests(unittest.TestCase):
    repo_root = Path(__file__).resolve().parents[1]

    def _plan(self, *, build_jobs: int | None = None, env: dict[str, str] | None = None, cpu_count: int = 32):
        return build_runpod_plan(
            repo_root=self.repo_root,
            persistent_root=None,
            storage_kind=None,
            cuda_arch_profile="blackwell-rtx",
            cuda_arch_list=None,
            base_image_contract=None,
            dry_run=True,
            doctor_only=False,
            skip_vllm_build=False,
            force_vllm_build=False,
            skip_gpu_smoke=True,
            build_jobs=build_jobs,
            cpu_count=cpu_count,
            env=env or {},
        )

    def test_runpod_defaults_to_nproc_and_mocked_32_resolves_to_32(self) -> None:
        cpus, jobs = resolve_build_jobs(None, env={}, cpu_count=32)
        self.assertEqual((cpus, jobs), (32, 32))
        plan = self._plan()
        self.assertEqual(plan.environment()["MAX_JOBS"], "32")
        self.assertEqual(plan.environment()["CMAKE_BUILD_PARALLEL_LEVEL"], "32")
        self.assertEqual(plan.environment()["NVCC_THREADS"], "1")
        self.assertEqual(plan.environment()["TMPDIR"], "/workspace/putpocket_dataset_mining/builds/tmp")

    def test_cli_overrides_environment_and_nproc(self) -> None:
        self.assertEqual(resolve_build_jobs(24, env={"PUTPOCKET_BUILD_JOBS": "12"}, cpu_count=32), (32, 24))

    def test_environment_overrides_nproc(self) -> None:
        self.assertEqual(resolve_build_jobs(None, env={"PUTPOCKET_BUILD_JOBS": "20"}, cpu_count=32), (32, 20))

    def test_invalid_job_values_fail(self) -> None:
        for value in (0, -1):
            with self.subTest(value=value), self.assertRaises(ConfigError):
                resolve_build_jobs(value, env={}, cpu_count=32)
        with self.assertRaises(ConfigError):
            resolve_build_jobs(None, env={"PUTPOCKET_BUILD_JOBS": "0"}, cpu_count=32)

    def test_manifest_records_effective_parallelism(self) -> None:
        plan = self._plan()
        manifest = build_manifest(plan, {"vllm_sha": "abc"})
        self.assertEqual(manifest["cpu_count_detected"], 32)
        self.assertEqual(manifest["build_jobs_requested"], 32)
        self.assertEqual(manifest["build_jobs_effective"], 32)
        self.assertEqual(manifest["max_jobs"], 32)
        self.assertEqual(manifest["cmake_build_parallel_level"], 32)
        self.assertEqual(manifest["nvcc_threads"], 1)

    def test_dry_run_cli_reports_override_without_building(self) -> None:
        with patch("putpocket_dataset_mining.runpod_runtime.detect_cpu_count", return_value=32), patch("builtins.print") as output:
            self.assertEqual(run_bootstrap(["--preset", "runpod-dev", "--build-jobs", "28", "--dry-run"]), 0)
        payload = json.loads(output.call_args.args[0])
        self.assertEqual(payload["cpu_count_detected"], 32)
        self.assertEqual(payload["build_jobs_effective"], 28)

    def test_server2_policy_defaults_to_8_and_server1_verifier_contract_skips_build(self) -> None:
        activation = (self.repo_root / "scripts" / "env" / "env_activate.sh").read_text(encoding="utf-8")
        self.assertIn('PUTPOCKET_BUILD_THREADS="${PUTPOCKET_BUILD_THREADS:-8}"', activation)
        self.assertIn('CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-${PUTPOCKET_BUILD_JOBS}}"', activation)
        bootstrap = (self.repo_root / "src" / "putpocket_dataset_mining" / "bootstrap_sr.py").read_text(encoding="utf-8")
        static_stage = bootstrap.split("def _stage_manifest", 1)[1].split("def _gpu_phase", 1)[0]
        self.assertIn('"vllm_build_requested": args.build_vllm', static_stage)
        self.assertNotIn("_run_runpod_vllm_build", static_stage)


if __name__ == "__main__":
    unittest.main()
