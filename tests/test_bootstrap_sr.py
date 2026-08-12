from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.bootstrap_sr import run_bootstrap


class BootstrapSrTests(unittest.TestCase):
    repo_root = Path(__file__).resolve().parents[1]

    def test_cpu_phase_with_no_visible_gpu_and_disabled_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            code = run_bootstrap(
                [
                    "--phase",
                    "cpu",
                    "--hardware-profile",
                    "cpu",
                    "--execution-role",
                    "local_controller",
                    "--verifier-backend",
                    "disabled",
                    "--manifest-dir",
                    tmp_name,
                    "--dry-run",
                ]
            )
            self.assertEqual(code, 0)
            manifests = list(Path(tmp_name).glob("core_bootstrap_manifest_*.json"))
            self.assertEqual(len(manifests), 1)

    def test_stage_core_server2_blackwell_no_gpu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}):
            self.assertEqual(
                run_bootstrap(
                    [
                        "--stage",
                        "core",
                        "--server-profile",
                        "server2_blackwell",
                        "--role",
                        "development",
                        "--manifest-dir",
                        tmp_name,
                        "--dry-run",
                    ]
                ),
                0,
            )
            text = next(Path(tmp_name).glob("core_bootstrap_manifest_*.json")).read_text()
            self.assertIn('"server_profile": "server2_blackwell"', text)
            self.assertIn('"cuda_arch_list": "12.0"', text)

    def test_runpod_model_server_skips_local_docker_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            self.assertEqual(
                run_bootstrap(
                    [
                        "--stage",
                        "all",
                        "--server-profile",
                        "runpod_hopper",
                        "--role",
                        "model_server",
                        "--verifier-backend",
                        "disabled",
                        "--manifest-dir",
                        tmp_name,
                        "--dry-run",
                    ]
                ),
                0,
            )
            self.assertTrue(list(Path(tmp_name).glob("*_bootstrap_manifest_*.json")))

    def test_server2_preset_dry_run_plans_canonical_environment(self) -> None:
        with patch("builtins.print") as mocked_print:
            self.assertEqual(run_bootstrap(["--preset", "server2", "--dry-run"]), 0)
        rendered = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list)
        self.assertIn('"preset": "server2"', rendered)
        self.assertIn('"environment":', rendered)
        self.assertIn("Putpocket_env", rendered)
        self.assertIn("qwen-runtime", rendered)

    def test_doctor_only_plan_has_no_mutations(self) -> None:
        with patch("builtins.print") as mocked_print:
            self.assertEqual(run_bootstrap(["--preset", "server2", "--doctor-only", "--dry-run"]), 0)
        rendered = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list)
        self.assertIn('"doctor_only": true', rendered)
        self.assertIn('"mutations": []', rendered)

    def test_bootstrap_env_delegates_to_canonical_preset(self) -> None:
        script = self.repo_root / "scripts" / "env" / "bootstrap_env.sh"
        result = subprocess.run([str(script), "--dry-run"], cwd=self.repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"preset": "server2"', result.stdout)
        self.assertIn("delegating to bootstrap_sr.sh --preset server2", result.stderr)

    def test_legacy_glm_bootstrap_requires_explicit_opt_in(self) -> None:
        for name in ("bootstrap_glm52_env.sh", "bootstrap_glm52_v025_env.sh"):
            script = self.repo_root / "scripts" / "env" / name
            result = subprocess.run([str(script), "--help"], cwd=self.repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 2)
            self.assertIn("PUTPOCKET_ALLOW_LEGACY_GLM_ENV=1", result.stderr)

    def test_activation_script_does_not_set_cuda_visible_devices(self) -> None:
        script = self.repo_root / "scripts" / "env" / "env_activate.sh"
        text = script.read_text(encoding="utf-8")
        self.assertNotIn("export CUDA_VISIBLE_DEVICES", text)
        self.assertIn("Putpocket_env_glm52", text)


if __name__ == "__main__":
    unittest.main()
