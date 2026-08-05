from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.bootstrap_sr import run_bootstrap


class BootstrapSrTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
