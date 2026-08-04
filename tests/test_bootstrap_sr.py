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
            manifests = list(Path(tmp_name).glob("cpu_bootstrap_manifest_*.json"))
            self.assertEqual(len(manifests), 1)


if __name__ == "__main__":
    unittest.main()
