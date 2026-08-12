from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.bootstrap_sr import run_bootstrap
from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.runpod_runtime import (
    build_runpod_plan,
    resolve_cuda_arch_list,
    validate_base_image_contract,
    validate_network_volume,
    validate_torch_contract,
)


class RunpodRuntimeContractTests(unittest.TestCase):
    repo_root = Path(__file__).resolve().parents[1]

    def test_base_image_contract_uses_plain_cuda_devel(self) -> None:
        contract = validate_base_image_contract(self.repo_root / "configs" / "env" / "runpod_base_image.lock.yaml")
        self.assertEqual(contract["image_repository"], "nvidia/cuda")
        self.assertEqual(contract["image_tag"], "12.9.1-devel-ubuntu22.04")
        self.assertEqual(
            contract["index_digest"],
            "sha256:bd4e2680a261c212f1e2fea241606f71497dc67a417f73175d794ec8212b5ba8",
        )
        self.assertIn("not_required", contract["cudnn_system_package"]["status"])

    def test_dockerfile_ownership_contract(self) -> None:
        text = (self.repo_root / "cloud" / "runpod" / "Dockerfile.dev-base").read_text(encoding="utf-8")
        self.assertIn(
            "FROM --platform=linux/amd64 nvidia/cuda:12.9.1-devel-ubuntu22.04@sha256:bd4e2680a261c212f1e2fea241606f71497dc67a417f73175d794ec8212b5ba8",
            text,
        )
        self.assertIn("COPY --from=uv /uv /usr/local/bin/uv", text)
        self.assertIn('CMD ["/usr/local/bin/putpocket-runpod-start"]', text)
        self.assertNotIn("pip install torch", text)
        self.assertNotIn("pip install vllm", text)
        self.assertNotIn("ENTRYPOINT", text)

    def test_architecture_profile_expansion(self) -> None:
        self.assertEqual(resolve_cuda_arch_list("portable-nvidia"), "8.6 9.0 10.0 12.0")
        self.assertEqual(resolve_cuda_arch_list("hopper"), "9.0")
        self.assertEqual(resolve_cuda_arch_list("blackwell-datacenter"), "10.0")
        self.assertEqual(resolve_cuda_arch_list("blackwell-rtx"), "12.0")
        self.assertEqual(resolve_cuda_arch_list("hopper", "9.0 10.0"), "9.0 10.0")

    def test_torch_contract_fails_closed_when_unresolved(self) -> None:
        path = self.repo_root / "configs" / "env" / "torch" / "torch_2_10_cu129.lock.yaml"
        contract = validate_torch_contract(path, require_resolved=False)
        self.assertEqual(contract["package"]["version"], "2.10.0+cu129")
        self.assertEqual(contract["provenance_status"], "unresolved")
        with self.assertRaisesRegex(ConfigError, "TORCH_CU129_PROVENANCE_UNRESOLVED"):
            validate_torch_contract(path, require_resolved=True)

    def test_runpod_dev_dry_run_emits_runtime_plan(self) -> None:
        with patch.dict(os.environ, {key: "" for key in ()}, clear=True), patch("builtins.print") as mocked_print:
            self.assertEqual(run_bootstrap(["--preset", "runpod-dev", "--dry-run"]), 0)
        rendered = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list)
        self.assertIn('"preset": "runpod-dev"', rendered)
        self.assertIn('"PUTPOCKET_ENV_PATH": "/workspace/putpocket_dataset_mining/Putpocket_env"', rendered)
        self.assertIn('"cuda_arch_list": "8.6 9.0 10.0 12.0"', rendered)
        self.assertIn('"torch_contract_status": "unresolved"', rendered)

    def test_runpod_dev_non_dry_run_stops_on_unresolved_torch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name, patch.dict(
            os.environ,
            {"RUNPOD_NETWORK_VOLUME_ID": "unit-test-volume"},
            clear=False,
        ):
            code = run_bootstrap(
                [
                    "--preset",
                    "runpod-dev",
                    "--persistent-root",
                    f"{tmp_name}/putpocket_dataset_mining",
                    "--storage-kind",
                    "network-volume",
                    "--skip-vllm-build",
                ]
            )
        self.assertEqual(code, 2)

    def test_network_volume_guard_requires_volume_identity(self) -> None:
        plan = build_runpod_plan(
            repo_root=self.repo_root,
            persistent_root="/workspace/putpocket_dataset_mining",
            storage_kind="network-volume",
            cuda_arch_profile="portable-nvidia",
            cuda_arch_list=None,
            base_image_contract=None,
            dry_run=False,
            doctor_only=False,
            skip_vllm_build=True,
            force_vllm_build=False,
            skip_gpu_smoke=True,
        )
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ConfigError, "RUNPOD_NETWORK_VOLUME_ID"):
            validate_network_volume(plan)

    def test_template_does_not_auto_launch_vllm(self) -> None:
        text = (self.repo_root / "cloud" / "runpod" / "template.dev-base.example.yaml").read_text(encoding="utf-8")
        self.assertIn("start_command: /usr/local/bin/putpocket-runpod-start", text)
        self.assertNotIn("api_server", text)

    def test_activation_uses_repo_relative_defaults_and_no_mutation(self) -> None:
        script = self.repo_root / "scripts" / "env" / "env_activate.sh"
        text = script.read_text(encoding="utf-8")
        self.assertIn('PUTPOCKET_REPO_ROOT', text)
        self.assertIn('${PUTPOCKET_DATASET_MINING_ROOT}/Putpocket_env', text)
        self.assertIn('UV_PYTHON_INSTALL_DIR', text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("docker ", text)
        self.assertNotIn("export CUDA_VISIBLE_DEVICES", text)

    def test_server2_preset_still_dry_runs(self) -> None:
        with patch("builtins.print") as mocked_print:
            self.assertEqual(run_bootstrap(["--preset", "server2", "--dry-run"]), 0)
        rendered = "\n".join(str(call.args[0]) for call in mocked_print.call_args_list)
        self.assertIn('"preset": "server2"', rendered)

    def test_server1_static_profile_still_dry_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            self.assertEqual(
                run_bootstrap(
                    [
                        "--stage",
                        "all",
                        "--server-profile",
                        "server1_rtx3090",
                        "--role",
                        "verifier",
                        "--vllm-profile",
                        "skip",
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
