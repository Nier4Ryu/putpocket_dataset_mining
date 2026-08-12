from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from putpocket_dataset_mining.bootstrap_sr import run_bootstrap
from putpocket_dataset_mining.docker_workspace import DockerImageManager
from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.execution_config import HardwareProfile, cuda_arch_for_profile
from putpocket_dataset_mining.runpod_runtime import (
    ARCH_PROFILES,
    build_manifest,
    build_runpod_plan,
    docker_build_args,
    inspect_binary_architectures,
    normalize_cuda_arch_list,
    parse_cuobjdump_arches,
    resolve_cuda_arch_contract,
    resolve_cuda_arch_list,
    vllm_editable_build_command,
)


class CudaArchitectureContractTests(unittest.TestCase):
    repo_root = Path(__file__).resolve().parents[1]

    def test_profile_registry(self) -> None:
        self.assertEqual(ARCH_PROFILES["portable-nvidia"], "8.6 9.0 10.0 12.0")
        self.assertEqual(ARCH_PROFILES["rtx3090"], "8.6")
        self.assertEqual(ARCH_PROFILES["hopper"], "9.0")
        self.assertEqual(ARCH_PROFILES["blackwell-datacenter"], "10.0")
        self.assertEqual(ARCH_PROFILES["blackwell-rtx"], "12.0")
        self.assertEqual(cuda_arch_for_profile(HardwareProfile.SM100), "10.0")

    def test_arch_list_validation_and_normalization(self) -> None:
        self.assertEqual(normalize_cuda_arch_list("8.6   9.0  10.0   12.0"), "8.6 9.0 10.0 12.0")
        for bad in ("", "9", "9.0 9.0", "7.5", "9.0+PTX"):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                normalize_cuda_arch_list(bad)
        with self.assertRaises(ConfigError):
            resolve_cuda_arch_list("does-not-exist")

    def test_precedence_rules(self) -> None:
        env = {"PUTPOCKET_CUDA_ARCH_PROFILE": "hopper", "PUTPOCKET_CUDA_ARCH_LIST": "10.0"}
        self.assertEqual(
            resolve_cuda_arch_contract(
                preset_default_profile="portable-nvidia",
                cli_arch_profile="blackwell-rtx",
                cli_arch_list="9.0",
                env=env,
            ),
            ("blackwell-rtx", "9.0"),
        )
        self.assertEqual(
            resolve_cuda_arch_contract(
                preset_default_profile="portable-nvidia",
                cli_arch_profile="blackwell-rtx",
                cli_arch_list=None,
                env=env,
            ),
            ("blackwell-rtx", "12.0"),
        )
        self.assertEqual(resolve_cuda_arch_contract(preset_default_profile="portable-nvidia", env=env), ("hopper", "10.0"))
        self.assertEqual(resolve_cuda_arch_contract(preset_default_profile="portable-nvidia", env={}), ("portable-nvidia", "8.6 9.0 10.0 12.0"))

    def test_native_detection_and_unknown_capability(self) -> None:
        completed = MagicMock(returncode=0, stdout="12.0\n9.0\n", stderr="")
        with patch("putpocket_dataset_mining.runpod_runtime.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "putpocket_dataset_mining.runpod_runtime.subprocess.run", return_value=completed
        ):
            self.assertEqual(resolve_cuda_arch_list("native"), "9.0 12.0")
        bad = MagicMock(returncode=0, stdout="8.9\n", stderr="")
        with patch("putpocket_dataset_mining.runpod_runtime.shutil.which", return_value="/usr/bin/nvidia-smi"), patch(
            "putpocket_dataset_mining.runpod_runtime.subprocess.run", return_value=bad
        ), self.assertRaisesRegex(ConfigError, "unsupported native"):
            resolve_cuda_arch_list("native")

    def test_runpod_dev_overrides(self) -> None:
        plan = build_runpod_plan(
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
            env={},
        )
        self.assertEqual(plan.cuda_arch_profile, "blackwell-rtx")
        self.assertEqual(plan.cuda_arch_list, "12.0")

    def test_vllm_and_docker_build_propagation(self) -> None:
        command = vllm_editable_build_command(Path("/workspace/putpocket_dataset_mining"), "8.6 9.0 10.0 12.0")
        self.assertIn('export TORCH_CUDA_ARCH_LIST="8.6 9.0 10.0 12.0"', command)
        self.assertIn("uv pip install --no-build-isolation --no-deps -e .", command)
        self.assertEqual(docker_build_args("8.6 9.0 10.0 12.0"), ["--build-arg", "torch_cuda_arch_list=8.6 9.0 10.0 12.0"])

    def test_docker_manager_build_arg(self) -> None:
        manager = DockerImageManager("img", Path("/tmp/Dockerfile"))
        with patch.object(DockerImageManager, "image_exists", return_value=False), patch.object(Path, "exists", return_value=True), patch(
            "putpocket_dataset_mining.docker_workspace.shutil.which", return_value="docker"
        ), patch("putpocket_dataset_mining.docker_workspace.subprocess.run") as mocked:
            mocked.return_value = MagicMock(returncode=0, stderr="")
            with patch.dict(os.environ, {"PUTPOCKET_CUDA_ARCH_LIST": "8.6 9.0 10.0 12.0"}):
                manager.ensure_image()
        argv = mocked.call_args.args[0]
        self.assertIn("torch_cuda_arch_list=8.6 9.0 10.0 12.0", argv)

    def test_manifest_architecture_fields(self) -> None:
        plan = build_runpod_plan(
            repo_root=self.repo_root,
            persistent_root=None,
            storage_kind=None,
            cuda_arch_profile=None,
            cuda_arch_list=None,
            base_image_contract=None,
            dry_run=True,
            doctor_only=False,
            skip_vllm_build=False,
            force_vllm_build=False,
            skip_gpu_smoke=True,
            env={},
        )
        manifest = build_manifest(plan, {"base_image_digest": "d", "python": "3.13", "torch": "2.10", "torch_cuda": "12.9", "cuda_toolkit": "12.9"})
        self.assertEqual(manifest["cuda_arch_profile"], "portable-nvidia")
        self.assertEqual(manifest["requested_cuda_arch_list"], ["8.6", "9.0", "10.0", "12.0"])
        self.assertEqual(manifest["torch_cuda_arch_list"], "8.6 9.0 10.0 12.0")

    def test_cuobjdump_parsing(self) -> None:
        evidence = parse_cuobjdump_arches("code for sm_86\ncode for sm_120\n")
        self.assertEqual(evidence["sm_86"], "PRESENT")
        self.assertEqual(evidence["sm_90"], "MISSING")
        self.assertEqual(evidence["sm_120"], "PRESENT")
        self.assertEqual(inspect_binary_architectures([])["sm_100"], "NOT_APPLICABLE")

    def test_static_preset_regressions(self) -> None:
        cases = [
            ("server1_rtx3090", "verifier", "8.6"),
            ("server2_blackwell", "development", "12.0"),
            ("server2_rtxpro6000_blackwell", "development", "12.0"),
            ("runpod_hopper", "model_server", "9.0"),
        ]
        for server, role, arch in cases:
            with self.subTest(server=server), tempfile.TemporaryDirectory() as tmp:
                self.assertEqual(run_bootstrap(["--stage", "core", "--server-profile", server, "--role", role, "--verifier-backend", "disabled", "--manifest-dir", tmp, "--dry-run"]), 0)
                text = next(Path(tmp).glob("core_bootstrap_manifest_*.json")).read_text(encoding="utf-8")
                self.assertIn(f'"torch_cuda_arch_list": "{arch}"', text)

    def test_runpod_dry_run_profiles(self) -> None:
        with patch("builtins.print") as mocked_print:
            self.assertEqual(run_bootstrap(["--preset", "runpod-dev", "--cuda-arch-profile", "blackwell-rtx", "--dry-run"]), 0)
        self.assertIn('"TORCH_CUDA_ARCH_LIST": "12.0"', "\n".join(str(c.args[0]) for c in mocked_print.call_args_list))
        with patch("builtins.print") as mocked_print:
            self.assertEqual(run_bootstrap(["--preset", "runpod-dev", "--cuda-arch-profile", "portable-nvidia", "--dry-run"]), 0)
        self.assertIn('"TORCH_CUDA_ARCH_LIST": "8.6 9.0 10.0 12.0"', "\n".join(str(c.args[0]) for c in mocked_print.call_args_list))


if __name__ == "__main__":
    unittest.main()
