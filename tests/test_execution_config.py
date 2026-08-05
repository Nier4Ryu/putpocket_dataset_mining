from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.execution_config import (
    E_CLOUD_LOCAL_DOCKER_FORBIDDEN,
    ExecutionConfig,
    ExecutionRole,
    HardwareProfile,
    cuda_arch_for_profile,
    default_hardware_for_server,
)


class ExecutionConfigTests(unittest.TestCase):
    def test_explicit_mapping_overrides_environment(self) -> None:
        with patch.dict(os.environ, {"SR_EXECUTION_ROLE": "cloud_controller"}):
            cfg = ExecutionConfig.from_env_and_mapping({"execution_role": "local_controller"})
        self.assertEqual(cfg.execution_role, ExecutionRole.LOCAL_CONTROLLER)

    def test_runpod_detection_defaults_cloud_controller(self) -> None:
        with patch.dict(os.environ, {"RUNPOD_POD_ID": "pod"}, clear=True):
            cfg = ExecutionConfig.from_env_and_mapping()
        self.assertEqual(cfg.execution_role, ExecutionRole.CLOUD_CONTROLLER)

    def test_cloud_local_docker_guard(self) -> None:
        cfg = ExecutionConfig.from_env_and_mapping(
            {
                "execution_role": "cloud_controller",
                "workspace_backend": "local_docker",
                "verifier_backend": "local_docker",
            }
        )
        with self.assertRaisesRegex(ConfigError, E_CLOUD_LOCAL_DOCKER_FORBIDDEN):
            cfg.guard_cloud_local_docker()

    def test_remote_requires_complete_config(self) -> None:
        cfg = ExecutionConfig.from_env_and_mapping(
            {
                "workspace_backend": "remote_ssh_docker",
                "verifier_backend": "remote_ssh_docker",
                "remote": {"host": "example", "user": "u"},
            }
        )
        with self.assertRaisesRegex(ConfigError, "remote_ssh_docker requires repository_root"):
            cfg.validate_for_evaluation_start()

    def test_cuda_arch_selection(self) -> None:
        self.assertEqual(cuda_arch_for_profile("sm86"), "8.6")
        self.assertEqual(cuda_arch_for_profile("sm90"), "9.0")
        self.assertEqual(cuda_arch_for_profile("sm120"), "12.0")
        self.assertEqual(default_hardware_for_server("runpod_hopper"), HardwareProfile.SM90)


if __name__ == "__main__":
    unittest.main()
