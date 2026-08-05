from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.execution_config import (
    E_CLOUD_LOCAL_DOCKER_FORBIDDEN,
    DEFAULT_VERIFIER_TIMEOUT_SEC,
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

    def test_default_verifier_timeout_is_one_hour(self) -> None:
        cfg = ExecutionConfig.from_env_and_mapping()
        self.assertEqual(cfg.verifier_timeout_sec, DEFAULT_VERIFIER_TIMEOUT_SEC)
        self.assertEqual(cfg.verifier_timeout_sec, 3600)

    def test_explicit_verifier_timeout_override(self) -> None:
        cfg = ExecutionConfig.from_env_and_mapping({"verifier_timeout_sec": 2})
        self.assertEqual(cfg.verifier_timeout_sec, 2)

    def test_remote_command_timeout_budget_and_connect_timeout_are_separate(self) -> None:
        cfg = ExecutionConfig.from_remote_verifier_mapping(
            {
                "backend": "remote_ssh_docker",
                "target": {"host": "10.0.0.5", "user": "dyryu", "port": 42},
                "repository_root": "/repo",
                "job_root": "/repo/data/remote_verifier",
                "connection_timeout_sec": 10,
                "command_timeout_sec": 3720,
                "verifier": {"timeout_sec": 3600},
            }
        )
        cfg.validate_for_evaluation_start()
        self.assertEqual(cfg.remote.connection_timeout_sec, 10)
        self.assertEqual(cfg.remote.command_timeout_sec, 3720)

    def test_remote_command_timeout_too_short_is_rejected(self) -> None:
        cfg = ExecutionConfig.from_remote_verifier_mapping(
            {
                "backend": "remote_ssh_docker",
                "target": {"host": "10.0.0.5", "user": "dyryu", "port": 42},
                "repository_root": "/repo",
                "job_root": "/repo/data/remote_verifier",
                "command_timeout_sec": 120,
                "verifier": {"timeout_sec": 3600},
            }
        )
        with self.assertRaisesRegex(ConfigError, "verifier timeout plus grace"):
            cfg.validate_for_evaluation_start()

    def test_remote_wrapper_config_and_env(self) -> None:
        cfg = ExecutionConfig.from_remote_verifier_mapping(
            {
                "backend": "remote_ssh_docker",
                "target": {"host": "10.0.0.5", "user": "dyryu", "port": 42},
                "repository_root": "/repo",
                "job_root": "/repo/data/remote_verifier",
                "wrapper": "/repo/Putpocket_env/bin/putpocket-remote-verifier",
            }
        )
        self.assertEqual(cfg.remote.wrapper, "/repo/Putpocket_env/bin/putpocket-remote-verifier")
        with patch.dict(os.environ, {"SR_REMOTE_WRAPPER": "/env/bin/putpocket-remote-verifier"}):
            self.assertEqual(ExecutionConfig.from_env_and_mapping().remote.wrapper, "/env/bin/putpocket-remote-verifier")


if __name__ == "__main__":
    unittest.main()
