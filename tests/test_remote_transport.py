from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.execution_config import RemoteDockerConfig
from putpocket_dataset_mining.ssh_transport import SshRsyncTransport, validate_relative_path, validate_safe_id


class RemoteTransportTests(unittest.TestCase):
    def test_rejects_unsafe_ids_and_paths(self) -> None:
        with self.assertRaises(ConfigError):
            validate_safe_id("../bad")
        with self.assertRaises(ConfigError):
            validate_relative_path("../bad")
        with self.assertRaises(ConfigError):
            validate_relative_path("/absolute")

    def test_ssh_argv_uses_batchmode_and_identity(self) -> None:
        t = SshRsyncTransport(RemoteDockerConfig.from_env_and_mapping({"host": "host", "user": "user", "port": 2222, "repository_root": "/srv/sr", "job_root": "/srv/sr/data/remote_verifier", "identity_file": "/id", "known_hosts_file": "/kh"}))
        argv = t.ssh_base_argv()
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("2222", argv)
        self.assertIn("/id", argv)
        self.assertIn("UserKnownHostsFile=/kh", argv)

    def test_direct_server1_port_42_argv(self) -> None:
        t = SshRsyncTransport(RemoteDockerConfig.from_env_and_mapping({"host": "10.0.0.5", "user": "dyryu", "port": 42, "repository_root": "/repo", "job_root": "/jobs"}))
        argv = t.ssh_base_argv()
        self.assertIn("42", argv)
        self.assertNotIn("-J", argv)

    def test_two_hop_proxyjump_argv(self) -> None:
        t = SshRsyncTransport(
            RemoteDockerConfig.from_env_and_mapping(
                {
                    "host": "10.0.0.5",
                    "user": "dyryu",
                    "port": 42,
                    "route": "proxy_jump",
                    "repository_root": "/repo",
                    "job_root": "/jobs",
                    "jump_hosts": [
                        {"host": "141.223.145.88", "user": "dyryu", "port": 4500},
                        {"host": "141.223.25.156", "user": "dyryu", "port": 42},
                    ],
                }
            )
        )
        argv = t.ssh_base_argv()
        self.assertIn("-J", argv)
        self.assertIn("dyryu@141.223.145.88:4500,dyryu@141.223.25.156:42", argv)

    def test_invalid_port_rejected(self) -> None:
        with self.assertRaises(Exception):
            RemoteDockerConfig.from_env_and_mapping({"host": "h", "user": "u", "port": 70000})

    def test_preflight_success_is_structured(self) -> None:
        t = SshRsyncTransport(RemoteDockerConfig.from_env_and_mapping({"host": "host", "user": "user", "repository_root": "/srv/sr", "job_root": "/srv/sr/data/remote_verifier"}))
        fake = type("R", (), {"returncode": 0, "stdout": '{"wrapper_ok": true, "rsync_ok": true, "docker_ok": true, "staging_root_ok": true, "image_ok": true}', "stderr": "", "json_stdout": lambda self: __import__("json").loads(self.stdout)})
        with patch.object(t, "run_wrapper", return_value=fake()):
            result = t.lightweight_preflight("image")
        self.assertTrue(result.docker_ok)
        self.assertEqual(result.status, "REMOTE_DOCKER_PREFLIGHT_PASSED")

    def test_configured_absolute_wrapper_is_used(self) -> None:
        t = SshRsyncTransport(
            RemoteDockerConfig.from_env_and_mapping(
                {
                    "host": "host",
                    "user": "user",
                    "repository_root": "/srv/sr",
                    "job_root": "/srv/sr/data/remote_verifier",
                    "wrapper": "/srv/sr/Putpocket_env/bin/putpocket-remote-verifier",
                }
            )
        )
        with patch("subprocess.run") as run:
            run.return_value = type("R", (), {"returncode": 0, "stdout": "{}", "stderr": ""})()
            t.run_wrapper("protocol-version")
        remote_cmd = run.call_args.args[0][-1]
        self.assertIn("/srv/sr/Putpocket_env/bin/putpocket-remote-verifier protocol-version", remote_cmd)

    def test_configured_wrapper_reaches_remote_command_builders(self) -> None:
        t = SshRsyncTransport(
            RemoteDockerConfig.from_env_and_mapping(
                {
                    "host": "host",
                    "user": "user",
                    "repository_root": "/srv/sr",
                    "job_root": "/srv/sr/data/remote_verifier",
                    "wrapper": "/srv/sr/Putpocket_env/bin/putpocket-remote-verifier",
                }
            )
        )
        commands = [
            ("protocol-version", []),
            ("preflight", []),
            ("promote", ["--job-id", "job1"]),
            ("verify", ["--job-id", "job1"]),
            ("result-status", ["--job-id", "job1"]),
        ]
        with patch("subprocess.run") as run:
            run.return_value = type("R", (), {"returncode": 0, "stdout": "{}", "stderr": ""})()
            for command, extra_args in commands:
                t.run_wrapper(command, extra_args=extra_args)
        remote_cmds = [call.args[0][-1] for call in run.call_args_list]
        for (command, _), remote_cmd in zip(commands, remote_cmds, strict=True):
            self.assertIn("/srv/sr/Putpocket_env/bin/putpocket-remote-verifier", remote_cmd)
            self.assertIn(command, remote_cmd)

    def test_unsafe_wrapper_is_rejected(self) -> None:
        with self.assertRaises(ConfigError):
            SshRsyncTransport(
                RemoteDockerConfig.from_env_and_mapping(
                    {
                        "host": "host",
                        "user": "user",
                        "repository_root": "/srv/sr",
                        "job_root": "/srv/sr/data/remote_verifier",
                        "wrapper": "bad;wrapper",
                    }
                )
            )

    def test_rsync_construction_does_not_shell_interpolate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = SshRsyncTransport(RemoteDockerConfig.from_env_and_mapping({"host": "host", "user": "user", "repository_root": "/srv/sr", "job_root": "/srv/sr/data/remote_verifier"}))
            with patch("subprocess.run") as run:
                run.return_value = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
                t.rsync_to_remote(Path(tmp), "jobs/job1/workspace/")
        args = run.call_args.args[0]
        self.assertEqual(args[0], "rsync")
        self.assertIn("--partial", args)


if __name__ == "__main__":
    unittest.main()
