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
        t = SshRsyncTransport(RemoteDockerConfig(host="host", user="user", port=2222, root="/srv/sr", identity_file="/id", known_hosts_file="/kh"))
        argv = t.ssh_base_argv()
        self.assertIn("BatchMode=yes", argv)
        self.assertIn("2222", argv)
        self.assertIn("/id", argv)
        self.assertIn("UserKnownHostsFile=/kh", argv)

    def test_preflight_success_is_structured(self) -> None:
        t = SshRsyncTransport(RemoteDockerConfig(host="host", user="user", root="/srv/sr"))
        fake = type("R", (), {"returncode": 0, "stdout": '{"wrapper_ok": true, "rsync_ok": true, "docker_ok": true, "staging_root_ok": true, "image_ok": true}', "stderr": "", "json_stdout": lambda self: __import__("json").loads(self.stdout)})
        with patch.object(t, "run_wrapper", return_value=fake()):
            result = t.lightweight_preflight("image")
        self.assertTrue(result.docker_ok)
        self.assertEqual(result.status, "REMOTE_DOCKER_PREFLIGHT_PASSED")

    def test_rsync_construction_does_not_shell_interpolate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = SshRsyncTransport(RemoteDockerConfig(host="host", user="user", root="/srv/sr"))
            with patch("subprocess.run") as run:
                run.return_value = type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
                t.rsync_to_remote(Path(tmp), "jobs/job1/workspace/")
        args = run.call_args.args[0]
        self.assertEqual(args[0], "rsync")
        self.assertIn("--partial", args)


if __name__ == "__main__":
    unittest.main()
