from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.cli import main
from putpocket_dataset_mining.verifier import VerificationResult


REMOTE_CONFIG = """\
backend: remote_ssh_docker
target:
  host: 10.0.0.5
  user: dyryu
  port: 42
route: direct
repository_root: /home/dyryu/putpocket_dataset_mining
job_root: /home/dyryu/putpocket_dataset_mining/data/remote_verifier
connection_timeout_sec: 10
command_timeout_sec: 3720
rsync_timeout_sec: 300
wrapper: /home/dyryu/putpocket_dataset_mining/Putpocket_env/bin/putpocket-remote-verifier
docker_image: putpocket-classeval-python:ubuntu22.04-py313-v1
dockerfile: docker/classeval_python/Dockerfile
max_concurrent_jobs: 1
verifier:
  timeout_sec: 3600
"""


class RemoteFixtureCliTests(unittest.TestCase):
    def _config(self, root: Path) -> Path:
        path = root / "remote.yaml"
        path.write_text(REMOTE_CONFIG, encoding="utf-8")
        return path

    def test_remote_test_dry_run_writes_summary_without_connecting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self._config(root)
            out = root / "out"
            rc = main(["remote-test", "--config", str(cfg), "--fixtures", "pass,timeout", "--timeout-fixture-sec", "2", "--output-dir", str(out), "--dry-run"])
            self.assertEqual(rc, 0)
            summary = json.loads((out / "summary.json").read_text())
            self.assertTrue(summary["dry_run"])
            self.assertEqual([row["fixture"] for row in summary["fixtures"]], ["pass", "timeout"])
            self.assertEqual(summary["fixtures"][0]["timeout_sec"], 3600)
            self.assertEqual(summary["fixtures"][1]["timeout_sec"], 2)
            self.assertEqual(summary["fixtures"][0]["wrapper"], "/home/dyryu/putpocket_dataset_mining/Putpocket_env/bin/putpocket-remote-verifier")

    def test_remote_preflight_config_reaches_transport(self) -> None:
        fake = type(
            "P",
            (),
            {
                "status": "REMOTE_DOCKER_PREFLIGHT_PASSED",
                "ssh_ok": True,
                "wrapper_ok": True,
                "rsync_ok": True,
                "docker_ok": True,
                "staging_root_ok": True,
                "image_ok": True,
                "error_class": None,
                "detail": None,
            },
        )()
        with tempfile.TemporaryDirectory() as tmp, patch("putpocket_dataset_mining.ssh_transport.SshRsyncTransport.lightweight_preflight", return_value=fake) as preflight:
            cfg = self._config(Path(tmp))
            self.assertEqual(main(["remote-preflight", "--config", str(cfg)]), 0)
            self.assertEqual(preflight.call_count, 1)

    def test_remote_preflight_uses_configured_wrapper(self) -> None:
        payload = {
            "wrapper_ok": True,
            "rsync_ok": True,
            "docker_ok": True,
            "staging_root_ok": True,
            "image_ok": True,
        }
        with tempfile.TemporaryDirectory() as tmp, patch("subprocess.run") as run:
            run.return_value = type("R", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()
            cfg = self._config(Path(tmp))
            self.assertEqual(main(["remote-preflight", "--config", str(cfg)]), 0)
        remote_cmd = run.call_args.args[0][-1]
        self.assertIn("/home/dyryu/putpocket_dataset_mining/Putpocket_env/bin/putpocket-remote-verifier", remote_cmd)
        self.assertIn("preflight", remote_cmd)

    def test_remote_test_uses_transport_and_timeout_override(self) -> None:
        calls = []

        def fake_run(self, **kwargs):
            calls.append(kwargs)
            fixture = kwargs["task"].raw["fixture"]
            return VerificationResult(
                stage=kwargs["stage"],
                passed=fixture == "pass",
                final_status="timeout" if fixture == "timeout" else "passed" if fixture == "pass" else "failed",
                failure_class=None,
                returncode=124 if fixture == "timeout" else 0 if fixture == "pass" else 1,
                stdout="",
                stderr="",
                timeout=fixture == "timeout",
                timeout_sec=kwargs["timeout_sec"],
                workspace=str(kwargs["verifier_workspace"]),
            )

        with tempfile.TemporaryDirectory() as tmp, patch("putpocket_dataset_mining.verifier.SshRsyncVerifierTransport.run", new=fake_run):
            root = Path(tmp)
            cfg = self._config(root)
            out = root / "out"
            rc = main(["remote-test", "--config", str(cfg), "--fixtures", "pass,fail,timeout", "--timeout-fixture-sec", "2", "--output-dir", str(out)])
            self.assertEqual(rc, 0)
            self.assertEqual([call["timeout_sec"] for call in calls], [3600, 3600, 2])
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual([row["status"] for row in summary["fixtures"]], ["passed", "failed", "timeout"])


if __name__ == "__main__":
    unittest.main()
