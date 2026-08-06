from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.dataset import SourceTask
from putpocket_dataset_mining.docker_workspace import CommandResult
from putpocket_dataset_mining.execution_config import ExecutionConfig
from putpocket_dataset_mining.remote_verifier.image import ImageStatus
from putpocket_dataset_mining.remote_verifier.cli import main as remote_main
from putpocket_dataset_mining.remote_verifier.manifest import result_sha256, sha256_tree, write_json_atomic
from putpocket_dataset_mining.remote_verifier.runner import _test_command, promote_incoming, result_status, verify
from putpocket_dataset_mining.verifier import SshRsyncVerifierTransport


class RemoteVerifierWrapperTests(unittest.TestCase):
    def test_protocol_version(self) -> None:
        self.assertEqual(remote_main(["protocol-version"]), 0)

    def test_list_test_command_is_shell_quoted_for_local_runner(self) -> None:
        self.assertEqual(_test_command(["python3", "-m", "pytest", "-q", "/workspace/a b.py"]), "python3 -m pytest -q '/workspace/a b.py'")

    def test_timeout_output_bytes_are_decoded_for_result_files(self) -> None:
        from putpocket_dataset_mining.docker_workspace import run_verifier_container

        with tempfile.TemporaryDirectory() as tmp, patch(
            "putpocket_dataset_mining.docker_workspace.host_uid_gid", return_value=(1000, 1000)
        ), patch(
            "putpocket_dataset_mining.docker_workspace.subprocess.run",
            side_effect=subprocess.TimeoutExpired(["docker"], 1, output=b"stdout bytes", stderr=b"stderr bytes"),
        ):
            result = run_verifier_container(
                Path(tmp),
                "image",
                "pytest -q tests/test_solution.py",
                cpus=1,
                memory="512m",
                timeout_sec=1,
            )
        self.assertTrue(result.timeout)
        self.assertEqual(result.returncode, 124)
        self.assertEqual(result.stdout, "stdout bytes")
        self.assertEqual(result.stderr, "stderr bytes")

    def test_timeout_output_text_normalization_variants(self) -> None:
        from putpocket_dataset_mining.docker_workspace import _timeout_output_text

        self.assertEqual(_timeout_output_text("already text"), "already text")
        self.assertEqual(_timeout_output_text(None), "")
        self.assertEqual(_timeout_output_text(b"bad:\xff"), "bad:\ufffd")

    def _job(self, root: Path, job_id: str, expected: int = 1) -> Path:
        job = root / "incoming" / f"{job_id}.partial"
        ws = job / "workspace"
        tests = ws / "tests"
        tests.mkdir(parents=True)
        (ws / "solution.py").write_text("def f():\n    return 1\n", encoding="utf-8")
        (tests / "test_solution.py").write_text(f"from solution import f\n\ndef test_f():\n    assert f() == {expected}\n", encoding="utf-8")
        manifest = {
            "schema_version": 1,
            "protocol_version": "sr-remote-verifier-v1",
            "job_id": job_id,
            "workspace_sha256": sha256_tree(ws),
            "docker_image": "image",
            "dockerfile": "docker/classeval_python/Dockerfile",
            "test_command": "pytest -q tests/test_solution.py",
            "timeout_sec": 1,
        }
        write_json_atomic(job / "manifest.json", manifest)
        return job

    def test_partial_job_is_never_executed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SR_REMOTE_JOB_ROOT": tmp}):
            self._job(Path(tmp), "j1")
            result = verify("j1")
            self.assertEqual(result["status"], "infra_failed")
            self.assertEqual(result["error_class"], "partial_job_not_executable")

    def test_duplicate_completed_result_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SR_REMOTE_JOB_ROOT": tmp}):
            completed = Path(tmp) / "completed" / "j2"
            completed.mkdir(parents=True)
            write_json_atomic(completed / "result.json", {"schema_version": 1, "job_id": "j2", "status": "passed"})
            self.assertEqual(result_status("j2")["status"], "passed")
            self.assertEqual(verify("j2")["status"], "passed")

    def test_checksum_mismatch_is_integrity_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SR_REMOTE_JOB_ROOT": tmp}):
            self._job(Path(tmp), "j3")
            ready = promote_incoming("j3")
            manifest = json.loads((ready / "manifest.json").read_text())
            manifest["workspace_sha256"] = "0" * 64
            write_json_atomic(ready / "manifest.json", manifest)
            result = verify("j3")
            self.assertEqual(result["status"], "infra_failed")
            self.assertEqual(result["error_class"], "REMOTE_RESULT_INTEGRITY_FAILED")

    def test_timeout_result_maps_to_structured_timeout_status(self) -> None:
        image_status = ImageStatus(
            image="image",
            image_id="sha256:image",
            dockerfile_sha256="0" * 64,
            built=False,
        )
        timed_out = CommandResult(["docker"], 124, "out", "err", timeout=True)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SR_REMOTE_JOB_ROOT": tmp}), patch(
            "putpocket_dataset_mining.remote_verifier.runner.ensure_image",
            return_value=image_status,
        ), patch(
            "putpocket_dataset_mining.remote_verifier.runner.run_verifier_container",
            return_value=timed_out,
        ):
            self._job(Path(tmp), "j4")
            promote_incoming("j4")
            result = verify("j4")
            self.assertEqual(result["status"], "timeout")
            self.assertEqual(result["process_exit_code"], 124)
            self.assertTrue(result["timed_out"])
            self.assertEqual(result["timeout_sec"], 1)

    def test_image_ensure_uses_mocked_docker(self) -> None:
        from putpocket_dataset_mining.remote_verifier.image import ensure_image

        with tempfile.TemporaryDirectory() as tmp, patch("shutil.which", return_value="docker"), patch("subprocess.run") as run, patch.dict(os.environ, {"SR_REMOTE_JOB_ROOT": tmp}):
            dockerfile = Path(tmp) / "docker" / "classeval_python" / "Dockerfile"
            dockerfile.parent.mkdir(parents=True)
            dockerfile.write_text("FROM scratch\n", encoding="utf-8")
            run.return_value = type("R", (), {"returncode": 0, "stdout": "sha256:image\n", "stderr": ""})()
            status = ensure_image("image", dockerfile)
            self.assertFalse(status.built)
            self.assertEqual(status.image_id, "sha256:image")

    def test_remote_manifest_records_effective_timeout_and_command(self) -> None:
        payload = {
            "schema_version": 1,
            "protocol_version": "sr-remote-verifier-v1",
            "status": "passed",
            "verifier_passed": True,
            "process_exit_code": 0,
            "timed_out": False,
            "stdout": "ok",
            "stderr": "",
        }
        payload["result_sha256"] = result_sha256(payload)
        fake_ok = type("R", (), {"returncode": 0, "stdout": "{}", "stderr": ""})()
        fake_verify = type("R", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()

        cfg = ExecutionConfig.from_env_and_mapping(
            {
                "verifier_backend": "remote_ssh_docker",
                "verifier_timeout_sec": 2,
                "remote": {
                    "host": "host",
                    "user": "user",
                    "repository_root": "/repo",
                    "job_root": "/repo/data/remote_verifier",
                    "command_timeout_sec": 122,
                },
            }
        )
        task = SourceTask("fixture", "fixture", "split", 0, "task", "", "", [], "", {})
        with tempfile.TemporaryDirectory() as tmp, patch(
            "putpocket_dataset_mining.verifier.SshRsyncTransport.rsync_to_remote",
            return_value=fake_ok,
        ), patch(
            "putpocket_dataset_mining.verifier.SshRsyncTransport.run_wrapper",
            side_effect=[fake_ok, fake_verify, fake_verify],
        ) as run_wrapper:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "solution.py").write_text("", encoding="utf-8")
            attempt = root / "run" / "per_sample" / "sample" / "attempt"
            result = SshRsyncVerifierTransport(cfg).run(
                stage="history1",
                verifier_workspace=workspace,
                task=task,
                docker_image="image",
                test_command="pytest -q tests/test_solution.py",
                cpus=1,
                memory="512m",
                timeout_sec=2,
                attempt_dir=attempt,
            )
            self.assertTrue(result.passed)
            manifest = json.loads((attempt / "verification" / "history1" / "remote_job" / "manifest.json").read_text())
            self.assertEqual(manifest["timeout_sec"], 2)
            self.assertEqual(manifest["test_command"], "pytest -q tests/test_solution.py")
            self.assertEqual(result.timeout_sec, 2)
            self.assertEqual([call.args[0] for call in run_wrapper.call_args_list], ["promote", "verify", "result-status"])


if __name__ == "__main__":
    unittest.main()
