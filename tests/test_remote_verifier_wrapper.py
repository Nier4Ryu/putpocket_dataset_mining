from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.remote_verifier.cli import main as remote_main
from putpocket_dataset_mining.remote_verifier.manifest import sha256_tree, write_json_atomic
from putpocket_dataset_mining.remote_verifier.runner import _test_command, promote_incoming, result_status, verify


class RemoteVerifierWrapperTests(unittest.TestCase):
    def test_protocol_version(self) -> None:
        self.assertEqual(remote_main(["protocol-version"]), 0)

    def test_list_test_command_is_shell_quoted_for_local_runner(self) -> None:
        self.assertEqual(_test_command(["python3", "-m", "pytest", "-q", "/workspace/a b.py"]), "python3 -m pytest -q '/workspace/a b.py'")

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


if __name__ == "__main__":
    unittest.main()
