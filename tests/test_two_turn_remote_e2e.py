from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.errors import InfraError
from putpocket_dataset_mining.two_turn_e2e import ScriptedTwoTurnEngine, build_scripted_config, scripted_task
from putpocket_dataset_mining.single import SingleSampleRunner
from putpocket_dataset_mining.verifier import VerificationResult


class _FakeWorkspace:
    def __init__(self, host_workspace: Path) -> None:
        self.host_workspace = host_workspace

    def __enter__(self) -> "_FakeWorkspace":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def write_file(self, path: str, content: str) -> None:
        target = self.host_workspace / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def read_file(self, path: str) -> str:
        return (self.host_workspace / path).read_text(encoding="utf-8")

    def list_files(self, path: str = ".", recursive: bool = False) -> list[str]:
        root = self.host_workspace / path
        iterator = root.rglob("*") if recursive else root.iterdir()
        return [str(item.relative_to(self.host_workspace)) for item in iterator]


def _fake_workspace_from_execution_config(**kwargs):
    return _FakeWorkspace(Path(kwargs["host_workspace"]))


def _remote_result(stage: str, attempt_dir: Path, passed: bool = True) -> VerificationResult:
    remote_result = {
        "status": "passed" if passed else "failed",
        "verification_policy": "history2_pytest_then_judge" if stage == "history2" else "history1_pytest_only",
        "pytest": {"status": "passed" if passed else "failed"},
        "judge": {
            "executed": stage == "history2" and passed,
            "backend": "codex_cli",
            "decision": "pass" if stage == "history2" and passed else None,
            "infrastructure_status": "passed" if stage == "history2" and passed else None,
            "reason": "fixture",
        },
    }
    return VerificationResult(
        stage=stage,
        passed=passed,
        final_status="passed" if passed else "failed",
        failure_class=None if passed else f"{stage}.unit_test.failed",
        returncode=0 if passed else 1,
        stdout="ok" if passed else "failed",
        stderr="",
        timeout=False,
        timeout_sec=3600,
        workspace=str(attempt_dir / "verification" / stage / "workspace"),
        backend="remote_ssh_docker",
        remote_job_id=f"job-{stage}",
        remote_protocol="sr-remote-verifier-v1",
        verifier_host="cerrotorre",
        docker_image_id="sha256:test",
        workspace_sha256=f"workspace-{stage}",
        result_sha256=f"result-{stage}",
        verifier_revision="sr-remote-verifier-v1",
        remote_result=remote_result,
    )


class TwoTurnRemoteE2ETests(unittest.TestCase):
    def _runner_config(self, tmp: Path) -> dict:
        remote = tmp / "remote.yaml"
        remote.write_text(
            """
backend: remote_ssh_docker
target: {host: host, user: user, port: 22}
repository_root: /repo
job_root: /repo/data/remote_verifier
command_timeout_sec: 3720
verifier: {timeout_sec: 3600}
""".strip(),
            encoding="utf-8",
        )
        cfg = build_scripted_config(remote)
        cfg["run"]["output_root"] = str(tmp / "runs")
        cfg["docker"]["build_if_missing"] = False
        return cfg

    def test_complete_two_turn_remote_path(self) -> None:
        calls: list[str] = []

        def fake_run(self, **kwargs):
            calls.append(kwargs["stage"])
            return _remote_result(kwargs["stage"], kwargs["attempt_dir"], passed=True)

        with tempfile.TemporaryDirectory() as td, \
            patch("putpocket_dataset_mining.single.DockerImageManager.ensure_image"), \
            patch("putpocket_dataset_mining.single.workspace_from_execution_config", side_effect=_fake_workspace_from_execution_config), \
            patch("putpocket_dataset_mining.single.SshRsyncTransport.lightweight_preflight") as preflight, \
            patch("putpocket_dataset_mining.verifier.SshRsyncVerifierTransport.run", new=fake_run), \
            patch("putpocket_dataset_mining.verifier.LocalDockerVerifierTransport.run") as local_run:
            preflight.return_value.status = "REMOTE_DOCKER_PREFLIGHT_PASSED"
            runner = SingleSampleRunner(self._runner_config(Path(td)))
            summary = runner.run_task(scripted_task(), run_id="run", attempt_id="attempt", write_index=False, engine=ScriptedTwoTurnEngine())
            attempt = Path(summary["artifact_path"])
            self.assertEqual(summary["final_status"], "accepted")
            self.assertEqual(calls, ["history1", "history2"])
            local_run.assert_not_called()
            messages2 = json.loads((attempt / "prepared/messages_history2.json").read_text())
            self.assertTrue(any("History-1 implementation complete" in item.get("content", "") for item in messages2))
            self.assertIn("Return the sum", (attempt / "workspace_snapshots/after_history2/solution.py").read_text())
            self.assertEqual(json.loads((attempt / "verification/history1/checklist.json").read_text())["backend"], "remote_ssh_docker")
            self.assertEqual(json.loads((attempt / "verification/history2/checklist.json").read_text())["backend"], "remote_ssh_docker")
            judge = json.loads((attempt / "judge/judge_decision.json").read_text())
            self.assertTrue(judge["remote_verification2"])
            self.assertEqual(judge["decision"], "pass")

    def test_remote_history2_pytest_pass_without_judge_is_not_accepted(self) -> None:
        def fake_run(self, **kwargs):
            result = _remote_result(kwargs["stage"], kwargs["attempt_dir"], passed=True)
            if kwargs["stage"] == "history2":
                result.remote_result["judge"] = {"executed": False, "decision": None}
            return result

        with tempfile.TemporaryDirectory() as td, \
            patch("putpocket_dataset_mining.single.DockerImageManager.ensure_image"), \
            patch("putpocket_dataset_mining.single.workspace_from_execution_config", side_effect=_fake_workspace_from_execution_config), \
            patch("putpocket_dataset_mining.single.SshRsyncTransport.lightweight_preflight") as preflight, \
            patch("putpocket_dataset_mining.verifier.SshRsyncVerifierTransport.run", new=fake_run):
            preflight.return_value.status = "REMOTE_DOCKER_PREFLIGHT_PASSED"
            summary = SingleSampleRunner(self._runner_config(Path(td))).run_task(
                scripted_task(), run_id="run", attempt_id="attempt", write_index=False, engine=ScriptedTwoTurnEngine()
            )
            self.assertEqual(summary["final_status"], "uncertain")
            self.assertEqual(summary["failure_class"], "judge.not_run")

    def test_history1_verification_failure_skips_history2(self) -> None:
        calls: list[str] = []

        def fake_run(self, **kwargs):
            calls.append(kwargs["stage"])
            return _remote_result(kwargs["stage"], kwargs["attempt_dir"], passed=False)

        with tempfile.TemporaryDirectory() as td, \
            patch("putpocket_dataset_mining.single.DockerImageManager.ensure_image"), \
            patch("putpocket_dataset_mining.single.workspace_from_execution_config", side_effect=_fake_workspace_from_execution_config), \
            patch("putpocket_dataset_mining.single.SshRsyncTransport.lightweight_preflight") as preflight, \
            patch("putpocket_dataset_mining.verifier.SshRsyncVerifierTransport.run", new=fake_run):
            preflight.return_value.status = "REMOTE_DOCKER_PREFLIGHT_PASSED"
            summary = SingleSampleRunner(self._runner_config(Path(td))).run_task(
                scripted_task(), run_id="run", attempt_id="attempt", write_index=False, engine=ScriptedTwoTurnEngine()
            )
            self.assertEqual(summary["final_status"], "rejected")
            self.assertEqual(summary["failure_class"], "history1.unit_test.failed")
            self.assertEqual(calls, ["history1"])
            attempt = Path(summary["artifact_path"])
            self.assertIn("SKIPPED", (attempt / "prepared/rendered_prompt_history2.txt").read_text())

    def test_history1_remote_infra_failure_skips_history2_without_local_fallback(self) -> None:
        def fake_run(self, **kwargs):
            raise InfraError("REMOTE_RESULT_INTEGRITY_FAILED")

        with tempfile.TemporaryDirectory() as td, \
            patch("putpocket_dataset_mining.single.DockerImageManager.ensure_image"), \
            patch("putpocket_dataset_mining.single.workspace_from_execution_config", side_effect=_fake_workspace_from_execution_config), \
            patch("putpocket_dataset_mining.single.SshRsyncTransport.lightweight_preflight") as preflight, \
            patch("putpocket_dataset_mining.verifier.SshRsyncVerifierTransport.run", new=fake_run), \
            patch("putpocket_dataset_mining.verifier.LocalDockerVerifierTransport.run") as local_run:
            preflight.return_value.status = "REMOTE_DOCKER_PREFLIGHT_PASSED"
            summary = SingleSampleRunner(self._runner_config(Path(td))).run_task(
                scripted_task(), run_id="run", attempt_id="attempt", write_index=False, engine=ScriptedTwoTurnEngine()
            )
            self.assertEqual(summary["final_status"], "failed_infra")
            local_run.assert_not_called()

    def test_history2_verification_failure_records_history2_stage(self) -> None:
        calls: list[str] = []

        def fake_run(self, **kwargs):
            stage = kwargs["stage"]
            calls.append(stage)
            return _remote_result(stage, kwargs["attempt_dir"], passed=stage == "history1")

        with tempfile.TemporaryDirectory() as td, \
            patch("putpocket_dataset_mining.single.DockerImageManager.ensure_image"), \
            patch("putpocket_dataset_mining.single.workspace_from_execution_config", side_effect=_fake_workspace_from_execution_config), \
            patch("putpocket_dataset_mining.single.SshRsyncTransport.lightweight_preflight") as preflight, \
            patch("putpocket_dataset_mining.verifier.SshRsyncVerifierTransport.run", new=fake_run):
            preflight.return_value.status = "REMOTE_DOCKER_PREFLIGHT_PASSED"
            summary = SingleSampleRunner(self._runner_config(Path(td))).run_task(
                scripted_task(), run_id="run", attempt_id="attempt", write_index=False, engine=ScriptedTwoTurnEngine()
            )
            self.assertEqual(calls, ["history1", "history2"])
            self.assertEqual(summary["final_status"], "rejected")
            self.assertEqual(summary["failure_class"], "history2.unit_test.failed")

    def test_artifact_completeness(self) -> None:
        def fake_run(self, **kwargs):
            return _remote_result(kwargs["stage"], kwargs["attempt_dir"], passed=True)

        with tempfile.TemporaryDirectory() as td, \
            patch("putpocket_dataset_mining.single.DockerImageManager.ensure_image"), \
            patch("putpocket_dataset_mining.single.workspace_from_execution_config", side_effect=_fake_workspace_from_execution_config), \
            patch("putpocket_dataset_mining.single.SshRsyncTransport.lightweight_preflight") as preflight, \
            patch("putpocket_dataset_mining.verifier.SshRsyncVerifierTransport.run", new=fake_run):
            preflight.return_value.status = "REMOTE_DOCKER_PREFLIGHT_PASSED"
            summary = SingleSampleRunner(self._runner_config(Path(td))).run_task(
                scripted_task(), run_id="run", attempt_id="attempt", write_index=False, engine=ScriptedTwoTurnEngine()
            )
            attempt = Path(summary["artifact_path"])
            for rel in [
                "prepared/messages_history1.json",
                "prepared/rendered_prompt_history1.txt",
                "trajectories/history1_trajectory.jsonl",
                "workspace_snapshots/after_history1/solution.py",
                "verification/history1/checklist.json",
                "prepared/messages_history2.json",
                "prepared/rendered_prompt_history2.txt",
                "trajectories/history2_trajectory.jsonl",
                "workspace_snapshots/after_history2/solution.py",
                "verification/history2/checklist.json",
                "episode_summary.json",
                "result.json",
                "summary.json",
                "summary.md",
            ]:
                self.assertTrue((attempt / rel).exists(), rel)


if __name__ == "__main__":
    unittest.main()
