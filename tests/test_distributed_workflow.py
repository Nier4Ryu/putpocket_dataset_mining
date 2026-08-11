from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.cli import main
from putpocket_dataset_mining.distributed_workflow import WorkflowCheckpointStore, manual_run_stage, run_workflow
from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.two_turn_e2e import build_scripted_config, scripted_task


class DistributedWorkflowTests(unittest.TestCase):
    def _config(self, tmp: Path) -> Path:
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
        cfg["dataset"] = {"adapter": "classeval"}
        path = tmp / "config.yaml"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return path

    def test_state_machine_rejects_illegal_transition(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "run"
            store = WorkflowCheckpointStore(root)
            store.initialize(run_uuid="u", mode="sequential", sample_ids=["s"], config={})
            with self.assertRaises(ConfigError):
                store.transition("HISTORY2_READY")

    def test_manual_stages_write_safe_stop_markers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "run"
            run_workflow(mode="manual", config_path=self._config(Path(td)), remote_config=None, sample_ids=["test_ClassEval_76"], run_root=root, run_uuid="u")
            first = manual_run_stage(root, "history1-infer-submit")
            self.assertEqual(first["status"], "SAFE_TO_STOP_SERVER2")
            marker = Path(first["marker"])
            self.assertTrue(marker.exists())
            payload = json.loads(marker.read_text())
            self.assertTrue(payload["can_stop_server2"])
            self.assertTrue(payload["engine_stopped"])
            self.assertIn("history1-retrieve", payload["next_command"])
            state = json.loads((root / "workflow_state.json").read_text())
            self.assertEqual(state["current_state"], "VERIFICATION1_SUBMITTED")

    def test_workflow_cli_manual_status(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "run"
            rc = main(["workflow", "manual", "init", "--config", str(self._config(Path(td))), "--run-root", str(root), "--samples", "test_ClassEval_76", "--run-uuid", "u"])
            self.assertEqual(rc, 0)
            rc = main(["workflow", "manual", "status", "--run-root", str(root)])
            self.assertEqual(rc, 0)

    def test_sequential_uses_shared_artifact_contract(self) -> None:
        calls: list[str] = []

        def fake_task(config, sample_id):
            self.assertEqual(sample_id, "test_ClassEval_76")
            return scripted_task()

        def fake_run_task(self, task, **kwargs):
            calls.append(task.sample_id)
            artifact = Path(kwargs["run_id"]) if Path(kwargs["run_id"]).is_absolute() else Path(tempfile.gettempdir()) / kwargs["run_id"]
            attempt = Path(tempfile.mkdtemp()) / "attempt"
            (attempt / "verification/history2").mkdir(parents=True)
            (attempt / "verification/history2/remote_result.json").write_text("{}", encoding="utf-8")
            return {"sample_id": "test_ClassEval_76", "final_status": "accepted", "artifact_path": str(attempt)}

        with tempfile.TemporaryDirectory() as td, patch("putpocket_dataset_mining.distributed_workflow._task_by_sample_id", new=fake_task), patch(
            "putpocket_dataset_mining.distributed_workflow.SingleSampleRunner.run_task",
            new=fake_run_task,
        ):
            root = Path(td) / "run"
            result = run_workflow(mode="sequential", config_path=self._config(Path(td)), remote_config=None, sample_ids=["test_ClassEval_76"], run_root=root, run_uuid="u")
            self.assertEqual(result["counts"]["accepted"], 1)
            self.assertEqual(calls, ["scripted_two_turn_remote_e2e"])
            self.assertTrue((root / "events.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
