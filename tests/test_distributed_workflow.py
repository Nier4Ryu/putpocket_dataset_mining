from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.cli import main
from putpocket_dataset_mining.distributed_workflow import WorkflowCheckpointStore, _mode_engine, manual_run_stage, run_workflow
from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.two_turn_e2e import build_scripted_config, scripted_task
from putpocket_dataset_mining.verifier import VerificationResult


class DistributedWorkflowTests(unittest.TestCase):
    def test_mode_engine_propagates_pp3_tp2_and_six_devices(self) -> None:
        cfg = {
            "model": {
                "generation_model_id": "/immutable/model",
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 3,
            },
            "gpu": {"allowed_cuda_devices": [0, 1, 2, 3, 4, 5]},
        }
        engine = _mode_engine(cfg, None)
        self.assertIsNotNone(engine)
        self.assertEqual(engine.tensor_parallel_size, 2)
        self.assertEqual(engine.pipeline_parallel_size, 3)
        self.assertEqual(engine.gpu_devices, [0, 1, 2, 3, 4, 5])

    def test_mode_engine_rejects_device_count_mismatch(self) -> None:
        cfg = {
            "model": {
                "generation_model_id": "/immutable/model",
                "tensor_parallel_size": 2,
                "pipeline_parallel_size": 3,
            },
            "gpu": {"allowed_cuda_devices": [0, 1, 2]},
        }
        with self.assertRaisesRegex(ConfigError, "requires 6 GPU devices"):
            _mode_engine(cfg, None)

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
            detail = {
                "attempt_dir": str(root / "manual/samples/test_ClassEval_76/manual_attempt"),
                "stage": "history1",
                "job_id": "job-history1",
                "receipt": str(root / "manual/samples/test_ClassEval_76/manual_attempt/verification/history1/submission_receipt.json"),
                "checkpoint": str(root / "manual/samples/test_ClassEval_76/manual_attempt/checkpoints/after_history1/workspace"),
                "checkpoint_sha256": "abc123",
                "inference_sec": 1.0,
                "submit_sec": 0.1,
                "submit_end_perf": 2.0,
                "controller_pid": 12345,
            }
            with patch("putpocket_dataset_mining.distributed_workflow._run_history_and_submit", return_value=detail) as submit:
                first = manual_run_stage(root, "history1-infer-submit")
            self.assertEqual(first["status"], "SAFE_TO_STOP_SERVER2")
            submit.assert_called_once()
            marker = Path(first["marker"])
            self.assertTrue(marker.exists())
            payload = json.loads(marker.read_text())
            self.assertTrue(payload["can_stop_server2"])
            self.assertTrue(payload["safe_to_stop_server2"])
            self.assertTrue(payload["engine_stopped"])
            self.assertEqual(payload["remote_job_id"], "job-history1")
            self.assertEqual(payload["checkpoint_sha256"], "abc123")
            self.assertIn("remote_job_durably_accepted", payload["satisfied_proof_conditions"])
            self.assertIn("history1-retrieve", payload["next_command"])
            state = json.loads((root / "workflow_state.json").read_text())
            self.assertEqual(state["current_state"], "VERIFICATION1_SUBMITTED")

    def test_manual_retrieve_uses_remote_result_and_skips_h2_on_v1_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "run"
            run_workflow(mode="manual", config_path=self._config(Path(td)), remote_config=None, sample_ids=["test_ClassEval_76"], run_root=root, run_uuid="u")
            store = WorkflowCheckpointStore(root)
            store.transition("HISTORY1_INFERENCE_READY", sample_id="test_ClassEval_76")
            store.transition("HISTORY1_INFERENCE_RUNNING", sample_id="test_ClassEval_76")
            store.transition("HISTORY1_INFERENCE_COMPLETED", sample_id="test_ClassEval_76")
            store.transition("VERIFICATION1_BUNDLE_READY", sample_id="test_ClassEval_76")
            store.transition("VERIFICATION1_SUBMITTED", detail={"history1_job_id": "job-history1"}, sample_id="test_ClassEval_76")
            failed = VerificationResult(
                stage="history1",
                passed=False,
                final_status="failed",
                failure_class="verifier.failed",
                returncode=1,
                stdout="",
                stderr="",
                timeout=False,
                timeout_sec=3600,
                workspace="workspace",
                backend="remote_ssh_docker",
                remote_job_id="job-history1",
                verifier_host="cerrotorre",
            )
            with patch("putpocket_dataset_mining.distributed_workflow._retrieve_stage", return_value=failed) as retrieve:
                result = manual_run_stage(root, "history1-retrieve")
            retrieve.assert_called_once()
            self.assertEqual(result["status"], "REJECTED")
            state = json.loads((root / "workflow_state.json").read_text())
            self.assertEqual(state["current_state"], "REJECTED")

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

    def test_pipeline_uses_async_stage_boundaries_and_records_overlap(self) -> None:
        submit_calls: list[tuple[str, str]] = []
        retrieve_calls: list[tuple[str, str]] = []
        status_counts: dict[str, int] = {}

        def fake_submit(*, cfg, run_root, sample_id, stage, mode, gpu_device, async_submit, engine=None):
            self.assertTrue(async_submit)
            self.assertEqual(mode, "pipeline")
            time.sleep(0.35)
            submit_calls.append((sample_id, stage))
            attempt = root / "pipeline" / "samples" / sample_id / "pipeline_attempt" / "verification" / stage
            attempt.mkdir(parents=True, exist_ok=True)
            receipt = {"job_id": f"{sample_id}-{stage}", "verification_policy": "history2_pytest_then_judge" if stage == "history2" else "history1_pytest_only"}
            (attempt / "submission_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            return {"job_id": receipt["job_id"], "submit_end_perf": 10.0 if stage == "history1" else 20.0, "receipt": str(attempt / "submission_receipt.json")}

        def fake_status(cfg, receipt_path):
            job_id = json.loads(receipt_path.read_text(encoding="utf-8"))["job_id"]
            status_counts[job_id] = status_counts.get(job_id, 0) + 1
            return {"job_id": job_id, "status": "passed"}

        def fake_retrieve(*, cfg, run_root, sample_id, stage, mode):
            retrieve_calls.append((sample_id, stage))
            remote = {"judge": {"executed": stage == "history2", "decision": "pass"}}
            return VerificationResult(
                stage=stage,
                passed=True,
                final_status="passed",
                failure_class=None,
                returncode=0,
                stdout="",
                stderr="",
                timeout=False,
                timeout_sec=3600,
                workspace="workspace",
                backend="remote_ssh_docker",
                remote_job_id=f"{sample_id}-{stage}",
                verifier_host="cerrotorre",
                remote_result=remote,
            )

        with tempfile.TemporaryDirectory() as td, patch("putpocket_dataset_mining.distributed_workflow._mode_engine", return_value=object()), patch(
            "putpocket_dataset_mining.distributed_workflow._run_history_and_submit",
            side_effect=fake_submit,
        ), patch("putpocket_dataset_mining.distributed_workflow._remote_status", side_effect=fake_status), patch(
            "putpocket_dataset_mining.distributed_workflow._retrieve_stage",
            side_effect=fake_retrieve,
        ), patch("putpocket_dataset_mining.distributed_workflow.SingleSampleRunner.run_task") as run_task:
            root = Path(td) / "run"
            result = run_workflow(
                mode="pipeline",
                config_path=self._config(Path(td)),
                remote_config=None,
                sample_ids=["test_ClassEval_76", "test_ClassEval_37"],
                run_root=root,
                run_uuid="u",
            )
            run_task.assert_not_called()
            self.assertEqual(submit_calls[:2], [("test_ClassEval_76", "history1"), ("test_ClassEval_37", "history1")])
            self.assertIn(("test_ClassEval_76", "history2"), submit_calls)
            self.assertEqual(len(retrieve_calls), 4)
            overlap = json.loads((root / "pipeline" / "overlap.json").read_text(encoding="utf-8"))
            self.assertGreater(overlap["total_overlap_sec"], 0.5)
            self.assertEqual(result["counts"]["accepted"], 2)


if __name__ == "__main__":
    unittest.main()
