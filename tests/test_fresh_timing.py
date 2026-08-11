from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.dataset import SourceTask
from putpocket_dataset_mining.execution_config import ExecutionConfig
from putpocket_dataset_mining.fresh_timing import _top_level_rows
from putpocket_dataset_mining.timing import TimingRecorder
from putpocket_dataset_mining.verifier import SshRsyncVerifierTransport
from putpocket_dataset_mining.remote_verifier.manifest import result_sha256


class FreshTimingTests(unittest.TestCase):
    def test_remote_job_id_prefix_is_used(self) -> None:
        payload = {
            "schema_version": 1,
            "protocol_version": "sr-remote-verifier-v1",
            "status": "passed",
            "verifier_passed": True,
            "process_exit_code": 0,
            "timed_out": False,
            "stdout": "ok",
            "stderr": "",
            "wall_time_sec": 0.1,
            "completed_at_kst": "2026-08-11T00:00:00+0900",
        }
        payload["result_sha256"] = result_sha256(payload)
        fake_ok = type("R", (), {"returncode": 0, "stdout": "{}", "stderr": ""})()

        def fake_wrapper(command, *args, **kwargs):
            if command in {"verify", "result-status"}:
                return type("R", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()
            return fake_ok

        cfg = ExecutionConfig.from_env_and_mapping(
            {
                "verifier_backend": "remote_ssh_docker",
                "remote": {
                    "host": "host",
                    "user": "user",
                    "repository_root": "/repo",
                    "job_root": "/repo/data/remote_verifier",
                    "command_timeout_sec": 3720,
                },
            }
        )
        task = SourceTask("fixture", "fixture", "test", 0, "ClassEval_76", "", "", [], "", {})
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {"SR_REMOTE_JOB_ID_PREFIX": "fresh-uuid"}), patch(
            "putpocket_dataset_mining.verifier.SshRsyncTransport.rsync_to_remote",
            return_value=fake_ok,
        ), patch("putpocket_dataset_mining.verifier.SshRsyncTransport.run_wrapper", side_effect=fake_wrapper):
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "solution.py").write_text("", encoding="utf-8")
            result = SshRsyncVerifierTransport(cfg).run(
                stage="history1",
                verifier_workspace=workspace,
                task=task,
                docker_image="image",
                test_command="pytest -q tests/test_solution.py",
                cpus=1,
                memory="512m",
                timeout_sec=3600,
                attempt_dir=root / "attempt",
            )
        self.assertTrue(result.remote_job_id.startswith("fresh-uuid-test_ClassEval_76-history1-"))

    def test_top_level_rows_include_required_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = TimingRecorder(Path(tmp))
            for start, end in [
                ("e2e.start", "e2e.end"),
                ("model_engine.init.start", "model_engine.ready"),
                ("history1.rollout.start", "history1.rollout.end"),
                ("history1.verification_roundtrip.start", "history1.verification_roundtrip.end"),
                ("history2.prepare.start", "history2.prepare.end"),
                ("history2.rollout.start", "history2.rollout.end"),
                ("history2.verification_roundtrip.start", "history2.verification_roundtrip.end"),
                ("final_aggregation.start", "final_aggregation.end"),
                ("model_engine.shutdown.start", "model_engine.shutdown.end"),
            ]:
                recorder.mark(start)
                recorder.mark(end)
            recorder.durations.update(
                {
                    "remote_preflight": 0.1,
                    "history1.verify_bundle": 0.2,
                    "history1.rsync_upload": 0.3,
                    "history1.result_retrieval": 0.4,
                    "history2.verify_bundle": 0.5,
                    "history2.rsync_upload": 0.6,
                    "history2.result_retrieval": 0.7,
                }
            )
            recorder.vllm_requests.extend(
                [
                    {"stage": "history1", "elapsed_sec": 1.0},
                    {"stage": "history2", "elapsed_sec": 2.0},
                ]
            )
            recorder.tool_calls.extend(
                [
                    {"stage": "history1", "elapsed_sec": 0.1},
                    {"stage": "history2", "elapsed_sec": 0.2},
                ]
            )
            rows = _top_level_rows(recorder, {"wall_time_sec": 3.0}, {"wall_time_sec": 4.0})
        required = [
            "source_and_workspace_setup_sec",
            "remote_preflight_sec",
            "model_engine_initialization_sec",
            "history1_vllm_generation_sum_sec",
            "history1_local_tool_execution_sum_sec",
            "history1_server1_docker_validation_sec",
            "history2_vllm_generation_sum_sec",
            "history2_local_tool_execution_sum_sec",
            "history2_server1_docker_validation_sec",
            "total_e2e_including_engine_init_sec",
        ]
        for key in required:
            self.assertIn(key, rows)


if __name__ == "__main__":
    unittest.main()
