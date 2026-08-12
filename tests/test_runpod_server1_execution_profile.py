from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from putpocket_dataset_mining.cli import main
from putpocket_dataset_mining.config import load_yaml
from putpocket_dataset_mining.distributed_workflow import run_workflow
from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.execution_config import DockerBackend, ExecutionConfig, ExecutionRole, RemoteRoute
from putpocket_dataset_mining.runpod_execution import (
    PROFILE_NAME,
    RUNPOD_LOCAL_WORKSPACE_BACKEND_UNAVAILABLE,
    load_runpod_execution_profile,
    run_combined_preflight,
)
from putpocket_dataset_mining.two_turn_e2e import build_scripted_config
from putpocket_dataset_mining.verifier import VerificationResult


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs" / "runpod" / "runpod_controller_server1_verifier.example.yaml"


class RunpodServer1ExecutionProfileTests(unittest.TestCase):
    def test_profile_parses_runpod_local_inference_server1_verifier(self) -> None:
        profile = load_runpod_execution_profile(PROFILE)
        self.assertEqual(profile.name, PROFILE_NAME)
        self.assertEqual(profile.controller_host, "runpod")
        self.assertEqual(profile.inference_backend, "local_vllm")
        self.assertEqual(profile.verifier_backend, "ssh_rsync")
        self.assertFalse(profile.local_hidden_verifier_fallback)
        self.assertEqual(profile.execution_config.execution_role, ExecutionRole.RUNPOD_CONTROLLER)
        self.assertEqual(profile.execution_config.workspace_backend, DockerBackend.SSH_REMOTE_DOCKER)
        self.assertEqual(profile.execution_config.workspace_remote.job_root, "/home/dyryu/putpocket_dataset_mining/data/remote_workspace")
        self.assertEqual(profile.execution_config.verifier_backend, DockerBackend.REMOTE_SSH_DOCKER)
        self.assertEqual(profile.execution_config.remote.route, RemoteRoute.PROXY_JUMP)
        self.assertEqual(profile.execution_config.remote.host, "10.0.0.5")
        self.assertEqual(profile.execution_config.remote.port, 42)
        self.assertEqual([host.host for host in profile.execution_config.remote.jump_hosts], ["141.223.145.88", "141.223.25.156"])
        self.assertNotIn("/home/dyryu", profile.run_root)
        self.assertEqual(profile.topology["verification1_policy"], "history1_pytest_only")
        self.assertEqual(profile.topology["verification2_policy"], "history2_pytest_then_judge")

    def test_runpod_controller_allows_local_workspace_but_rejects_local_verifier(self) -> None:
        cfg = ExecutionConfig.from_env_and_mapping(
            {
                "execution_role": "runpod_controller",
                "workspace_backend": "local_docker",
                "verifier_backend": "remote_ssh_docker",
                "remote": {
                    "host": "10.0.0.5",
                    "user": "dyryu",
                    "repository_root": "/repo",
                    "job_root": "/jobs",
                },
            }
        )
        cfg.guard_cloud_local_docker()
        bad = ExecutionConfig.from_env_and_mapping(
            {"execution_role": "runpod_controller", "workspace_backend": "local_docker", "verifier_backend": "local_docker"}
        )
        with self.assertRaisesRegex(ConfigError, "must not run hidden verification locally"):
            bad.guard_cloud_local_docker()

    def test_direct_private_route_is_rejected_for_runpod_profile(self) -> None:
        raw = load_yaml(PROFILE)
        raw["execution"]["remote"]["route"] = "direct"
        raw["execution"]["remote"]["jump_hosts"] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "proxy_jump"):
                load_runpod_execution_profile(path)

    def test_server2_ssh_paths_are_rejected_for_runpod_secrets(self) -> None:
        raw = load_yaml(PROFILE)
        raw["execution"]["remote"]["identity_file"] = "/home/dyryu/.ssh/private"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "runtime materialized"):
                load_runpod_execution_profile(path)

    def test_remote_workspace_does_not_require_local_docker_without_remote_contact(self) -> None:
        with patch("putpocket_dataset_mining.runpod_execution.shutil.which", return_value=None), patch(
            "putpocket_dataset_mining.runpod_execution.SshRsyncTransport"
        ) as transport:
            result = run_combined_preflight(PROFILE, live_remote=False, live_workspace=False, import_checks=False)
        transport.assert_not_called()
        self.assertEqual(result["status"], "partial")
        self.assertIsNone(result["local"]["failure_class"])
        self.assertTrue(result["local"]["local_workspace_backend_ready"])
        self.assertFalse(result["local"]["local_docker_required"])
        self.assertFalse(result["server1"]["checked"])

    def test_static_preflight_cli_does_not_contact_remote_or_run_docker(self) -> None:
        with patch("putpocket_dataset_mining.runpod_execution.shutil.which", return_value="/usr/bin/docker"), patch(
            "putpocket_dataset_mining.runpod_execution.subprocess.run"
        ) as run, patch("putpocket_dataset_mining.runpod_execution.SshRsyncTransport") as transport:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(["distributed-preflight", "--config", str(PROFILE)])
        self.assertEqual(rc, 0)
        run.assert_not_called()
        transport.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["status"], "partial")
        self.assertTrue(payload["local"]["local_workspace_backend_ready"])
        self.assertFalse(payload["server1"]["checked"])

    def test_profile_does_not_persist_pod_or_container_identity(self) -> None:
        raw_text = PROFILE.read_text(encoding="utf-8")
        self.assertNotIn("RUNPOD_POD_ID", raw_text)
        self.assertNotIn("container_id", raw_text)
        self.assertNotIn("RUNPOD_POD_ID", raw_text)


class RunpodWorkflowTopologyTests(unittest.TestCase):
    def _scripted_config(self, tmp: Path) -> Path:
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
        cfg["execution"]["execution_role"] = "runpod_controller"
        cfg["execution"]["inference_host_role"] = "runpod"
        cfg["execution"]["inference_backend"] = "local_vllm"
        cfg["execution"]["verifier_host_role"] = "server1"
        path = tmp / "config.yaml"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return path

    def test_manual_init_writes_runpod_server1_topology_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "run"
            result = run_workflow(mode="manual", config_path=self._scripted_config(Path(td)), remote_config=None, sample_ids=["test_ClassEval_76"], run_root=root, run_uuid="u")
            self.assertIn("history1-infer-submit", result["next_command"])
            manifest = json.loads((root / "common" / "environment_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["inference_host_role"], "runpod")
        self.assertEqual(manifest["inference_backend"], "local_vllm")
        self.assertEqual(manifest["verifier_host_role"], "server1")
        self.assertEqual(manifest["verifier_backend"], "remote_ssh_docker")

    def test_pipeline_intervals_use_runpod_and_server1_host_roles(self) -> None:
        def fake_submit(*, cfg, run_root, sample_id, stage, mode, gpu_device, async_submit, engine=None):
            attempt = root / "pipeline" / "samples" / sample_id / "pipeline_attempt" / "verification" / stage
            attempt.mkdir(parents=True, exist_ok=True)
            receipt = {"job_id": f"{sample_id}-{stage}", "verification_policy": "history2_pytest_then_judge" if stage == "history2" else "history1_pytest_only"}
            (attempt / "submission_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
            return {"job_id": receipt["job_id"], "submit_end_perf": 10.0, "receipt": str(attempt / "submission_receipt.json")}

        def fake_retrieve(*, cfg, run_root, sample_id, stage, mode):
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
                verifier_host="server1",
                remote_result={"judge": {"executed": stage == "history2", "decision": "pass"}},
            )

        with tempfile.TemporaryDirectory() as td, patch("putpocket_dataset_mining.distributed_workflow._mode_engine", return_value=object()), patch(
            "putpocket_dataset_mining.distributed_workflow._run_history_and_submit",
            side_effect=fake_submit,
        ), patch("putpocket_dataset_mining.distributed_workflow._remote_status", return_value={"status": "passed"}), patch(
            "putpocket_dataset_mining.distributed_workflow._retrieve_stage",
            side_effect=fake_retrieve,
        ):
            root = Path(td) / "run"
            run_workflow(
                mode="pipeline",
                config_path=self._scripted_config(Path(td)),
                remote_config=None,
                sample_ids=["test_ClassEval_76"],
                run_root=root,
                run_uuid="u",
            )
            intervals = json.loads((root / "pipeline" / "intervals.json").read_text(encoding="utf-8"))["intervals"]
        self.assertIn("runpod", {row["host"] for row in intervals})
        self.assertIn("server1", {row["host"] for row in intervals})
        self.assertIn("local_vllm_gpu", {row["resource"] for row in intervals})


if __name__ == "__main__":
    unittest.main()
