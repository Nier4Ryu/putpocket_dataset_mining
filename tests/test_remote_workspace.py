from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining import remote_workspace
from putpocket_dataset_mining.docker_workspace import RemoteDockerWorkspace
from putpocket_dataset_mining.execution_config import DockerBackend, ExecutionConfig


class RemoteWorkspaceWorkerTests(unittest.TestCase):
    def test_client_and_wrapper_use_same_session_layout(self) -> None:
        cfg = ExecutionConfig.from_env_and_mapping({
            "execution_role": "runpod_controller", "workspace_backend": "ssh_remote_docker", "verifier_backend": "remote_ssh_docker",
            "remote": {"host": "b", "user": "u", "repository_root": "/repo", "job_root": "/verify"},
            "workspace_remote": {"host": "b", "user": "u", "repository_root": "/repo", "job_root": "/workspace", "wrapper": "/repo/bin/putpocket-remote-workspace"},
        })
        with tempfile.TemporaryDirectory() as td:
            workspace = RemoteDockerWorkspace(Path(td), name="run.sample", execution_config=cfg)
            with patch.object(workspace.transport, "rsync_to_remote") as push, patch.object(workspace.transport, "rsync_from_remote") as pull:
                push.return_value = type("R", (), {"returncode": 0, "stderr": ""})()
                pull.return_value = type("R", (), {"returncode": 0, "stderr": ""})()
                workspace._push_workspace()
                workspace._pull_workspace()
        self.assertEqual(push.call_args.args[1], "run.sample/workspace/")
        self.assertEqual(pull.call_args.args[0], "run.sample/workspace/")
    def test_exec_places_untrusted_command_only_after_docker_exec(self) -> None:
        payload = {"session_id": "run.sample.episode", "command": "touch marker", "timeout_sec": 3}
        with patch.object(remote_workspace, "_docker") as docker:
            docker.return_value = type("R", (), {"returncode": 0, "stdout": "ok", "stderr": ""})()
            rc = remote_workspace.execute(payload)
        self.assertEqual(rc, 0)
        args = docker.call_args.args
        self.assertEqual(args[:2], ("exec", "-w"))
        self.assertIn("bash", args)
        self.assertEqual(args[-1], "touch marker")

    def test_snapshot_preserves_h1_state_for_h2(self) -> None:
        with tempfile.TemporaryDirectory() as td, patch.dict("os.environ", {"SR_REMOTE_WORKSPACE_ROOT": td}):
            ws = Path(td) / "episode" / "workspace"
            ws.mkdir(parents=True)
            (ws / "x.txt").write_text("h1", encoding="utf-8")
            with patch("sys.stdout"):
                remote_workspace.snapshot({"session_id": "episode", "snapshot_id": "after-h1"})
            self.assertEqual((Path(td) / "episode" / "snapshots" / "after-h1" / "x.txt").read_text(), "h1")
            (ws / "x.txt").write_text("h1+h2", encoding="utf-8")
            self.assertEqual((ws / "x.txt").read_text(), "h1+h2")

    def test_workspace_uses_separate_remote_wrapper_and_root(self) -> None:
        cfg = ExecutionConfig.from_env_and_mapping({
            "execution_role": "runpod_controller",
            "workspace_backend": "ssh_remote_docker",
            "verifier_backend": "remote_ssh_docker",
            "remote": {"host": "b", "user": "u", "repository_root": "/repo", "job_root": "/verify"},
            "workspace_remote": {"host": "b", "user": "u", "repository_root": "/repo", "job_root": "/workspace", "wrapper": "/repo/bin/putpocket-remote-workspace"},
        })
        self.assertEqual(cfg.workspace_backend, DockerBackend.SSH_REMOTE_DOCKER)
        with tempfile.TemporaryDirectory() as td:
            workspace = RemoteDockerWorkspace(Path(td), execution_config=cfg)
        self.assertEqual(workspace.transport.remote.job_root, "/workspace")
        self.assertIn("remote-workspace", workspace.transport.wrapper)


if __name__ == "__main__":
    unittest.main()
