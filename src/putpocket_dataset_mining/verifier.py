from __future__ import annotations

import json
import shutil
import time
import uuid
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import SourceTask, verifier_materializer_for_task
from .docker_workspace import run_verifier_container
from .errors import InfraError
from .execution_config import DockerBackend, ExecutionConfig
from .ssh_transport import SshRsyncTransport, validate_safe_id


@dataclass(frozen=True)
class VerificationResult:
    stage: str
    passed: bool
    final_status: str
    failure_class: str | None
    returncode: int
    stdout: str
    stderr: str
    timeout: bool
    workspace: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "checks": [
                {
                    "name": "hidden_mbpp_pytest",
                    "passed": self.passed,
                    "command": "pytest -q tests/test_solution.py",
                    "returncode": self.returncode,
                    "timeout": self.timeout,
                }
            ],
            "final_status": self.final_status,
            "failure_class": self.failure_class,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "workspace": self.workspace,
        }


class VerifierTransport(ABC):
    @abstractmethod
    def run(
        self,
        *,
        stage: str,
        verifier_workspace: Path,
        task: SourceTask,
        docker_image: str,
        test_command: str,
        cpus: int | float,
        memory: str,
        timeout_sec: int,
        attempt_dir: Path,
    ) -> VerificationResult:
        raise NotImplementedError


class LocalDockerVerifierTransport(VerifierTransport):
    def run(
        self,
        *,
        stage: str,
        verifier_workspace: Path,
        task: SourceTask,
        docker_image: str,
        test_command: str,
        cpus: int | float,
        memory: str,
        timeout_sec: int,
        attempt_dir: Path,
    ) -> VerificationResult:
        result = run_verifier_container(
            workspace=verifier_workspace,
            image=docker_image,
            command=test_command,
            cpus=cpus,
            memory=memory,
            timeout_sec=timeout_sec,
        )
        passed = result.returncode == 0
        failure_class = None if passed else f"{stage}.unit_test.timeout" if result.timeout else f"{stage}.unit_test.failed"
        return VerificationResult(
            stage=stage,
            passed=passed,
            final_status="passed" if passed else "failed",
            failure_class=failure_class,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timeout=result.timeout,
            workspace=str(verifier_workspace),
        )


class SshRsyncVerifierTransport(VerifierTransport):
    def __init__(self, execution_config: ExecutionConfig | None = None) -> None:
        self.execution_config = execution_config or ExecutionConfig.from_env_and_mapping()
        self.transport = SshRsyncTransport(self.execution_config.remote)

    def run(
        self,
        *,
        stage: str,
        verifier_workspace: Path,
        task: SourceTask,
        docker_image: str,
        test_command: str,
        cpus: int | float,
        memory: str,
        timeout_sec: int,
        attempt_dir: Path,
    ) -> VerificationResult:
        job_id = validate_safe_id(f"{task.sample_id}-{stage}-{uuid.uuid4().hex[:10]}", "job_id")
        workspace_sha = _sha256_tree(verifier_workspace)
        job_dir = attempt_dir / "verification" / stage / "remote_job"
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "protocol_version": "sr-remote-docker-v1",
            "run_id": attempt_dir.parents[2].name if len(attempt_dir.parents) > 2 else "unknown",
            "job_id": job_id,
            "dataset_version": task.split,
            "sample_id": task.sample_id,
            "source_task_id": task.task_id,
            "verifier_stage": stage,
            "workspace_sha256": workspace_sha,
            "timeout_sec": timeout_sec,
            "docker_image": docker_image,
            "network_disabled": True,
            "controller_revision": _git_head_or_unknown(),
            "canonical_dataset_sha256": None,
            "created_at": time.time(),
        }
        (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        remote_rel = f"jobs/{job_id}/workspace/"
        sync = self.transport.rsync_to_remote(verifier_workspace, remote_rel)
        if sync.returncode != 0:
            raise InfraError(f"Remote verifier workspace transfer failed: {sync.stderr.strip()}")
        manifest_remote = self.transport.rsync_to_remote(job_dir / "manifest.json", f"jobs/{job_id}/manifest.json")
        if manifest_remote.returncode != 0:
            raise InfraError(f"Remote verifier manifest transfer failed: {manifest_remote.stderr.strip()}")
        result = self.transport.run_wrapper(
            "verify",
            {
                **manifest,
                "test_command": test_command,
                "cpus": cpus,
                "memory": memory,
            },
            timeout_sec=timeout_sec + 60,
        )
        if result.returncode != 0:
            raise InfraError(f"Remote verifier failed: {(result.stderr or result.stdout).strip()}")
        payload = json.loads(result.stdout or "{}")
        remote_result = job_dir / "result.json"
        remote_result.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        stdout = str(payload.get("stdout", ""))
        stderr = str(payload.get("stderr", ""))
        if not stdout and payload.get("stdout_path"):
            self.transport.rsync_from_remote(f"jobs/{job_id}/{payload['stdout_path']}", job_dir / "stdout.txt")
            stdout = (job_dir / "stdout.txt").read_text(encoding="utf-8") if (job_dir / "stdout.txt").exists() else ""
        if not stderr and payload.get("stderr_path"):
            self.transport.rsync_from_remote(f"jobs/{job_id}/{payload['stderr_path']}", job_dir / "stderr.txt")
            stderr = (job_dir / "stderr.txt").read_text(encoding="utf-8") if (job_dir / "stderr.txt").exists() else ""
        passed = bool(payload.get("verifier_passed"))
        timeout = bool(payload.get("timeout"))
        failure_class = None if passed else f"{stage}.unit_test.timeout" if timeout else f"{stage}.unit_test.failed"
        return VerificationResult(
            stage=stage,
            passed=passed,
            final_status="passed" if passed else "failed",
            failure_class=failure_class,
            returncode=int(payload.get("process_exit_code", 1)),
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            workspace=str(verifier_workspace),
        )


def verifier_transport_from_execution_config(config: ExecutionConfig | None = None) -> VerifierTransport:
    config = config or ExecutionConfig.from_env_and_mapping()
    config.guard_cloud_local_docker()
    if config.verifier_backend == DockerBackend.LOCAL_DOCKER:
        return LocalDockerVerifierTransport()
    if config.verifier_backend == DockerBackend.REMOTE_SSH_DOCKER:
        return SshRsyncVerifierTransport(config)
    raise InfraError("EVALUATION_BLOCKED_NO_DOCKER_BACKEND: verifier Docker backend is disabled.")


class HiddenVerifier:
    def __init__(
        self,
        attempt_dir: Path,
        docker_image: str,
        cpus: int | float = 8,
        memory: str = "8g",
        test_command: str = "pytest -q tests/test_solution.py",
        timeout_sec: int = 120,
        transport: VerifierTransport | None = None,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        self.attempt_dir = attempt_dir
        self.docker_image = docker_image
        self.cpus = cpus
        self.memory = memory
        self.test_command = test_command
        self.timeout_sec = timeout_sec
        self.execution_config = execution_config or ExecutionConfig.from_env_and_mapping()
        self.transport = transport

    def verify(self, stage: str, snapshot_dir: Path, task: SourceTask) -> VerificationResult:
        self.execution_config.guard_cloud_local_docker()
        verification_dir = self.attempt_dir / "verification" / stage
        verification_dir.mkdir(parents=True, exist_ok=True)
        verifier_workspace = verification_dir / "workspace"
        if verifier_workspace.exists():
            shutil.rmtree(verifier_workspace)
        shutil.copytree(snapshot_dir, verifier_workspace, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        verifier_materializer_for_task(task).write(task, verifier_workspace)
        if (snapshot_dir / "tests").exists():
            raise InfraError("Hidden tests leaked into agent-visible workspace snapshot before verifier materialization.")
        transport = self.transport or verifier_transport_from_execution_config(self.execution_config)
        verification = transport.run(
            stage=stage,
            verifier_workspace=verifier_workspace,
            task=task,
            docker_image=self.docker_image,
            test_command=self.test_command,
            cpus=self.cpus,
            memory=self.memory,
            timeout_sec=self.timeout_sec,
            attempt_dir=self.attempt_dir,
        )
        (verification_dir / "checklist.json").write_text(
            json.dumps(verification.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (verification_dir / "stdout.txt").write_text(verification.stdout, encoding="utf-8")
        (verification_dir / "stderr.txt").write_text(verification.stderr, encoding="utf-8")
        return verification


def _sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _git_head_or_unknown() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"
