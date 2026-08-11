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
from .execution_config import DEFAULT_VERIFIER_TIMEOUT_SEC, DockerBackend, ExecutionConfig
from .remote_verifier.manifest import result_sha256
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
    timeout_sec: int
    workspace: str
    backend: str = "local_docker"
    remote_job_id: str | None = None
    remote_protocol: str | None = None
    verifier_host: str | None = None
    docker_image_id: str | None = None
    workspace_sha256: str | None = None
    result_sha256: str | None = None
    verifier_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "backend": self.backend,
            "remote_job_id": self.remote_job_id,
            "remote_protocol": self.remote_protocol,
            "verifier_host": self.verifier_host,
            "docker_image_id": self.docker_image_id,
            "workspace_sha256": self.workspace_sha256,
            "result_sha256": self.result_sha256,
            "verifier_revision": self.verifier_revision,
            "checks": [
                {
                    "name": "hidden_mbpp_pytest",
                    "passed": self.passed,
                    "command": "pytest -q tests/test_solution.py",
                    "returncode": self.returncode,
                    "timeout": self.timeout,
                    "timeout_sec": self.timeout_sec,
                }
            ],
            "final_status": self.final_status,
            "failure_class": self.failure_class,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timeout_sec": self.timeout_sec,
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
            timeout_sec=timeout_sec,
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
            "protocol_version": "sr-remote-verifier-v1",
            "run_id": attempt_dir.parents[2].name if len(attempt_dir.parents) > 2 else "unknown",
            "job_id": job_id,
            "dataset_version": task.split,
            "sample_id": task.sample_id,
            "source_task_id": task.task_id,
            "verifier_stage": stage,
            "workspace_sha256": workspace_sha,
            "test_command": test_command,
            "timeout_sec": timeout_sec,
            "docker_image": docker_image,
            "dockerfile": self.execution_config.remote.dockerfile,
            "network_disabled": True,
            "controller_revision": _git_head_or_unknown(),
            "canonical_dataset_sha256": None,
            "created_at_kst": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cpus": cpus,
            "memory": memory,
        }
        self.execution_config.validate_remote_timeout_budget(timeout_sec)
        (job_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        remote_rel = f"incoming/{job_id}.partial/workspace/"
        sync = self.transport.rsync_to_remote(verifier_workspace, remote_rel)
        if sync.returncode != 0:
            raise InfraError(f"Remote verifier workspace transfer failed: {sync.stderr.strip()}")
        manifest_remote = self.transport.rsync_to_remote(job_dir / "manifest.json", f"incoming/{job_id}.partial/manifest.json")
        if manifest_remote.returncode != 0:
            raise InfraError(f"Remote verifier manifest transfer failed: {manifest_remote.stderr.strip()}")
        promoted = self.transport.run_wrapper("promote", timeout_sec=30, extra_args=["--job-id", job_id])
        if promoted.returncode != 0:
            raise InfraError(f"Remote verifier promote failed: {(promoted.stderr or promoted.stdout).strip()}")
        result = self.transport.run_wrapper(
            "verify",
            None,
            timeout_sec=timeout_sec + self.execution_config.verifier_remote_grace_sec,
            extra_args=["--job-id", job_id],
        )
        if result.returncode != 0:
            raise InfraError(f"Remote verifier failed: {(result.stderr or result.stdout).strip()}")
        status_result = self.transport.run_wrapper("result-status", timeout_sec=30, extra_args=["--job-id", job_id])
        if status_result.returncode != 0:
            raise InfraError(f"Remote verifier result-status failed: {(status_result.stderr or status_result.stdout).strip()}")
        payload = json.loads(status_result.stdout or result.stdout or "{}")
        expected_result_sha = payload.get("result_sha256")
        if expected_result_sha and expected_result_sha != result_sha256(payload):
            raise InfraError("REMOTE_RESULT_INTEGRITY_FAILED: remote verifier result checksum mismatch.")
        remote_result = job_dir / "result.json"
        remote_result.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        (job_dir.parent / "remote_result.json").write_text(
            json.dumps({"backend": "remote_ssh_docker", **payload}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        stdout = str(payload.get("stdout", ""))
        stderr = str(payload.get("stderr", ""))
        if not stdout and payload.get("stdout_path"):
            self.transport.rsync_from_remote(f"completed/{job_id}/{payload['stdout_path']}", job_dir / "stdout.txt")
            stdout = (job_dir / "stdout.txt").read_text(encoding="utf-8") if (job_dir / "stdout.txt").exists() else ""
        elif not stdout and payload.get("stdout_file"):
            self.transport.rsync_from_remote(f"completed/{job_id}/{payload['stdout_file']}", job_dir / "stdout.txt")
            stdout = (job_dir / "stdout.txt").read_text(encoding="utf-8") if (job_dir / "stdout.txt").exists() else ""
        if not stderr and payload.get("stderr_path"):
            self.transport.rsync_from_remote(f"completed/{job_id}/{payload['stderr_path']}", job_dir / "stderr.txt")
            stderr = (job_dir / "stderr.txt").read_text(encoding="utf-8") if (job_dir / "stderr.txt").exists() else ""
        elif not stderr and payload.get("stderr_file"):
            self.transport.rsync_from_remote(f"completed/{job_id}/{payload['stderr_file']}", job_dir / "stderr.txt")
            stderr = (job_dir / "stderr.txt").read_text(encoding="utf-8") if (job_dir / "stderr.txt").exists() else ""
        passed = bool(payload.get("verifier_passed"))
        timeout = bool(payload.get("timeout") or payload.get("timed_out"))
        failure_class = None if passed else f"{stage}.unit_test.timeout" if timeout else f"{stage}.unit_test.failed"
        return VerificationResult(
            stage=stage,
            passed=passed,
            final_status=str(payload.get("status") or ("passed" if passed else "failed")),
            failure_class=failure_class if payload.get("status") != "infra_failed" else "infra.remote_verifier_failed",
            returncode=int(payload.get("process_exit_code") if payload.get("process_exit_code") is not None else 1),
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
            timeout_sec=int(payload.get("timeout_sec") if payload.get("timeout_sec") is not None else timeout_sec),
            workspace=str(verifier_workspace),
            backend="remote_ssh_docker",
            remote_job_id=str(payload.get("job_id") or job_id),
            remote_protocol=str(payload.get("protocol_version") or manifest["protocol_version"]),
            verifier_host=str(payload.get("verifier_host") or ""),
            docker_image_id=str(payload.get("docker_image_id") or ""),
            workspace_sha256=str(payload.get("workspace_sha256") or workspace_sha),
            result_sha256=str(payload.get("result_sha256") or ""),
            verifier_revision=str(payload.get("verifier_revision") or ""),
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
        timeout_sec: int | None = None,
        transport: VerifierTransport | None = None,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        self.attempt_dir = attempt_dir
        self.docker_image = docker_image
        self.cpus = cpus
        self.memory = memory
        self.test_command = test_command
        self.execution_config = execution_config or ExecutionConfig.from_env_and_mapping()
        self.timeout_sec = int(timeout_sec if timeout_sec is not None else self.execution_config.verifier_timeout_sec or DEFAULT_VERIFIER_TIMEOUT_SEC)
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
