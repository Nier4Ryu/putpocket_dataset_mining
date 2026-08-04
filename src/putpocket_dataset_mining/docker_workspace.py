from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import CONTAINER_HOME, DEFAULT_DOCKERFILE, DEFAULT_DOCKER_IMAGE, DOCKER_WORKSPACE_ROOT
from .errors import InfraError
from .execution_config import DockerBackend, ExecutionConfig
from .fs import host_uid_gid, safe_relative_path
from .ssh_transport import SshRsyncTransport, validate_safe_id


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timeout: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timeout": self.timeout,
        }


class DockerImageManager:
    def __init__(self, image: str, dockerfile: Path) -> None:
        self.image = image
        self.dockerfile = dockerfile

    @classmethod
    def from_default(cls) -> "DockerImageManager":
        return cls(DEFAULT_DOCKER_IMAGE, DEFAULT_DOCKERFILE)

    @staticmethod
    def _docker_cmd() -> str:
        ExecutionConfig.from_env_and_mapping().guard_cloud_local_docker()
        docker = shutil.which("docker")
        if docker is None:
            raise InfraError("docker command is missing. Install Docker or rerun with --skip-docker where supported.")
        return docker

    def image_exists(self) -> bool:
        docker = self._docker_cmd()
        result = subprocess.run(
            [docker, "image", "inspect", self.image],
            text=True,
            capture_output=True,
        )
        return result.returncode == 0

    def ensure_image(self, build_if_missing: bool = True, timeout_sec: int = 900) -> None:
        if self.image_exists():
            return
        if not build_if_missing:
            raise InfraError(f"Docker image is missing and build_if_missing=false: {self.image}")
        if not self.dockerfile.exists():
            raise InfraError(f"Dockerfile is missing: {self.dockerfile}")
        python_build_jobs = os.environ.get("PUTPOCKET_BUILD_THREADS", "16")
        docker = self._docker_cmd()
        result = subprocess.run(
            [
                docker,
                "build",
                "--build-arg",
                f"PYTHON_BUILD_JOBS={python_build_jobs}",
                "-t",
                self.image,
                "-f",
                str(self.dockerfile),
                str(self.dockerfile.parents[1]),
            ],
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
        if result.returncode != 0:
            raise InfraError(f"Docker image build failed: {result.stderr[-4000:]}")


class DockerWorkspace:
    """Episode workspace container backed by a host-mounted directory."""

    def __init__(
        self,
        host_workspace: Path,
        image: str = DEFAULT_DOCKER_IMAGE,
        cpus: int | float = 8,
        memory: str = "8g",
        name: str | None = None,
        startup_timeout_sec: int = 120,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        self.host_workspace = Path(host_workspace)
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.name = name or f"putpocket-episode-{uuid.uuid4().hex[:12]}"
        self.startup_timeout_sec = startup_timeout_sec
        self.execution_config = execution_config or ExecutionConfig.from_env_and_mapping()
        self._started = False

    def __enter__(self) -> "DockerWorkspace":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop(remove=True)

    def start(self) -> None:
        self.execution_config.guard_cloud_local_docker()
        self.host_workspace.mkdir(parents=True, exist_ok=True)
        uid, gid = host_uid_gid()
        cmd = [
            "docker",
            "run",
            "-d",
            "--name",
            self.name,
            "--network",
            "none",
            "--cpus",
            str(self.cpus),
            "--memory",
            self.memory,
            "--user",
            f"{uid}:{gid}",
            "-e",
            f"HOME={CONTAINER_HOME}",
            "-v",
            f"{self.host_workspace}:{DOCKER_WORKSPACE_ROOT}:rw",
            "-w",
            DOCKER_WORKSPACE_ROOT,
            self.image,
            "sleep",
            "infinity",
        ]
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=self.startup_timeout_sec)
        if result.returncode != 0:
            raise InfraError(f"Docker workspace start failed: {result.stderr.strip()}")
        self._started = True

    def stop(self, remove: bool = True) -> None:
        if not self._started:
            return
        subprocess.run(["docker", "stop", self.name], text=True, capture_output=True, timeout=30)
        if remove:
            subprocess.run(["docker", "rm", "-f", self.name], text=True, capture_output=True, timeout=30)
        self._started = False

    def exec(self, command: str, timeout_sec: int = 120) -> CommandResult:
        cmd = ["docker", "exec", "-w", DOCKER_WORKSPACE_ROOT, self.name, "bash", "-lc", command]
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec)
            return CommandResult(cmd, result.returncode, result.stdout, result.stderr)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(cmd, 124, exc.stdout or "", exc.stderr or "", timeout=True)

    def read_file(self, path: str) -> str:
        rel = safe_relative_path(path)
        target = self.host_workspace / rel
        if not target.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        rel = safe_relative_path(path)
        target = self.host_workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def list_files(self, path: str = ".", recursive: bool = False) -> list[str]:
        rel = safe_relative_path(path)
        root = (self.host_workspace / rel).resolve()
        if not root.exists():
            return []
        if root.is_file():
            return [str(rel)]
        iterator = root.rglob("*") if recursive else root.iterdir()
        rows = []
        for item in sorted(iterator):
            rows.append(str(item.relative_to(self.host_workspace)))
        return rows

    def search_files(self, path: str, regex: str, file_pattern: str | None = None, timeout_sec: int = 120) -> CommandResult:
        rel = safe_relative_path(path)
        pattern_arg = ""
        if file_pattern:
            encoded_pattern = base64.b64encode(file_pattern.encode("utf-8")).decode("ascii")
            pattern_arg = f" --glob \"$(echo {encoded_pattern} | base64 -d)\""
        encoded_regex = base64.b64encode(regex.encode("utf-8")).decode("ascii")
        command = f"rg --line-number --color never{pattern_arg} \"$(echo {encoded_regex} | base64 -d)\" {rel}"
        return self.exec(command, timeout_sec=timeout_sec)


class RemoteDockerWorkspace:
    """Remote episode workspace backed by the restricted SSH Docker wrapper.

    The implementation follows the public methods used by HeadlessClineRuntime.
    File reads/writes operate on the controller-side mirror and are synced for
    command execution so the existing Cline tool semantics remain unchanged.
    """

    def __init__(
        self,
        host_workspace: Path,
        image: str = DEFAULT_DOCKER_IMAGE,
        cpus: int | float = 8,
        memory: str = "8g",
        name: str | None = None,
        startup_timeout_sec: int = 120,
        execution_config: ExecutionConfig | None = None,
    ) -> None:
        self.host_workspace = Path(host_workspace)
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.name = validate_safe_id(name or f"putpocket-remote-{uuid.uuid4().hex[:12]}", "workspace_session")
        self.startup_timeout_sec = startup_timeout_sec
        self.execution_config = execution_config or ExecutionConfig.from_env_and_mapping()
        self.transport = SshRsyncTransport(self.execution_config.remote)
        self._started = False

    def __enter__(self) -> "RemoteDockerWorkspace":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop(remove=True)

    def start(self) -> None:
        if self.execution_config.workspace_backend != DockerBackend.REMOTE_SSH_DOCKER:
            raise InfraError("RemoteDockerWorkspace requires workspace_backend=remote_ssh_docker.")
        self.host_workspace.mkdir(parents=True, exist_ok=True)
        preflight = self.transport.lightweight_preflight(self.image)
        if not preflight.docker_ok:
            raise InfraError(f"REMOTE_DOCKER_PREFLIGHT_FAILED: {preflight.detail or preflight.error_class}")
        result = self.transport.run_wrapper(
            "workspace-create",
            {
                "session_id": self.name,
                "docker_image": self.image,
                "cpus": self.cpus,
                "memory": self.memory,
            },
            timeout_sec=self.startup_timeout_sec,
        )
        if result.returncode != 0:
            raise InfraError(f"Remote Docker workspace start failed: {(result.stderr or result.stdout).strip()}")
        sync = self.transport.rsync_to_remote(self.host_workspace, f"sessions/{self.name}/workspace/")
        if sync.returncode != 0:
            raise InfraError(f"Remote workspace initial sync failed: {sync.stderr.strip()}")
        self._started = True

    def stop(self, remove: bool = True) -> None:
        if not self._started:
            return
        if remove:
            self.transport.run_wrapper("workspace-destroy", {"session_id": self.name}, timeout_sec=30)
        self._started = False

    def _push_workspace(self) -> None:
        result = self.transport.rsync_to_remote(self.host_workspace, f"sessions/{self.name}/workspace/")
        if result.returncode != 0:
            raise InfraError(f"Remote workspace sync failed: {result.stderr.strip()}")

    def _pull_workspace(self) -> None:
        result = self.transport.rsync_from_remote(f"sessions/{self.name}/workspace/", self.host_workspace)
        if result.returncode != 0:
            raise InfraError(f"Remote workspace result sync failed: {result.stderr.strip()}")

    def exec(self, command: str, timeout_sec: int = 120) -> CommandResult:
        self._push_workspace()
        result = self.transport.run_wrapper(
            "workspace-exec",
            {
                "session_id": self.name,
                "command": command,
                "docker_image": self.image,
                "cpus": self.cpus,
                "memory": self.memory,
                "timeout_sec": timeout_sec,
            },
            timeout_sec=timeout_sec + 30,
        )
        self._pull_workspace()
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return CommandResult(result.command, result.returncode, result.stdout, result.stderr, result.timeout)
        return CommandResult(
            result.command,
            int(payload.get("returncode", result.returncode)),
            str(payload.get("stdout", "")),
            str(payload.get("stderr", result.stderr)),
            bool(payload.get("timeout", result.timeout)),
        )

    def read_file(self, path: str) -> str:
        rel = safe_relative_path(path)
        target = self.host_workspace / rel
        if not target.exists():
            raise FileNotFoundError(f"File does not exist: {path}")
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        rel = safe_relative_path(path)
        target = self.host_workspace / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def list_files(self, path: str = ".", recursive: bool = False) -> list[str]:
        rel = safe_relative_path(path)
        root = (self.host_workspace / rel).resolve()
        if not root.exists():
            return []
        if root.is_file():
            return [str(rel)]
        iterator = root.rglob("*") if recursive else root.iterdir()
        return [str(item.relative_to(self.host_workspace)) for item in sorted(iterator)]

    def search_files(self, path: str, regex: str, file_pattern: str | None = None, timeout_sec: int = 120) -> CommandResult:
        rel = safe_relative_path(path)
        pattern_arg = ""
        if file_pattern:
            encoded_pattern = base64.b64encode(file_pattern.encode("utf-8")).decode("ascii")
            pattern_arg = f" --glob \"$(echo {encoded_pattern} | base64 -d)\""
        encoded_regex = base64.b64encode(regex.encode("utf-8")).decode("ascii")
        return self.exec(f"rg --line-number --color never{pattern_arg} \"$(echo {encoded_regex} | base64 -d)\" {rel}", timeout_sec=timeout_sec)

    def apply_unified_diff(self, diff: str, timeout_sec: int = 120) -> CommandResult:
        encoded = base64.b64encode(diff.encode("utf-8")).decode("ascii")
        return self.exec(f"printf '%s' {json.dumps(encoded)} | base64 -d | patch -p0", timeout_sec=timeout_sec)


def workspace_from_execution_config(
    *,
    host_workspace: Path,
    image: str,
    cpus: int | float,
    memory: str,
    startup_timeout_sec: int,
    execution_config: ExecutionConfig | None = None,
) -> DockerWorkspace | RemoteDockerWorkspace:
    execution_config = execution_config or ExecutionConfig.from_env_and_mapping()
    execution_config.guard_cloud_local_docker()
    if execution_config.workspace_backend == DockerBackend.REMOTE_SSH_DOCKER:
        return RemoteDockerWorkspace(
            host_workspace=host_workspace,
            image=image,
            cpus=cpus,
            memory=memory,
            startup_timeout_sec=startup_timeout_sec,
            execution_config=execution_config,
        )
    if execution_config.workspace_backend == DockerBackend.LOCAL_DOCKER:
        return DockerWorkspace(
            host_workspace=host_workspace,
            image=image,
            cpus=cpus,
            memory=memory,
            startup_timeout_sec=startup_timeout_sec,
            execution_config=execution_config,
        )
    raise InfraError("EVALUATION_BLOCKED_NO_DOCKER_BACKEND: workspace Docker backend is disabled.")

    def apply_unified_diff(self, diff: str, timeout_sec: int = 120) -> CommandResult:
        encoded = base64.b64encode(diff.encode("utf-8")).decode("ascii")
        command = f"printf '%s' {json.dumps(encoded)} | base64 -d | patch -p0"
        return self.exec(command, timeout_sec=timeout_sec)


def run_verifier_container(
    workspace: Path,
    image: str,
    command: str,
    cpus: int | float,
    memory: str,
    timeout_sec: int,
) -> CommandResult:
    ExecutionConfig.from_env_and_mapping().guard_cloud_local_docker()
    uid, gid = host_uid_gid()
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        str(cpus),
        "--memory",
        memory,
        "--user",
        f"{uid}:{gid}",
        "-e",
        f"HOME={CONTAINER_HOME}",
        "-e",
        f"PYTHONPATH={DOCKER_WORKSPACE_ROOT}",
        "-v",
        f"{Path(workspace)}:{DOCKER_WORKSPACE_ROOT}:rw",
        "-w",
        DOCKER_WORKSPACE_ROOT,
        image,
        "bash",
        "-lc",
        command,
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_sec)
        return CommandResult(cmd, result.returncode, result.stdout, result.stderr)
    except subprocess.TimeoutExpired as exc:
        return CommandResult(cmd, 124, exc.stdout or "", exc.stderr or "", timeout=True)


def snapshot_workspace(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
