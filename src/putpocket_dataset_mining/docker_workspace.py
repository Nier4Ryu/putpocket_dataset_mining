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
from .fs import host_uid_gid, safe_relative_path


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

    def image_exists(self) -> bool:
        result = subprocess.run(
            ["docker", "image", "inspect", self.image],
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
        python_build_jobs = os.environ.get("PUTPOCKET_BUILD_THREADS", "32")
        result = subprocess.run(
            [
                "docker",
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
    ) -> None:
        self.host_workspace = Path(host_workspace)
        self.image = image
        self.cpus = cpus
        self.memory = memory
        self.name = name or f"putpocket-episode-{uuid.uuid4().hex[:12]}"
        self.startup_timeout_sec = startup_timeout_sec
        self._started = False

    def __enter__(self) -> "DockerWorkspace":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.stop(remove=True)

    def start(self) -> None:
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
