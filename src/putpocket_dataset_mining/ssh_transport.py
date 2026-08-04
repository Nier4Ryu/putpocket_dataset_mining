from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError, InfraError
from .execution_config import RemoteDockerConfig

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class TransportResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    timeout: bool = False

    def json_stdout(self) -> dict[str, Any]:
        return json.loads(self.stdout or "{}")


@dataclass(frozen=True)
class RemotePreflightResult:
    status: str
    ssh_ok: bool
    wrapper_ok: bool = False
    rsync_ok: bool = False
    docker_ok: bool = False
    staging_root_ok: bool = False
    image_ok: bool | None = None
    error_class: str | None = None
    detail: str | None = None


def validate_safe_id(value: str, field_name: str = "id") -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise ConfigError(f"Invalid {field_name}: {value!r}. Use alphanumeric, '.', '_' or '-' only.")
    return value


def validate_relative_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise ConfigError(f"Unsafe relative path: {path}")
    return p


class SshRsyncTransport:
    def __init__(
        self,
        remote: RemoteDockerConfig,
        *,
        connect_timeout_sec: int = 10,
        command_timeout_sec: int = 120,
        wrapper: str = "python -m putpocket_dataset_mining.remote_worker",
    ) -> None:
        remote.require_complete()
        self.remote = remote
        self.connect_timeout_sec = connect_timeout_sec
        self.command_timeout_sec = command_timeout_sec
        self.wrapper = wrapper

    @property
    def target(self) -> str:
        assert self.remote.host and self.remote.user
        return f"{self.remote.user}@{self.remote.host}"

    def ssh_base_argv(self) -> list[str]:
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout_sec}",
            "-p",
            str(self.remote.port),
        ]
        if self.remote.identity_file:
            argv.extend(["-i", self.remote.identity_file])
        if self.remote.known_hosts_file:
            argv.extend(["-o", f"UserKnownHostsFile={self.remote.known_hosts_file}"])
        return argv

    def rsync_base_argv(self) -> list[str]:
        ssh_cmd = " ".join(self.ssh_base_argv())
        return ["rsync", "-a", "--partial", "--delay-updates", "-e", ssh_cmd]

    def run_wrapper(self, command: str, payload: dict[str, Any] | None = None, timeout_sec: int | None = None) -> TransportResult:
        validate_safe_id(command, "wrapper_command")
        remote_cmd = f"cd {self.remote.root} && {self.wrapper} {command}"
        argv = [*self.ssh_base_argv(), self.target, remote_cmd]
        try:
            result = subprocess.run(
                argv,
                input=json.dumps(payload or {}),
                text=True,
                capture_output=True,
                timeout=timeout_sec or self.command_timeout_sec,
            )
            return TransportResult(argv, result.returncode, result.stdout, result.stderr)
        except subprocess.TimeoutExpired as exc:
            return TransportResult(argv, 124, exc.stdout or "", exc.stderr or "", timeout=True)

    def rsync_to_remote(self, source: Path, remote_rel: str | Path, *, dry_run: bool = False) -> TransportResult:
        rel = validate_relative_path(remote_rel)
        assert self.remote.root
        destination = f"{self.target}:{self.remote.root}/{rel.as_posix()}"
        argv = [*self.rsync_base_argv()]
        if dry_run:
            argv.append("--dry-run")
        argv.extend([str(source), destination])
        result = subprocess.run(argv, text=True, capture_output=True)
        return TransportResult(argv, result.returncode, result.stdout, result.stderr)

    def rsync_from_remote(self, remote_rel: str | Path, destination: Path, *, dry_run: bool = False) -> TransportResult:
        rel = validate_relative_path(remote_rel)
        assert self.remote.root
        source = f"{self.target}:{self.remote.root}/{rel.as_posix()}"
        argv = [*self.rsync_base_argv()]
        if dry_run:
            argv.append("--dry-run")
        argv.extend([source, str(destination)])
        result = subprocess.run(argv, text=True, capture_output=True)
        return TransportResult(argv, result.returncode, result.stdout, result.stderr)

    def lightweight_preflight(self, docker_image: str | None = None) -> RemotePreflightResult:
        result = self.run_wrapper("preflight", {"docker_image": docker_image or self.remote.docker_image})
        if result.returncode != 0:
            return RemotePreflightResult(
                status="REMOTE_DOCKER_PREFLIGHT_FAILED",
                ssh_ok=result.returncode != 255,
                error_class="remote.preflight_failed",
                detail=(result.stderr or result.stdout)[-2000:],
            )
        try:
            data = result.json_stdout()
        except json.JSONDecodeError as exc:
            raise InfraError(f"Remote preflight returned invalid JSON: {exc}") from exc
        return RemotePreflightResult(
            status="REMOTE_DOCKER_PREFLIGHT_PASSED" if data.get("docker_ok") else "REMOTE_DOCKER_PREFLIGHT_FAILED",
            ssh_ok=True,
            wrapper_ok=bool(data.get("wrapper_ok")),
            rsync_ok=bool(data.get("rsync_ok")),
            docker_ok=bool(data.get("docker_ok")),
            staging_root_ok=bool(data.get("staging_root_ok")),
            image_ok=data.get("image_ok"),
            error_class=data.get("error_class"),
            detail=data.get("detail"),
        )
