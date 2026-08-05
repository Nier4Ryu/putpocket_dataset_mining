from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError, InfraError
from .execution_config import RemoteDockerConfig, RemoteRoute

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
SAFE_HOST_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
SAFE_USER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
SAFE_WRAPPER_RE = re.compile(r"^[A-Za-z0-9_./:-]+$")


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


class TransportErrorClass:
    SSH_CONNECTION_FAILED = "SSH_CONNECTION_FAILED"
    SSH_HOST_KEY_FAILED = "SSH_HOST_KEY_FAILED"
    SSH_COMMAND_FAILED = "SSH_COMMAND_FAILED"
    RSYNC_TRANSFER_FAILED = "RSYNC_TRANSFER_FAILED"
    REMOTE_PROTOCOL_MISMATCH = "REMOTE_PROTOCOL_MISMATCH"
    REMOTE_RESULT_MISSING = "REMOTE_RESULT_MISSING"
    REMOTE_RESULT_INTEGRITY_FAILED = "REMOTE_RESULT_INTEGRITY_FAILED"


def validate_safe_id(value: str, field_name: str = "id") -> str:
    if not SAFE_ID_RE.fullmatch(value):
        raise ConfigError(f"Invalid {field_name}: {value!r}. Use alphanumeric, '.', '_' or '-' only.")
    return value


def validate_relative_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise ConfigError(f"Unsafe relative path: {path}")
    return p


def validate_remote_path(path: str | Path) -> str:
    text = str(path)
    if any(ord(ch) < 32 for ch in text) or "\n" in text or "\r" in text:
        raise ConfigError("Unsafe remote path contains control characters.")
    if ".." in Path(text).parts:
        raise ConfigError(f"Unsafe remote path traversal: {path}")
    return text.rstrip("/")


def validate_user_host(user: str, host: str) -> None:
    if not SAFE_USER_RE.fullmatch(user):
        raise ConfigError(f"Unsafe SSH user: {user!r}")
    if not SAFE_HOST_RE.fullmatch(host):
        raise ConfigError(f"Unsafe SSH host: {host!r}")


def validate_wrapper_command(value: str) -> str:
    if not value or not SAFE_WRAPPER_RE.fullmatch(value):
        raise ConfigError(f"Unsafe remote wrapper command/path: {value!r}")
    return value


class SshRsyncTransport:
    def __init__(
        self,
        remote: RemoteDockerConfig,
        *,
        connect_timeout_sec: int | None = None,
        command_timeout_sec: int | None = None,
        wrapper: str | None = None,
    ) -> None:
        remote.require_complete()
        assert remote.host and remote.user
        validate_user_host(remote.user, remote.host)
        self.remote = remote
        self.connect_timeout_sec = connect_timeout_sec or remote.connection_timeout_sec
        self.command_timeout_sec = command_timeout_sec or remote.command_timeout_sec
        self.wrapper = validate_wrapper_command(wrapper or remote.wrapper)

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
            "IdentitiesOnly=yes",
            "-o",
            f"StrictHostKeyChecking={'yes' if self.remote.strict_host_key_checking else 'no'}",
            "-o",
            f"ConnectTimeout={self.connect_timeout_sec}",
            "-p",
            str(self.remote.port),
        ]
        if self.remote.identity_file:
            argv.extend(["-i", self.remote.identity_file])
        if self.remote.known_hosts_file:
            argv.extend(["-o", f"UserKnownHostsFile={self.remote.known_hosts_file}"])
        if self.remote.route == RemoteRoute.PROXY_JUMP:
            proxy_jump = ",".join(host.as_proxyjump_target() for host in self.remote.jump_hosts)
            argv.extend(["-J", proxy_jump])
        return argv

    def rsync_base_argv(self) -> list[str]:
        ssh_cmd = " ".join(self.ssh_base_argv())
        return ["rsync", "-a", "--partial", "--delay-updates", "-e", ssh_cmd]

    def run_wrapper(self, command: str, payload: dict[str, Any] | None = None, timeout_sec: int | None = None, extra_args: list[str] | None = None) -> TransportResult:
        validate_safe_id(command, "wrapper_command")
        repository_root = validate_remote_path(self.remote.repository_root or self.remote.root or "")
        job_root = validate_remote_path(self.remote.job_root or "")
        safe_args = []
        for item in extra_args or []:
            if any(ord(ch) < 32 for ch in item) or "\n" in item or "\r" in item:
                raise ConfigError("Unsafe wrapper argument contains control characters.")
            safe_args.append(shlex.quote(item))
        remote_cmd = f"cd {shlex.quote(repository_root)} && SR_REMOTE_JOB_ROOT={shlex.quote(job_root)} {shlex.quote(self.wrapper)} {shlex.quote(command)}"
        if safe_args:
            remote_cmd += " " + " ".join(safe_args)
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
        root = validate_remote_path(self.remote.job_root or "")
        destination = f"{self.target}:{root}/{rel.as_posix()}"
        argv = [*self.rsync_base_argv()]
        if dry_run:
            argv.append("--dry-run")
        argv.extend([str(source), destination])
        try:
            result = subprocess.run(argv, text=True, capture_output=True, timeout=self.remote.rsync_timeout_sec)
        except subprocess.TimeoutExpired as exc:
            return TransportResult(argv, 124, exc.stdout or "", exc.stderr or "", timeout=True)
        return TransportResult(argv, result.returncode, result.stdout, result.stderr)

    def rsync_from_remote(self, remote_rel: str | Path, destination: Path, *, dry_run: bool = False) -> TransportResult:
        rel = validate_relative_path(remote_rel)
        root = validate_remote_path(self.remote.job_root or "")
        source = f"{self.target}:{root}/{rel.as_posix()}"
        argv = [*self.rsync_base_argv()]
        if dry_run:
            argv.append("--dry-run")
        argv.extend([source, str(destination)])
        try:
            result = subprocess.run(argv, text=True, capture_output=True, timeout=self.remote.rsync_timeout_sec)
        except subprocess.TimeoutExpired as exc:
            return TransportResult(argv, 124, exc.stdout or "", exc.stderr or "", timeout=True)
        return TransportResult(argv, result.returncode, result.stdout, result.stderr)

    def lightweight_preflight(self, docker_image: str | None = None) -> RemotePreflightResult:
        image = docker_image or self.remote.docker_image
        extra_args = ["--docker-image", image] if image else []
        result = self.run_wrapper("preflight", None, extra_args=extra_args)
        if result.returncode != 0:
            return RemotePreflightResult(
                status="REMOTE_DOCKER_PREFLIGHT_FAILED",
                ssh_ok=result.returncode != 255,
                error_class=classify_ssh_failure(result),
                detail=(result.stderr or result.stdout)[-2000:],
            )
        try:
            data = result.json_stdout()
        except json.JSONDecodeError as exc:
            raise InfraError(f"{TransportErrorClass.REMOTE_PROTOCOL_MISMATCH}: Remote preflight returned invalid JSON: {exc}") from exc
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


def classify_ssh_failure(result: TransportResult) -> str:
    text = f"{result.stderr}\n{result.stdout}".lower()
    if "host key" in text or "known_hosts" in text:
        return TransportErrorClass.SSH_HOST_KEY_FAILED
    if result.returncode == 255 or "connection refused" in text or "timed out" in text or "no route" in text:
        return TransportErrorClass.SSH_CONNECTION_FAILED
    return TransportErrorClass.SSH_COMMAND_FAILED


def classify_rsync_failure(result: TransportResult) -> str:
    return TransportErrorClass.RSYNC_TRANSFER_FAILED
