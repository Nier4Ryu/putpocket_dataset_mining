from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .errors import ConfigError


class ExecutionRole(StrEnum):
    LOCAL_CONTROLLER = "local_controller"
    CLOUD_CONTROLLER = "cloud_controller"
    VERIFIER_HOST = "verifier_host"
    CONTROLLER = "controller"
    VERIFIER = "verifier"
    MODEL_SERVER = "model_server"
    DEVELOPMENT = "development"


class DockerBackend(StrEnum):
    LOCAL_DOCKER = "local_docker"
    REMOTE_SSH_DOCKER = "remote_ssh_docker"
    DISABLED = "disabled"


class HardwareProfile(StrEnum):
    CPU = "cpu"
    SM86 = "sm86"
    SM90 = "sm90"
    SM120 = "sm120"
    AUTO = "auto"


class ServerProfile(StrEnum):
    SERVER1_RTX3090 = "server1_rtx3090"
    SERVER2_RTXPRO6000_BLACKWELL = "server2_rtxpro6000_blackwell"
    SERVER2_BLACKWELL = "server2_blackwell"
    RUNPOD_HOPPER = "runpod_hopper"
    CUSTOM = "custom"


class RemoteRoute(StrEnum):
    DIRECT = "direct"
    PROXY_JUMP = "proxy_jump"


E_CLOUD_LOCAL_DOCKER_FORBIDDEN = "E_CLOUD_LOCAL_DOCKER_FORBIDDEN"
LOCAL_DOCKER_SKIPPED_EXPECTED = "LOCAL_DOCKER_SKIPPED_EXPECTED"
REMOTE_DOCKER_PREFLIGHT_PASSED = "REMOTE_DOCKER_PREFLIGHT_PASSED"
REMOTE_DOCKER_PREFLIGHT_FAILED = "REMOTE_DOCKER_PREFLIGHT_FAILED"
DOCKER_DISABLED_FOR_STATIC_ONLY = "DOCKER_DISABLED_FOR_STATIC_ONLY"
EVALUATION_BLOCKED_NO_DOCKER_BACKEND = "EVALUATION_BLOCKED_NO_DOCKER_BACKEND"
DEFAULT_VERIFIER_TIMEOUT_SEC = 3600
DEFAULT_VERIFIER_REMOTE_GRACE_SEC = 120


@dataclass(frozen=True)
class JumpHost:
    host: str
    user: str
    port: int = 22

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "JumpHost":
        host = _safe_text(mapping.get("host"), "jump_host.host")
        user = _safe_text(mapping.get("user"), "jump_host.user")
        port = _safe_port(mapping.get("port", 22), "jump_host.port")
        return cls(host=host, user=user, port=port)

    def as_proxyjump_target(self) -> str:
        return f"{self.user}@{self.host}:{self.port}"


@dataclass(frozen=True)
class RemoteDockerConfig:
    host: str | None = None
    user: str | None = None
    port: int = 22
    route: RemoteRoute = RemoteRoute.DIRECT
    jump_hosts: tuple[JumpHost, ...] = ()
    repository_root: str | None = None
    job_root: str | None = None
    root: str | None = None
    identity_file: str | None = None
    known_hosts_file: str | None = None
    strict_host_key_checking: bool = True
    connection_timeout_sec: int = 10
    command_timeout_sec: int = DEFAULT_VERIFIER_TIMEOUT_SEC + DEFAULT_VERIFIER_REMOTE_GRACE_SEC
    rsync_timeout_sec: int = 300
    wrapper: str = "putpocket-remote-verifier"
    docker_image: str | None = None
    dockerfile: str = "docker/classeval_python/Dockerfile"
    max_concurrent_jobs: int = 1

    @classmethod
    def from_env_and_mapping(cls, mapping: dict[str, Any] | None = None) -> "RemoteDockerConfig":
        mapping = mapping or {}
        jump_hosts_raw = mapping.get("jump_hosts") or []
        jump_hosts = tuple(JumpHost.from_mapping(item) for item in jump_hosts_raw)
        repository_root = str(mapping.get("repository_root") or mapping.get("root") or os.environ.get("SR_REMOTE_REPOSITORY_ROOT") or os.environ.get("SR_REMOTE_ROOT") or "") or None
        job_root = str(mapping.get("job_root") or os.environ.get("SR_REMOTE_JOB_ROOT") or (f"{repository_root}/data/remote_verifier" if repository_root else "")) or None
        return cls(
            host=_safe_optional_text(mapping.get("host") or os.environ.get("SR_REMOTE_HOST"), "remote.host"),
            user=_safe_optional_text(mapping.get("user") or os.environ.get("SR_REMOTE_USER"), "remote.user"),
            port=_safe_port(mapping.get("port") or os.environ.get("SR_REMOTE_PORT") or 22, "remote.port"),
            route=RemoteRoute(str(mapping.get("route") or os.environ.get("SR_REMOTE_ROUTE") or "direct")),
            jump_hosts=jump_hosts,
            repository_root=repository_root,
            job_root=job_root,
            root=repository_root,
            identity_file=str(mapping.get("identity_file") or os.environ.get("SR_REMOTE_IDENTITY_FILE") or "") or None,
            known_hosts_file=str(mapping.get("known_hosts_file") or os.environ.get("SR_REMOTE_KNOWN_HOSTS_FILE") or "") or None,
            strict_host_key_checking=_bool(mapping.get("strict_host_key_checking"), default=True),
            connection_timeout_sec=int(mapping.get("connection_timeout_sec") or os.environ.get("SR_REMOTE_CONNECTION_TIMEOUT_SEC") or 10),
            command_timeout_sec=int(mapping.get("command_timeout_sec") or os.environ.get("SR_REMOTE_COMMAND_TIMEOUT_SEC") or (DEFAULT_VERIFIER_TIMEOUT_SEC + DEFAULT_VERIFIER_REMOTE_GRACE_SEC)),
            rsync_timeout_sec=int(mapping.get("rsync_timeout_sec") or os.environ.get("SR_REMOTE_RSYNC_TIMEOUT_SEC") or 300),
            wrapper=_safe_text(mapping.get("wrapper") or os.environ.get("SR_REMOTE_WRAPPER") or "putpocket-remote-verifier", "remote.wrapper"),
            docker_image=str(mapping.get("docker_image") or os.environ.get("SR_REMOTE_DOCKER_IMAGE") or "") or None,
            dockerfile=str(mapping.get("dockerfile") or os.environ.get("SR_REMOTE_DOCKERFILE") or "docker/classeval_python/Dockerfile"),
            max_concurrent_jobs=int(mapping.get("max_concurrent_jobs") or os.environ.get("SR_REMOTE_MAX_CONCURRENT_JOBS") or 1),
        )

    def require_complete(self) -> None:
        missing = [name for name in ("host", "user", "repository_root", "job_root") if not getattr(self, name)]
        if missing:
            raise ConfigError(
                f"{EVALUATION_BLOCKED_NO_DOCKER_BACKEND}: remote_ssh_docker requires {', '.join(missing)}."
            )
        if self.route == RemoteRoute.PROXY_JUMP and not self.jump_hosts:
            raise ConfigError("remote_ssh_docker route=proxy_jump requires at least one jump host.")


@dataclass(frozen=True)
class ExecutionConfig:
    execution_role: ExecutionRole = ExecutionRole.LOCAL_CONTROLLER
    workspace_backend: DockerBackend = DockerBackend.LOCAL_DOCKER
    verifier_backend: DockerBackend = DockerBackend.LOCAL_DOCKER
    hardware_profile: HardwareProfile = HardwareProfile.AUTO
    server_profile: ServerProfile = ServerProfile.CUSTOM
    model_id: str | None = None
    model_path: str | None = None
    vllm_profile: str | None = None
    cuda_arch_list: str | None = None
    verifier_timeout_sec: int = DEFAULT_VERIFIER_TIMEOUT_SEC
    verifier_remote_grace_sec: int = DEFAULT_VERIFIER_REMOTE_GRACE_SEC
    remote: RemoteDockerConfig = RemoteDockerConfig()

    @classmethod
    def from_env_and_mapping(cls, mapping: dict[str, Any] | None = None) -> "ExecutionConfig":
        mapping = mapping or {}
        remote_config_path = (
            mapping.get("remote_config")
            or mapping.get("remote_config_path")
            or (mapping.get("verifier", {}).get("remote_config") if isinstance(mapping.get("verifier"), dict) else None)
            or os.environ.get("SR_REMOTE_CONFIG")
        )
        remote_mapping: dict[str, Any] | None = None
        if remote_config_path:
            from .config import load_yaml

            remote_mapping = _remote_mapping_from_config(load_yaml(Path(str(remote_config_path))))

        def pick(key: str, env: str, default: str) -> str:
            return str(mapping.get(key) or os.environ.get(env) or default)

        role_default = "model_server" if _looks_like_runpod() else "controller"
        return cls(
            execution_role=_role_from_text(pick("execution_role", "SR_EXECUTION_ROLE", role_default)),
            workspace_backend=DockerBackend(pick("workspace_backend", "SR_WORKSPACE_BACKEND", "local_docker")),
            verifier_backend=DockerBackend(pick("verifier_backend", "SR_VERIFIER_BACKEND", "local_docker")),
            hardware_profile=HardwareProfile(pick("hardware_profile", "SR_HARDWARE_PROFILE", "auto")),
            server_profile=ServerProfile(pick("server_profile", "SR_SERVER_PROFILE", "custom")),
            model_id=str(mapping.get("model_id") or os.environ.get("SR_MODEL_ID") or "") or None,
            model_path=str(mapping.get("model_path") or os.environ.get("SR_MODEL_PATH") or "") or None,
            vllm_profile=str(mapping.get("vllm_profile") or os.environ.get("SR_VLLM_PROFILE") or "") or None,
            cuda_arch_list=str(mapping.get("cuda_arch_list") or os.environ.get("SR_CUDA_ARCH_LIST") or "") or None,
            verifier_timeout_sec=int(mapping.get("verifier_timeout_sec") or os.environ.get("SR_VERIFIER_TIMEOUT_SEC") or DEFAULT_VERIFIER_TIMEOUT_SEC),
            verifier_remote_grace_sec=int(mapping.get("verifier_remote_grace_sec") or os.environ.get("SR_VERIFIER_REMOTE_GRACE_SEC") or DEFAULT_VERIFIER_REMOTE_GRACE_SEC),
            remote=RemoteDockerConfig.from_env_and_mapping(
                remote_mapping
                or (mapping.get("remote") if isinstance(mapping.get("remote"), dict) else None)
            ),
        )

    @classmethod
    def from_remote_verifier_mapping(cls, mapping: dict[str, Any]) -> "ExecutionConfig":
        remote = _remote_mapping_from_config(mapping)
        verifier = mapping.get("verifier") if isinstance(mapping.get("verifier"), dict) else {}
        timeout_sec = mapping.get("timeout_sec") or verifier.get("timeout_sec")
        return cls.from_env_and_mapping(
            {
                "verifier_backend": mapping.get("backend", "remote_ssh_docker"),
                "verifier_timeout_sec": timeout_sec or DEFAULT_VERIFIER_TIMEOUT_SEC,
                "remote": remote,
            }
        )

    def validate_for_evaluation_start(self) -> None:
        self.guard_cloud_local_docker()
        if self.workspace_backend == DockerBackend.DISABLED or self.verifier_backend == DockerBackend.DISABLED:
            raise ConfigError(f"{EVALUATION_BLOCKED_NO_DOCKER_BACKEND}: evaluation requires workspace and verifier Docker backends.")
        if self.workspace_backend == DockerBackend.REMOTE_SSH_DOCKER or self.verifier_backend == DockerBackend.REMOTE_SSH_DOCKER:
            self.remote.require_complete()
            self.validate_remote_timeout_budget(self.verifier_timeout_sec)

    def validate_remote_timeout_budget(self, verifier_timeout_sec: int | None = None) -> None:
        if self.verifier_backend != DockerBackend.REMOTE_SSH_DOCKER and self.workspace_backend != DockerBackend.REMOTE_SSH_DOCKER:
            return
        effective = int(verifier_timeout_sec or self.verifier_timeout_sec)
        minimum = effective + int(self.verifier_remote_grace_sec)
        if self.remote.command_timeout_sec < minimum:
            raise ConfigError(
                "remote.command_timeout_sec must be at least verifier timeout plus grace "
                f"({minimum}s required, got {self.remote.command_timeout_sec}s)."
            )

    def guard_cloud_local_docker(self) -> None:
        if self.execution_role not in {ExecutionRole.CLOUD_CONTROLLER, ExecutionRole.MODEL_SERVER}:
            return
        local_backends = {
            "workspace_backend": self.workspace_backend,
            "verifier_backend": self.verifier_backend,
        }
        offenders = [name for name, backend in local_backends.items() if backend == DockerBackend.LOCAL_DOCKER]
        if offenders:
            raise ConfigError(
                f"{E_CLOUD_LOCAL_DOCKER_FORBIDDEN}: Local Docker execution is unavailable in cloud-controller mode. "
                "Configure remote_ssh_docker and pass the remote preflight before starting the episode or verifier. "
                f"Offending settings: {', '.join(offenders)}."
            )


def _looks_like_runpod() -> bool:
    return any(os.environ.get(name) for name in ("RUNPOD_POD_ID", "RUNPOD_PUBLIC_IP", "RUNPOD_DC_ID"))


def _remote_mapping_from_config(mapping: dict[str, Any]) -> dict[str, Any]:
    target = mapping.get("target") if isinstance(mapping.get("target"), dict) else {}
    remote = dict(mapping)
    remote.pop("target", None)
    remote.update(target)
    return remote


def cuda_arch_for_profile(profile: HardwareProfile | str) -> str | None:
    profile = HardwareProfile(profile)
    return {
        HardwareProfile.CPU: None,
        HardwareProfile.SM86: "8.6",
        HardwareProfile.SM90: "9.0",
        HardwareProfile.SM120: "12.0",
        HardwareProfile.AUTO: None,
    }[profile]


def default_hardware_for_server(server: ServerProfile | str) -> HardwareProfile:
    server = ServerProfile(server)
    return {
        ServerProfile.SERVER1_RTX3090: HardwareProfile.SM86,
        ServerProfile.SERVER2_RTXPRO6000_BLACKWELL: HardwareProfile.SM120,
        ServerProfile.SERVER2_BLACKWELL: HardwareProfile.SM120,
        ServerProfile.RUNPOD_HOPPER: HardwareProfile.SM90,
        ServerProfile.CUSTOM: HardwareProfile.AUTO,
    }[server]


def _role_from_text(text: str) -> ExecutionRole:
    aliases = {
        "controller": ExecutionRole.LOCAL_CONTROLLER,
        "development": ExecutionRole.LOCAL_CONTROLLER,
        "verifier": ExecutionRole.VERIFIER_HOST,
        "model_server": ExecutionRole.CLOUD_CONTROLLER,
    }
    return aliases.get(text, ExecutionRole(text))


def _safe_optional_text(value: Any, field: str) -> str | None:
    if value is None or value == "":
        return None
    return _safe_text(value, field)


def _safe_text(value: Any, field: str) -> str:
    text = str(value)
    if any(ord(ch) < 32 for ch in text) or "\n" in text or "\r" in text:
        raise ConfigError(f"Invalid control character in {field}.")
    return text


def _safe_port(value: Any, field: str) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid {field}: {value!r}") from exc
    if port < 1 or port > 65535:
        raise ConfigError(f"Invalid {field}: {port}")
    return port


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
