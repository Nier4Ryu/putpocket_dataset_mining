from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .errors import ConfigError


class ExecutionRole(StrEnum):
    LOCAL_CONTROLLER = "local_controller"
    CLOUD_CONTROLLER = "cloud_controller"
    VERIFIER_HOST = "verifier_host"


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
    RUNPOD_HOPPER = "runpod_hopper"
    CUSTOM = "custom"


E_CLOUD_LOCAL_DOCKER_FORBIDDEN = "E_CLOUD_LOCAL_DOCKER_FORBIDDEN"
LOCAL_DOCKER_SKIPPED_EXPECTED = "LOCAL_DOCKER_SKIPPED_EXPECTED"
REMOTE_DOCKER_PREFLIGHT_PASSED = "REMOTE_DOCKER_PREFLIGHT_PASSED"
REMOTE_DOCKER_PREFLIGHT_FAILED = "REMOTE_DOCKER_PREFLIGHT_FAILED"
DOCKER_DISABLED_FOR_STATIC_ONLY = "DOCKER_DISABLED_FOR_STATIC_ONLY"
EVALUATION_BLOCKED_NO_DOCKER_BACKEND = "EVALUATION_BLOCKED_NO_DOCKER_BACKEND"


@dataclass(frozen=True)
class RemoteDockerConfig:
    host: str | None = None
    user: str | None = None
    port: int = 22
    root: str | None = None
    identity_file: str | None = None
    known_hosts_file: str | None = None
    docker_image: str | None = None

    @classmethod
    def from_env_and_mapping(cls, mapping: dict[str, Any] | None = None) -> "RemoteDockerConfig":
        mapping = mapping or {}
        return cls(
            host=str(mapping.get("host") or os.environ.get("SR_REMOTE_HOST") or "") or None,
            user=str(mapping.get("user") or os.environ.get("SR_REMOTE_USER") or "") or None,
            port=int(mapping.get("port") or os.environ.get("SR_REMOTE_PORT") or 22),
            root=str(mapping.get("root") or os.environ.get("SR_REMOTE_ROOT") or "") or None,
            identity_file=str(mapping.get("identity_file") or os.environ.get("SR_REMOTE_IDENTITY_FILE") or "") or None,
            known_hosts_file=str(mapping.get("known_hosts_file") or os.environ.get("SR_REMOTE_KNOWN_HOSTS_FILE") or "") or None,
            docker_image=str(mapping.get("docker_image") or os.environ.get("SR_REMOTE_DOCKER_IMAGE") or "") or None,
        )

    def require_complete(self) -> None:
        missing = [name for name in ("host", "user", "root") if not getattr(self, name)]
        if missing:
            raise ConfigError(
                f"{EVALUATION_BLOCKED_NO_DOCKER_BACKEND}: remote_ssh_docker requires {', '.join(missing)}."
            )


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
    remote: RemoteDockerConfig = RemoteDockerConfig()

    @classmethod
    def from_env_and_mapping(cls, mapping: dict[str, Any] | None = None) -> "ExecutionConfig":
        mapping = mapping or {}

        def pick(key: str, env: str, default: str) -> str:
            return str(mapping.get(key) or os.environ.get(env) or default)

        role_default = "cloud_controller" if _looks_like_runpod() else "local_controller"
        return cls(
            execution_role=ExecutionRole(pick("execution_role", "SR_EXECUTION_ROLE", role_default)),
            workspace_backend=DockerBackend(pick("workspace_backend", "SR_WORKSPACE_BACKEND", "local_docker")),
            verifier_backend=DockerBackend(pick("verifier_backend", "SR_VERIFIER_BACKEND", "local_docker")),
            hardware_profile=HardwareProfile(pick("hardware_profile", "SR_HARDWARE_PROFILE", "auto")),
            server_profile=ServerProfile(pick("server_profile", "SR_SERVER_PROFILE", "custom")),
            model_id=str(mapping.get("model_id") or os.environ.get("SR_MODEL_ID") or "") or None,
            model_path=str(mapping.get("model_path") or os.environ.get("SR_MODEL_PATH") or "") or None,
            vllm_profile=str(mapping.get("vllm_profile") or os.environ.get("SR_VLLM_PROFILE") or "") or None,
            cuda_arch_list=str(mapping.get("cuda_arch_list") or os.environ.get("SR_CUDA_ARCH_LIST") or "") or None,
            remote=RemoteDockerConfig.from_env_and_mapping(mapping.get("remote") if isinstance(mapping.get("remote"), dict) else None),
        )

    def validate_for_evaluation_start(self) -> None:
        self.guard_cloud_local_docker()
        if self.workspace_backend == DockerBackend.DISABLED or self.verifier_backend == DockerBackend.DISABLED:
            raise ConfigError(f"{EVALUATION_BLOCKED_NO_DOCKER_BACKEND}: evaluation requires workspace and verifier Docker backends.")
        if self.workspace_backend == DockerBackend.REMOTE_SSH_DOCKER or self.verifier_backend == DockerBackend.REMOTE_SSH_DOCKER:
            self.remote.require_complete()

    def guard_cloud_local_docker(self) -> None:
        if self.execution_role != ExecutionRole.CLOUD_CONTROLLER:
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
        ServerProfile.RUNPOD_HOPPER: HardwareProfile.SM90,
        ServerProfile.CUSTOM: HardwareProfile.AUTO,
    }[server]
