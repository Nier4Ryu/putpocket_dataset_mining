from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import load_yaml
from .constants import REPO_ROOT
from .errors import ConfigError
from .execution_config import DockerBackend, ExecutionConfig, RemoteRoute
from .ssh_transport import SshRsyncTransport


PROFILE_NAME = "runpod_controller_server1_verifier"
V1_POLICY = "history1_pytest_only"
V2_POLICY = "history2_pytest_then_judge"

RUNPOD_LOCAL_INFERENCE_PREFLIGHT_FAILED = "RUNPOD_LOCAL_INFERENCE_PREFLIGHT_FAILED"
RUNPOD_LOCAL_WORKSPACE_BACKEND_UNAVAILABLE = "RUNPOD_LOCAL_WORKSPACE_BACKEND_UNAVAILABLE"
SERVER_B_WORKSPACE_PREFLIGHT_FAILED = "SERVER_B_WORKSPACE_PREFLIGHT_FAILED"
SERVER1_SSH_CONNECTION_FAILED = "SERVER1_SSH_CONNECTION_FAILED"
SERVER1_HOST_KEY_FAILED = "SERVER1_HOST_KEY_FAILED"
SERVER1_WRAPPER_PREFLIGHT_FAILED = "SERVER1_WRAPPER_PREFLIGHT_FAILED"
SERVER1_DOCKER_PREFLIGHT_FAILED = "SERVER1_DOCKER_PREFLIGHT_FAILED"
SERVER1_JUDGE_PREFLIGHT_FAILED = "SERVER1_JUDGE_PREFLIGHT_FAILED"
REMOTE_VERIFICATION_SUBMIT_FAILED = "REMOTE_VERIFICATION_SUBMIT_FAILED"
REMOTE_VERIFICATION_RETRIEVE_FAILED = "REMOTE_VERIFICATION_RETRIEVE_FAILED"
WORKFLOW_STATE_INVALID = "WORKFLOW_STATE_INVALID"

SERVER1_HOST = "10.0.0.5"
SERVER1_USER = "dyryu"
SERVER1_PORT = 42
PROXY_A = ("141.223.145.88", "dyryu", 4500)
PROXY_C = ("141.223.25.156", "dyryu", 42)


@dataclass(frozen=True)
class RunpodExecutionProfile:
    name: str
    controller_host: str
    inference_backend: str
    verifier_backend: str
    verifier_target: str
    run_root: str
    execution_config: ExecutionConfig
    local_hidden_verifier_fallback: bool
    local_workspace_backend: str

    @property
    def topology(self) -> dict[str, Any]:
        return {
            "controller_host": self.controller_host,
            "controller_role": "server_a",
            "inference_host_role": "server_a",
            "inference_backend": self.inference_backend,
            "verifier_host_role": "server_b",
            "workspace_host_role": "server_b",
            "workspace_backend": str(self.execution_config.workspace_backend),
            "verifier_backend": self.verifier_backend,
            "verifier_route": "Proxy-A -> Proxy-C -> Server-1",
            "verifier_target": self.verifier_target,
            "deployment_mapping": {"server_a": "runpod", "server_b": "server1"},
            "verification1_policy": V1_POLICY,
            "verification2_policy": V2_POLICY,
            "local_hidden_verifier_fallback": self.local_hidden_verifier_fallback,
            "sr_reuse": "not_implemented",
            "kv_continuity": "not_claimed",
        }

    def workflow_execution_mapping(self) -> dict[str, Any]:
        remote = self.execution_config.remote
        return {
            "execution_role": "runpod_controller",
            "workspace_backend": str(self.execution_config.workspace_backend),
            "verifier_backend": str(self.execution_config.verifier_backend),
            "allow_local_fallback": False,
            "inference_host_role": "server_a",
            "inference_backend": "local_vllm",
            "workspace_host_role": "server_b",
            "verifier_host_role": "server_b",
            "remote": {
                "host": remote.host,
                "user": remote.user,
                "port": remote.port,
                "route": str(remote.route),
                "jump_hosts": [
                    {"host": host.host, "user": host.user, "port": host.port}
                    for host in remote.jump_hosts
                ],
                "repository_root": remote.repository_root,
                "job_root": remote.job_root,
                "identity_file": remote.identity_file,
                "known_hosts_file": remote.known_hosts_file,
                "strict_host_key_checking": remote.strict_host_key_checking,
                "connection_timeout_sec": remote.connection_timeout_sec,
                "command_timeout_sec": remote.command_timeout_sec,
                "rsync_timeout_sec": remote.rsync_timeout_sec,
                "wrapper": remote.wrapper,
                "docker_image": remote.docker_image,
                "dockerfile": remote.dockerfile,
                "max_concurrent_jobs": remote.max_concurrent_jobs,
            },
            "workspace_remote": _remote_mapping(self.execution_config.workspace_remote or remote),
        }


def load_runpod_execution_profile(path: str | Path) -> RunpodExecutionProfile:
    raw = load_yaml(path)
    if raw.get("name") != PROFILE_NAME:
        raise ConfigError(f"unsupported RunPod execution profile: {raw.get('name')!r}")
    execution = raw.get("execution")
    if not isinstance(execution, dict):
        raise ConfigError("RunPod execution profile requires execution mapping.")
    profile = RunpodExecutionProfile(
        name=str(raw["name"]),
        controller_host=str(raw.get("controller_host") or ""),
        inference_backend=str(raw.get("inference_backend") or ""),
        verifier_backend=str(raw.get("verifier_backend") or ""),
        verifier_target=str(raw.get("verifier_target") or ""),
        run_root=str(raw.get("run_root") or ""),
        execution_config=ExecutionConfig.from_env_and_mapping(execution),
        local_hidden_verifier_fallback=bool(raw.get("local_hidden_verifier_fallback", True)),
        local_workspace_backend=str(raw.get("local_workspace_backend") or execution.get("workspace_backend") or ""),
    )
    validate_runpod_execution_profile(profile)
    return profile


def validate_runpod_execution_profile(profile: RunpodExecutionProfile) -> None:
    if profile.controller_host != "runpod":
        raise ConfigError("RunPod execution profile requires controller_host=runpod.")
    if profile.inference_backend != "local_vllm":
        raise ConfigError("RunPod execution profile requires inference_backend=local_vllm.")
    if profile.verifier_backend != "ssh_rsync":
        raise ConfigError("RunPod execution profile requires verifier_backend=ssh_rsync.")
    if profile.local_hidden_verifier_fallback:
        raise ConfigError("RunPod execution profile must disable local hidden-verifier fallback.")
    if "/home/dyryu" in profile.run_root:
        raise ConfigError("RunPod run_root must not depend on a Server-2 /home/dyryu path.")
    remote = profile.execution_config.remote
    if profile.execution_config.verifier_backend != DockerBackend.REMOTE_SSH_DOCKER:
        raise ConfigError("RunPod execution profile requires remote_ssh_docker verifier backend.")
    if profile.execution_config.workspace_backend not in {DockerBackend.SSH_REMOTE_DOCKER, DockerBackend.REMOTE_SSH_DOCKER}:
        raise ConfigError("RunPod controller profile requires ssh_remote_docker workspace backend.")
    workspace_remote = profile.execution_config.workspace_remote
    if workspace_remote is None:
        raise ConfigError("RunPod controller profile requires workspace_remote configuration.")
    workspace_remote.require_complete()
    remote.require_complete()
    if remote.route != RemoteRoute.PROXY_JUMP:
        raise ConfigError("RunPod-to-Server-1 verifier route must be proxy_jump.")
    if remote.host != SERVER1_HOST or remote.user != SERVER1_USER or remote.port != SERVER1_PORT:
        raise ConfigError("RunPod verifier target must be Server-1 dyryu@10.0.0.5:42.")
    jumps = tuple((host.host, host.user, host.port) for host in remote.jump_hosts)
    if jumps != (PROXY_A, PROXY_C):
        raise ConfigError("RunPod verifier route must be Proxy-A -> Proxy-C -> Server-1.")
    if remote.identity_file and remote.identity_file.startswith("/home/dyryu/"):
        raise ConfigError("RunPod SSH identity file must be runtime materialized, not a Server-2 path.")
    if remote.known_hosts_file and remote.known_hosts_file.startswith("/home/dyryu/"):
        raise ConfigError("RunPod known_hosts file must be runtime materialized, not a Server-2 path.")


def redact_profile_for_logs(profile: RunpodExecutionProfile) -> dict[str, Any]:
    payload = {
        "name": profile.name,
        **profile.topology,
        "run_root": profile.run_root,
        "local_workspace_backend": profile.local_workspace_backend,
        "remote": profile.workflow_execution_mapping()["remote"],
        "workspace_remote": profile.workflow_execution_mapping()["workspace_remote"],
    }
    remote = payload["remote"]
    for key in ("identity_file", "known_hosts_file"):
        if remote.get(key):
            remote[key] = f"<runtime-path:{Path(str(remote[key])).name}>"
    workspace_remote = payload.get("workspace_remote", {})
    for key in ("identity_file", "known_hosts_file"):
        if workspace_remote.get(key):
            workspace_remote[key] = f"<runtime-path:{Path(str(workspace_remote[key])).name}>"
    return payload


def run_combined_preflight(
    profile_path: str | Path,
    *,
    live_remote: bool = False,
    live_workspace: bool = False,
    import_checks: bool = False,
) -> dict[str, Any]:
    profile = load_runpod_execution_profile(profile_path)
    local = _local_preflight(profile, live_workspace=live_workspace, import_checks=import_checks)
    server1 = _server1_preflight(profile, live_remote=live_remote, live_workspace=live_workspace)
    failure_classes = [
        item
        for item in (
            local.get("failure_class"),
            server1.get("failure_class"),
        )
        if item
    ]
    status = "passed" if not failure_classes and local["local_workspace_backend_ready"] and server1["checked"] else "partial"
    if failure_classes:
        status = "failed"
    return {
        "schema_version": 1,
        "profile": redact_profile_for_logs(profile),
        "status": status,
        "local": local,
        "server1": server1,
        "failure_classes": failure_classes,
    }


def _local_preflight(profile: RunpodExecutionProfile, *, live_workspace: bool, import_checks: bool) -> dict[str, Any]:
    repo_root = Path(os.environ.get("PUTPOCKET_REPO_ROOT") or REPO_ROOT)
    run_root = _resolve_repo_relative(profile.run_root, repo_root)
    python_ok = Path(sys.executable).exists()
    imports = {
        "torch": _import_available("torch") if import_checks else "not_checked",
        "vllm": _import_available("vllm") if import_checks else "not_checked",
    }
    remote_workspace = profile.execution_config.workspace_backend in {DockerBackend.SSH_REMOTE_DOCKER, DockerBackend.REMOTE_SSH_DOCKER}
    workspace_ready = True if remote_workspace else _local_workspace_backend_ready(live_workspace)
    failure_class = None
    if not python_ok or (import_checks and not all(value is True for value in imports.values())):
        failure_class = RUNPOD_LOCAL_INFERENCE_PREFLIGHT_FAILED
    if not workspace_ready:
        failure_class = failure_class or RUNPOD_LOCAL_WORKSPACE_BACKEND_UNAVAILABLE
    return {
        "controller_host": "runpod",
        "repo_root": str(repo_root),
        "run_root": str(run_root),
        "python_executable": sys.executable,
        "python_ok": python_ok,
        "inference_backend": "local_vllm",
        "inference_imports": imports,
        "gpu_visibility": "not_checked",
        "local_workspace_backend": profile.local_workspace_backend,
        "local_workspace_backend_ready": workspace_ready,
        "local_workspace_backend_checked": live_workspace and not remote_workspace,
        "local_docker_required": not remote_workspace,
        "failure_class": failure_class,
    }


def _server1_preflight(profile: RunpodExecutionProfile, *, live_remote: bool, live_workspace: bool = False) -> dict[str, Any]:
    remote = profile.execution_config.remote
    static = {
        "checked": live_remote,
        "verifier_host_role": "server1",
        "route": "Proxy-A -> Proxy-C -> Server-1",
        "target": f"{remote.user}@{remote.host}:{remote.port}",
        "wrapper": remote.wrapper,
        "job_root": remote.job_root,
        "ssh_ok": "not_checked",
        "rsync_ok": "not_checked",
        "wrapper_ok": "not_checked",
        "docker_ok": "not_checked",
        "judge_preflight": "not_checked",
        "failure_class": None,
    }
    if not live_remote:
        return static
    workspace_payload: dict[str, Any] = {"checked": True, "ready": False}
    if profile.execution_config.workspace_backend in {DockerBackend.SSH_REMOTE_DOCKER, DockerBackend.REMOTE_SSH_DOCKER}:
        workspace_remote = profile.execution_config.workspace_remote
        assert workspace_remote is not None
        try:
            workspace_result = SshRsyncTransport(workspace_remote).run_wrapper(
                "preflight", {"docker_image": workspace_remote.docker_image}
            )
            data = workspace_result.json_stdout()
            workspace_payload = {"checked": True, "ready": workspace_result.returncode == 0 and bool(data.get("docker_ok")), **data}
            if workspace_payload["ready"] and live_workspace:
                session_id = f"preflight-{uuid.uuid4().hex[:12]}"
                transport = SshRsyncTransport(workspace_remote)
                create = transport.run_wrapper("create", {"session_id": session_id, "docker_image": workspace_remote.docker_image, "cpus": 1, "memory": "512m"}, timeout_sec=120)
                execute = transport.run_wrapper("exec", {"session_id": session_id, "command": "printf SERVER_B_WORKSPACE_READY", "timeout_sec": 30}) if create.returncode == 0 else create
                snap = transport.run_wrapper("snapshot", {"session_id": session_id, "snapshot_id": "preflight"}) if execute.returncode == 0 else execute
                destroy = transport.run_wrapper("destroy", {"session_id": session_id})
                workspace_payload["disposable_smoke"] = {
                    "create": create.returncode == 0,
                    "exec": execute.returncode == 0 and "SERVER_B_WORKSPACE_READY" in execute.stdout,
                    "snapshot": snap.returncode == 0,
                    "destroy": destroy.returncode == 0,
                }
                workspace_payload["ready"] = all(workspace_payload["disposable_smoke"].values())
        except Exception as exc:  # noqa: BLE001
            workspace_payload = {"checked": True, "ready": False, "detail": _redact(str(exc))}
    try:
        result = SshRsyncTransport(remote).lightweight_preflight(remote.docker_image)
    except Exception as exc:  # noqa: BLE001 - preflight must classify external failures.
        text = str(exc).lower()
        failure = SERVER1_HOST_KEY_FAILED if "host key" in text or "known_hosts" in text else SERVER1_SSH_CONNECTION_FAILED
        return {**static, "checked": True, "ssh_ok": False, "failure_class": failure, "detail": _redact(str(exc))}
    failure = None
    if not workspace_payload.get("ready"):
        failure = SERVER_B_WORKSPACE_PREFLIGHT_FAILED
    elif not result.wrapper_ok:
        failure = SERVER1_WRAPPER_PREFLIGHT_FAILED
    elif not result.docker_ok:
        failure = SERVER1_DOCKER_PREFLIGHT_FAILED
    return {
        **static,
        "checked": True,
        "ssh_ok": result.ssh_ok,
        "rsync_ok": result.rsync_ok,
        "wrapper_ok": result.wrapper_ok,
        "docker_ok": result.docker_ok,
        "image_ok": result.image_ok,
        "failure_class": failure,
        "detail": _redact(result.detail or ""),
        "workspace": workspace_payload,
    }


def _resolve_repo_relative(value: str, repo_root: Path) -> Path:
    text = value.replace("${PUTPOCKET_REPO_ROOT}", str(repo_root)).replace("$PUTPOCKET_REPO_ROOT", str(repo_root))
    return Path(text)


def _import_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _local_workspace_backend_ready(live_workspace: bool) -> bool:
    docker = shutil.which("docker")
    if docker is None:
        return False
    if not live_workspace:
        return True
    result = subprocess.run([docker, "info", "--format", "{{json .ServerVersion}}"], text=True, capture_output=True, timeout=10)
    return result.returncode == 0


def _redact(text: str) -> str:
    redacted = text
    for name in ("OPENAI_API_KEY", "HF_TOKEN", "RUNPOD_API_KEY", "GITHUB_TOKEN"):
        value = os.environ.get(name)
        if value:
            redacted = redacted.replace(value, f"<redacted:{name}>")
    return redacted[-2000:]


def dumps_preflight(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)


def _remote_mapping(remote: Any) -> dict[str, Any]:
    return {
        "host": remote.host, "user": remote.user, "port": remote.port,
        "route": str(remote.route),
        "jump_hosts": [{"host": h.host, "user": h.user, "port": h.port} for h in remote.jump_hosts],
        "repository_root": remote.repository_root, "job_root": remote.job_root,
        "identity_file": remote.identity_file, "known_hosts_file": remote.known_hosts_file,
        "strict_host_key_checking": remote.strict_host_key_checking,
        "connection_timeout_sec": remote.connection_timeout_sec,
        "command_timeout_sec": remote.command_timeout_sec, "rsync_timeout_sec": remote.rsync_timeout_sec,
        "wrapper": remote.wrapper, "docker_image": remote.docker_image,
    }
