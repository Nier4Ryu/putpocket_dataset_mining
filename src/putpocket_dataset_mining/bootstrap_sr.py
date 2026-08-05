from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .constants import REPO_ROOT
from .errors import ConfigError
from .execution_config import (
    DOCKER_DISABLED_FOR_STATIC_ONLY,
    HardwareProfile,
    ServerProfile,
    cuda_arch_for_profile,
    default_hardware_for_server,
    ExecutionConfig,
)
from .finalized_dataset import load_finalized_lock, validate_finalized_dataset


def run_bootstrap(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bootstrap_sr")
    parser.add_argument("--phase", choices=["cpu", "gpu", "all"], default=None, help="Compatibility alias for --stage core|validate|all.")
    parser.add_argument("--stage", choices=["preflight", "system", "core", "verifier", "vllm_source", "vllm_build", "validate", "all"], default=None)
    parser.add_argument("--server-profile", choices=[x.value for x in ServerProfile], default="custom")
    parser.add_argument("--hardware-profile", choices=[x.value for x in HardwareProfile], default="auto")
    parser.add_argument("--role", choices=["controller", "verifier", "model_server", "development"], default=None)
    parser.add_argument("--execution-role", choices=["local_controller", "cloud_controller", "verifier_host"], default=None)
    parser.add_argument("--workspace-backend", choices=["local_docker", "remote_ssh_docker"], default=None)
    parser.add_argument("--verifier-backend", choices=["local_docker", "remote_ssh_docker", "disabled"], default=None)
    parser.add_argument("--vllm-profile", choices=["clean", "patched", "skip"], default="patched")
    parser.add_argument("--build-vllm", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--allow-system-install", action="store_true")
    parser.add_argument("--allow-docker-build", action="store_true")
    parser.add_argument("--allow-vllm-build", action="store_true")
    parser.add_argument("--runtime-checks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-dir", default="logs/bootstrap_sr")
    args = parser.parse_args(argv)

    mapping: dict[str, Any] = {}
    if args.role and not args.execution_role:
        mapping["execution_role"] = args.role
    for key in ("execution_role", "workspace_backend", "verifier_backend", "vllm_profile"):
        value = getattr(args, key)
        if value:
            mapping[key] = value
    server = ServerProfile(args.server_profile)
    hardware = HardwareProfile(args.hardware_profile)
    if hardware == HardwareProfile.AUTO:
        hardware = default_hardware_for_server(server)
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        mapping["cuda_arch_list"] = os.environ["TORCH_CUDA_ARCH_LIST"]
    mapping["hardware_profile"] = hardware.value
    mapping["server_profile"] = server.value
    config = ExecutionConfig.from_env_and_mapping(mapping)
    stages = _stages_from_args(args)
    for stage in stages:
        if stage in {"preflight", "system", "core", "verifier", "vllm_source", "vllm_build"}:
            _stage_manifest(stage, config, args)
        elif stage == "validate":
            if args.runtime_checks:
                _gpu_phase(config, args)
            else:
                _stage_manifest("validate", config, args, extra={"runtime_checks": "disabled"})
    return 0


def _stage_manifest(stage: str, config: ExecutionConfig, args: argparse.Namespace, extra: dict[str, Any] | None = None) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        gpu_state = "hidden"
    else:
        gpu_state = "not_required"
    if config.execution_role.value in {"cloud_controller", "model_server"} and config.verifier_backend.value == "disabled":
        docker_state = "LOCAL_DOCKER_SKIPPED_EXPECTED"
    elif config.verifier_backend.value == "disabled":
        docker_state = DOCKER_DISABLED_FOR_STATIC_ONLY
    else:
        config.guard_cloud_local_docker()
        docker_state = "configured"
    lock_status: dict[str, Any]
    try:
        lock_status = validate_finalized_dataset(load_finalized_lock("configs/dataset_mining/classeval_stateful_working_v0.lock.yaml"))
    except Exception as exc:  # noqa: BLE001 - clean source checkouts may not include ignored datasets.
        lock_status = {"status": "skipped_or_failed", "error": f"{exc.__class__.__name__}: {exc}"}
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "timestamp": time.time(),
        "repo": str(REPO_ROOT),
        "git_head": _git(["rev-parse", "HEAD"]),
        "execution": _execution_dict(config),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_state": gpu_state,
        "docker_state": docker_state,
        "class_eval_lock": lock_status,
        "vllm_build_requested": args.build_vllm,
        "vllm_profile": args.vllm_profile,
        "allow_system_install": args.allow_system_install,
        "allow_docker_build": args.allow_docker_build,
        "allow_vllm_build": args.allow_vllm_build,
        "runtime_checks": args.runtime_checks,
        "tools": _tool_report(),
        "cuda_arch_list": config.cuda_arch_list or cuda_arch_for_profile(config.hardware_profile),
        "dry_run": args.dry_run,
    }
    if extra:
        manifest.update(extra)
    _write_manifest(args.manifest_dir, stage, manifest)


def _gpu_phase(config: ExecutionConfig, args: argparse.Namespace) -> None:
    if config.hardware_profile == HardwareProfile.CPU:
        raise ConfigError("GPU bootstrap phase requires a non-cpu hardware profile.")
    actual = _nvidia_smi_query()
    arch = cuda_arch_for_profile(config.hardware_profile)
    manifest = {
        "schema_version": 1,
        "stage": "validate",
        "timestamp": time.time(),
        "git_head": _git(["rev-parse", "HEAD"]),
        "hardware_profile": config.hardware_profile.value,
        "server_profile": config.server_profile.value,
        "target_cuda_arch_list": config.cuda_arch_list or arch,
        "nvidia_smi": actual,
        "vllm_profile": args.vllm_profile,
        "build_vllm": args.build_vllm,
        "dry_run": args.dry_run,
    }
    _write_manifest(args.manifest_dir, "gpu", manifest)


def _stages_from_args(args: argparse.Namespace) -> list[str]:
    if args.stage:
        if args.stage == "all":
            return ["preflight", "system", "core", "verifier", "vllm_source", "vllm_build", "validate"]
        return [args.stage]
    if args.phase == "cpu":
        return ["core"]
    if args.phase == "gpu":
        return ["validate"]
    if args.phase == "all":
        return ["core", "validate"]
    raise SystemExit("--stage is required")


def _write_manifest(root: str, phase: str, manifest: dict[str, Any]) -> None:
    path = Path(root) / f"{phase}_bootstrap_manifest_{int(time.time())}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(path)


def _tool_report() -> dict[str, Any]:
    return {name: {"path": shutil.which(name)} for name in ["git", "ssh", "rsync", "docker", "nvcc", "nvidia-smi"]}


def _execution_dict(config: ExecutionConfig) -> dict[str, Any]:
    remote = config.remote
    return {
        "execution_role": config.execution_role.value,
        "workspace_backend": config.workspace_backend.value,
        "verifier_backend": config.verifier_backend.value,
        "hardware_profile": config.hardware_profile.value,
        "server_profile": config.server_profile.value,
        "model_id": config.model_id,
        "model_path": config.model_path,
        "vllm_profile": config.vllm_profile,
        "cuda_arch_list": config.cuda_arch_list,
        "remote": {
            "host": remote.host,
            "user": remote.user,
            "port": remote.port,
            "route": remote.route.value,
            "jump_hosts": [host.__dict__ for host in remote.jump_hosts],
            "repository_root": remote.repository_root,
            "job_root": remote.job_root,
            "strict_host_key_checking": remote.strict_host_key_checking,
            "docker_image": remote.docker_image,
            "max_concurrent_jobs": remote.max_concurrent_jobs,
        },
    }


def _git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, cwd=REPO_ROOT).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _nvidia_smi_query() -> list[dict[str, str]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
            text=True,
        )
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for line in out.splitlines():
        name, cap, memory = [x.strip() for x in line.split(",", 2)]
        rows.append({"name": name, "compute_capability": cap, "memory_total": memory})
    return rows


if __name__ == "__main__":
    raise SystemExit(run_bootstrap())
