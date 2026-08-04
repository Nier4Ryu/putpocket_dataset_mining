from __future__ import annotations

import argparse
import json
import os
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
    parser.add_argument("--phase", choices=["cpu", "gpu", "all"], required=True)
    parser.add_argument("--server-profile", choices=[x.value for x in ServerProfile], default="custom")
    parser.add_argument("--hardware-profile", choices=[x.value for x in HardwareProfile], default="auto")
    parser.add_argument("--execution-role", choices=["local_controller", "cloud_controller", "verifier_host"], default=None)
    parser.add_argument("--workspace-backend", choices=["local_docker", "remote_ssh_docker"], default=None)
    parser.add_argument("--verifier-backend", choices=["local_docker", "remote_ssh_docker", "disabled"], default=None)
    parser.add_argument("--vllm-profile", choices=["clean", "patched"], default="patched")
    parser.add_argument("--build-vllm", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-dir", default="logs/bootstrap_sr")
    args = parser.parse_args(argv)

    mapping: dict[str, Any] = {}
    for key in ("execution_role", "workspace_backend", "verifier_backend", "vllm_profile"):
        value = getattr(args, key)
        if value:
            mapping[key] = value
    server = ServerProfile(args.server_profile)
    hardware = HardwareProfile(args.hardware_profile)
    if hardware == HardwareProfile.AUTO:
        hardware = default_hardware_for_server(server)
    mapping["hardware_profile"] = hardware.value
    mapping["server_profile"] = server.value
    config = ExecutionConfig.from_env_and_mapping(mapping)
    if args.phase in {"cpu", "all"}:
        _cpu_phase(config, args)
    if args.phase in {"gpu", "all"}:
        _gpu_phase(config, args)
    return 0


def _cpu_phase(config: ExecutionConfig, args: argparse.Namespace) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        gpu_state = "hidden"
    else:
        gpu_state = "not_required"
    if config.verifier_backend.value == "disabled":
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
        "phase": "cpu",
        "timestamp": time.time(),
        "repo": str(REPO_ROOT),
        "git_head": _git(["rev-parse", "HEAD"]),
        "execution": config.__dict__ | {"remote": config.remote.__dict__},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_state": gpu_state,
        "docker_state": docker_state,
        "class_eval_lock": lock_status,
        "vllm_build_requested": args.build_vllm,
        "dry_run": args.dry_run,
    }
    _write_manifest(args.manifest_dir, "cpu", manifest)


def _gpu_phase(config: ExecutionConfig, args: argparse.Namespace) -> None:
    if config.hardware_profile == HardwareProfile.CPU:
        raise ConfigError("GPU bootstrap phase requires a non-cpu hardware profile.")
    actual = _nvidia_smi_query()
    arch = cuda_arch_for_profile(config.hardware_profile)
    manifest = {
        "schema_version": 1,
        "phase": "gpu",
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


def _write_manifest(root: str, phase: str, manifest: dict[str, Any]) -> None:
    path = Path(root) / f"{phase}_bootstrap_manifest_{int(time.time())}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(path)


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
