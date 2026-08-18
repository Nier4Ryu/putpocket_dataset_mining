from __future__ import annotations

import importlib.metadata
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .cluster_config import ClusterProfile
from .cluster_safety import (
    allocated_gpu_selector,
    bounded_text,
    require_slurm_allocation,
    safe_absolute_path,
    validate_secret_free_command,
)


@dataclass(frozen=True)
class ProbeResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class ProbeRunner:
    def run(self, command: Sequence[str]) -> ProbeResult:
        completed = subprocess.run(list(command), text=True, capture_output=True, check=False, timeout=60)
        return ProbeResult(tuple(command), completed.returncode, completed.stdout, completed.stderr)


def capture_run_manifest(
    *,
    profile: ClusterProfile,
    command: Sequence[str],
    artifact_root: str | Path,
    git_executable: str | Path,
    nvidia_smi_executable: str | Path,
    nvcc_executable: str | Path,
    model_revision: str | None,
    env: Mapping[str, str] | None = None,
    runner: ProbeRunner | None = None,
) -> dict[str, object]:
    allocation = require_slurm_allocation(env)
    root = safe_absolute_path(artifact_root, "artifact_root")
    git = safe_absolute_path(git_executable, "git_executable")
    nvidia_smi = safe_absolute_path(nvidia_smi_executable, "nvidia_smi_executable")
    nvcc = safe_absolute_path(nvcc_executable, "nvcc_executable")
    safe_command = validate_secret_free_command(command)
    run = runner or ProbeRunner()
    gpu_selector = allocated_gpu_selector(env)
    git_sha = _probe(run, [str(git), "rev-parse", "HEAD"])
    inventory_command = [
        str(nvidia_smi),
        "--query-gpu=index,name,uuid,driver_version,pci.bus_id,compute_cap",
        "--format=csv,noheader",
    ]
    if gpu_selector:
        inventory_command.append(f"--id={gpu_selector}")
    gpu_inventory = _probe(
        run,
        inventory_command,
    )
    gpu_topology = _probe(run, [str(nvidia_smi), "topo", "-m"])
    nvcc_version = _probe(run, [str(nvcc), "--version"])
    versions = {}
    for distribution in ("torch", "vllm", "flashinfer-python", "flashinfer-cubin", "nvidia-nccl-cu12"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return {
        "schema_version": 1,
        "captured_at_unix": int(time.time()),
        "git_sha": git_sha["stdout"].strip() if git_sha["returncode"] == 0 else "unavailable",
        "profile_id": profile.profile_id,
        "model": {"id": profile.model_id, "revision": model_revision or "unresolved"},
        "slurm": allocation,
        "gpu": {"allocated_selector": gpu_selector or "not-reported", "inventory": gpu_inventory, "topology": gpu_topology},
        "versions": {"packages": versions, "cuda_toolkit": nvcc_version},
        "exact_command": safe_command,
        "artifact_root": str(root),
        "checkpoint_hash_policy": "metadata_only_no_full_tensor_hash",
        "environment_capture_policy": "allowlisted_fields_only_no_environment_dump",
    }


def write_manifest(path: str | Path, payload: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def execute_guarded(
    *,
    action: str,
    profile: ClusterProfile,
    command: Sequence[str],
    artifact_root: str | Path,
    git_executable: str | Path,
    nvidia_smi_executable: str | Path,
    nvcc_executable: str | Path,
    model_revision: str | None,
    env: Mapping[str, str] | None = None,
) -> int:
    allowed = {
        "environment-build",
        "dependency-install",
        "checkpoint-stage",
        "gpu-smoke",
        "model-load",
        "benchmark",
        "one-shot-generation",
    }
    if action not in allowed:
        raise ValueError(f"Unknown guarded Cluster action: {action}")
    root = safe_absolute_path(artifact_root, "artifact_root")
    manifest = capture_run_manifest(
        profile=profile,
        command=command,
        artifact_root=root,
        git_executable=git_executable,
        nvidia_smi_executable=nvidia_smi_executable,
        nvcc_executable=nvcc_executable,
        model_revision=model_revision,
        env=env,
    )
    manifest["action"] = action
    manifest["status"] = "started"
    write_manifest(root / "run_manifest.json", manifest)
    result = subprocess.run(validate_secret_free_command(command), env=dict(os.environ if env is None else env), check=False)
    manifest["status"] = "passed" if result.returncode == 0 else "failed"
    manifest["returncode"] = result.returncode
    write_manifest(root / "run_manifest.json", manifest)
    return result.returncode


def _probe(runner: ProbeRunner, command: Sequence[str]) -> dict[str, object]:
    result = runner.run(command)
    return {
        "returncode": result.returncode,
        "stdout": bounded_text(result.stdout),
        "stderr": bounded_text(result.stderr),
    }
