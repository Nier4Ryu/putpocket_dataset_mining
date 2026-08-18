from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .cluster_config import VLLM_026_SHA, validate_environment_lock
from .cluster_safety import require_slurm_allocation, safe_absolute_path, validate_secret_free_command
from .errors import ConfigError


@dataclass(frozen=True)
class BootstrapPlan:
    lock_id: str
    environment_root: Path
    vllm_source_root: Path
    cache_root: Path
    build_jobs: int
    commands: tuple[tuple[str, ...], ...]
    environment: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "preset": "cluster-h200-sm90",
            "lock_id": self.lock_id,
            "allocation_required_for_execute": True,
            "environment_root": str(self.environment_root),
            "vllm_source_root": str(self.vllm_source_root),
            "cache_root": str(self.cache_root),
            "build_jobs": self.build_jobs,
            "commands": [list(command) for command in self.commands],
            "environment": dict(self.environment),
        }


def build_bootstrap_plan(
    *,
    lock_path: str | Path,
    repository_root: str | Path,
    environment_root: str | Path,
    vllm_source_root: str | Path,
    cache_root: str | Path,
    python_executable: str | Path,
    uv_executable: str | Path,
    git_executable: str | Path,
    build_jobs: int,
) -> BootstrapPlan:
    lock = validate_environment_lock(lock_path)
    repo = safe_absolute_path(repository_root, "repository_root")
    env_root = safe_absolute_path(environment_root, "environment_root")
    source_root = safe_absolute_path(vllm_source_root, "vllm_source_root")
    cache = safe_absolute_path(cache_root, "cache_root")
    python = safe_absolute_path(python_executable, "python_executable")
    uv = safe_absolute_path(uv_executable, "uv_executable")
    git = safe_absolute_path(git_executable, "git_executable")
    if build_jobs < 1:
        raise ConfigError("build_jobs must be positive")
    runtime = lock["runtime"]
    torch = runtime["torch"]
    vllm = lock["vllm"]
    torch_requirement = f"{torch['wheel_url']}#sha256={torch['wheel_sha256']}"
    env_python = env_root / "bin" / "python"
    commands: list[tuple[str, ...]] = [
        (str(uv), "venv", "--python", str(runtime["python"]["version"]), str(env_root)),
        (str(uv), "pip", "install", "--python", str(env_python), torch_requirement),
        (str(uv), "pip", "install", "--python", str(env_python), "-e", f"{repo}[dev]"),
        (
            str(git),
            "clone",
            "--filter=blob:none",
            "--branch",
            str(vllm["ref"]),
            "--single-branch",
            "--no-checkout",
            str(vllm["repository"]),
            str(source_root),
        ),
        (str(git), "-C", str(source_root), "checkout", "--detach", str(vllm["commit"])),
        (
            str(uv),
            "pip",
            "install",
            "--python",
            str(env_python),
            "-r",
            str(source_root / "requirements" / "cuda.txt"),
            torch_requirement,
        ),
        (
            str(uv),
            "pip",
            "install",
            "--python",
            str(env_python),
            "-r",
            str(source_root / "requirements" / "build.txt"),
        ),
        (
            str(uv),
            "pip",
            "install",
            "--python",
            str(env_python),
            "--no-build-isolation",
            "--no-deps",
            "-e",
            str(source_root),
        ),
    ]
    for command in commands:
        validate_secret_free_command(command)
    return BootstrapPlan(
        lock_id=str(lock["lock_id"]),
        environment_root=env_root,
        vllm_source_root=source_root,
        cache_root=cache,
        build_jobs=build_jobs,
        commands=tuple(commands),
        environment={
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "MAX_JOBS": str(build_jobs),
            "CMAKE_BUILD_PARALLEL_LEVEL": str(build_jobs),
            "NVCC_THREADS": "1",
            "UV_CACHE_DIR": str(cache / "uv"),
            "VLLM_CACHE_ROOT": str(cache / "vllm"),
            "HF_HOME": str(cache / "huggingface"),
            "DG_JIT_CACHE_DIR": str(cache / "deep_gemm"),
        },
    )


def execute_bootstrap_plan(
    plan: BootstrapPlan,
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    require_slurm_allocation(env)
    plan.environment_root.parent.mkdir(parents=True, exist_ok=True)
    plan.vllm_source_root.parent.mkdir(parents=True, exist_ok=True)
    plan.cache_root.mkdir(parents=True, exist_ok=True)
    execution_env = dict(os.environ if env is None else env)
    execution_env.update(plan.environment)

    commands = list(plan.commands)
    clone_index = 3
    checkout_index = 4
    source = plan.vllm_source_root
    if source.exists():
        if not (source / ".git").exists():
            raise ConfigError(f"vLLM source path exists but is not a Git checkout: {source}")
        git = commands[clone_index][0]
        status = _run((git, "-C", str(source), "status", "--porcelain"), execution_env, capture=True)
        if status.stdout.strip():
            raise ConfigError(f"vLLM source must be clean before Cluster bootstrap: {source}")
        head = _run((git, "-C", str(source), "rev-parse", "HEAD"), execution_env, capture=True).stdout.strip()
        if head != VLLM_026_SHA:
            raise ConfigError(
                f"vLLM source commit mismatch: expected {VLLM_026_SHA}, found {head}; "
                "reconcile the commit outside this bootstrap without overwriting local work"
            )
        del commands[checkout_index]
        del commands[clone_index]
    for command in commands:
        _run(command, execution_env, capture=False)


def bootstrap_plan_json(plan: BootstrapPlan) -> str:
    return json.dumps(plan.as_dict(), indent=2, sort_keys=True)


def _run(command: tuple[str, ...], env: Mapping[str, str], *, capture: bool) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        text=True,
        env=dict(env),
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise ConfigError(f"Cluster environment bootstrap command failed ({result.returncode}): {command[0]} {detail}")
    return result
