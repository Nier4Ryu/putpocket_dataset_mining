from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from .errors import ConfigError


RUNPOD_REPO_ROOT = Path("/workspace/putpocket_dataset_mining")
RUNPOD_ENV_PATH = RUNPOD_REPO_ROOT / "Putpocket_env"
RUNPOD_UV_CACHE = RUNPOD_REPO_ROOT / ".cache" / "uv"
RUNPOD_UV_PYTHON = RUNPOD_UV_CACHE / "python"
RUNPOD_HF_HOME = RUNPOD_REPO_ROOT / "models" / "hf"
RUNPOD_BASE_IMAGE_CONTRACT = Path("configs/env/runpod_base_image.lock.yaml")
RUNPOD_DEV_CONTRACT = Path("configs/env/runpod_dev.lock.yaml")

ARCH_PROFILES = {
    "portable-nvidia": "8.6 9.0 10.0 12.0",
    "rtx3090": "8.6",
    "hopper": "9.0",
    "blackwell-datacenter": "10.0",
    "blackwell-rtx": "12.0",
}
SUPPORTED_CUDA_ARCHES = ("8.6", "9.0", "10.0", "12.0")
SM_EVIDENCE_KEYS = ("sm_86", "sm_90", "sm_100", "sm_120")
NATIVE_CAPABILITY_TO_ARCH = {
    "8.6": "8.6",
    "9.0": "9.0",
    "10.0": "10.0",
    "12.0": "12.0",
}
PRESET_DEFAULT_ARCH_PROFILE = {
    "server1_rtx3090": "rtx3090",
    "runpod_hopper": "hopper",
    "runpod-dev": "portable-nvidia",
    "server2_blackwell": "blackwell-rtx",
    "server2_rtxpro6000_blackwell": "blackwell-rtx",
}


@dataclass(frozen=True)
class RunpodPlan:
    repo_root: Path
    env_path: Path
    persistent_root: Path
    storage_kind: str
    cuda_arch_profile: str
    cuda_arch_list: str
    base_image_contract: Path
    torch_contract: Path
    dry_run: bool
    doctor_only: bool
    skip_vllm_build: bool
    force_vllm_build: bool
    skip_gpu_smoke: bool
    cpu_count_detected: int
    build_jobs_requested: int
    build_jobs_effective: int
    nvcc_threads: int

    def as_dict(self) -> dict[str, Any]:
        env = self.environment()
        return {
            "schema_version": 1,
            "preset": "runpod-dev",
            "repo_root": str(self.repo_root),
            "environment": str(self.env_path),
            "persistent_root": str(self.persistent_root),
            "storage_kind": self.storage_kind,
            "dry_run": self.dry_run,
            "doctor_only": self.doctor_only,
            "skip_vllm_build": self.skip_vllm_build,
            "force_vllm_build": self.force_vllm_build,
            "skip_gpu_smoke": self.skip_gpu_smoke,
            "cpu_count_detected": self.cpu_count_detected,
            "build_jobs_requested": self.build_jobs_requested,
            "build_jobs_effective": self.build_jobs_effective,
            "max_jobs": self.build_jobs_effective,
            "cmake_build_parallel_level": self.build_jobs_effective,
            "nvcc_threads": self.nvcc_threads,
            "stages": [
                "base-image-contract",
                "network-volume-guard",
                "uv-python",
                "environment",
                "torch-contract",
                "project-editable",
                "externals-editable",
                "runtime-fingerprint",
                "doctor",
                "manifest",
            ],
            "environment_variables": env,
            "mutations": [] if self.dry_run or self.doctor_only else [
                "create/reuse uv-managed Python under persistent UV_PYTHON_INSTALL_DIR",
                "create/reuse Putpocket_env under the repository root",
                "install exact RunPod torch wheel by URL and SHA-256",
                "install project, vLLM, and LMCache editable",
                "write runtime fingerprint and build manifests",
            ],
            "vllm_developer_commands": vllm_developer_commands(self.repo_root, self.build_jobs_effective, self.nvcc_threads),
            "docker_build_args": docker_build_args(self.cuda_arch_list),
        }

    def environment(self) -> dict[str, str]:
        return {
            "PUTPOCKET_REPO_ROOT": str(self.repo_root),
            "PUTPOCKET_ENV_PATH": str(self.env_path),
            "PUTPOCKET_STORAGE_KIND": self.storage_kind,
            "UV_PROJECT_ENVIRONMENT": str(self.env_path),
            "UV_CACHE_DIR": str(self.repo_root / ".cache" / "uv"),
            "UV_PYTHON_INSTALL_DIR": str(self.repo_root / ".cache" / "uv" / "python"),
            "VLLM_CACHE_ROOT": str(self.repo_root / ".cache" / "vllm"),
            "TORCH_HOME": str(self.repo_root / ".cache" / "torch"),
            "HF_HOME": str(self.repo_root / "models" / "hf"),
            "TMPDIR": str(self.repo_root / "builds" / "tmp"),
            "PUTPOCKET_CUDA_ARCH_PROFILE": self.cuda_arch_profile,
            "PUTPOCKET_CUDA_ARCH_LIST": self.cuda_arch_list,
            "TORCH_CUDA_ARCH_LIST": self.cuda_arch_list,
            "PUTPOCKET_BUILD_JOBS": str(self.build_jobs_effective),
            "MAX_JOBS": str(self.build_jobs_effective),
            "CMAKE_BUILD_PARALLEL_LEVEL": str(self.build_jobs_effective),
            "NVCC_THREADS": str(self.nvcc_threads),
            "CCACHE_NOHASHDIR": "true",
        }


def detect_cpu_count() -> int:
    """Return the build-visible CPU count using the same contract as `nproc`."""

    tool = shutil.which("nproc")
    if not tool:
        raise ConfigError("runpod-dev build parallelism requires nproc")
    proc = subprocess.run([tool], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise ConfigError(f"nproc failed: {proc.stderr.strip()}")
    return normalize_build_jobs(proc.stdout.strip(), source="nproc")


def normalize_build_jobs(value: str | int, *, source: str) -> int:
    try:
        jobs = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{source} must be a positive integer, got {value!r}") from exc
    if jobs <= 0:
        raise ConfigError(f"{source} must be a positive integer, got {value!r}")
    return jobs


def resolve_build_jobs(
    cli_build_jobs: int | None,
    *,
    env: dict[str, str] | None = None,
    cpu_count: int | None = None,
) -> tuple[int, int]:
    """Resolve CLI > PUTPOCKET_BUILD_JOBS > nproc and return (CPUs, jobs)."""

    env = env if env is not None else os.environ
    detected = normalize_build_jobs(cpu_count if cpu_count is not None else detect_cpu_count(), source="nproc")
    if cli_build_jobs is not None:
        return detected, normalize_build_jobs(cli_build_jobs, source="--build-jobs")
    if env.get("PUTPOCKET_BUILD_JOBS") is not None:
        return detected, normalize_build_jobs(env["PUTPOCKET_BUILD_JOBS"], source="PUTPOCKET_BUILD_JOBS")
    return detected, detected


def normalize_cuda_arch_list(value: str) -> str:
    values = value.split()
    if not values:
        raise ConfigError("CUDA architecture list must not be empty")
    seen: set[str] = set()
    for arch in values:
        if not re.fullmatch(r"\d+\.\d+", arch):
            raise ConfigError(f"invalid CUDA architecture syntax: {arch}")
        if arch not in SUPPORTED_CUDA_ARCHES:
            raise ConfigError(f"unsupported CUDA architecture: {arch}")
        if arch in seen:
            raise ConfigError(f"duplicate CUDA architecture: {arch}")
        seen.add(arch)
    return " ".join(values)


def resolve_cuda_arch_profile(profile: str | None) -> str:
    profile = profile or "portable-nvidia"
    if profile == "native":
        return profile
    if profile not in ARCH_PROFILES:
        raise ConfigError(f"unknown CUDA architecture profile: {profile}")
    return profile


def resolve_cuda_arch_list(profile: str, explicit: str | None = None) -> str:
    if explicit is not None:
        return normalize_cuda_arch_list(explicit)
    profile = resolve_cuda_arch_profile(profile)
    if profile == "native":
        return detect_native_cuda_arch_list()
    return normalize_cuda_arch_list(ARCH_PROFILES[profile])


def resolve_cuda_arch_contract(
    *,
    preset_default_profile: str,
    cli_arch_profile: str | None = None,
    cli_arch_list: str | None = None,
    env: dict[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve CUDA arch settings with CLI list > CLI profile > env list > env profile > preset."""

    env = env if env is not None else os.environ
    if cli_arch_list is not None:
        profile = cli_arch_profile or env.get("PUTPOCKET_CUDA_ARCH_PROFILE") or preset_default_profile
        return resolve_cuda_arch_profile(profile), normalize_cuda_arch_list(cli_arch_list)
    if cli_arch_profile is not None:
        profile = resolve_cuda_arch_profile(cli_arch_profile)
        return profile, resolve_cuda_arch_list(profile)
    if env.get("PUTPOCKET_CUDA_ARCH_LIST"):
        profile = env.get("PUTPOCKET_CUDA_ARCH_PROFILE") or preset_default_profile
        return resolve_cuda_arch_profile(profile), normalize_cuda_arch_list(str(env["PUTPOCKET_CUDA_ARCH_LIST"]))
    profile = resolve_cuda_arch_profile(env.get("PUTPOCKET_CUDA_ARCH_PROFILE") or preset_default_profile)
    return profile, resolve_cuda_arch_list(profile)


def detect_native_cuda_arch_list() -> str:
    smi = shutil.which("nvidia-smi")
    if not smi:
        raise ConfigError("native CUDA architecture detection requires nvidia-smi")
    proc = subprocess.run(
        [smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise ConfigError(f"native CUDA architecture detection failed: {proc.stderr.strip()}")
    caps = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    if not caps:
        raise ConfigError("native CUDA architecture detection found no visible GPUs")
    unknown = [cap for cap in caps if cap not in NATIVE_CAPABILITY_TO_ARCH]
    if unknown:
        raise ConfigError(f"unsupported native CUDA capability: {', '.join(unknown)}")
    detected_arches = {NATIVE_CAPABILITY_TO_ARCH[cap] for cap in caps}
    return normalize_cuda_arch_list(" ".join(arch for arch in SUPPORTED_CUDA_ARCHES if arch in detected_arches))


def build_runpod_plan(
    *,
    repo_root: Path,
    persistent_root: str | None,
    storage_kind: str | None,
    cuda_arch_profile: str | None,
    cuda_arch_list: str | None,
    base_image_contract: str | None,
    dry_run: bool,
    doctor_only: bool,
    skip_vllm_build: bool,
    force_vllm_build: bool,
    skip_gpu_smoke: bool,
    build_jobs: int | None = None,
    cpu_count: int | None = None,
    env: dict[str, str] | None = None,
) -> RunpodPlan:
    env = env if env is not None else os.environ
    root = Path(persistent_root or env.get("PUTPOCKET_REPO_ROOT", str(RUNPOD_REPO_ROOT))).expanduser()
    env_path = Path(env.get("PUTPOCKET_ENV_PATH", str(root / "Putpocket_env"))).expanduser()
    storage = storage_kind or env.get("PUTPOCKET_STORAGE_KIND", "network-volume")
    arch_profile, arch_list = resolve_cuda_arch_contract(
        preset_default_profile=PRESET_DEFAULT_ARCH_PROFILE["runpod-dev"],
        cli_arch_profile=cuda_arch_profile,
        cli_arch_list=cuda_arch_list,
        env=env,
    )
    detected_cpus, resolved_build_jobs = resolve_build_jobs(build_jobs, env=env, cpu_count=cpu_count)
    return RunpodPlan(
        repo_root=root,
        env_path=env_path,
        persistent_root=root.parent,
        storage_kind=storage,
        cuda_arch_profile=arch_profile,
        cuda_arch_list=arch_list,
        base_image_contract=repo_root / (base_image_contract or str(RUNPOD_BASE_IMAGE_CONTRACT)),
        torch_contract=repo_root / RUNPOD_DEV_CONTRACT,
        dry_run=dry_run,
        doctor_only=doctor_only,
        skip_vllm_build=skip_vllm_build,
        force_vllm_build=force_vllm_build,
        skip_gpu_smoke=skip_gpu_smoke,
        cpu_count_detected=detected_cpus,
        build_jobs_requested=resolved_build_jobs,
        build_jobs_effective=resolved_build_jobs,
        nvcc_threads=1,
    )


def validate_base_image_contract(path: Path) -> dict[str, Any]:
    data = read_structured(path)
    required = {
        "schema_version",
        "image_repository",
        "image_tag",
        "index_digest",
        "amd64_manifest_digest",
        "cuda_version",
        "uv",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ConfigError(f"base-image contract is missing fields: {', '.join(missing)}")
    expected_index = "sha256:bd4e2680a261c212f1e2fea241606f71497dc67a417f73175d794ec8212b5ba8"
    expected_amd64 = "sha256:38804006c937a83f28f63a959abcee688042072319c8614ad57b350958a30bd3"
    if data["index_digest"] != expected_index or data["amd64_manifest_digest"] != expected_amd64:
        raise ConfigError("base-image contract digest does not match the pinned CUDA 12.9.1 devel image")
    return data


def validate_torch_contract(path: Path, *, require_resolved: bool) -> dict[str, Any]:
    data = read_structured(path)
    torch = data.get("torch", {})
    if data.get("python", {}).get("version") != "3.13.14" or torch.get("version") != "2.10.0":
        raise ConfigError("RunPod contract must pin Python 3.13.14 and torch 2.10.0")
    if not torch.get("wheel_url") or not re.fullmatch(r"[0-9a-f]{64}", str(torch.get("wheel_sha256", ""))):
        raise ConfigError("RunPod torch contract requires a wheel URL and SHA-256")
    if require_resolved and data.get("provenance_status") != "resolved":
        raise ConfigError("RUNPOD_TORCH_CONTRACT_BLOCKED")
    return data


def validate_network_volume(plan: RunpodPlan) -> dict[str, Any]:
    if plan.storage_kind != "network-volume":
        raise ConfigError("runpod-dev requires --storage-kind network-volume")
    if plan.persistent_root != Path("/workspace"):
        raise ConfigError("runpod-dev requires the repository to live under /workspace")
    volume_id = os.environ.get("RUNPOD_NETWORK_VOLUME_ID") or os.environ.get("RUNPOD_VOLUME_ID")
    if not volume_id:
        raise ConfigError("RUNPOD_VOLUME_ID (or legacy RUNPOD_NETWORK_VOLUME_ID) must be set for runpod-dev before mutating the environment")
    if not plan.persistent_root.exists() or not os.access(plan.persistent_root, os.W_OK):
        raise ConfigError("/workspace must exist and be writable")
    usage = shutil.disk_usage(plan.persistent_root)
    return {
        "path": str(plan.persistent_root),
        "volume_id": volume_id,
        "free_bytes": usage.free,
        "total_bytes": usage.total,
    }


def runtime_fingerprint(plan: RunpodPlan, base: dict[str, Any], torch_contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "base_image_digest": base["index_digest"],
        "cpu_architecture": platform.machine(),
        "glibc": platform.libc_ver(),
        "python": torch_contract.get("python", {}).get("version"),
        "torch": torch_contract.get("torch", {}).get("version"),
        "torch_cuda": torch_contract.get("torch", {}).get("torch_cuda"),
        "cuda_toolkit": base.get("cuda_version"),
        "vllm_sha": _git_head(plan.repo_root / "externals" / "vllm"),
        "lmcache_sha": _git_head(plan.repo_root / "externals" / "lmcache"),
        "project_lock_hash": _sha256(plan.repo_root / RUNPOD_DEV_CONTRACT),
        "cuda_arch_profile": plan.cuda_arch_profile,
        "cuda_arch_list": plan.cuda_arch_list,
        "torch_cuda_arch_list": plan.cuda_arch_list,
        "cpu_count_detected": plan.cpu_count_detected,
        "build_jobs_requested": plan.build_jobs_requested,
        "build_jobs_effective": plan.build_jobs_effective,
        "max_jobs": plan.build_jobs_effective,
        "cmake_build_parallel_level": plan.build_jobs_effective,
        "nvcc_threads": plan.nvcc_threads,
    }


def build_manifest(plan: RunpodPlan, fingerprint: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "classification": "SOURCE_PORTABLE_REBUILD_REQUIRED",
        "project_sha": _git_head(plan.repo_root),
        "vllm_sha": fingerprint.get("vllm_sha"),
        "lmcache_sha": fingerprint.get("lmcache_sha"),
        "base_image_digest": fingerprint.get("base_image_digest"),
        "python": fingerprint.get("python"),
        "torch": fingerprint.get("torch"),
        "torch_cuda": fingerprint.get("torch_cuda"),
        "cuda_toolkit": fingerprint.get("cuda_toolkit"),
        "cuda_arch_profile": plan.cuda_arch_profile,
        "requested_cuda_arch_list": plan.cuda_arch_list.split(),
        "torch_cuda_arch_list": plan.cuda_arch_list,
        "cpu_count_detected": plan.cpu_count_detected,
        "build_jobs_requested": plan.build_jobs_requested,
        "build_jobs_effective": plan.build_jobs_effective,
        "max_jobs": plan.build_jobs_effective,
        "cmake_build_parallel_level": plan.build_jobs_effective,
        "nvcc_threads": plan.nvcc_threads,
        "requested_architecture_profile": plan.cuda_arch_profile,
        "requested_architecture_list": plan.cuda_arch_list,
        "actual_compiled_architecture_evidence": {key: "NOT_INSPECTED" for key in SM_EVIDENCE_KEYS},
        "heavy_multiarch_build_executed": os.environ.get("PUTPOCKET_ALLOW_HEAVY_MULTIARCH_BUILD") == "1",
        "vllm_editable_build": {
            "environment": {
                "TORCH_CUDA_ARCH_LIST": plan.cuda_arch_list,
                "PUTPOCKET_BUILD_JOBS": str(plan.build_jobs_effective),
                "MAX_JOBS": str(plan.build_jobs_effective),
                "CMAKE_BUILD_PARALLEL_LEVEL": str(plan.build_jobs_effective),
                "NVCC_THREADS": str(plan.nvcc_threads),
            },
            "command": vllm_editable_build_command(plan.repo_root, plan.cuda_arch_list, plan.build_jobs_effective, plan.nvcc_threads),
            "heavy_build_required_for_portable_multiarch": plan.cuda_arch_list == ARCH_PROFILES["portable-nvidia"],
            "build_start_time": None,
            "build_end_time": None,
            "build_wall_time_seconds": None,
            "ram_before_build": None,
            "ram_after_build": None,
            "memory_related_build_fallback_used": False,
            "fallback_effective_jobs": None,
        },
        "docker_build_args": docker_build_args(plan.cuda_arch_list),
        "native_scratch": plan.environment()["TMPDIR"],
    }


def vllm_editable_build_command(repo_root: Path, cuda_arch_list: str, build_jobs: int = 1, nvcc_threads: int = 1) -> str:
    prefix = f"cd {repo_root}/externals/vllm"
    return (
        f"{prefix} && export TORCH_CUDA_ARCH_LIST=\"{cuda_arch_list}\" && "
        f"export PUTPOCKET_BUILD_JOBS={build_jobs} MAX_JOBS={build_jobs} "
        f"CMAKE_BUILD_PARALLEL_LEVEL={build_jobs} NVCC_THREADS={nvcc_threads} && "
        "uv pip install -r requirements/build.txt && "
        "CCACHE_NOHASHDIR=true uv pip install --no-build-isolation --no-deps -e ."
    )


def docker_build_args(cuda_arch_list: str) -> list[str]:
    return ["--build-arg", f"torch_cuda_arch_list={normalize_cuda_arch_list(cuda_arch_list)}"]


def vllm_developer_commands(repo_root: Path, build_jobs: int = 1, nvcc_threads: int = 1) -> dict[str, str]:
    prefix = f"cd {repo_root}/externals/vllm"
    portable = ARCH_PROFILES["portable-nvidia"]
    return {
        "python_only_change": "edit externals/vllm Python files; no reinstall required for editable install",
        "cpp_or_cuda_change": vllm_editable_build_command(repo_root, portable, build_jobs, nvcc_threads),
        "clean_rebuild": f"{prefix} && rm -rf build && export TORCH_CUDA_ARCH_LIST=\"{portable}\" PUTPOCKET_BUILD_JOBS={build_jobs} MAX_JOBS={build_jobs} CMAKE_BUILD_PARALLEL_LEVEL={build_jobs} NVCC_THREADS={nvcc_threads} && CCACHE_NOHASHDIR=true uv pip install --no-build-isolation -e .",
        "build_doctor": f"{prefix} && export TORCH_CUDA_ARCH_LIST=\"{portable}\" PUTPOCKET_BUILD_JOBS={build_jobs} MAX_JOBS={build_jobs} CMAKE_BUILD_PARALLEL_LEVEL={build_jobs} NVCC_THREADS={nvcc_threads} && uv pip install -r requirements/build.txt",
        "serve_later": "python -m vllm.entrypoints.openai.api_server --help",
    }


def parse_cuobjdump_arches(output: str) -> dict[str, str]:
    found = {key: "MISSING" for key in SM_EVIDENCE_KEYS}
    for match in re.findall(r"sm_(86|90|100|120)", output):
        found[f"sm_{match}"] = "PRESENT"
    return found


def inspect_binary_architectures(paths: list[Path], *, cuobjdump: str | None = None) -> dict[str, str]:
    if not paths:
        return {key: "NOT_APPLICABLE" for key in SM_EVIDENCE_KEYS}
    tool = cuobjdump or shutil.which("cuobjdump")
    if not tool:
        return {key: "NOT_INSPECTED" for key in SM_EVIDENCE_KEYS}
    combined = {key: "MISSING" for key in SM_EVIDENCE_KEYS}
    for path in paths:
        proc = subprocess.run([tool, "--list-elf", str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        evidence = parse_cuobjdump_arches(proc.stdout + "\n" + proc.stderr) if proc.returncode == 0 else {key: "NOT_INSPECTED" for key in SM_EVIDENCE_KEYS}
        for key, value in evidence.items():
            if combined[key] != "PRESENT":
                combined[key] = value
    return combined


def read_structured(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"missing contract file: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"} and yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ConfigError(f"contract must be a mapping: {path}")
    return data


def _git_head(path: Path) -> str:
    if not path.exists():
        return "missing"
    proc = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
