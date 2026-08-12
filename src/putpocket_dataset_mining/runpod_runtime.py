from __future__ import annotations

import json
import os
import platform
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
TORCH_CU129_CONTRACT = Path("configs/env/torch/torch_2_10_cu129.lock.yaml")

ARCH_PROFILES = {
    "portable-nvidia": "8.6 9.0 10.0 12.0",
    "rtx3090": "8.6",
    "hopper": "9.0",
    "blackwell-datacenter": "10.0",
    "blackwell-rtx": "12.0",
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
                "install exact torch contract only when provenance is resolved",
                "install project, vLLM, and LMCache editable",
                "write runtime fingerprint and build manifests",
            ],
            "vllm_developer_commands": vllm_developer_commands(self.repo_root),
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
            "PUTPOCKET_CUDA_ARCH_PROFILE": self.cuda_arch_profile,
            "PUTPOCKET_CUDA_ARCH_LIST": self.cuda_arch_list,
            "TORCH_CUDA_ARCH_LIST": self.cuda_arch_list,
            "CCACHE_NOHASHDIR": "true",
        }


def resolve_cuda_arch_list(profile: str, explicit: str | None = None) -> str:
    if explicit:
        return " ".join(explicit.split())
    if profile == "native":
        return detect_native_cuda_arch_list()
    if profile not in ARCH_PROFILES:
        raise ConfigError(f"unknown CUDA architecture profile: {profile}")
    return ARCH_PROFILES[profile]


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
    caps = sorted({line.strip() for line in proc.stdout.splitlines() if line.strip()})
    if not caps:
        raise ConfigError("native CUDA architecture detection found no visible GPUs")
    return " ".join(caps)


def build_runpod_plan(
    *,
    repo_root: Path,
    persistent_root: str | None,
    storage_kind: str | None,
    cuda_arch_profile: str,
    cuda_arch_list: str | None,
    base_image_contract: str | None,
    dry_run: bool,
    doctor_only: bool,
    skip_vllm_build: bool,
    force_vllm_build: bool,
    skip_gpu_smoke: bool,
) -> RunpodPlan:
    root = Path(persistent_root or os.environ.get("PUTPOCKET_REPO_ROOT", str(RUNPOD_REPO_ROOT))).expanduser()
    env_path = Path(os.environ.get("PUTPOCKET_ENV_PATH", str(root / "Putpocket_env"))).expanduser()
    storage = storage_kind or os.environ.get("PUTPOCKET_STORAGE_KIND", "network-volume")
    arch_list = resolve_cuda_arch_list(cuda_arch_profile, cuda_arch_list)
    return RunpodPlan(
        repo_root=root,
        env_path=env_path,
        persistent_root=root.parent,
        storage_kind=storage,
        cuda_arch_profile=cuda_arch_profile,
        cuda_arch_list=arch_list,
        base_image_contract=repo_root / (base_image_contract or str(RUNPOD_BASE_IMAGE_CONTRACT)),
        torch_contract=repo_root / TORCH_CU129_CONTRACT,
        dry_run=dry_run,
        doctor_only=doctor_only,
        skip_vllm_build=skip_vllm_build,
        force_vllm_build=force_vllm_build,
        skip_gpu_smoke=skip_gpu_smoke,
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
    if data.get("package", {}).get("version") != "2.10.0+cu129":
        raise ConfigError("torch contract must describe torch 2.10.0+cu129")
    if require_resolved and data.get("provenance_status") != "resolved":
        raise ConfigError("TORCH_CU129_PROVENANCE_UNRESOLVED")
    return data


def validate_network_volume(plan: RunpodPlan) -> dict[str, Any]:
    if plan.storage_kind != "network-volume":
        raise ConfigError("runpod-dev requires --storage-kind network-volume")
    if plan.persistent_root != Path("/workspace"):
        raise ConfigError("runpod-dev requires the repository to live under /workspace")
    volume_id = os.environ.get("RUNPOD_NETWORK_VOLUME_ID")
    if not volume_id:
        raise ConfigError("RUNPOD_NETWORK_VOLUME_ID must be set for runpod-dev before mutating the environment")
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
        "torch": torch_contract.get("package", {}).get("version"),
        "torch_cuda": torch_contract.get("cuda", {}).get("torch_cuda"),
        "cuda_toolkit": base.get("cuda_version"),
        "vllm_sha": _git_head(plan.repo_root / "externals" / "vllm"),
        "lmcache_sha": _git_head(plan.repo_root / "externals" / "lmcache"),
        "project_lock_hash": _sha256(plan.repo_root / "configs" / "env" / "server2_blackwell.lock.yaml"),
        "cuda_arch_profile": plan.cuda_arch_profile,
        "cuda_arch_list": plan.cuda_arch_list,
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
        "requested_architecture_profile": plan.cuda_arch_profile,
        "requested_architecture_list": plan.cuda_arch_list,
        "actual_compiled_architecture_evidence": {
            "sm_86": "NOT_RUN",
            "sm_90": "NOT_RUN",
            "sm_100": "NOT_RUN",
            "sm_120": "NOT_RUN",
        },
        "heavy_multiarch_build_executed": os.environ.get("PUTPOCKET_ALLOW_HEAVY_MULTIARCH_BUILD") == "1",
    }


def vllm_developer_commands(repo_root: Path) -> dict[str, str]:
    prefix = f"cd {repo_root}/externals/vllm"
    return {
        "python_only_change": "edit externals/vllm Python files; no reinstall required for editable install",
        "cpp_or_cuda_change": f"{prefix} && CCACHE_NOHASHDIR=true uv pip install --no-build-isolation -e .",
        "clean_rebuild": f"{prefix} && rm -rf build && CCACHE_NOHASHDIR=true uv pip install --no-build-isolation -e .",
        "build_doctor": f"{prefix} && python use_existing_torch.py && uv pip install -r requirements/build/cuda.txt",
        "serve_later": "python -m vllm.entrypoints.openai.api_server --help",
    }


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
