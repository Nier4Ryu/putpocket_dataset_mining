from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .cluster_safety import reject_secret_fields, safe_absolute_path
from .config import load_yaml
from .constants import REPO_ROOT
from .errors import ConfigError


PROFILE_DIR = REPO_ROOT / "configs" / "cluster" / "profiles"
CLUSTER_ENV_LOCK = REPO_ROOT / "configs" / "env" / "cluster_h200_sm90_vllm026.lock.yaml"
VLLM_026_SHA = "568afb3a13806beb53bb2e6bd518269357b237c0"


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"Cluster profile field must be a mapping: {field}")
    return value


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ConfigError(f"Cluster profile field must be a positive integer: {field}")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Cluster profile field must be a positive integer: {field}") from exc
    if result < 1:
        raise ConfigError(f"Cluster profile field must be a positive integer: {field}")
    return result


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or any(ord(ch) < 32 for ch in value):
        raise ConfigError(f"Cluster profile field must be non-empty text: {field}")
    return value.strip()


@dataclass(frozen=True)
class ClusterProfile:
    profile_id: str
    model_id: str
    model_path: str | None
    model_revision: str | None
    quantization: str
    accelerator_name_pattern: str
    compute_capability: str
    nodes: int
    gpus_per_node: int
    tensor_parallel_size: int
    prefill_context_parallel_size: int
    expert_parallel: bool
    environment_lock: str
    engine_args: tuple[str, ...]
    quantization_markers: tuple[str, ...]
    required_imports: dict[str, tuple[str, ...]]

    @property
    def world_size(self) -> int:
        return self.nodes * self.gpus_per_node

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ClusterProfile":
        reject_secret_fields(data, path="profile")
        if data.get("schema_version") != 1:
            raise ConfigError("Cluster profile schema_version must be 1")
        if data.get("phase") != "1_foundation":
            raise ConfigError("Cluster profile phase must be 1_foundation")
        profile_id = _required_text(data.get("profile_id"), "profile_id")
        if not all(ch.islower() or ch.isdigit() or ch == "_" for ch in profile_id):
            raise ConfigError("profile_id must contain only lowercase letters, digits, and underscores")
        model = _mapping(data.get("model"), "model")
        hardware = _mapping(data.get("hardware"), "hardware")
        parallel = _mapping(data.get("parallelism"), "parallelism")
        runtime = _mapping(data.get("runtime"), "runtime")
        readiness = _mapping(data.get("readiness"), "readiness")
        if runtime.get("engine") != "vllm":
            raise ConfigError("Phase-1 Cluster profiles require runtime.engine=vllm")
        tp = _positive_int(parallel.get("tensor_parallel_size"), "parallelism.tensor_parallel_size")
        pcp = _positive_int(
            parallel.get("prefill_context_parallel_size"),
            "parallelism.prefill_context_parallel_size",
        )
        nodes = _positive_int(hardware.get("nodes"), "hardware.nodes")
        gpus = _positive_int(hardware.get("gpus_per_node"), "hardware.gpus_per_node")
        if nodes != 1:
            raise ConfigError("Phase-1 Cluster profiles are single-node only")
        if tp * pcp != nodes * gpus:
            raise ConfigError(
                "Cluster profile parallelism mismatch: tensor_parallel_size * "
                "prefill_context_parallel_size must equal allocated GPUs"
            )
        ep = parallel.get("expert_parallel")
        if not isinstance(ep, bool):
            raise ConfigError("parallelism.expert_parallel must be boolean")
        engine_args_raw = runtime.get("engine_args")
        if not isinstance(engine_args_raw, list) or not all(isinstance(item, str) for item in engine_args_raw):
            raise ConfigError("runtime.engine_args must be a list of strings")
        expected_args = ["--tensor-parallel-size", str(tp)]
        if pcp > 1:
            expected_args.extend(["--prefill-context-parallel-size", str(pcp)])
        if ep:
            expected_args.append("--enable-expert-parallel")
        if engine_args_raw != expected_args:
            raise ConfigError(f"runtime.engine_args must exactly match validated parallelism: {expected_args}")
        imports_raw = _mapping(readiness.get("required_imports"), "readiness.required_imports")
        imports: dict[str, tuple[str, ...]] = {}
        for module, symbols in imports_raw.items():
            if not isinstance(symbols, list) or not symbols or not all(isinstance(item, str) and item for item in symbols):
                raise ConfigError(f"readiness.required_imports.{module} must be a non-empty string list")
            imports[_required_text(module, "readiness.required_imports module")] = tuple(symbols)
        markers = readiness.get("accepted_quantization_markers")
        if not isinstance(markers, list) or not markers or not all(isinstance(item, str) and item for item in markers):
            raise ConfigError("readiness.accepted_quantization_markers must be a non-empty string list")
        model_path_value = model.get("path")
        revision_value = model.get("revision")
        return cls(
            profile_id=profile_id,
            model_id=_required_text(model.get("id"), "model.id"),
            model_path=str(model_path_value) if model_path_value else None,
            model_revision=str(revision_value) if revision_value else None,
            quantization=_required_text(model.get("quantization"), "model.quantization").lower(),
            accelerator_name_pattern=_required_text(
                hardware.get("accelerator_name_pattern"), "hardware.accelerator_name_pattern"
            ),
            compute_capability=_required_text(hardware.get("compute_capability"), "hardware.compute_capability"),
            nodes=nodes,
            gpus_per_node=gpus,
            tensor_parallel_size=tp,
            prefill_context_parallel_size=pcp,
            expert_parallel=ep,
            environment_lock=_required_text(runtime.get("environment_lock"), "runtime.environment_lock"),
            engine_args=tuple(engine_args_raw),
            quantization_markers=tuple(str(item).lower() for item in markers),
            required_imports=imports,
        )

    def model_load_command(self, model_path: str, revision: str | None = None) -> list[str]:
        command = ["vllm", "serve", model_path, *self.engine_args]
        if revision:
            command.extend(["--revision", revision])
        return command


@dataclass(frozen=True)
class ClusterSite:
    partition: str | None
    account: str | None
    constraint: str | None
    wall_time: str
    cpus_per_task: int
    repository_root: Path
    python_executable: Path
    uv_executable: Path
    git_executable: Path
    nvidia_smi_executable: Path
    nvcc_executable: Path
    environment_root: Path
    vllm_source_root: Path
    cache_root: Path
    checkpoint_root: Path
    artifact_root: Path
    slurm_log_root: Path
    models: dict[str, dict[str, str | None]]

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ClusterSite":
        reject_secret_fields(data, path="site_config")
        if data.get("schema_version") != 1:
            raise ConfigError("Cluster site schema_version must be 1")
        site = _mapping(data.get("site"), "site")
        models = _mapping(data.get("models"), "models")
        wall_time = _required_text(site.get("wall_time"), "site.wall_time")
        if not _valid_wall_time(wall_time):
            raise ConfigError("site.wall_time must use Slurm D-HH:MM:SS or HH:MM:SS syntax")
        model_values: dict[str, dict[str, str | None]] = {}
        for profile_id, raw in models.items():
            item = _mapping(raw, f"models.{profile_id}")
            model_values[str(profile_id)] = {
                "path": str(item["path"]) if item.get("path") else None,
                "revision": str(item["revision"]) if item.get("revision") else None,
            }
        return cls(
            partition=_optional_text(site.get("partition"), "site.partition"),
            account=_optional_text(site.get("account"), "site.account"),
            constraint=_optional_text(site.get("constraint"), "site.constraint"),
            wall_time=wall_time,
            cpus_per_task=_positive_int(site.get("cpus_per_task"), "site.cpus_per_task"),
            repository_root=safe_absolute_path(site.get("repository_root") or "", "site.repository_root"),
            python_executable=safe_absolute_path(site.get("python_executable") or "", "site.python_executable"),
            uv_executable=safe_absolute_path(site.get("uv_executable") or "", "site.uv_executable"),
            git_executable=safe_absolute_path(site.get("git_executable") or "", "site.git_executable"),
            nvidia_smi_executable=safe_absolute_path(
                site.get("nvidia_smi_executable") or "", "site.nvidia_smi_executable"
            ),
            nvcc_executable=safe_absolute_path(site.get("nvcc_executable") or "", "site.nvcc_executable"),
            environment_root=safe_absolute_path(site.get("environment_root") or "", "site.environment_root"),
            vllm_source_root=safe_absolute_path(site.get("vllm_source_root") or "", "site.vllm_source_root"),
            cache_root=safe_absolute_path(site.get("cache_root") or "", "site.cache_root"),
            checkpoint_root=safe_absolute_path(site.get("checkpoint_root") or "", "site.checkpoint_root"),
            artifact_root=safe_absolute_path(site.get("artifact_root") or "", "site.artifact_root"),
            slurm_log_root=safe_absolute_path(
                site.get("slurm_log_root") or "", "site.slurm_log_root", slurm_directive=True
            ),
            models=model_values,
        )

    def model_for(self, profile: ClusterProfile) -> tuple[str, str | None]:
        override = self.models.get(profile.profile_id, {})
        model_path = override.get("path") or profile.model_path
        if not model_path:
            raise ConfigError(f"models.{profile.profile_id}.path must be supplied by the Cluster site config")
        safe = safe_absolute_path(model_path, f"models.{profile.profile_id}.path")
        try:
            safe.relative_to(self.checkpoint_root)
        except ValueError as exc:
            raise ConfigError(
                f"models.{profile.profile_id}.path must be inside site.checkpoint_root"
            ) from exc
        return str(safe), override.get("revision") or profile.model_revision


def load_cluster_profile(path_or_id: str | Path) -> ClusterProfile:
    path = Path(path_or_id)
    if not path.suffix and not path.exists():
        path = PROFILE_DIR / f"{path}.yaml"
    return ClusterProfile.from_mapping(load_yaml(path))


def load_cluster_site(path: str | Path) -> ClusterSite:
    return ClusterSite.from_mapping(load_yaml(path))


def validate_environment_lock(path: str | Path = CLUSTER_ENV_LOCK) -> dict[str, Any]:
    lock = load_yaml(path)
    reject_secret_fields(lock, path="environment_lock")
    if lock.get("schema_version") != 1 or lock.get("provenance_status") != "commit_addressed":
        raise ConfigError("Cluster environment lock must be schema v1 and commit_addressed")
    hardware = _mapping(lock.get("hardware"), "hardware")
    vllm = _mapping(lock.get("vllm"), "vllm")
    if hardware.get("compute_capability") != "9.0" or hardware.get("torch_cuda_arch_list") != "9.0":
        raise ConfigError("Cluster environment lock must target H200/SM90 only")
    if vllm.get("commit") != VLLM_026_SHA or vllm.get("tag") != "v0.26.0":
        raise ConfigError("Cluster environment lock must pin clean vLLM v0.26.0 source")
    return lock


def _optional_text(value: Any, field: str) -> str | None:
    if value is None or value == "":
        return None
    return _required_text(value, field)


def _valid_wall_time(value: str) -> bool:
    import re

    return re.fullmatch(r"(?:\d+-)?\d{1,2}:\d{2}:\d{2}", value) is not None
