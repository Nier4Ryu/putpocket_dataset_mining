from __future__ import annotations

import importlib
import json
import subprocess
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Mapping

from .cluster_config import ClusterProfile, validate_environment_lock
from .cluster_safety import allocated_gpu_selector, require_slurm_allocation, safe_absolute_path
from .constants import REPO_ROOT
from .errors import ConfigError


class CheckStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"
    READY = "ready"


class FailureClass(StrEnum):
    STATIC_CONFIG_INVALID = "STATIC_CONFIG_INVALID"
    NOT_IN_SLURM_ALLOCATION = "NOT_IN_SLURM_ALLOCATION"
    GPU_INVENTORY_UNAVAILABLE = "GPU_INVENTORY_UNAVAILABLE"
    GPU_COUNT_MISMATCH = "GPU_COUNT_MISMATCH"
    GPU_TYPE_MISMATCH = "GPU_TYPE_MISMATCH"
    PACKAGE_IMPORT_MISSING = "PACKAGE_IMPORT_MISSING"
    PACKAGE_SYMBOL_MISSING = "PACKAGE_SYMBOL_MISSING"
    CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"
    CHECKPOINT_LAYOUT_INVALID = "CHECKPOINT_LAYOUT_INVALID"
    QUANTIZATION_BACKEND_INCOMPATIBLE = "QUANTIZATION_BACKEND_INCOMPATIBLE"
    MODEL_LOAD_NOT_READY = "MODEL_LOAD_NOT_READY"
    GENERATION_HANDOFF_BLOCKED = "GENERATION_HANDOFF_BLOCKED"


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: CheckStatus
    failure_class: FailureClass | None = None
    detail: str = ""
    evidence: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status.value,
            "failure_class": self.failure_class.value if self.failure_class else None,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ReadinessReport:
    profile_id: str
    requested_stage: str
    checks: tuple[ReadinessCheck, ...]
    handoff: dict[str, object] | None = None

    @property
    def succeeded(self) -> bool:
        return all(check.status != CheckStatus.FAIL for check in self.checks)

    @property
    def status(self) -> str:
        if not self.succeeded:
            return "failed"
        if any(check.status == CheckStatus.READY for check in self.checks):
            return "handoff_ready"
        return "passed"

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "requested_stage": self.requested_stage,
            "status": self.status,
            "checks": [check.as_dict() for check in self.checks],
            "handoff": self.handoff,
            "claim_boundary": "no model load, GPU workload, or generation result is claimed by readiness",
        }


class ReadinessProbe:
    def gpu_inventory(
        self,
        nvidia_smi_executable: str | Path,
        allocated_devices: str | None = None,
    ) -> tuple[int, str, str]:
        command = [
            str(nvidia_smi_executable),
            "--query-gpu=name,compute_cap",
            "--format=csv,noheader",
        ]
        if allocated_devices:
            command.append(f"--id={allocated_devices}")
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        return completed.returncode, completed.stdout, completed.stderr

    def import_expectations(self, expected: Mapping[str, tuple[str, ...]]) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for module_name, symbols in expected.items():
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:  # noqa: BLE001 - readiness classifies native/import failures.
                result[module_name] = {"imported": False, "error": f"{type(exc).__name__}: {exc}", "missing": list(symbols)}
                continue
            missing = [symbol for symbol in symbols if not hasattr(module, symbol)]
            result[module_name] = {"imported": True, "missing": missing}
        return result


def run_readiness(
    profile: ClusterProfile,
    *,
    stage: str,
    model_path: str | Path | None = None,
    model_revision: str | None = None,
    nvidia_smi_executable: str | Path | None = None,
    env: Mapping[str, str] | None = None,
    probe: ReadinessProbe | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> ReadinessReport:
    stages = {
        "static",
        "allocation",
        "gpu",
        "imports",
        "checkpoint",
        "model-load",
        "generation-handoff",
        "all",
    }
    if stage not in stages:
        raise ConfigError(f"Unknown Cluster readiness stage: {stage}")
    checks: list[ReadinessCheck] = []
    root = Path(repo_root)
    try:
        lock_path = root / profile.environment_lock
        validate_environment_lock(lock_path)
        checks.append(
            ReadinessCheck(
                "static_config",
                CheckStatus.PASS,
                detail="profile and commit-addressed SM90 environment lock are valid",
                evidence={"environment_lock": profile.environment_lock, "world_size": profile.world_size},
            )
        )
    except ConfigError as exc:
        checks.append(ReadinessCheck("static_config", CheckStatus.FAIL, FailureClass.STATIC_CONFIG_INVALID, str(exc)))
        return ReadinessReport(profile.profile_id, stage, tuple(checks))
    if stage == "static":
        return ReadinessReport(profile.profile_id, stage, tuple(checks))

    try:
        allocation = require_slurm_allocation(env)
        checks.append(ReadinessCheck("allocation", CheckStatus.PASS, evidence=allocation))
    except ConfigError as exc:
        checks.append(ReadinessCheck("allocation", CheckStatus.FAIL, FailureClass.NOT_IN_SLURM_ALLOCATION, str(exc)))
        return ReadinessReport(profile.profile_id, stage, tuple(checks))
    if stage == "allocation":
        return ReadinessReport(profile.profile_id, stage, tuple(checks))

    active_probe = probe or ReadinessProbe()
    if nvidia_smi_executable is None:
        checks.append(
            ReadinessCheck(
                "gpu_inventory",
                CheckStatus.FAIL,
                FailureClass.GPU_INVENTORY_UNAVAILABLE,
                "nvidia_smi_executable must be explicitly supplied",
            )
        )
        return ReadinessReport(profile.profile_id, stage, tuple(checks))
    nvidia_smi = safe_absolute_path(nvidia_smi_executable, "nvidia_smi_executable")
    gpu_check = _check_gpus(profile, active_probe, nvidia_smi, allocated_gpu_selector(env))
    checks.append(gpu_check)
    if gpu_check.status == CheckStatus.FAIL or stage == "gpu":
        return ReadinessReport(profile.profile_id, stage, tuple(checks))

    import_check = _check_imports(profile, active_probe)
    checks.append(import_check)
    if import_check.status == CheckStatus.FAIL or stage == "imports":
        return ReadinessReport(profile.profile_id, stage, tuple(checks))

    checkpoint_check, compatibility_check = _check_checkpoint(profile, model_path)
    checks.extend([checkpoint_check, compatibility_check])
    if checkpoint_check.status == CheckStatus.FAIL or compatibility_check.status == CheckStatus.FAIL:
        return ReadinessReport(profile.profile_id, stage, tuple(checks))
    if stage == "checkpoint":
        return ReadinessReport(profile.profile_id, stage, tuple(checks))

    assert model_path is not None
    model_path_text = str(safe_absolute_path(model_path, "model_path"))
    model_command = profile.model_load_command(model_path_text, model_revision)
    model_ready = ReadinessCheck(
        "model_load_readiness",
        CheckStatus.READY,
        detail="prerequisites passed; model load has not been executed",
        evidence={"command": model_command},
    )
    checks.append(model_ready)
    handoff: dict[str, object] = {
        "model_load": {
            "status": "ready_not_executed",
            "guarded_action": "model-load",
            "command": model_command,
        }
    }
    if stage == "model-load":
        return ReadinessReport(profile.profile_id, stage, tuple(checks), handoff)

    generation = ReadinessCheck(
        "one_shot_generation_handoff",
        CheckStatus.READY,
        detail="model-load command and guarded one-shot action are ready; no prompt was run",
        evidence={
            "guarded_action": "one-shot-generation",
            "required_future_inputs": ["started model endpoint or offline generation command", "prompt artifact", "output artifact"],
        },
    )
    checks.append(generation)
    handoff["one_shot_generation"] = {
        "status": "ready_not_executed",
        "guarded_action": "one-shot-generation",
        "extension_point": "phase-2 adapter supplies prompt/request and quality evaluation",
    }
    return ReadinessReport(profile.profile_id, stage, tuple(checks), handoff)


def _check_gpus(
    profile: ClusterProfile,
    probe: ReadinessProbe,
    executable: Path,
    allocated_devices: str | None,
) -> ReadinessCheck:
    returncode, stdout, stderr = probe.gpu_inventory(executable, allocated_devices)
    if returncode != 0:
        return ReadinessCheck(
            "gpu_inventory",
            CheckStatus.FAIL,
            FailureClass.GPU_INVENTORY_UNAVAILABLE,
            stderr.strip() or "nvidia-smi inventory failed",
        )
    rows: list[tuple[str, str]] = []
    for raw in stdout.splitlines():
        if not raw.strip():
            continue
        parts = [item.strip() for item in raw.rsplit(",", 1)]
        if len(parts) != 2:
            return ReadinessCheck(
                "gpu_inventory",
                CheckStatus.FAIL,
                FailureClass.GPU_INVENTORY_UNAVAILABLE,
                f"unparseable GPU inventory row: {raw}",
            )
        rows.append((parts[0], parts[1]))
    if len(rows) != profile.world_size:
        return ReadinessCheck(
            "gpu_inventory",
            CheckStatus.FAIL,
            FailureClass.GPU_COUNT_MISMATCH,
            f"expected {profile.world_size} allocated GPUs, found {len(rows)}",
            {"allocated_selector": allocated_devices, "gpus": [{"name": name, "compute_capability": cap} for name, cap in rows]},
        )
    wrong = [row for row in rows if profile.accelerator_name_pattern.lower() not in row[0].lower() or row[1] != profile.compute_capability]
    if wrong:
        return ReadinessCheck(
            "gpu_inventory",
            CheckStatus.FAIL,
            FailureClass.GPU_TYPE_MISMATCH,
            f"allocated GPUs must match {profile.accelerator_name_pattern} compute capability {profile.compute_capability}",
            {"mismatches": [{"name": name, "compute_capability": cap} for name, cap in wrong]},
        )
    return ReadinessCheck(
        "gpu_inventory",
        CheckStatus.PASS,
        evidence={"allocated_selector": allocated_devices, "gpus": [{"name": name, "compute_capability": cap} for name, cap in rows]},
    )


def _check_imports(profile: ClusterProfile, probe: ReadinessProbe) -> ReadinessCheck:
    results = probe.import_expectations(profile.required_imports)
    missing_imports = [name for name, result in results.items() if not result.get("imported")]
    if missing_imports:
        return ReadinessCheck(
            "package_imports",
            CheckStatus.FAIL,
            FailureClass.PACKAGE_IMPORT_MISSING,
            f"required imports failed: {', '.join(missing_imports)}",
            results,
        )
    missing_symbols = {
        name: result.get("missing", [])
        for name, result in results.items()
        if result.get("missing")
    }
    if missing_symbols:
        return ReadinessCheck(
            "package_imports",
            CheckStatus.FAIL,
            FailureClass.PACKAGE_SYMBOL_MISSING,
            "required runtime symbols are missing",
            missing_symbols,
        )
    return ReadinessCheck("package_imports", CheckStatus.PASS, evidence=results)


def _check_checkpoint(
    profile: ClusterProfile,
    model_path: str | Path | None,
) -> tuple[ReadinessCheck, ReadinessCheck]:
    if not model_path:
        failure = ReadinessCheck(
            "checkpoint_layout",
            CheckStatus.FAIL,
            FailureClass.CHECKPOINT_NOT_FOUND,
            "model_path is required for checkpoint readiness",
        )
        return failure, ReadinessCheck(
            "quantization_backend",
            CheckStatus.SKIP,
            detail="checkpoint layout did not pass",
        )
    path = safe_absolute_path(model_path, "model_path")
    if not path.is_dir():
        failure = ReadinessCheck(
            "checkpoint_layout",
            CheckStatus.FAIL,
            FailureClass.CHECKPOINT_NOT_FOUND,
            f"checkpoint directory does not exist: {path}",
        )
        return failure, ReadinessCheck("quantization_backend", CheckStatus.SKIP, detail="checkpoint not found")
    config_path = path / "config.json"
    if not config_path.is_file() or config_path.stat().st_size > 16 * 1024 * 1024:
        failure = ReadinessCheck(
            "checkpoint_layout",
            CheckStatus.FAIL,
            FailureClass.CHECKPOINT_LAYOUT_INVALID,
            "checkpoint requires a bounded config.json",
        )
        return failure, ReadinessCheck("quantization_backend", CheckStatus.SKIP, detail="checkpoint config invalid")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        failure = ReadinessCheck(
            "checkpoint_layout",
            CheckStatus.FAIL,
            FailureClass.CHECKPOINT_LAYOUT_INVALID,
            f"config.json is unreadable: {exc}",
        )
        return failure, ReadinessCheck("quantization_backend", CheckStatus.SKIP, detail="checkpoint config invalid")
    if not isinstance(config, dict):
        failure = ReadinessCheck(
            "checkpoint_layout",
            CheckStatus.FAIL,
            FailureClass.CHECKPOINT_LAYOUT_INVALID,
            "config.json root must be a mapping",
        )
        return failure, ReadinessCheck("quantization_backend", CheckStatus.SKIP, detail="checkpoint config invalid")
    index_files = sorted(item.name for item in path.glob("*.safetensors.index.json") if item.is_file())
    weight_files = sorted(item for item in path.glob("*.safetensors") if item.is_file())
    tokenizer_files = [name for name in ("tokenizer.json", "tokenizer.model", "vocab.json") if (path / name).is_file()]
    if not (index_files or weight_files) or not tokenizer_files:
        failure = ReadinessCheck(
            "checkpoint_layout",
            CheckStatus.FAIL,
            FailureClass.CHECKPOINT_LAYOUT_INVALID,
            "checkpoint requires safetensors weights/index and tokenizer metadata",
            {"index_files": index_files, "weight_file_count": len(weight_files), "tokenizer_files": tokenizer_files},
        )
        return failure, ReadinessCheck("quantization_backend", CheckStatus.SKIP, detail="checkpoint layout invalid")
    metadata = {
        "config": "config.json",
        "index_files": index_files,
        "weight_files": [{"name": item.name, "size": item.stat().st_size} for item in weight_files],
        "tokenizer_files": tokenizer_files,
        "hash_policy": "no checkpoint tensor hashing",
    }
    layout = ReadinessCheck("checkpoint_layout", CheckStatus.PASS, evidence=metadata)
    config_text = json.dumps(config, sort_keys=True).lower()
    markers = [marker for marker in profile.quantization_markers if marker in config_text]
    if not markers:
        compatibility = ReadinessCheck(
            "quantization_backend",
            CheckStatus.FAIL,
            FailureClass.QUANTIZATION_BACKEND_INCOMPATIBLE,
            f"checkpoint config has no accepted {profile.quantization} backend marker",
            {"accepted_markers": list(profile.quantization_markers)},
        )
    else:
        compatibility = ReadinessCheck(
            "quantization_backend",
            CheckStatus.PASS,
            evidence={"quantization": profile.quantization, "matched_markers": markers},
        )
    return layout, compatibility
