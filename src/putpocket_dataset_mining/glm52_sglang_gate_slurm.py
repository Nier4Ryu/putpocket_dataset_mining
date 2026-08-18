from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ConfigError
from .glm52_sglang_gate import SITE_PROFILE, safe_directive, validate_public_project


@dataclass(frozen=True)
class GateSite:
    partition: str
    account: str
    qos: str
    gpu_directive: str
    nodes: int
    ntasks: int
    cpus_per_task: int
    memory: str
    wall_time: str
    storage_root: Path
    artifact_root: Path
    slurm_log_root: Path
    bootstrap_python: Path
    git_executable: Path
    nvidia_smi_executable: Path
    cuda_module: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "GateSite":
        if data.get("schema_version") != 1:
            raise ConfigError("SGLang gate site schema_version must be 1")
        slurm = _mapping(data.get("slurm"), "slurm")
        paths = _mapping(data.get("paths"), "paths")
        tools = _mapping(data.get("tools"), "tools")
        exact = {
            "partition": "H200",
            "account": "gsai-account",
            "qos": "hpgpu",
            "gpu_directive": "--gres=gpu:H200:4",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 32,
            "memory": "512G",
            "wall_time": "06:00:00",
            "export": "NONE",
        }
        for key, expected in exact.items():
            if slurm.get(key) != expected:
                raise ConfigError(f"Site {key} must equal authoritative value {expected!r}")
        storage = _absolute(paths.get("storage_root"), "storage_root")
        artifact = _absolute(paths.get("artifact_root"), "artifact_root")
        try:
            artifact.relative_to(storage)
        except ValueError as exc:
            raise ConfigError("artifact_root must be inside storage_root") from exc
        log_root = _absolute(paths.get("slurm_log_root"), "slurm_log_root")
        if str(log_root) != "/home2/jslee202403/putpocket-slurm":
            raise ConfigError("Slurm logs must use the observed Login home log directory")
        return cls(
            partition="H200",
            account="gsai-account",
            qos="hpgpu",
            gpu_directive="--gres=gpu:H200:4",
            nodes=1,
            ntasks=1,
            cpus_per_task=32,
            memory="512G",
            wall_time="06:00:00",
            storage_root=storage,
            artifact_root=artifact,
            slurm_log_root=log_root,
            bootstrap_python=_absolute(tools.get("bootstrap_python"), "bootstrap_python"),
            git_executable=_absolute(tools.get("git"), "git"),
            nvidia_smi_executable=_absolute(tools.get("nvidia_smi"), "nvidia_smi"),
            cuda_module=safe_directive(tools.get("cuda_module"), "cuda_module"),
        )


def load_gate_site(path: str | Path = SITE_PROFILE) -> GateSite:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError("SGLang gate site profile must be a JSON object")
    return GateSite.from_mapping(data)


def render_compact_gate_submission(*, site: GateSite, project_url: str, project_commit: str) -> str:
    validate_public_project(project_url, project_commit)
    source_root = site.storage_root / "source" / f"putpocket-{project_commit}"
    wrapper = "; ".join(
        (
            "set -euo pipefail",
            "umask 077",
            f"PROJECT_URL={shlex.quote(project_url)}",
            f"PROJECT_COMMIT={project_commit}",
            f"SOURCE_ROOT={shlex.quote(str(source_root))}",
            f"RUN_ROOT={shlex.quote(str(site.artifact_root))}/\"${{SLURM_JOB_ID:-unknown}}\"",
            "[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || { echo E_SLURM_ALLOCATION_REQUIRED >&2; exit 20; }",
            "[[ ${SLURM_JOB_NUM_NODES:-0} == 1 && -n ${SLURM_JOB_NODELIST:-} ]] || { echo E_NODE_COUNT_MISMATCH >&2; exit 20; }",
            "[[ ${SLURM_GPUS_ON_NODE:-0} == 4 ]] || { echo E_GPU_ALLOCATION_COUNT_MISMATCH >&2; exit 20; }",
            "[[ -d /local-data/jslee202403 ]] || { echo E_COMPUTE_LOCAL_STORAGE_MISSING >&2; exit 21; }",
            f"[[ -x {shlex.quote(str(site.git_executable))} ]] || {{ echo E_COMPUTE_GIT_MISSING >&2; exit 21; }}",
            "mkdir -p \"$RUN_ROOT\" \"$SOURCE_ROOT\"",
            f"if [[ ! -d \"$SOURCE_ROOT/.git\" ]]; then {shlex.quote(str(site.git_executable))} -C \"$SOURCE_ROOT\" init; fi",
            f"{shlex.quote(str(site.git_executable))} -C \"$SOURCE_ROOT\" diff --quiet --ignore-submodules -- || {{ echo E_DIRTY_COMPUTE_CHECKOUT >&2; exit 22; }}",
            f"{shlex.quote(str(site.git_executable))} -C \"$SOURCE_ROOT\" fetch --depth=1 \"$PROJECT_URL\" \"$PROJECT_COMMIT\"",
            f"{shlex.quote(str(site.git_executable))} -C \"$SOURCE_ROOT\" checkout --detach FETCH_HEAD",
            f"[[ $({shlex.quote(str(site.git_executable))} -C \"$SOURCE_ROOT\" rev-parse HEAD) == \"$PROJECT_COMMIT\" ]] || {{ echo E_PROJECT_COMMIT_MISMATCH >&2; exit 23; }}",
            "exec /bin/bash \"$SOURCE_ROOT/scripts/cluster/run_glm52_sglang_feasibility_gate.sh\" \"$PROJECT_COMMIT\"",
        )
    )
    flags = [
        "sbatch",
        "--parsable",
        "--job-name=pp-glm52-sglang-gate",
        f"--partition={safe_directive(site.partition, 'partition')}",
        f"--account={safe_directive(site.account, 'account')}",
        f"--qos={safe_directive(site.qos, 'qos')}",
        f"--nodes={site.nodes}",
        f"--ntasks={site.ntasks}",
        site.gpu_directive,
        f"--cpus-per-task={site.cpus_per_task}",
        f"--mem={safe_directive(site.memory, 'memory')}",
        f"--time={safe_directive(site.wall_time, 'wall_time')}",
        f"--output={site.slurm_log_root}/%x-%j.out",
        f"--error={site.slurm_log_root}/%x-%j.err",
        "--export=NONE",
        f"--wrap={wrapper}",
    ]
    command = f"mkdir -p {shlex.quote(str(site.slurm_log_root))} && " + shlex.join(flags)
    lowered = command.lower()
    if "swe-bench" in lowered or "swebench" in lowered or "swepro" in lowered:
        raise AssertionError("SGLang feasibility command must not contain a benchmark transition")
    return command


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _absolute(value: Any, field: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or ".." in path.parts:
        raise ConfigError(f"{field} must be an absolute normalized path")
    return path
