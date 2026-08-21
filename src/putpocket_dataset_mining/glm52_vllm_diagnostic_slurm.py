from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from .errors import ConfigError
from .glm52_vllm_diagnostic import load_lock, validate_lock, validate_source_identities


_SHA40 = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class CpuBuildSite:
    partition: str
    account: str
    qos: str
    cpus_per_task: int
    memory: str
    wall_time: str
    local_scratch_root: Path
    container_executable: Path


@dataclass(frozen=True)
class VllmDiagnosticSite:
    cpu: CpuBuildSite
    h200: Mapping[str, Any]
    h200_storage_parent: Path
    h200_work_root: Path
    h200_artifact_root: Path
    shared_build_root: Path
    slurm_log_root: Path
    git_executable: Path
    nvidia_smi: Path


def load_site(
    path: str | Path,
    *,
    cpu_partition: str | None = None,
    cpu_account: str | None = None,
    cpu_qos: str | None = None,
    cpu_cpus_per_task: int | None = None,
    cpu_memory: str | None = None,
    cpu_wall_time: str | None = None,
    cpu_local_scratch_root: str | None = None,
    container_executable: str | None = None,
) -> VllmDiagnosticSite:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    cpu = _mapping(value.get("cpu_build"), "cpu_build")
    resolved = {
        "partition": cpu_partition or cpu.get("partition"),
        "account": cpu_account or cpu.get("account"),
        "qos": cpu_qos or cpu.get("qos"),
        "cpus_per_task": cpu_cpus_per_task or cpu.get("cpus_per_task"),
        "memory": cpu_memory or cpu.get("memory"),
        "wall_time": cpu_wall_time or cpu.get("wall_time"),
        "local_scratch_root": cpu_local_scratch_root or cpu.get("local_scratch_root"),
        "container_executable": container_executable or cpu.get("container_executable"),
    }
    missing = [key for key, item in resolved.items() if item in (None, "")]
    if missing:
        raise ConfigError(f"CPU_BUILD_SITE_FIELDS_UNSET:{','.join(sorted(missing))}")
    if not isinstance(resolved["cpus_per_task"], int) or resolved["cpus_per_task"] < 1:
        raise ConfigError("CPU_BUILD_CPUS_INVALID")
    h200 = _mapping(value.get("h200_run"), "h200_run")
    exact = {
        "partition": "H200",
        "account": "gsai-account",
        "qos": "hpgpu",
        "nodes": 1,
        "ntasks": 1,
        "gpu_directive": "--gres=gpu:H200:4",
        "cpus_per_task": 32,
        "memory": "512G",
        "wall_time": "06:00:00",
    }
    for key, expected in exact.items():
        if h200.get(key) != expected:
            raise ConfigError(f"H200_SITE_CONTRACT_MISMATCH:{key}")
    paths = [
        Path(str(resolved["local_scratch_root"])),
        Path(str(resolved["container_executable"])),
        Path(str(value["shared_build_root"])),
        Path(str(value["slurm_log_root"])),
        Path(str(value["git_executable"])),
        Path(str(value["nvidia_smi"])),
        Path(str(h200["local_storage_parent"])),
        Path(str(h200["work_root"])),
        Path(str(h200["artifact_root"])),
    ]
    if not all(item.is_absolute() for item in paths):
        raise ConfigError("SITE_PATHS_MUST_BE_ABSOLUTE")
    if paths[1].name != "docker":
        raise ConfigError("OFFICIAL_DOCKER_RUNTIME_REQUIRED")
    storage_parent, work_root, artifact_root = paths[6:9]
    if work_root == storage_parent or storage_parent not in work_root.parents:
        raise ConfigError("H200_WORK_ROOT_MUST_BE_INSIDE_STORAGE_PARENT")
    if artifact_root != work_root / "artifacts":
        raise ConfigError("H200_ARTIFACT_ROOT_MUST_EQUAL_WORK_ROOT_ARTIFACTS")
    return VllmDiagnosticSite(
        cpu=CpuBuildSite(
            partition=str(resolved["partition"]),
            account=str(resolved["account"]),
            qos=str(resolved["qos"]),
            cpus_per_task=int(resolved["cpus_per_task"]),
            memory=str(resolved["memory"]),
            wall_time=str(resolved["wall_time"]),
            local_scratch_root=paths[0],
            container_executable=paths[1],
        ),
        h200=h200,
        h200_storage_parent=storage_parent,
        h200_work_root=work_root,
        h200_artifact_root=artifact_root,
        shared_build_root=paths[2],
        slurm_log_root=paths[3],
        git_executable=paths[4],
        nvidia_smi=paths[5],
    )


def render_two_stage_submission(
    *,
    site: VllmDiagnosticSite,
    project_url: str,
    runtime_source_commit: str,
    build_source_commit: str,
    allow_runtime_source_split: bool,
    lock_path: str | Path,
) -> str:
    clean_url = _public_project_url(project_url)
    if not _SHA40.fullmatch(runtime_source_commit) or not _SHA40.fullmatch(build_source_commit):
        raise ConfigError("PROJECT_COMMITS_MUST_BE_FULL_SHA")
    lock = load_lock(lock_path)
    validate_lock(lock)
    validate_source_identities(
        pinned_build_source_commit=lock["build"]["project_source_commit"],
        expected_build_source_commit=build_source_commit,
        runtime_source_commit=runtime_source_commit,
        observed_runtime_source_commit=runtime_source_commit,
        wrapper_source_commit=runtime_source_commit,
        allow_runtime_source_split=allow_runtime_source_split,
    )
    bundle_key = lock["build"]["bundle_key"]
    if str(site.shared_build_root / bundle_key) != lock["build"]["immutable_bundle_root"]:
        raise ConfigError("SITE_IMMUTABLE_BUNDLE_ROOT_MISMATCH")
    build_body = _build_wrapper(site, clean_url, runtime_source_commit, build_source_commit, bundle_key, allow_runtime_source_split)
    run_body = _run_wrapper(site, clean_url, runtime_source_commit, build_source_commit, bundle_key, allow_runtime_source_split)
    build_flags = [
        "sbatch", "--parsable", "--job-name=pp-vllm-bundle-validate",
        f"--partition={site.cpu.partition}", f"--account={site.cpu.account}", f"--qos={site.cpu.qos}",
        "--nodes=1", "--ntasks=1", f"--cpus-per-task={site.cpu.cpus_per_task}",
        f"--mem={site.cpu.memory}", f"--time={site.cpu.wall_time}",
        f"--output={site.slurm_log_root}/%x-%j.out", f"--error={site.slurm_log_root}/%x-%j.err",
        "--export=NONE", f"--wrap={build_body}",
    ]
    run_flags = [
        "sbatch", "--parsable", "--job-name=pp-glm52-vllm-dsa",
        "--partition=H200", "--account=gsai-account", "--qos=hpgpu",
        "--nodes=1", "--ntasks=1", "--gres=gpu:H200:4", "--cpus-per-task=32",
        "--mem=512G", "--time=06:00:00", "--dependency=afterok:$VALIDATOR_JOB_ID",
        f"--output={site.slurm_log_root}/%x-%j.out", f"--error={site.slurm_log_root}/%x-%j.err",
        "--export=NONE", f"--wrap={run_body}",
    ]
    # shlex.join must not quote the dependency variable itself; it is expanded by
    # the Login control shell only after the build ID has been validated.
    run_command = shlex.join(run_flags).replace("'--dependency=afterok:$VALIDATOR_JOB_ID'", "--dependency=afterok:$VALIDATOR_JOB_ID")
    command = " && ".join(
        (
            "set -euo pipefail",
            f"mkdir -p {shlex.quote(str(site.slurm_log_root))} {shlex.quote(str(site.shared_build_root))}",
            f"VALIDATOR_JOB_ID=$({shlex.join(build_flags)})",
            "[[ $VALIDATOR_JOB_ID =~ ^[0-9]+$ ]]",
            f"RUN_JOB_ID=$({run_command})",
            "[[ $RUN_JOB_ID =~ ^[0-9]+$ ]]",
            "printf 'VALIDATOR_JOB_ID=%s\\nRUN_JOB_ID=%s\\n' \"$VALIDATOR_JOB_ID\" \"$RUN_JOB_ID\"",
        )
    )
    lowered = command.lower()
    forbidden = ("sglang", "swebench_pro_full", "--selection full", "run_swebench", "swe_bench_pro_eval")
    if any(item in lowered for item in forbidden):
        raise AssertionError("Active vLLM two-stage wrapper contains a forbidden transition")
    return command


def _build_wrapper(
    site: VllmDiagnosticSite,
    url: str,
    runtime_commit: str,
    build_commit: str,
    bundle_key: str,
    allow_source_split: bool,
) -> str:
    source = f'{shlex.quote(str(site.cpu.local_scratch_root / "putpocket-vllm-build"))}/"${{SLURM_JOB_ID}}-{runtime_commit[:12]}"'
    split = "1" if allow_source_split else "0"
    parts = (
        "set -euo pipefail", "umask 077",
        "[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_JOB_NUM_NODES:-0} == 1 ]] || { echo E_CPU_SLURM_ALLOCATION_REQUIRED >&2; exit 20; }",
        "[[ -z ${SLURM_JOB_GPUS:-} && -z ${SLURM_GPUS:-} && -z ${CUDA_VISIBLE_DEVICES:-} ]] || { echo E_CPU_BUILD_GPU_ALLOCATION_FORBIDDEN >&2; exit 20; }",
        f"[[ -x {shlex.quote(str(site.cpu.container_executable))} ]] && {shlex.quote(str(site.cpu.container_executable))} info >/dev/null 2>&1 || {{ echo E_CPU_CONTAINER_RUNTIME_UNAVAILABLE >&2; exit 21; }}",
        f"mkdir -p {source}",
        f"{shlex.quote(str(site.git_executable))} -C {source} init",
        f"{shlex.quote(str(site.git_executable))} -C {source} fetch --depth=1 {shlex.quote(url)} {runtime_commit}",
        f"{shlex.quote(str(site.git_executable))} -C {source} checkout --detach FETCH_HEAD",
        f"[[ $({shlex.quote(str(site.git_executable))} -C {source} rev-parse HEAD) == {runtime_commit} ]] || exit 22",
        f"exec env PUTPOCKET_CONTAINER_EXECUTABLE={shlex.quote(str(site.cpu.container_executable))} PUTPOCKET_CPU_LOCAL_SCRATCH_ROOT={shlex.quote(str(site.cpu.local_scratch_root))} PUTPOCKET_SHARED_BUILD_ROOT={shlex.quote(str(site.shared_build_root))} PUTPOCKET_EXPECTED_BUNDLE_KEY={bundle_key} PUTPOCKET_BUILD_SOURCE_COMMIT={build_commit} PUTPOCKET_RUNTIME_SOURCE_COMMIT={runtime_commit} PUTPOCKET_WRAPPER_SOURCE_COMMIT={runtime_commit} PUTPOCKET_ALLOW_RUNTIME_SOURCE_SPLIT={split} PUTPOCKET_IMMUTABLE_BUNDLE_REUSE_ONLY=1 /bin/bash {source}/scripts/cluster/build_glm52_vllm_sm90.sh",
    )
    return "; ".join(parts)


def _run_wrapper(
    site: VllmDiagnosticSite,
    url: str,
    runtime_commit: str,
    build_commit: str,
    bundle_key: str,
    allow_source_split: bool,
) -> str:
    source = f'{shlex.quote(str(site.h200_work_root / "source"))}/"${{SLURM_JOB_ID}}-{runtime_commit[:12]}"'
    split = "1" if allow_source_split else "0"
    parts = (
        "set -euo pipefail", "umask 077",
        "[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_JOB_NUM_NODES:-0} == 1 && ${SLURM_GPUS_ON_NODE:-0} == 4 ]] || { echo E_H200_SLURM_ALLOCATION_REQUIRED >&2; exit 20; }",
        f"[[ -x {shlex.quote(str(site.cpu.container_executable))} ]] && {shlex.quote(str(site.cpu.container_executable))} info >/dev/null 2>&1 || {{ echo E_H200_CONTAINER_RUNTIME_UNAVAILABLE >&2; exit 21; }}",
        f"[[ -d {shlex.quote(str(site.h200_storage_parent))} && -w {shlex.quote(str(site.h200_storage_parent))} ]] || {{ echo E_H200_LOCAL_STORAGE_PARENT_UNWRITABLE >&2; exit 21; }}",
        f"mkdir -p {shlex.quote(str(site.h200_work_root))} {shlex.quote(str(site.h200_artifact_root))}",
        f"[[ -d {shlex.quote(str(site.h200_work_root))} && -w {shlex.quote(str(site.h200_work_root))} && -d {shlex.quote(str(site.h200_artifact_root))} && -w {shlex.quote(str(site.h200_artifact_root))} ]] || {{ echo E_H200_RUN_ROOT_UNWRITABLE >&2; exit 21; }}",
        f"mkdir -p {source}",
        f"{shlex.quote(str(site.git_executable))} -C {source} init",
        f"{shlex.quote(str(site.git_executable))} -C {source} fetch --depth=1 {shlex.quote(url)} {runtime_commit}",
        f"{shlex.quote(str(site.git_executable))} -C {source} checkout --detach FETCH_HEAD",
        f"[[ $({shlex.quote(str(site.git_executable))} -C {source} rev-parse HEAD) == {runtime_commit} ]] || exit 22",
        f"exec env PUTPOCKET_CONTAINER_EXECUTABLE={shlex.quote(str(site.cpu.container_executable))} PUTPOCKET_SHARED_BUILD_ROOT={shlex.quote(str(site.shared_build_root))} PUTPOCKET_EXPECTED_BUNDLE_KEY={bundle_key} PUTPOCKET_NVIDIA_SMI={shlex.quote(str(site.nvidia_smi))} PUTPOCKET_H200_STORAGE_PARENT={shlex.quote(str(site.h200_storage_parent))} PUTPOCKET_H200_WORK_ROOT={shlex.quote(str(site.h200_work_root))} PUTPOCKET_RUN_ARTIFACT_ROOT={shlex.quote(str(site.h200_artifact_root))} PUTPOCKET_BUILD_SOURCE_COMMIT={build_commit} PUTPOCKET_RUNTIME_SOURCE_COMMIT={runtime_commit} PUTPOCKET_WRAPPER_SOURCE_COMMIT={runtime_commit} PUTPOCKET_ALLOW_RUNTIME_SOURCE_SPLIT={split} /bin/bash {source}/scripts/cluster/run_glm52_vllm_diagnostic.sh",
    )
    return "; ".join(parts)


def _public_project_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.query or parsed.fragment:
        raise ConfigError("PUBLIC_GITHUB_HTTPS_PROJECT_URL_REQUIRED")
    if parsed.username or parsed.password:
        raise ConfigError("CREDENTIAL_BEARING_PROJECT_URL_FORBIDDEN")
    path = parsed.path.rstrip("/")
    if not re.fullmatch(r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?", path):
        raise ConfigError("PUBLIC_GITHUB_PROJECT_PATH_INVALID")
    return urlunsplit(("https", "github.com", path, "", ""))


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be an object")
    return value
