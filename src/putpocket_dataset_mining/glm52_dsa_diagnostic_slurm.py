from __future__ import annotations

import shlex

from .glm52_sglang_gate import validate_public_project
from .glm52_sglang_gate_slurm import GateSite


def render_compact_diagnostic_submission(*, site: GateSite, project_url: str, project_commit: str) -> str:
    """Render one Login-safe control-plane command for only the DSA diagnostic."""

    validate_public_project(project_url, project_commit)
    source_root = site.storage_root / "diagnostic-source" / f"putpocket-{project_commit}"
    artifact_root = site.storage_root / "artifacts" / "glm52-dsa-diagnostic"
    wrapper = "; ".join(
        (
            "set -euo pipefail",
            "umask 077",
            f"PROJECT_URL={shlex.quote(project_url)}",
            f"PROJECT_COMMIT={project_commit}",
            f"SOURCE_ROOT={shlex.quote(str(source_root))}",
            f"RUN_ROOT={shlex.quote(str(artifact_root))}/\"${{SLURM_JOB_ID:-unknown}}\"",
            "[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || { echo E_SLURM_ALLOCATION_REQUIRED >&2; exit 20; }",
            "[[ ${SLURM_JOB_NUM_NODES:-0} == 1 && -n ${SLURM_JOB_NODELIST:-} ]] || { echo E_NODE_COUNT_MISMATCH >&2; exit 20; }",
            "[[ ${SLURM_GPUS_ON_NODE:-0} == 4 ]] || { echo E_GPU_ALLOCATION_COUNT_MISMATCH >&2; exit 20; }",
            "[[ -d /local-data/jslee202403 ]] || { echo E_COMPUTE_LOCAL_STORAGE_MISSING >&2; exit 21; }",
            "command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 || { echo E_CONTAINER_RUNTIME_UNAVAILABLE >&2; exit 30; }",
            f"[[ -x {shlex.quote(str(site.git_executable))} ]] || {{ echo E_COMPUTE_GIT_MISSING >&2; exit 21; }}",
            "mkdir -p \"$RUN_ROOT\" \"$SOURCE_ROOT\"",
            f"if [[ ! -d \"$SOURCE_ROOT/.git\" ]]; then {shlex.quote(str(site.git_executable))} -C \"$SOURCE_ROOT\" init; fi",
            f"{shlex.quote(str(site.git_executable))} -C \"$SOURCE_ROOT\" diff --quiet --ignore-submodules -- || {{ echo E_DIRTY_COMPUTE_CHECKOUT >&2; exit 22; }}",
            f"{shlex.quote(str(site.git_executable))} -C \"$SOURCE_ROOT\" fetch --depth=1 \"$PROJECT_URL\" \"$PROJECT_COMMIT\"",
            f"{shlex.quote(str(site.git_executable))} -C \"$SOURCE_ROOT\" checkout --detach FETCH_HEAD",
            f"[[ $({shlex.quote(str(site.git_executable))} -C \"$SOURCE_ROOT\" rev-parse HEAD) == \"$PROJECT_COMMIT\" ]] || {{ echo E_PROJECT_COMMIT_MISMATCH >&2; exit 23; }}",
            "exec /bin/bash \"$SOURCE_ROOT/scripts/cluster/run_glm52_dsa_diagnostic.sh\" \"$PROJECT_COMMIT\"",
        )
    )
    flags = [
        "sbatch",
        "--parsable",
        "--job-name=pp-glm52-dsa-diagnostic",
        f"--partition={site.partition}",
        f"--account={site.account}",
        f"--qos={site.qos}",
        "--nodes=1",
        "--ntasks=1",
        "--gres=gpu:H200:4",
        "--cpus-per-task=32",
        "--mem=512G",
        "--time=06:00:00",
        f"--output={site.slurm_log_root}/%x-%j.out",
        f"--error={site.slurm_log_root}/%x-%j.err",
        "--export=NONE",
        f"--wrap={wrapper}",
    ]
    command = f"mkdir -p {shlex.quote(str(site.slurm_log_root))} && " + shlex.join(flags)
    lowered = command.lower()
    forbidden = ("swe_bench_pro_eval", "swebench_pro_full", "--selection full", "run_swebench")
    if any(value in lowered for value in forbidden):
        raise AssertionError("Diagnostic wrapper must not contain a benchmark/full transition")
    return command
