from __future__ import annotations

import re
import shlex

from .cluster_config import ClusterProfile, ClusterSite
from .cluster_safety import safe_absolute_path
from .errors import ConfigError


SLURM_JOB_KINDS = ("environment", "readiness", "generation-handoff")
_DIRECTIVE_VALUE = re.compile(r"^[A-Za-z0-9_.:@+,&|!%=-]+$")


def render_slurm_job(profile: ClusterProfile, site: ClusterSite, job_kind: str) -> str:
    if job_kind not in SLURM_JOB_KINDS:
        raise ConfigError(f"Unsupported phase-1 Slurm job kind: {job_kind}")
    model_path, model_revision = site.model_for(profile)
    profile_path = site.repository_root / "configs" / "cluster" / "profiles" / f"{profile.profile_id}.yaml"
    lock_path = site.repository_root / profile.environment_lock
    job_name = f"pp-{profile.profile_id}-{job_kind}"[:120]
    log_root = safe_absolute_path(site.slurm_log_root, "site.slurm_log_root", slurm_directive=True)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={_directive(job_name, 'job_name')}",
        f"#SBATCH --nodes={profile.nodes}",
        "#SBATCH --ntasks=1",
        f"#SBATCH --gpus-per-node={profile.gpus_per_node}",
        f"#SBATCH --cpus-per-task={site.cpus_per_task}",
        f"#SBATCH --time={_directive(site.wall_time, 'wall_time')}",
        f"#SBATCH --output={log_root.as_posix()}/%x-%j.out",
        f"#SBATCH --error={log_root.as_posix()}/%x-%j.err",
        "#SBATCH --export=NONE",
    ]
    if site.partition:
        lines.append(f"#SBATCH --partition={_directive(site.partition, 'partition')}")
    if site.account:
        lines.append(f"#SBATCH --account={_directive(site.account, 'account')}")
    if site.constraint:
        lines.append(f"#SBATCH --constraint={_directive(site.constraint, 'constraint')}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "umask 077",
            f"REPOSITORY_ROOT={shlex.quote(str(site.repository_root))}",
            f"PROFILE_PATH={shlex.quote(str(profile_path))}",
            f"MODEL_PATH={shlex.quote(model_path)}",
            f"MODEL_REVISION={shlex.quote(model_revision or '')}",
            f"ARTIFACT_BASE={shlex.quote(str(site.artifact_root))}",
            f"CACHE_ROOT={shlex.quote(str(site.cache_root))}",
            f"BASE_PYTHON={shlex.quote(str(site.python_executable))}",
            f"RUNTIME_PYTHON={shlex.quote(str(site.environment_root / 'bin' / 'python'))}",
            f"GIT_EXECUTABLE={shlex.quote(str(site.git_executable))}",
            f"NVIDIA_SMI_EXECUTABLE={shlex.quote(str(site.nvidia_smi_executable))}",
            f"NVCC_EXECUTABLE={shlex.quote(str(site.nvcc_executable))}",
            f"RUN_ROOT=\"$ARTIFACT_BASE/{profile.profile_id}/{job_kind}/$SLURM_JOB_ID\"",
            "export HF_HOME=\"$CACHE_ROOT/huggingface\"",
            "export VLLM_CACHE_ROOT=\"$CACHE_ROOT/vllm\"",
            "export TORCH_HOME=\"$CACHE_ROOT/torch\"",
            "export DG_JIT_CACHE_DIR=\"$CACHE_ROOT/deep_gemm\"",
            "export PYTHONPATH=\"$REPOSITORY_ROOT/src\"",
            "mkdir -p \"$RUN_ROOT\"",
            "cd \"$REPOSITORY_ROOT\"",
            '"$BASE_PYTHON" -m putpocket_dataset_mining.cluster_cli allocation-check',
        ]
    )
    if job_kind == "environment":
        command = [
            '"$BASE_PYTHON"',
            "-m putpocket_dataset_mining.cluster_cli run-guarded",
            "--action environment-build",
            '--profile "$PROFILE_PATH"',
            '--artifact-root "$RUN_ROOT"',
            '--git-executable "$GIT_EXECUTABLE"',
            '--nvidia-smi-executable "$NVIDIA_SMI_EXECUTABLE"',
            '--nvcc-executable "$NVCC_EXECUTABLE"',
            '--model-revision "$MODEL_REVISION"',
            "--",
            '"$BASE_PYTHON" -m putpocket_dataset_mining.cluster_cli env-bootstrap',
            f"--lock {shlex.quote(str(lock_path))}",
            '--repository-root "$REPOSITORY_ROOT"',
            f"--environment-root {shlex.quote(str(site.environment_root))}",
            f"--vllm-source-root {shlex.quote(str(site.vllm_source_root))}",
            '--cache-root "$CACHE_ROOT"',
            '--python-executable "$BASE_PYTHON"',
            f"--uv-executable {shlex.quote(str(site.uv_executable))}",
            '--git-executable "$GIT_EXECUTABLE"',
            f"--build-jobs {site.cpus_per_task}",
            "--execute",
        ]
    else:
        stage = "all" if job_kind == "readiness" else "generation-handoff"
        command = [
            '"$RUNTIME_PYTHON"',
            "-m putpocket_dataset_mining.cluster_cli readiness",
            '--profile "$PROFILE_PATH"',
            f"--stage {stage}",
            '--model-path "$MODEL_PATH"',
            '--model-revision "$MODEL_REVISION"',
            '--artifact-root "$RUN_ROOT"',
            '--git-executable "$GIT_EXECUTABLE"',
            '--nvidia-smi-executable "$NVIDIA_SMI_EXECUTABLE"',
            '--nvcc-executable "$NVCC_EXECUTABLE"',
        ]
    separator = " \\" + "\n  "
    lines.extend(["", separator.join(command), ""])
    rendered = "\n".join(lines)
    if re.search(r"(^|\s)(?:sbatch|salloc)(?:\s|$)", rendered):
        raise AssertionError("Renderer must never submit or allocate Slurm jobs")
    return rendered


def _directive(value: str, field: str) -> str:
    if not _DIRECTIVE_VALUE.fullmatch(value):
        raise ConfigError(f"Unsafe Slurm directive value for {field}: {value!r}")
    return value
