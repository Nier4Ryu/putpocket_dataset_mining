from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .cluster_config import _optional_text, _positive_int, _required_text, _valid_wall_time
from .cluster_safety import reject_secret_fields, safe_absolute_path
from .config import load_yaml
from .errors import ConfigError
from .swebench_pro import DATASET_REVISION, HARNESS_REPOSITORY, HARNESS_SHA, MODEL_ID


_DIRECTIVE = re.compile(r"^[A-Za-z0-9_.:@+,&|!%=/:-]+$")
_GPU_DIRECTIVES = (
    re.compile(r"^--gres=gpu:h200:4$", re.IGNORECASE),
    re.compile(r"^--gpus-per-node=h200:4$", re.IGNORECASE),
    re.compile(r"^--gpus=h200:4$", re.IGNORECASE),
)
_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class BaselineSite:
    partition: str | None
    account: str | None
    qos: str | None
    wall_time: str
    memory: str
    cpus_per_task: int
    h200_gpu_directive: str
    base_python: Path
    uv_executable: Path
    git_executable: Path
    nvidia_smi_executable: Path
    nvcc_executable: Path
    curl_executable: Path
    storage_root: Path
    cache_root: Path
    artifact_root: Path
    slurm_log_root: Path
    model_path: Path
    model_revision: str
    model_source: str
    experiment_id: str
    evaluation_workers: int
    agent_workers: int

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "BaselineSite":
        reject_secret_fields(data, path="swebench_pro_site")
        if data.get("schema_version") != 1:
            raise ConfigError("SWE-bench Pro site schema_version must be 1")
        site = data.get("site")
        model = data.get("model")
        job = data.get("job")
        if not isinstance(site, dict) or not isinstance(model, dict) or not isinstance(job, dict):
            raise ConfigError("SWE-bench Pro site, model, and job fields must be mappings")
        wall_time = _required_text(site.get("wall_time"), "site.wall_time")
        if not _valid_wall_time(wall_time):
            raise ConfigError("site.wall_time must use Slurm D-HH:MM:SS or HH:MM:SS syntax")
        gpu = _required_text(site.get("h200_gpu_directive"), "site.h200_gpu_directive")
        if not any(pattern.fullmatch(gpu) for pattern in _GPU_DIRECTIVES):
            raise ConfigError("site.h200_gpu_directive must request exactly four H200 GPUs")
        source = _required_text(model.get("source"), "model.source")
        if source not in {"huggingface", "local"}:
            raise ConfigError("model.source must be huggingface or local")
        model_path = safe_absolute_path(model.get("path") or "", "model.path")
        storage = safe_absolute_path(site.get("storage_root") or "", "site.storage_root")
        for field in ("cache_root", "artifact_root", "slurm_log_root"):
            value = safe_absolute_path(site.get(field) or "", f"site.{field}", slurm_directive=field == "slurm_log_root")
            try:
                value.relative_to(storage)
            except ValueError as exc:
                raise ConfigError(f"site.{field} must be inside site.storage_root") from exc
        try:
            model_path.relative_to(storage)
        except ValueError as exc:
            raise ConfigError("model.path must be inside site.storage_root") from exc
        memory = _required_text(site.get("memory"), "site.memory")
        if not re.fullmatch(r"[1-9][0-9]*(?:[KMGTP])?", memory, re.IGNORECASE):
            raise ConfigError("site.memory must use a simple Slurm size such as 256G")
        experiment_id = _required_text(job.get("experiment_id"), "job.experiment_id")
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,79}", experiment_id):
            raise ConfigError("job.experiment_id must be a lowercase Slurm-safe identifier")
        return cls(
            partition=_optional_text(site.get("partition"), "site.partition"),
            account=_optional_text(site.get("account"), "site.account"),
            qos=_optional_text(site.get("qos"), "site.qos"),
            wall_time=wall_time,
            memory=memory,
            cpus_per_task=_positive_int(site.get("cpus_per_task"), "site.cpus_per_task"),
            h200_gpu_directive=gpu,
            base_python=safe_absolute_path(site.get("base_python") or "", "site.base_python"),
            uv_executable=safe_absolute_path(site.get("uv_executable") or "", "site.uv_executable"),
            git_executable=safe_absolute_path(site.get("git_executable") or "", "site.git_executable"),
            nvidia_smi_executable=safe_absolute_path(
                site.get("nvidia_smi_executable") or "", "site.nvidia_smi_executable"
            ),
            nvcc_executable=safe_absolute_path(site.get("nvcc_executable") or "", "site.nvcc_executable"),
            curl_executable=safe_absolute_path(site.get("curl_executable") or "", "site.curl_executable"),
            storage_root=storage,
            cache_root=safe_absolute_path(site["cache_root"], "site.cache_root"),
            artifact_root=safe_absolute_path(site["artifact_root"], "site.artifact_root"),
            slurm_log_root=safe_absolute_path(site["slurm_log_root"], "site.slurm_log_root", slurm_directive=True),
            model_path=model_path,
            model_revision=_required_text(model.get("revision"), "model.revision"),
            model_source=source,
            experiment_id=experiment_id,
            evaluation_workers=_positive_int(job.get("evaluation_workers"), "job.evaluation_workers"),
            agent_workers=_positive_int(job.get("agent_workers"), "job.agent_workers"),
        )


def load_baseline_site(path: str | Path) -> BaselineSite:
    return BaselineSite.from_mapping(load_yaml(path))


def render_baseline_job(
    *, site: BaselineSite, project_url: str, project_commit: str, preflight_only: bool = False
) -> str:
    _validate_project(project_url, project_commit)
    directives = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name=pp-glm52-swepro-{'preflight' if preflight_only else 'baseline'}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH {site.h200_gpu_directive}",
        f"#SBATCH --cpus-per-task={site.cpus_per_task}",
        f"#SBATCH --mem={_directive(site.memory, 'memory')}",
        f"#SBATCH --time={_directive(site.wall_time, 'wall_time')}",
        f"#SBATCH --output={site.slurm_log_root}/%x-%j.out",
        f"#SBATCH --error={site.slurm_log_root}/%x-%j.err",
        "#SBATCH --export=NONE",
    ]
    if site.partition:
        directives.append(f"#SBATCH --partition={_directive(site.partition, 'partition')}")
    if site.account:
        directives.append(f"#SBATCH --account={_directive(site.account, 'account')}")
    if site.qos:
        directives.append(f"#SBATCH --qos={_directive(site.qos, 'qos')}")
    values = [
        "",
        "set -euo pipefail",
        "umask 077",
        f"PROJECT_URL={shlex.quote(project_url)}",
        f"PROJECT_COMMIT={shlex.quote(project_commit)}",
        f"HARNESS_URL={shlex.quote(HARNESS_REPOSITORY)}",
        f"HARNESS_COMMIT={HARNESS_SHA}",
        f"DATASET_REVISION={DATASET_REVISION}",
        f"MODEL_ID={shlex.quote(MODEL_ID)}",
        f"MODEL_REVISION={shlex.quote(site.model_revision)}",
        f"MODEL_SOURCE={site.model_source}",
        f"MODEL_PATH={shlex.quote(str(site.model_path))}",
        f"STORAGE_ROOT={shlex.quote(str(site.storage_root))}",
        f"CACHE_ROOT={shlex.quote(str(site.cache_root))}",
        f"ARTIFACT_BASE={shlex.quote(str(site.artifact_root))}",
        f"BASE_PYTHON={shlex.quote(str(site.base_python))}",
        f"UV_EXECUTABLE={shlex.quote(str(site.uv_executable))}",
        f"GIT_EXECUTABLE={shlex.quote(str(site.git_executable))}",
        f"NVIDIA_SMI_EXECUTABLE={shlex.quote(str(site.nvidia_smi_executable))}",
        f"NVCC_EXECUTABLE={shlex.quote(str(site.nvcc_executable))}",
        f"CURL_EXECUTABLE={shlex.quote(str(site.curl_executable))}",
        f"EXPERIMENT_ID={site.experiment_id}",
        f"AGENT_WORKERS={site.agent_workers}",
        f"EVAL_WORKERS={site.evaluation_workers}",
        f"RUN_ROOT=\"$ARTIFACT_BASE/$EXPERIMENT_ID/$SLURM_JOB_ID\"",
        "PROJECT_ROOT=\"$STORAGE_ROOT/work/$EXPERIMENT_ID/project-$PROJECT_COMMIT\"",
        "HARNESS_ROOT=\"$STORAGE_ROOT/work/$EXPERIMENT_ID/SWE-bench_Pro-os-$HARNESS_COMMIT\"",
        "ENV_ROOT=\"$STORAGE_ROOT/envs/glm52-swepro\"",
        "VLLM_SOURCE_ROOT=\"$STORAGE_ROOT/src/vllm-026\"",
        "RUNTIME_PYTHON=\"$ENV_ROOT/bin/python\"",
        "PROFILE_PRIMARY=glm52_nvfp4_tp1_pcp4_ep",
        "PROFILE_FALLBACK=glm52_nvfp4_tp2_pcp2_ep",
        "SERVER_PID=",
        "JOB_STATUS=failed",
        "export HF_HOME=\"$CACHE_ROOT/huggingface\"",
        "export VLLM_CACHE_ROOT=\"$CACHE_ROOT/vllm\"",
        "export TORCH_HOME=\"$CACHE_ROOT/torch\"",
        "export DG_JIT_CACHE_DIR=\"$CACHE_ROOT/deep_gemm\"",
        "export UV_CACHE_DIR=\"$CACHE_ROOT/uv\"",
        "export OPENAI_API_KEY=local-vllm-no-auth",
        "",
        "finish() {",
        "  rc=$?",
        "  if [[ -n \"$SERVER_PID\" ]]; then kill \"$SERVER_PID\" >/dev/null 2>&1 || true; wait \"$SERVER_PID\" >/dev/null 2>&1 || true; fi",
        "  printf '{\"schema_version\":1,\"status\":\"%s\",\"returncode\":%s,\"job_id\":\"%s\"}\\n' \"$JOB_STATUS\" \"$rc\" \"${SLURM_JOB_ID:-unknown}\" > \"$RUN_ROOT/final_status.json.partial\" 2>/dev/null || true",
        "  mv \"$RUN_ROOT/final_status.json.partial\" \"$RUN_ROOT/final_status.json\" 2>/dev/null || true",
        "}",
        "trap finish EXIT INT TERM",
        "",
        "[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || { echo E_SLURM_ALLOCATION_REQUIRED >&2; exit 20; }",
        "[[ -n ${SLURM_JOB_NODELIST:-} && ${SLURM_JOB_NUM_NODES:-0} == 1 ]] || { echo E_SLURM_ALLOCATION_REQUIRED >&2; exit 20; }",
        "[[ -n ${SLURM_JOB_NAME:-}${SLURM_STEP_ID:-} ]] || { echo E_SLURM_ALLOCATION_REQUIRED >&2; exit 20; }",
        "mkdir -p \"$RUN_ROOT\" \"$CACHE_ROOT\" \"$STORAGE_ROOT/work/$EXPERIMENT_ID\"",
        "printf '{\"schema_version\":1,\"job_id\":\"%s\",\"nodelist\":\"%s\",\"requested_gpus\":4,\"requested_type\":\"H200\"}\\n' \"$SLURM_JOB_ID\" \"$SLURM_JOB_NODELIST\" > \"$RUN_ROOT/allocation.json\"",
        "",
        "docker_present=false; docker_usable=false; podman_present=false; apptainer_present=false; singularity_present=false",
        "command -v docker >/dev/null 2>&1 && docker_present=true",
        "command -v podman >/dev/null 2>&1 && podman_present=true",
        "command -v apptainer >/dev/null 2>&1 && apptainer_present=true",
        "command -v singularity >/dev/null 2>&1 && singularity_present=true",
        "if $docker_present && docker info >/dev/null 2>&1; then docker_usable=true; fi",
        "if $docker_usable; then",
        "  docker version > \"$RUN_ROOT/docker_version.txt\" 2>&1",
        "  printf '{\"schema_version\":1,\"status\":\"passed\",\"runtime\":\"docker\",\"official_evaluation_supported\":true}\\n' > \"$RUN_ROOT/container_preflight.json\"",
        "else",
        "  printf '{\"schema_version\":1,\"status\":\"failed\",\"failure_class\":\"OFFICIAL_EVALUATION_DOCKER_REQUIRED\",\"docker_present\":%s,\"podman_present\":%s,\"apptainer_present\":%s,\"singularity_present\":%s,\"official_evaluation_supported\":false}\\n' \"$docker_present\" \"$podman_present\" \"$apptainer_present\" \"$singularity_present\" > \"$RUN_ROOT/container_preflight.json\"",
        "  JOB_STATUS=blocked_container_runtime",
        "  exit 42",
        "fi",
    ]
    if preflight_only:
        values.extend(["JOB_STATUS=preflight_passed", "exit 0", ""])
        return "\n".join(directives + values)
    body = [
        "",
        "if [[ ! -d \"$PROJECT_ROOT/.git\" ]]; then",
        "  mkdir -p \"$PROJECT_ROOT\"",
        "  \"$GIT_EXECUTABLE\" -C \"$PROJECT_ROOT\" init",
        "  \"$GIT_EXECUTABLE\" -C \"$PROJECT_ROOT\" remote add origin \"$PROJECT_URL\"",
        "  \"$GIT_EXECUTABLE\" -C \"$PROJECT_ROOT\" fetch --depth=1 origin \"$PROJECT_COMMIT\"",
        "  \"$GIT_EXECUTABLE\" -C \"$PROJECT_ROOT\" checkout --detach FETCH_HEAD",
        "fi",
        "[[ $(\"$GIT_EXECUTABLE\" -C \"$PROJECT_ROOT\" rev-parse HEAD) == \"$PROJECT_COMMIT\" ]] || { echo project commit mismatch >&2; exit 31; }",
        "export PYTHONPATH=\"$PROJECT_ROOT/src\"",
        "\"$BASE_PYTHON\" -m putpocket_dataset_mining.swebench_pro_cli validate",
        "",
        "\"$BASE_PYTHON\" -m putpocket_dataset_mining.cluster_cli env-bootstrap \\",
        "  --lock \"$PROJECT_ROOT/configs/env/cluster_h200_sm90_vllm026.lock.yaml\" \\",
        "  --repository-root \"$PROJECT_ROOT\" --environment-root \"$ENV_ROOT\" \\",
        "  --vllm-source-root \"$VLLM_SOURCE_ROOT\" --cache-root \"$CACHE_ROOT\" \\",
        "  --python-executable \"$BASE_PYTHON\" --uv-executable \"$UV_EXECUTABLE\" \\",
        "  --git-executable \"$GIT_EXECUTABLE\" --build-jobs \"$SLURM_CPUS_PER_TASK\" --execute",
        "",
        "if [[ ! -d \"$HARNESS_ROOT/.git\" ]]; then",
        "  \"$GIT_EXECUTABLE\" clone --filter=blob:none --no-checkout \"$HARNESS_URL\" \"$HARNESS_ROOT\"",
        "  \"$GIT_EXECUTABLE\" -C \"$HARNESS_ROOT\" checkout --detach \"$HARNESS_COMMIT\"",
        "  \"$GIT_EXECUTABLE\" -C \"$HARNESS_ROOT\" submodule update --init --recursive",
        "fi",
        "[[ $(\"$GIT_EXECUTABLE\" -C \"$HARNESS_ROOT\" rev-parse HEAD) == \"$HARNESS_COMMIT\" ]] || { echo harness commit mismatch >&2; exit 32; }",
        "\"$UV_EXECUTABLE\" pip install --python \"$RUNTIME_PYTHON\" -r \"$HARNESS_ROOT/requirements.txt\"",
        "\"$UV_EXECUTABLE\" pip install --python \"$RUNTIME_PYTHON\" -e \"$HARNESS_ROOT/mini-swe-agent\"",
        "",
        "if [[ \"$MODEL_SOURCE\" == huggingface ]]; then",
        "  \"$RUNTIME_PYTHON\" -c \"from huggingface_hub import snapshot_download; snapshot_download(repo_id='$MODEL_ID', revision='$MODEL_REVISION', local_dir='$MODEL_PATH')\"",
        "fi",
        "[[ -d \"$MODEL_PATH\" ]] || { echo model checkpoint missing >&2; exit 33; }",
        "",
        "RUNTIME_MANIFEST_ROOT=\"$RUN_ROOT/runtime\"",
        "\"$RUNTIME_PYTHON\" -m putpocket_dataset_mining.cluster_cli readiness \\",
        "  --profile \"$PROJECT_ROOT/configs/cluster/profiles/$PROFILE_PRIMARY.yaml\" --stage checkpoint \\",
        "  --model-path \"$MODEL_PATH\" --model-revision \"$MODEL_REVISION\" \\",
        "  --artifact-root \"$RUNTIME_MANIFEST_ROOT\" --git-executable \"$GIT_EXECUTABLE\" \\",
        "  --nvidia-smi-executable \"$NVIDIA_SMI_EXECUTABLE\" --nvcc-executable \"$NVCC_EXECUTABLE\"",
        "",
        "start_server() {",
        "  profile=$1",
        "  if [[ \"$profile\" == \"$PROFILE_PRIMARY\" ]]; then args=(--tensor-parallel-size 1 --prefill-context-parallel-size 4 --enable-expert-parallel); else args=(--tensor-parallel-size 2 --prefill-context-parallel-size 2 --enable-expert-parallel); fi",
        "  printf '%s\\n' \"$profile\" > \"$RUN_ROOT/parallel_profile_used.txt\"",
        "  \"$ENV_ROOT/bin/vllm\" serve \"$MODEL_PATH\" --revision \"$MODEL_REVISION\" --served-model-name \"$MODEL_ID\" \"${args[@]}\" > \"$RUN_ROOT/vllm-$profile.log\" 2>&1 &",
        "  SERVER_PID=$!",
        "}",
        "wait_server() {",
        "  for _ in $(seq 1 120); do",
        "    \"$CURL_EXECUTABLE\" --fail --silent --show-error http://127.0.0.1:8000/health >/dev/null 2>&1 && return 0",
        "    kill -0 \"$SERVER_PID\" >/dev/null 2>&1 || return 1",
        "    sleep 10",
        "  done",
        "  return 1",
        "}",
        "start_server \"$PROFILE_PRIMARY\"",
        "if ! wait_server; then",
        "  wait \"$SERVER_PID\" >/dev/null 2>&1 || true",
        "  SERVER_PID=",
        "  if grep -Eiq 'prefill.?context.?parallel|PCP.*(unsupported|not supported|invalid)|parallel.*(unsupported|not supported)' \"$RUN_ROOT/vllm-$PROFILE_PRIMARY.log\"; then",
        "    printf '%s\\n' primary_startup_compatibility_failure > \"$RUN_ROOT/fallback_reason.txt\"",
        "    start_server \"$PROFILE_FALLBACK\"",
        "    wait_server || { echo fallback vLLM startup failed >&2; exit 34; }",
        "  else",
        "    echo primary vLLM startup failed without a classified compatibility reason >&2",
        "    exit 34",
        "  fi",
        "fi",
        "",
        "\"$CURL_EXECUTABLE\" --fail --silent --show-error -H 'Content-Type: application/json' \\",
        "  -d '{\"model\":\"nvidia/GLM-5.2-NVFP4\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with READY only.\"}],\"max_tokens\":8,\"temperature\":0}' \\",
        "  http://127.0.0.1:8000/v1/chat/completions > \"$RUN_ROOT/one_shot_generation.json\"",
        "",
        "for selection in smoke full; do",
        "  selection_root=\"$RUN_ROOT/$selection\"",
        "  \"$RUNTIME_PYTHON\" -m putpocket_dataset_mining.swebench_pro_cli stage --stage prepare \\",
        "    --artifact-root \"$selection_root\" --fingerprint \"$DATASET_REVISION-$selection\" -- \\",
        "    \"$RUNTIME_PYTHON\" -m putpocket_dataset_mining.swebench_pro_cli prepare \\",
        "    --selection \"$selection\" --harness-root \"$HARNESS_ROOT\" --output-root \"$selection_root/prepared\"",
        "  \"$RUNTIME_PYTHON\" -m putpocket_dataset_mining.swebench_pro_cli agent-config \\",
        "    --harness-root \"$HARNESS_ROOT\" --runtime docker --output \"$selection_root/agent.yaml\"",
        "  \"$RUNTIME_PYTHON\" -m putpocket_dataset_mining.swebench_pro_cli stage --stage inference \\",
        "    --artifact-root \"$selection_root\" --fingerprint \"$PROJECT_COMMIT-$selection-inference\" -- \\",
        "    \"$ENV_ROOT/bin/mini-extra\" swebench --subset \"$selection_root/prepared/mini_dataset\" --split test \\",
        "    --workers \"$AGENT_WORKERS\" --model openai/nvidia/GLM-5.2-NVFP4 \\",
        "    --config \"$selection_root/agent.yaml\" --environment-class docker --output \"$selection_root/inference\"",
        "  \"$RUNTIME_PYTHON\" -m putpocket_dataset_mining.swebench_pro_cli stage --stage gather \\",
        "    --artifact-root \"$selection_root\" --fingerprint \"$PROJECT_COMMIT-$selection-gather\" -- \\",
        "    \"$RUNTIME_PYTHON\" -m putpocket_dataset_mining.swebench_pro_cli gather \\",
        "    --harness-root \"$HARNESS_ROOT\" --inference-root \"$selection_root/inference\" \\",
        "    --prefix glm52-nvfp4-baseline --output \"$selection_root/patches.json\"",
        "  \"$RUNTIME_PYTHON\" -m putpocket_dataset_mining.swebench_pro_cli stage --stage evaluate \\",
        "    --artifact-root \"$selection_root\" --fingerprint \"$HARNESS_COMMIT-$selection-evaluate\" -- \\",
        "    \"$RUNTIME_PYTHON\" \"$HARNESS_ROOT/swe_bench_pro_eval.py\" \\",
        "    --raw_sample_path \"$selection_root/prepared/raw_samples.jsonl\" --patch_path \"$selection_root/patches.json\" \\",
        "    --output_dir \"$selection_root/evaluation\" --scripts_dir \"$HARNESS_ROOT/run_scripts\" \\",
        "    --num_workers \"$EVAL_WORKERS\" --dockerhub_username jefzda --use_local_docker",
        "  \"$RUNTIME_PYTHON\" -m putpocket_dataset_mining.swebench_pro_cli stage --stage finalize \\",
        "    --artifact-root \"$selection_root\" --fingerprint \"$HARNESS_COMMIT-$selection-finalize\" -- \\",
        "    \"$RUNTIME_PYTHON\" -m putpocket_dataset_mining.swebench_pro_cli finalize \\",
        "    --selection \"$selection\" --selection-manifest \"$selection_root/prepared/selection_manifest.json\" \\",
        "    --eval-results \"$selection_root/evaluation/eval_results.json\" --output \"$selection_root/acceptance_report.json\"",
        "done",
        "",
        "PROFILE_USED=$(cat \"$RUN_ROOT/parallel_profile_used.txt\")",
        "\"$RUNTIME_PYTHON\" -m putpocket_dataset_mining.swebench_pro_cli provenance \\",
        "  --project-root \"$PROJECT_ROOT\" --harness-root \"$HARNESS_ROOT\" --model-revision \"$MODEL_REVISION\" \\",
        "  --parallel-profile \"$PROFILE_USED\" --container-runtime docker \\",
        "  --runtime-manifest \"$RUNTIME_MANIFEST_ROOT/readiness_manifest.json\" --artifact-root \"$RUN_ROOT\" \\",
        "  --output \"$RUN_ROOT/provenance.json\"",
        "JOB_STATUS=complete",
        "",
    ]
    rendered = "\n".join(directives + values + body)
    if re.search(r"(^|\s)(?:sbatch|salloc)(?:\s|$)", rendered):
        raise AssertionError("SWE-bench Pro renderer must never submit or allocate")
    return rendered


def _validate_project(url: str, commit: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.username or parsed.password:
        raise ConfigError("Project checkout URL must be a public credential-free GitHub HTTPS URL")
    if parsed.query or parsed.fragment or not parsed.path.endswith(".git"):
        raise ConfigError("Project checkout URL must not contain query/fragment data and must end in .git")
    if not _SHA.fullmatch(commit):
        raise ConfigError("Project checkout commit must be a full Git SHA")


def _directive(value: str, field: str) -> str:
    if not _DIRECTIVE.fullmatch(value):
        raise ConfigError(f"Unsafe Slurm directive value for {field}: {value!r}")
    return value
