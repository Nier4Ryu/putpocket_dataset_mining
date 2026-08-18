from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from .cluster_safety import require_slurm_allocation, safe_absolute_path, validate_secret_free_command
from .config import load_yaml
from .constants import REPO_ROOT
from .errors import ConfigError


SOURCE_LOCK = REPO_ROOT / "configs" / "cluster" / "swebench_pro_sources.lock.yaml"
AGENT_OVERLAY = REPO_ROOT / "configs" / "cluster" / "swebench_pro_agent_overlay.yaml"
SELECTION_DIR = REPO_ROOT / "configs" / "cluster" / "selections"
HARNESS_REPOSITORY = "https://github.com/scaleapi/SWE-bench_Pro-os.git"
HARNESS_SHA = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
SWE_AGENT_SHA = "402a7b8fdac8193f3f255bb53859ba274234f596"
MINI_SWE_AGENT_SHA = "d74716a3c8104a113f77cc9ab94cf407ecdcf1e9"
DATASET_ID = "ScaleAI/SWE-bench_Pro"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
DATASET_SPLIT = "test"
FULL_TEST_ROWS = 731
DOCKERHUB_NAMESPACE = "jefzda/sweap-images"
MODEL_ID = "nvidia/GLM-5.2-NVFP4"
ACCEPTANCE_THRESHOLD_PERCENT = 40.0
_SHA = re.compile(r"^[0-9a-f]{40}$")
_DOCKER_TAG = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$")
_STAGES = {"prepare", "inference", "gather", "evaluate", "finalize"}


def validate_source_lock(path: str | Path = SOURCE_LOCK) -> dict[str, Any]:
    lock = load_yaml(path)
    expected = {
        ("harness", "repository"): HARNESS_REPOSITORY,
        ("harness", "commit"): HARNESS_SHA,
        ("submodules", "SWE-agent", "commit"): SWE_AGENT_SHA,
        ("submodules", "mini-swe-agent", "commit"): MINI_SWE_AGENT_SHA,
        ("dataset", "id"): DATASET_ID,
        ("dataset", "split"): DATASET_SPLIT,
        ("dataset", "revision"): DATASET_REVISION,
        ("dataset", "expected_rows"): FULL_TEST_ROWS,
        ("containers", "dockerhub_namespace"): DOCKERHUB_NAMESPACE,
        ("contract", "acceptance_threshold_percent"): ACCEPTANCE_THRESHOLD_PERCENT,
    }
    if lock.get("schema_version") != 1 or lock.get("provenance_status") != "commit_addressed":
        raise ConfigError("SWE-bench Pro source lock must be schema v1 and commit_addressed")
    for keys, value in expected.items():
        current: Any = lock
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                raise ConfigError(f"SWE-bench Pro source lock is missing {'.'.join(keys)}")
            current = current[key]
        if current != value:
            raise ConfigError(f"SWE-bench Pro source lock mismatch for {'.'.join(keys)}")
    for sha in (HARNESS_SHA, SWE_AGENT_SHA, MINI_SWE_AGENT_SHA, DATASET_REVISION):
        if not _SHA.fullmatch(sha):
            raise ConfigError("SWE-bench Pro source identities must be full Git SHAs")
    if lock["harness"].get("scorer") != "swe_bench_pro_eval.py":
        raise ConfigError("Official SWE-bench Pro scorer must remain swe_bench_pro_eval.py")
    if lock["dataset"].get("required_image_field") != "dockerhub_tag":
        raise ConfigError("Official image selection must use dataset.dockerhub_tag")
    if lock["contract"].get("official_scorer_unchanged") is not True:
        raise ConfigError("Official SWE-bench Pro scorer must be unchanged")
    return lock


@dataclass(frozen=True)
class Selection:
    selection_id: str
    mode: str
    expected_count: int
    score_eligible: bool


def load_selection(path_or_name: str | Path) -> Selection:
    path = Path(path_or_name)
    if not path.suffix and not path.exists():
        path = SELECTION_DIR / f"swebench_pro_{path}.yaml"
    data = load_yaml(path)
    if data.get("schema_version") != 1:
        raise ConfigError("SWE-bench Pro selection schema_version must be 1")
    if data.get("dataset") != DATASET_ID or data.get("split") != DATASET_SPLIT:
        raise ConfigError("Selection must target the pinned public SWE-bench Pro test split")
    if data.get("ordering") != "instance_id_ascending":
        raise ConfigError("Selection ordering must be deterministic instance_id_ascending")
    mode = data.get("mode")
    expected = data.get("expected_count")
    if mode == "smoke":
        if data.get("limit") != 1 or expected != 1 or data.get("score_eligible") is not False:
            raise ConfigError("Smoke selection must contain exactly one non-score-eligible instance")
    elif mode == "full":
        if data.get("limit") is not None or expected != FULL_TEST_ROWS or data.get("score_eligible") is not True:
            raise ConfigError("Full selection must cover all pinned public test rows")
    else:
        raise ConfigError("Selection mode must be smoke or full")
    return Selection(str(data.get("selection_id")), mode, int(expected), bool(data.get("score_eligible")))


def official_image_uri(row: Mapping[str, Any]) -> str:
    tag = row.get("dockerhub_tag")
    if not isinstance(tag, str) or not _DOCKER_TAG.fullmatch(tag):
        raise ConfigError(f"Invalid or missing dockerhub_tag for {row.get('instance_id', '<unknown>')}")
    return f"docker.io/{DOCKERHUB_NAMESPACE}:{tag}"


def select_rows(rows: Sequence[Mapping[str, Any]], selection: Selection) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id or instance_id in seen:
            raise ConfigError("Dataset rows require unique non-empty instance_id values")
        if not isinstance(row.get("repo"), str) or not row.get("repo"):
            raise ConfigError(f"Dataset row {instance_id} is missing repo")
        item = dict(row)
        item["image_name"] = official_image_uri(row)
        seen.add(instance_id)
        normalized.append(item)
    normalized.sort(key=lambda item: item["instance_id"])
    if selection.mode == "smoke":
        if not normalized:
            raise ConfigError("Smoke selection cannot be built from an empty dataset")
        normalized = normalized[:1]
    elif len(normalized) != FULL_TEST_ROWS:
        raise ConfigError(f"Full public selection requires {FULL_TEST_ROWS} rows, found {len(normalized)}")
    if len(normalized) != selection.expected_count:
        raise ConfigError("Selection row count does not match its committed manifest")
    return normalized


def validate_official_image_mapping(harness_root: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    root = safe_absolute_path(harness_root, "harness_root")
    helper = root / "helper_code" / "image_uri.py"
    if not helper.is_file():
        raise ConfigError("Pinned official harness image_uri helper is missing")
    spec = importlib.util.spec_from_file_location("putpocket_pinned_swepro_image_uri", helper)
    if spec is None or spec.loader is None:
        raise ConfigError("Pinned official harness image_uri helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for row in rows:
        expected = official_image_uri(row).removeprefix("docker.io/")
        actual = module.get_dockerhub_image_uri(row["instance_id"], "jefzda", row["repo"])
        if actual != expected:
            raise ConfigError(
                f"Official harness image mapping differs from dataset dockerhub_tag for {row['instance_id']}: "
                f"expected {expected}, got {actual}"
            )


def validate_agent_overlay(path: str | Path = AGENT_OVERLAY) -> dict[str, Any]:
    overlay = load_yaml(path)
    if overlay.get("schema_version") != 1:
        raise ConfigError("Agent overlay schema_version must be 1")
    model = overlay.get("model")
    endpoint = overlay.get("endpoint_contract")
    scaffold = overlay.get("scaffold")
    if not isinstance(model, dict) or model.get("id") != MODEL_ID:
        raise ConfigError("Agent overlay must use the unmodified GLM-5.2 NVFP4 model identity")
    if not isinstance(scaffold, dict) or scaffold.get("submodule") != "mini-swe-agent":
        raise ConfigError("Agent overlay must use the pinned official mini-swe-agent scaffold")
    api_base = str(model.get("api_base", ""))
    parsed = urlsplit(api_base)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.path != "/v1":
        raise ConfigError("Agent API base must be an HTTP loopback OpenAI-compatible /v1 endpoint")
    if not isinstance(endpoint, dict) or endpoint.get("cloud_api_forbidden") is not True:
        raise ConfigError("Agent overlay must explicitly forbid cloud model APIs")
    if model.get("api_key_sentinel") != "local-vllm-no-auth":
        raise ConfigError("Agent overlay may use only the fixed non-secret local vLLM sentinel")
    return overlay


def build_runtime_agent_config(base: Mapping[str, Any], overlay: Mapping[str, Any], runtime: str) -> dict[str, Any]:
    if runtime not in {"docker", "singularity"}:
        raise ConfigError("mini-swe-agent runtime must be docker or singularity")
    config = json.loads(json.dumps(base))
    model = overlay["model"]
    config["model"] = {
        "model_name": model["litellm_name"],
        "model_kwargs": {
            "custom_llm_provider": "openai",
            "api_base": model["api_base"],
            "api_key": model["api_key_sentinel"],
            "drop_params": True,
            "temperature": float(model["temperature"]),
        },
        "cost_tracking": "ignore_errors",
    }
    config.setdefault("environment", {})["environment_class"] = runtime
    return config


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    returncode: int
    marker: Path


def run_restartable_stage(
    *,
    stage: str,
    command: Sequence[str],
    artifact_root: str | Path,
    fingerprint: str,
    env: Mapping[str, str] | None = None,
    runner: Callable[[Sequence[str]], int] | None = None,
) -> StageResult:
    require_slurm_allocation(env)
    if stage not in _STAGES:
        raise ConfigError(f"Unsupported SWE-bench Pro stage: {stage}")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", fingerprint):
        raise ConfigError("Stage fingerprint contains unsafe characters")
    safe_command = validate_secret_free_command(command)
    root = safe_absolute_path(artifact_root, "artifact_root")
    marker_root = root / "markers"
    completed = marker_root / f"{stage}.complete.json"
    if completed.is_file():
        payload = json.loads(completed.read_text(encoding="utf-8"))
        if payload.get("fingerprint") == fingerprint and payload.get("returncode") == 0:
            return StageResult(stage, "skipped_complete", 0, completed)
        raise ConfigError(f"Completed stage fingerprint mismatch: {stage}")
    marker_root.mkdir(parents=True, exist_ok=True)
    partial = marker_root / f"{stage}.partial.json"
    _write_json(partial, {"schema_version": 1, "stage": stage, "fingerprint": fingerprint, "status": "running"})
    returncode = (runner or _subprocess_returncode)(safe_command)
    payload = {
        "schema_version": 1,
        "stage": stage,
        "fingerprint": fingerprint,
        "status": "passed" if returncode == 0 else "failed",
        "returncode": returncode,
        "exact_command": safe_command,
    }
    if returncode == 0:
        _write_json(completed, payload)
        partial.unlink(missing_ok=True)
        return StageResult(stage, "passed", 0, completed)
    failed = marker_root / f"{stage}.failed.json"
    _write_json(failed, payload)
    return StageResult(stage, "failed", returncode, failed)


def classify_container_preflight(
    *, docker_present: bool, docker_usable: bool, podman_present: bool, apptainer_present: bool, singularity_present: bool
) -> dict[str, Any]:
    if docker_present and docker_usable:
        return {"status": "passed", "runtime": "docker", "failure_class": None, "official_evaluation_supported": True}
    alternatives = [
        name
        for name, present in (("podman", podman_present), ("apptainer", apptainer_present), ("singularity", singularity_present))
        if present
    ]
    failure = "DOCKER_DAEMON_UNAVAILABLE" if docker_present else "DOCKER_EXECUTABLE_MISSING"
    if alternatives:
        failure = "OFFICIAL_EVALUATION_DOCKER_REQUIRED"
    return {
        "status": "failed",
        "runtime": None,
        "failure_class": failure,
        "alternatives_present": alternatives,
        "official_evaluation_supported": False,
        "detail": "mini-swe-agent can use Singularity, but unchanged official swe_bench_pro_eval.py requires Docker",
    }


def finalize_official_results(
    *, selection: Selection, expected_instance_ids: Sequence[str], eval_results: Mapping[str, Any]
) -> dict[str, Any]:
    expected = list(expected_instance_ids)
    if len(expected) != len(set(expected)) or len(expected) != selection.expected_count:
        raise ConfigError("Finalization expected IDs must exactly match the selection manifest")
    extra = sorted(set(eval_results) - set(expected))
    missing = sorted(set(expected) - set(eval_results))
    if extra:
        raise ConfigError("Official evaluation results contain instances outside the selection")
    invalid = [key for key, value in eval_results.items() if not isinstance(value, bool)]
    if invalid:
        raise ConfigError("Official evaluation result values must be booleans")
    resolved = sum(1 for key in expected if eval_results.get(key) is True)
    unresolved = sum(1 for key in expected if eval_results.get(key) is False)
    complete = not missing and len(eval_results) == selection.expected_count
    report: dict[str, Any] = {
        "schema_version": 1,
        "benchmark": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "selection_id": selection.selection_id,
        "selection_mode": selection.mode,
        "expected_count": selection.expected_count,
        "evaluated_count": len(eval_results),
        "resolved_count": resolved,
        "unresolved_count": unresolved,
        "missing_result_count": len(missing),
        "missing_instance_ids": missing,
        "official_scorer": f"{HARNESS_REPOSITORY}@{HARNESS_SHA}:swe_bench_pro_eval.py",
        "score_eligible": bool(selection.score_eligible and complete),
        "score_percent": None,
        "acceptance_threshold_percent": ACCEPTANCE_THRESHOLD_PERCENT,
        "acceptance_pass": None,
        "status": "complete" if complete else "incomplete",
    }
    if selection.mode == "full" and complete:
        score = 100.0 * resolved / selection.expected_count
        report["score_percent"] = score
        report["acceptance_pass"] = score >= ACCEPTANCE_THRESHOLD_PERCENT
    return report


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _subprocess_returncode(command: Sequence[str]) -> int:
    return subprocess.run(list(command), check=False).returncode
