from __future__ import annotations

import os
import json
import subprocess
import shlex
import shutil
import socket
import sys
import time
from pathlib import Path
from typing import Any

from putpocket_dataset_mining.constants import REPO_ROOT
from putpocket_dataset_mining.docker_workspace import run_verifier_container
from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.execution_config import DEFAULT_VERIFIER_TIMEOUT_SEC
from putpocket_dataset_mining.judge import CodexJudge

from . import PROTOCOL_VERSION
from .image import ensure_image
from .manifest import read_json, result_sha256, sha256_tree, write_json_atomic
from .paths import assert_no_symlink_escape, job_dir, safe_id


def protocol_version() -> dict[str, Any]:
    return {"schema_version": 1, "protocol_version": PROTOCOL_VERSION}


def preflight(image: str | None = None, dockerfile: str = "docker/classeval_python/Dockerfile") -> dict[str, Any]:
    dockerfile_path = REPO_ROOT / dockerfile
    status = None
    if image:
        try:
            status = ensure_image(image, dockerfile_path, build_if_missing=False)
        except Exception as exc:  # noqa: BLE001
            return protocol_version() | {
                "status": "infra_failed",
                "wrapper_ok": True,
                "docker_ok": shutil.which("docker") is not None,
                "rsync_ok": shutil.which("rsync") is not None,
                "image_ok": False,
                "error_class": "infra.image_missing",
                "error_message": str(exc),
            }
    return protocol_version() | {
        "status": "passed",
        "wrapper_ok": True,
        "docker_ok": shutil.which("docker") is not None,
        "rsync_ok": shutil.which("rsync") is not None,
        "image_ok": None if image is None else bool(status and status.image_id),
        "image_id": None if status is None else status.image_id,
        "verifier_host": socket.gethostname(),
    }


def promote_incoming(job_id: str) -> Path:
    job_id = safe_id(job_id, "job_id")
    incoming = job_dir("incoming", f"{job_id}.partial")
    ready = job_dir("ready", job_id)
    if not incoming.exists():
        raise ConfigError(f"partial job missing: {incoming}")
    if ready.exists():
        return ready
    incoming.replace(ready)
    return ready


def verify(job_id: str) -> dict[str, Any]:
    job_id = safe_id(job_id, "job_id")
    completed = job_dir("completed", job_id)
    result_path = completed / "result.json"
    if result_path.exists():
        return read_json(result_path)
    ready = job_dir("ready", job_id)
    if not ready.exists():
        partial = job_dir("incoming", f"{job_id}.partial")
        if partial.exists():
            return _infra(job_id, "partial_job_not_executable", "Partial jobs are never executed.")
        return _infra(job_id, "ready_job_missing", "Ready job directory is missing.")
    manifest = read_json(ready / "manifest.json")
    if manifest.get("protocol_version") != PROTOCOL_VERSION:
        return _infra(job_id, "REMOTE_PROTOCOL_MISMATCH", "Protocol version mismatch.")
    workspace = ready / "workspace"
    assert_no_symlink_escape(workspace)
    actual_sha = sha256_tree(workspace)
    if manifest.get("workspace_sha256") != actual_sha:
        return _infra(job_id, "REMOTE_RESULT_INTEGRITY_FAILED", "Workspace checksum mismatch.", workspace_sha256=actual_sha)
    image_status = ensure_image(str(manifest["docker_image"]), REPO_ROOT / str(manifest.get("dockerfile", "docker/classeval_python/Dockerfile")))
    running = job_dir("running", job_id)
    if running.exists():
        shutil.rmtree(running)
    ready.replace(running)
    total_start = time.monotonic()
    start = time.monotonic()
    result = run_verifier_container(
        running / "workspace",
        str(manifest["docker_image"]),
        _test_command(manifest.get("test_command", "pytest -q tests/test_solution.py")),
        cpus=manifest.get("cpus", 8),
        memory=str(manifest.get("memory", "8g")),
        timeout_sec=int(manifest.get("timeout_sec", DEFAULT_VERIFIER_TIMEOUT_SEC)),
    )
    pytest_wall = time.monotonic() - start
    policy = str(manifest.get("verification_policy") or ("history2_pytest_then_judge" if manifest.get("verifier_stage") == "history2" else "history1_pytest_only"))
    pytest_status = "passed" if result.returncode == 0 else "timeout" if result.timeout else "failed"
    status = pytest_status
    judge_payload: dict[str, Any] = {
        "executed": False,
        "backend": "codex_cli",
        "decision": None,
        "infrastructure_status": None,
        "reason": "not run",
        "wall_time_sec": None,
        "stdout_file": None,
        "stderr_file": None,
        "decision_file": None,
        "checksum": None,
    }
    judge_result = None
    if policy == "history2_pytest_then_judge" and pytest_status == "passed":
        judge_result = _run_remote_judge(running, result.returncode, manifest)
        judge_payload = judge_result["judge"]
        decision = judge_payload.get("decision")
        if judge_payload.get("infrastructure_status") == "infra_failed":
            status = "infra_failed"
        elif decision == "pass":
            status = "passed"
        elif decision == "fail":
            status = "failed"
        elif decision == "uncertain":
            status = "uncertain"
        else:
            status = "infra_failed"
    elif policy == "history2_pytest_then_judge":
        judge_payload["reason"] = "pytest did not pass"
    completed.parent.mkdir(parents=True, exist_ok=True)
    running.replace(completed)
    (completed / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (completed / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    if judge_result:
        judge_dir = completed / "judge"
        judge_dir.mkdir(parents=True, exist_ok=True)
        for name, content in judge_result["files"].items():
            (judge_dir / name).write_text(content, encoding="utf-8")
    pytest_payload = {
        "status": pytest_status,
        "process_exit_code": result.returncode,
        "timed_out": result.timeout,
        "stdout_file": "stdout.txt",
        "stderr_file": "stderr.txt",
        "wall_time_sec": pytest_wall,
    }
    data = protocol_version() | {
        "job_id": job_id,
        "verification_policy": policy,
        "sample_id": manifest.get("sample_id"),
        "stage": manifest.get("verifier_stage"),
        "status": status,
        "verifier_passed": status == "passed",
        "process_exit_code": result.returncode,
        "timed_out": result.timeout,
        "timeout_sec": int(manifest.get("timeout_sec", DEFAULT_VERIFIER_TIMEOUT_SEC)),
        "stdout_file": "stdout.txt",
        "stderr_file": "stderr.txt",
        "wall_time_sec": pytest_wall,
        "pytest": pytest_payload,
        "judge": judge_payload,
        "final": {
            "status": status,
            "failure_class": None if status == "passed" else "verifier.timeout" if result.timeout else "judge.infra_failed" if status == "infra_failed" else "judge.uncertain" if status == "uncertain" else "verifier.failed",
            "total_wall_time_sec": time.monotonic() - total_start,
        },
        "error_class": None if status == "passed" else "verifier.timeout" if result.timeout else "judge.infra_failed" if status == "infra_failed" else "judge.uncertain" if status == "uncertain" else "verifier.failed",
        "error_message": None,
        "verifier_host": socket.gethostname(),
        "verifier_revision": PROTOCOL_VERSION,
        "docker_image_tag": image_status.image,
        "docker_image_id": image_status.image_id,
        "dockerfile_sha256": image_status.dockerfile_sha256,
        "workspace_sha256": actual_sha,
        "completed_at_kst": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    data["result_sha256"] = result_sha256(data)
    write_json_atomic(completed / "result.json", data)
    write_json_atomic(completed / "combined_result.json", data)
    return data


def result_status(job_id: str) -> dict[str, Any]:
    job_id = safe_id(job_id, "job_id")
    result_path = job_dir("completed", job_id) / "result.json"
    if not result_path.exists():
        return protocol_version() | {"job_id": job_id, "status": "missing"}
    return read_json(result_path)


def job_status(job_id: str) -> dict[str, Any]:
    job_id = safe_id(job_id, "job_id")
    for state in ["completed", "running", "ready", "incoming"]:
        if state == "incoming":
            path = job_dir(state, f"{job_id}.partial")
        else:
            path = job_dir(state, job_id)
        if path.exists():
            payload = protocol_version() | {"job_id": job_id, "status": state, "path": str(path)}
            if state == "completed" and (path / "result.json").exists():
                payload["result"] = read_json(path / "result.json")
            return payload
    return protocol_version() | {"job_id": job_id, "status": "missing"}


def start_worker(job_id: str) -> dict[str, Any]:
    job_id = safe_id(job_id, "job_id")
    if result_status(job_id).get("status") != "missing":
        return protocol_version() | {"job_id": job_id, "status": "already_completed"}
    repo = str(REPO_ROOT)
    root = os.environ.get("SR_REMOTE_JOB_ROOT", "")
    cmd = [
        sys.executable,
        "-m",
        "putpocket_dataset_mining.remote_verifier.cli",
        "verify",
        "--job-id",
        job_id,
    ]
    env = dict(os.environ)
    env["SR_REMOTE_JOB_ROOT"] = root
    log_dir = job_dir("logs", job_id)
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "worker.stdout.txt").open("ab")
    stderr = (log_dir / "worker.stderr.txt").open("ab")
    proc = subprocess.Popen(cmd, cwd=repo, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
    write_json_atomic(log_dir / "worker.json", {"pid": proc.pid, "job_id": job_id, "started_at": time.time()})
    return protocol_version() | {"job_id": job_id, "status": "worker_started", "pid": proc.pid}


def worker_status(job_id: str) -> dict[str, Any]:
    job_id = safe_id(job_id, "job_id")
    path = job_dir("logs", job_id) / "worker.json"
    if not path.exists():
        return protocol_version() | {"job_id": job_id, "status": "missing"}
    data = read_json(path)
    pid = int(data.get("pid", 0))
    alive = pid > 0 and Path(f"/proc/{pid}").exists()
    return protocol_version() | {"job_id": job_id, "status": "running" if alive else "exited", **data}


def cleanup(job_ids: list[str], *, dry_run: bool = True) -> dict[str, Any]:
    selected = [safe_id(job_id, "job_id") for job_id in job_ids]
    return protocol_version() | {"status": "noop", "dry_run": dry_run, "job_ids": selected}


def _test_command(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return " ".join(shlex.quote(item) for item in value)
    raise ConfigError("test_command must be a string or list of strings")


def _run_remote_judge(running: Path, pytest_returncode: int, manifest: dict[str, Any]) -> dict[str, Any]:
    judge_dir = running / "judge_bundle"
    judge_input = read_json(judge_dir / "judge_input.json") if (judge_dir / "judge_input.json").exists() else {}
    attempt = running / "judge_attempt"
    attempt.mkdir(parents=True, exist_ok=True)
    (attempt / "judge").mkdir(parents=True, exist_ok=True)
    timeout_sec = int(manifest.get("judge_timeout_sec", 300))
    history2_summary = {
        "status": "passed" if pytest_returncode == 0 else "failed",
        "process_exit_code": pytest_returncode,
        "stage": "history2",
    }
    judge = CodexJudge(attempt, timeout_sec=timeout_sec, workdir=REPO_ROOT)
    started = time.monotonic()
    result = judge.run(
        cline_rules_v1=str(judge_input.get("cline_rules_v1", "")),
        files_after_history1=dict(judge_input.get("files_after_history1", {})),
        cline_rules_v2=str(judge_input.get("cline_rules_v2", "")),
        query2=str(judge_input.get("query2", "")),
        files_after_history2=dict(judge_input.get("files_after_history2", {})),
        history2_unit_test_summary=history2_summary,
    )
    elapsed = time.monotonic() - started
    decision_data = result.to_dict()
    checksum = result_sha256(decision_data)
    stdout = result.stdout
    stderr = result.stderr
    return {
        "judge": {
            "executed": True,
            "backend": result.backend,
            "model": "codex_cli",
            "decision": result.decision,
            "infrastructure_status": "infra_failed" if result.failure_class == "judge.cli_error" else "passed",
            "reason": result.reason,
            "wall_time_sec": elapsed,
            "stdout_file": "stdout.txt",
            "stderr_file": "stderr.txt",
            "decision_file": "judge_decision.json",
            "checksum": checksum,
        },
        "files": {
            "judge_input.json": json.dumps(judge_input, indent=2, sort_keys=True),
            "judge_prompt.txt": (attempt / "judge" / "judge_prompt.txt").read_text(encoding="utf-8") if (attempt / "judge" / "judge_prompt.txt").exists() else "",
            "stdout.txt": stdout,
            "stderr.txt": stderr,
            "judge_decision.json": json.dumps(decision_data, indent=2, sort_keys=True),
            "timing.json": json.dumps({"total_sec": elapsed}, indent=2, sort_keys=True),
        },
    }


def _infra(job_id: str, error_class: str, message: str, **extra: Any) -> dict[str, Any]:
    data = protocol_version() | {
        "job_id": job_id,
        "status": "infra_failed",
        "verifier_passed": None,
        "process_exit_code": None,
        "timed_out": False,
        "error_class": error_class,
        "error_message": message,
        **extra,
    }
    data["result_sha256"] = result_sha256(data)
    return data
