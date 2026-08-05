from __future__ import annotations

import os
import shlex
import shutil
import socket
import time
from pathlib import Path
from typing import Any

from putpocket_dataset_mining.constants import REPO_ROOT
from putpocket_dataset_mining.docker_workspace import run_verifier_container
from putpocket_dataset_mining.errors import ConfigError

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
    start = time.monotonic()
    result = run_verifier_container(
        running / "workspace",
        str(manifest["docker_image"]),
        _test_command(manifest.get("test_command", "pytest -q tests/test_solution.py")),
        cpus=manifest.get("cpus", 8),
        memory=str(manifest.get("memory", "8g")),
        timeout_sec=int(manifest.get("timeout_sec", 120)),
    )
    status = "passed" if result.returncode == 0 else "timeout" if result.timeout else "failed"
    completed.parent.mkdir(parents=True, exist_ok=True)
    running.replace(completed)
    (completed / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (completed / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    data = protocol_version() | {
        "job_id": job_id,
        "status": status,
        "verifier_passed": result.returncode == 0,
        "process_exit_code": result.returncode,
        "timed_out": result.timeout,
        "stdout_file": "stdout.txt",
        "stderr_file": "stderr.txt",
        "wall_time_sec": time.monotonic() - start,
        "error_class": None if result.returncode == 0 else "verifier.timeout" if result.timeout else "verifier.failed",
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
    return data


def result_status(job_id: str) -> dict[str, Any]:
    job_id = safe_id(job_id, "job_id")
    result_path = job_dir("completed", job_id) / "result.json"
    if not result_path.exists():
        return protocol_version() | {"job_id": job_id, "status": "missing"}
    return read_json(result_path)


def cleanup(job_ids: list[str], *, dry_run: bool = True) -> dict[str, Any]:
    selected = [safe_id(job_id, "job_id") for job_id in job_ids]
    return protocol_version() | {"status": "noop", "dry_run": dry_run, "job_ids": selected}


def _test_command(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return " ".join(shlex.quote(item) for item in value)
    raise ConfigError("test_command must be a string or list of strings")


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
