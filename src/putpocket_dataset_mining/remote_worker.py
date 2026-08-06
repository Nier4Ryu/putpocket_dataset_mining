from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .docker_workspace import run_verifier_container
from .errors import ConfigError
from .execution_config import DEFAULT_VERIFIER_TIMEOUT_SEC
from .ssh_transport import validate_safe_id

PROTOCOL_VERSION = "sr-remote-docker-v1"


def _remote_root() -> Path:
    root = Path(os.environ.get("SR_REMOTE_ROOT", ".")).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_child(root: Path, *parts: str) -> Path:
    current = root
    for part in parts:
        validate_safe_id(part)
        current = current / part
    resolved = current.resolve()
    if root not in resolved.parents and resolved != root:
        raise ConfigError(f"Path escapes remote root: {resolved}")
    return resolved


def _read_payload() -> dict[str, Any]:
    text = sys.stdin.read()
    return json.loads(text) if text.strip() else {}


def _write(data: dict[str, Any]) -> int:
    print(json.dumps(data, indent=2, sort_keys=True))
    return 0


def _sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def cmd_preflight(payload: dict[str, Any]) -> int:
    root = _remote_root()
    docker = shutil.which("docker")
    rsync = shutil.which("rsync")
    image = payload.get("docker_image") or os.environ.get("SR_REMOTE_DOCKER_IMAGE")
    image_ok: bool | None = None
    digest: str | None = None
    if docker and image:
        inspect = subprocess.run([docker, "image", "inspect", str(image), "--format", "{{index .RepoDigests 0}}"], text=True, capture_output=True)
        image_ok = inspect.returncode == 0
        digest = inspect.stdout.strip() or None
    return _write(
        {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "wrapper_ok": True,
            "rsync_ok": bool(rsync),
            "docker_ok": bool(docker),
            "staging_root_ok": os.access(root, os.W_OK),
            "image_ok": image_ok,
            "docker_image_digest": digest,
            "remote_root": str(root),
        }
    )


def cmd_fixture_pass(payload: dict[str, Any]) -> int:
    return _fixture(payload, expect_pass=True)


def cmd_fixture_fail(payload: dict[str, Any]) -> int:
    return _fixture(payload, expect_pass=False)


def _fixture(payload: dict[str, Any], *, expect_pass: bool) -> int:
    root = _remote_root()
    job_id = validate_safe_id(str(payload.get("job_id", f"fixture-{int(time.time())}")), "job_id")
    image = str(payload.get("docker_image") or os.environ.get("SR_REMOTE_DOCKER_IMAGE") or "putpocket-default-python:ubuntu22.04-py313-v1")
    job_dir = _safe_child(root, "jobs", job_id)
    ws = job_dir / "workspace"
    tests = ws / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (ws / "solution.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    expected = "1" if expect_pass else "2"
    (tests / "test_solution.py").write_text(f"from solution import f\n\ndef test_f():\n    assert f() == {expected}\n", encoding="utf-8")
    result = run_verifier_container(ws, image, "pytest -q tests/test_solution.py", cpus=1, memory="1g", timeout_sec=60)
    out = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "job_id": job_id,
        "status": "completed",
        "verifier_passed": result.returncode == 0,
        "process_exit_code": result.returncode,
        "timeout": result.timeout,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "workspace_sha256": _sha256_tree(ws),
    }
    (job_dir / "result.json").write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    return _write(out)


def cmd_verify(payload: dict[str, Any]) -> int:
    root = _remote_root()
    job_id = validate_safe_id(str(payload["job_id"]), "job_id")
    image = str(payload["docker_image"])
    command = str(payload.get("test_command", "pytest -q tests/test_solution.py"))
    timeout = int(payload.get("timeout_sec", DEFAULT_VERIFIER_TIMEOUT_SEC))
    job_dir = _safe_child(root, "jobs", job_id)
    workspace = job_dir / "workspace"
    manifest = job_dir / "manifest.json"
    result_path = job_dir / "result.json"
    if result_path.exists():
        return _write(json.loads(result_path.read_text(encoding="utf-8")))
    if not workspace.exists() or not manifest.exists():
        return _write({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "job_id": job_id, "status": "infra_failed", "error_class": "remote.missing_ready_job"})
    actual_sha = _sha256_tree(workspace)
    expected_sha = str(payload.get("workspace_sha256") or json.loads(manifest.read_text(encoding="utf-8")).get("workspace_sha256") or "")
    if expected_sha and expected_sha != actual_sha:
        return _write({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "job_id": job_id, "status": "infra_failed", "error_class": "remote.workspace_checksum_mismatch"})
    start = time.monotonic()
    result = run_verifier_container(workspace, image, command, cpus=payload.get("cpus", 8), memory=str(payload.get("memory", "8g")), timeout_sec=timeout)
    data = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "job_id": job_id,
        "status": "completed",
        "verifier_passed": result.returncode == 0,
        "infrastructure_status": "ok",
        "process_exit_code": result.returncode,
        "timeout": result.timeout,
        "stdout_path": "stdout.txt",
        "stderr_path": "stderr.txt",
        "wall_time_sec": time.monotonic() - start,
        "docker_image_digest": None,
        "verifier_host": os.uname().nodename,
        "verifier_revision": PROTOCOL_VERSION,
        "workspace_sha256": actual_sha,
        "error_class": None if result.returncode == 0 else "verifier.failed",
    }
    (job_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (job_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    data["result_sha256"] = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
    result_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return _write(data)


def cmd_workspace_create(payload: dict[str, Any]) -> int:
    root = _remote_root()
    session_id = validate_safe_id(str(payload["session_id"]), "session_id")
    session = _safe_child(root, "sessions", session_id)
    (session / "workspace").mkdir(parents=True, exist_ok=True)
    (session / "manifest.json").write_text(json.dumps({"protocol_version": PROTOCOL_VERSION, **payload}, indent=2, sort_keys=True), encoding="utf-8")
    return _write({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "session_id": session_id, "status": "created"})


def cmd_workspace_exec(payload: dict[str, Any]) -> int:
    root = _remote_root()
    session_id = validate_safe_id(str(payload["session_id"]), "session_id")
    command = str(payload["command"])
    image = str(payload["docker_image"])
    session = _safe_child(root, "sessions", session_id)
    ws = session / "workspace"
    result = run_verifier_container(ws, image, command, cpus=payload.get("cpus", 8), memory=str(payload.get("memory", "8g")), timeout_sec=int(payload.get("timeout_sec", 120)))
    return _write({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "session_id": session_id, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "timeout": result.timeout})


def cmd_workspace_snapshot(payload: dict[str, Any]) -> int:
    root = _remote_root()
    session_id = validate_safe_id(str(payload["session_id"]), "session_id")
    snapshot_id = validate_safe_id(str(payload["snapshot_id"]), "snapshot_id")
    session = _safe_child(root, "sessions", session_id)
    snapshot = _safe_child(root, "snapshots", snapshot_id)
    if snapshot.exists():
        shutil.rmtree(snapshot)
    shutil.copytree(session / "workspace", snapshot)
    return _write({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "session_id": session_id, "snapshot_id": snapshot_id, "workspace_sha256": _sha256_tree(snapshot)})


def cmd_workspace_destroy(payload: dict[str, Any]) -> int:
    root = _remote_root()
    session_id = validate_safe_id(str(payload["session_id"]), "session_id")
    session = _safe_child(root, "sessions", session_id)
    if session.exists():
        shutil.rmtree(session)
    return _write({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "session_id": session_id, "status": "destroyed"})


def cmd_result_status(payload: dict[str, Any]) -> int:
    root = _remote_root()
    job_id = validate_safe_id(str(payload["job_id"]), "job_id")
    result = _safe_child(root, "jobs", job_id) / "result.json"
    if not result.exists():
        return _write({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "job_id": job_id, "status": "missing"})
    return _write(json.loads(result.read_text(encoding="utf-8")))


def cmd_cleanup_stale(payload: dict[str, Any]) -> int:
    return _write({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "status": "noop", "retention_sec": payload.get("retention_sec")})


COMMANDS = {
    "preflight": cmd_preflight,
    "fixture-pass": cmd_fixture_pass,
    "fixture-fail": cmd_fixture_fail,
    "verify": cmd_verify,
    "workspace-create": cmd_workspace_create,
    "workspace-exec": cmd_workspace_exec,
    "workspace-snapshot": cmd_workspace_snapshot,
    "workspace-destroy": cmd_workspace_destroy,
    "result-status": cmd_result_status,
    "cleanup-stale": cmd_cleanup_stale,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m putpocket_dataset_mining.remote_worker")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](_read_payload())
    except Exception as exc:  # noqa: BLE001 - remote wrapper must return structured failure.
        print(json.dumps({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, "status": "infra_failed", "error_class": exc.__class__.__name__, "error": str(exc)}), file=sys.stdout)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
