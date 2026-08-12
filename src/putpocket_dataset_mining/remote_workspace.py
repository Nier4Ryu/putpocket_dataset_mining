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

from .errors import ConfigError
from .ssh_transport import validate_safe_id

PROTOCOL_VERSION = "sr-remote-workspace-v1"
CONTAINER_WORKSPACE = "/workspace"


def _root() -> Path:
    value = os.environ.get("SR_REMOTE_WORKSPACE_ROOT") or os.environ.get("SR_REMOTE_JOB_ROOT")
    if not value:
        raise ConfigError("SR_REMOTE_WORKSPACE_ROOT is required")
    root = Path(value).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _session(session_id: str) -> Path:
    root = _root()
    path = (root / validate_safe_id(session_id, "session_id")).resolve()
    if root not in path.parents:
        raise ConfigError("workspace session escapes configured root")
    return path


def _container_name(session_id: str) -> str:
    return f"putpocket-workspace-{validate_safe_id(session_id, 'session_id')}"


def _payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    return json.loads(raw) if raw.strip() else {}


def _emit(payload: dict[str, Any], rc: int = 0) -> int:
    print(json.dumps({"schema_version": 1, "protocol_version": PROTOCOL_VERSION, **payload}, sort_keys=True))
    return rc


def _docker(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], text=True, capture_output=True, timeout=timeout)


def protocol_version(_: dict[str, Any]) -> int:
    return _emit({"status": "ready"})


def preflight(payload: dict[str, Any]) -> int:
    docker = shutil.which("docker")
    image = str(payload.get("docker_image") or "")
    daemon_ok = False
    image_ok: bool | None = None
    if docker:
        daemon_ok = _docker("info", "--format", "{{.ServerVersion}}", timeout=15).returncode == 0
        if daemon_ok and image:
            image_ok = _docker("image", "inspect", image, timeout=30).returncode == 0
    root = _root()
    ok = bool(docker and daemon_ok and os.access(root, os.W_OK) and image_ok is not False)
    return _emit({
        "status": "passed" if ok else "infra_failed",
        "wrapper_ok": True,
        "docker_cli_ok": bool(docker),
        "docker_ok": daemon_ok,
        "image_ok": image_ok,
        "workspace_root_ok": os.access(root, os.W_OK),
        "remote_workspace_root": str(root),
    }, 0 if ok else 2)


def create(payload: dict[str, Any]) -> int:
    session_id = validate_safe_id(str(payload["session_id"]), "session_id")
    session = _session(session_id)
    workspace = session / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    name = _container_name(session_id)
    image = str(payload["docker_image"])
    existing = _docker("inspect", name, timeout=30)
    if existing.returncode != 0:
        result = _docker(
            "run", "-d", "--name", name, "--network", "none",
            "--cpus", str(payload.get("cpus", 8)), "--memory", str(payload.get("memory", "8g")),
            "-v", f"{workspace}:{CONTAINER_WORKSPACE}:rw", "-w", CONTAINER_WORKSPACE,
            image, "sleep", "infinity", timeout=int(payload.get("startup_timeout_sec", 120)),
        )
        if result.returncode:
            return _emit({"status": "infra_failed", "error_class": "workspace.container_create_failed", "stderr": result.stderr}, 2)
    (session / "metadata.json").write_text(json.dumps({"session_id": session_id, "container": name, "image": image}, indent=2), encoding="utf-8")
    return _emit({"status": "created", "session_id": session_id, "container": name})


def execute(payload: dict[str, Any]) -> int:
    session_id = validate_safe_id(str(payload["session_id"]), "session_id")
    name = _container_name(session_id)
    command = str(payload["command"])
    timeout = int(payload.get("timeout_sec", 120))
    started = time.perf_counter()
    try:
        result = _docker("exec", "-w", CONTAINER_WORKSPACE, name, "bash", "-lc", command, timeout=timeout)
        return _emit({"status": "completed", "session_id": session_id, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr, "timeout": False, "remote_docker_exec_sec": time.perf_counter() - started})
    except subprocess.TimeoutExpired as exc:
        return _emit({"status": "completed", "session_id": session_id, "returncode": 124, "stdout": exc.stdout or "", "stderr": exc.stderr or "", "timeout": True, "remote_docker_exec_sec": time.perf_counter() - started})


def _tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def snapshot(payload: dict[str, Any]) -> int:
    session_id = validate_safe_id(str(payload["session_id"]), "session_id")
    snapshot_id = validate_safe_id(str(payload["snapshot_id"]), "snapshot_id")
    source = _session(session_id) / "workspace"
    target = _session(session_id) / "snapshots" / snapshot_id
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    return _emit({"status": "completed", "session_id": session_id, "snapshot_id": snapshot_id, "snapshot_path": str(target), "workspace_sha256": _tree_sha(target)})


def status(payload: dict[str, Any]) -> int:
    session_id = validate_safe_id(str(payload["session_id"]), "session_id")
    result = _docker("inspect", "-f", "{{.State.Running}}", _container_name(session_id), timeout=30)
    return _emit({"status": "active" if result.returncode == 0 and result.stdout.strip() == "true" else "inactive", "session_id": session_id})


def destroy(payload: dict[str, Any]) -> int:
    session_id = validate_safe_id(str(payload["session_id"]), "session_id")
    _docker("rm", "-f", _container_name(session_id), timeout=30)
    if bool(payload.get("remove_state", True)):
        shutil.rmtree(_session(session_id), ignore_errors=True)
    return _emit({"status": "destroyed", "session_id": session_id})


COMMANDS = {
    "protocol-version": protocol_version,
    "preflight": preflight,
    "create": create, "workspace-create": create,
    "exec": execute, "workspace-exec": execute,
    "snapshot": snapshot, "workspace-snapshot": snapshot,
    "status": status,
    "destroy": destroy, "workspace-destroy": destroy,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="putpocket-remote-workspace")
    parser.add_argument("command", choices=sorted(COMMANDS))
    args = parser.parse_args(argv)
    try:
        return COMMANDS[args.command](_payload())
    except Exception as exc:  # noqa: BLE001
        return _emit({"status": "infra_failed", "error_class": exc.__class__.__name__, "error": str(exc)}, 2)


if __name__ == "__main__":
    raise SystemExit(main())
