from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any


TASK_RE = re.compile(r"^T(?P<date>\d{8})-(?P<seq>\d{3})__(?P<topic>[a-z0-9][a-z0-9-]*)$")
DEFAULT_FORBIDDEN_PATHS = [
    "Putpocket_env/",
    "data/",
    "logs/",
    "models/",
    ".ssh/",
]
LOCK_RESOURCE_ORDER = {
    "git-metadata": 10,
    "task-start": 20,
    "integration": 30,
    "runtime-sync": 40,
    "canonical-runtime": 50,
    "build": 60,
}
LOCK_POLL_SECONDS = 1.0


@dataclasses.dataclass(frozen=True)
class AgentConfig:
    canonical_root: Path
    worktree_root: Path
    artifact_root: Path
    active_env: Path
    external_vllm_root: Path
    external_lmcache_root: Path

    @classmethod
    def load(cls, *, environ: dict[str, str] | None = None) -> "AgentConfig":
        env = environ or os.environ
        cfg = _read_local_config(Path(env.get("PUTPOCKET_AGENT_CONFIG", Path.home() / ".config" / "putpocket" / "agent.toml")))
        root_base = Path("/workspace") if env.get("PUTPOCKET_STORAGE_KIND") == "network-volume" and Path("/workspace").exists() else Path.home()
        canonical = Path(env.get("PUTPOCKET_CANONICAL_ROOT") or cfg.get("canonical_root") or root_base / "putpocket_dataset_mining")
        worktrees = Path(env.get("PUTPOCKET_WORKTREE_ROOT") or cfg.get("worktree_root") or root_base / "putpocket_dataset_mining_worktrees")
        artifacts = Path(env.get("PUTPOCKET_ARTIFACT_ROOT") or cfg.get("artifact_root") or canonical / "data" / "model_evaluation" / "runs")
        active_env = Path(env.get("PUTPOCKET_ENV_PATH") or cfg.get("active_env") or canonical / "Putpocket_env")
        return cls(
            canonical_root=canonical,
            worktree_root=worktrees,
            artifact_root=artifacts,
            active_env=active_env,
            external_vllm_root=Path(cfg.get("external_vllm_root") or canonical / "externals" / "vllm"),
            external_lmcache_root=Path(cfg.get("external_lmcache_root") or canonical / "externals" / "lmcache"),
        )


@dataclasses.dataclass(frozen=True)
class ContextInfo:
    checkout_root: Path
    canonical_root: Path
    worktree_root: Path
    execution_context: str
    branch: str
    head: str
    origin_master: str
    active_env: Path
    source_ownership: str
    production_allowed: bool

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self) | {
            "checkout_root": str(self.checkout_root),
            "canonical_root": str(self.canonical_root),
            "worktree_root": str(self.worktree_root),
            "active_env": str(self.active_env),
        }


def allocate_task_id(topic: str, *, root: Path, date: str | None = None) -> str:
    safe_topic = slugify(topic)
    day = date or time.strftime("%Y%m%d", time.localtime())
    root.mkdir(parents=True, exist_ok=True)
    for seq in range(1, 1000):
        task_id = f"T{day}-{seq:03d}__{safe_topic}"
        if not (root / task_id).exists() and not _git_ref_exists(f"refs/heads/agent/{task_id}"):
            return task_id
    raise RuntimeError(f"no available task id for {day}")


def slugify(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return text or "task"


def detect_context(path: Path | None = None, config: AgentConfig | None = None) -> ContextInfo:
    cfg = config or AgentConfig.load()
    checkout = git_top_level(path or Path.cwd())
    branch = git(["-C", str(checkout), "branch", "--show-current"], check=False).stdout.strip() or "detached"
    head = git(["-C", str(checkout), "rev-parse", "HEAD"], check=False).stdout.strip()
    origin_master = git(["-C", str(cfg.canonical_root), "rev-parse", "refs/remotes/origin/master"], check=False).stdout.strip()
    try:
        checkout_resolved = checkout.resolve()
        canonical_resolved = cfg.canonical_root.resolve()
        worktree_resolved = cfg.worktree_root.resolve()
    except FileNotFoundError:
        checkout_resolved = checkout
        canonical_resolved = cfg.canonical_root
        worktree_resolved = cfg.worktree_root
    if checkout_resolved == canonical_resolved:
        execution_context = "canonical-runtime"
    elif str(checkout_resolved).startswith(str(worktree_resolved) + os.sep):
        execution_context = "task-worktree"
    else:
        execution_context = "unknown"
    ownership = classify_source_ownership(checkout, cfg)
    return ContextInfo(
        checkout_root=checkout,
        canonical_root=cfg.canonical_root,
        worktree_root=cfg.worktree_root,
        execution_context=execution_context,
        branch=branch,
        head=head,
        origin_master=origin_master,
        active_env=cfg.active_env,
        source_ownership=ownership,
        production_allowed=execution_context == "canonical-runtime" and ownership == "CANONICAL_SOURCE_OWNERSHIP_OK",
    )


def classify_source_ownership(checkout: Path | None = None, config: AgentConfig | None = None) -> str:
    cfg = config or AgentConfig.load()
    checkout = checkout or git_top_level(Path.cwd())
    details = source_ownership_details(checkout, cfg)
    if details["editable_worktree_leakage"]:
        return "EDITABLE_WORKTREE_LEAKAGE"
    if details["task_overlay_active"]:
        return "TASK_OVERLAY_ACTIVE"
    if details["putpocket_origin_ok"]:
        return "CANONICAL_SOURCE_OWNERSHIP_OK"
    return "UNKNOWN_SOURCE_OWNERSHIP"


def source_ownership_details(checkout: Path | None = None, config: AgentConfig | None = None) -> dict[str, Any]:
    cfg = config or AgentConfig.load()
    checkout = checkout or git_top_level(Path.cwd())
    paths = module_origins(["putpocket_dataset_mining", "vllm", "lmcache"])
    direct_urls = editable_direct_urls()
    sys_paths = [p for p in sys.path if p]
    worktree_root = str(cfg.worktree_root)
    task_leaks = [p for p in [*paths.values(), *direct_urls.values(), *sys_paths] if p and str(p).startswith(worktree_root)]
    checkout_is_task = str(checkout.resolve()).startswith(str(cfg.worktree_root.resolve()) + os.sep) if cfg.worktree_root.exists() else False
    return {
        "module_origins": paths,
        "editable_direct_urls": direct_urls,
        "sys_path_task_entries": [p for p in sys_paths if p.startswith(worktree_root)],
        "editable_worktree_leakage": bool(task_leaks and not checkout_is_task),
        "task_overlay_active": checkout_is_task and any(str(p).startswith(str(checkout)) for p in sys_paths),
        "putpocket_origin_ok": paths.get("putpocket_dataset_mining", "").startswith(str(cfg.canonical_root / "src")),
        "task_leaks": task_leaks,
    }


def module_origins(names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in names:
        try:
            mod = importlib.import_module(name)
            out[name] = str(getattr(mod, "__file__", ""))
        except Exception as exc:  # noqa: BLE001
            out[name] = f"UNAVAILABLE:{type(exc).__name__}:{exc}"
    return out


def editable_direct_urls() -> dict[str, str]:
    out: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "")
        if name not in {"putpocket-dataset-mining", "vllm", "lmcache"}:
            continue
        try:
            text = dist.read_text("direct_url.json")
        except Exception:
            text = None
        if not text:
            continue
        try:
            payload = json.loads(text)
            out[name] = payload.get("url", "")
        except Exception:
            out[name] = "UNPARSEABLE"
    return out


def require_production_allowed(operation: str, *, config: AgentConfig | None = None) -> None:
    info = detect_context(config=config)
    if not info.production_allowed and os.environ.get("PUTPOCKET_ALLOW_TASK_PRODUCTION") != "1":
        raise SystemExit(f"production operation refused from {info.execution_context}: {operation}")


@dataclasses.dataclass(frozen=True)
class AgentLockHandle:
    resource: str
    path: Path
    token: str

    def release(self) -> None:
        try:
            payload = _read_json_file(self.path)
        except OSError:
            return
        if payload.get("token") == self.token:
            self.path.unlink(missing_ok=True)


def agent_lock_root(config: AgentConfig | None = None) -> Path:
    cfg = config or AgentConfig.load()
    override = os.environ.get("PUTPOCKET_AGENT_LOCK_ROOT")
    if override:
        return Path(override).expanduser()
    proc = git(["-C", str(cfg.canonical_root), "rev-parse", "--git-common-dir"], check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        git_dir = Path(proc.stdout.strip())
        if not git_dir.is_absolute():
            git_dir = cfg.canonical_root / git_dir
    else:
        git_dir = cfg.canonical_root / ".git"
    return git_dir / "putpocket-locks"


@contextmanager
def acquire_agent_locks(
    config: AgentConfig,
    resources: list[str] | tuple[str, ...],
    *,
    operation: str,
    wait_seconds: float | None = None,
):
    ordered = sorted(dict.fromkeys(resources), key=lambda item: (LOCK_RESOURCE_ORDER.get(item, 1000), item))
    acquired: list[AgentLockHandle] = []
    try:
        for resource in ordered:
            acquired.append(acquire_agent_lock(config, resource, operation=operation, wait_seconds=wait_seconds))
        yield acquired
    finally:
        for handle in reversed(acquired):
            handle.release()


def acquire_agent_lock(
    config: AgentConfig,
    resource: str,
    *,
    operation: str,
    wait_seconds: float | None = None,
) -> AgentLockHandle:
    wait = _lock_wait_seconds(wait_seconds)
    root = agent_lock_root(config)
    root.mkdir(parents=True, exist_ok=True)
    pending_root = root / "pending"
    pending_root.mkdir(parents=True, exist_ok=True)
    lock_path = root / f"{_safe_lock_name(resource)}.lock"
    token = hashlib.sha256(f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}:{resource}".encode("utf-8")).hexdigest()
    owner = _lock_owner_payload(config, resource, operation, token)
    deadline = time.monotonic() + wait

    while True:
        try:
            fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            current = _read_json_file(lock_path)
            if _lock_payload_stale(current):
                lock_path.unlink(missing_ok=True)
                continue
            if time.monotonic() >= deadline:
                pending = _write_pending_lock_request(pending_root, owner, current)
                raise SystemExit(_format_lock_conflict(resource, operation, current, pending))
            time.sleep(min(LOCK_POLL_SECONDS, max(0.05, deadline - time.monotonic())))
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(owner, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return AgentLockHandle(resource=resource, path=lock_path, token=token)


def active_agent_locks(config: AgentConfig | None = None) -> list[dict[str, Any]]:
    root = agent_lock_root(config)
    if not root.exists():
        return []
    locks = []
    for path in sorted(root.glob("*.lock")):
        payload = _read_json_file(path)
        payload["path"] = str(path)
        payload["stale"] = _lock_payload_stale(payload)
        locks.append(payload)
    return locks


def pending_agent_lock_requests(config: AgentConfig | None = None) -> list[dict[str, Any]]:
    root = agent_lock_root(config) / "pending"
    if not root.exists():
        return []
    requests = []
    for path in sorted(root.glob("*.json")):
        payload = _read_json_file(path)
        payload["path"] = str(path)
        requests.append(payload)
    return requests


def locks_status(args: argparse.Namespace) -> int:
    cfg = AgentConfig.load()
    payload = {
        "schema_version": 1,
        "lock_root": str(agent_lock_root(cfg)),
        "active_locks": active_agent_locks(cfg),
        "pending_requests": pending_agent_lock_requests(cfg),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def task_start(args: argparse.Namespace) -> int:
    cfg = AgentConfig.load()
    with acquire_agent_locks(cfg, ["git-metadata", "task-start"], operation="task start", wait_seconds=args.wait_lock):
        task_id = args.task_id or allocate_task_id(args.topic, root=cfg.worktree_root)
        if not TASK_RE.match(task_id):
            raise SystemExit(f"invalid task id: {task_id}")
        branch = f"agent/{task_id}"
        worktree = cfg.worktree_root / task_id
        cfg.worktree_root.mkdir(parents=True, exist_ok=True)
        git(["fetch", "origin", "--prune"], cwd=cfg.canonical_root)
        base = git(["rev-parse", "refs/remotes/origin/master"], cwd=cfg.canonical_root).stdout.strip()
        git(["worktree", "add", "-b", branch, str(worktree), base], cwd=cfg.canonical_root)
        task_path = worktree / "agent" / "tasks" / task_id / "TASK.md"
        task_path.parent.mkdir(parents=True, exist_ok=True)
        task_path.write_text(render_task(task_id, args.topic, branch, worktree, base, args.runtime_mode), encoding="utf-8")
        print(json.dumps({"task_id": task_id, "branch": branch, "worktree": str(worktree), "task": str(task_path)}, indent=2))
    return 0


def task_status(args: argparse.Namespace) -> int:
    cfg = AgentConfig.load()
    info = detect_context(config=cfg)
    print(json.dumps(info.as_dict() | {"source_details": source_ownership_details(config=cfg)}, indent=2, sort_keys=True))
    return 0


def task_close(args: argparse.Namespace) -> int:
    checkout = git_top_level(Path.cwd())
    task = find_task_file(checkout)
    if not task:
        raise SystemExit("TASK.md not found under agent/tasks")
    handoffs = list(task.parent.glob("handoffs/TO_GPT_*.md"))
    if not handoffs:
        raise SystemExit("task close requires agent/tasks/<TASK>/handoffs/TO_GPT_*.md")
    status = git(["-C", str(checkout), "status", "--porcelain"], check=False).stdout.strip()
    if status:
        raise SystemExit("task close requires clean tracked/untracked state after committing handoff link")
    print(f"Task closeable. Integrate with: putpocket-agent task integrate --branch {git(['-C', str(checkout), 'branch', '--show-current']).stdout.strip()}")
    return 0


def task_integrate(args: argparse.Namespace) -> int:
    cfg = AgentConfig.load()
    branch = args.branch
    if not branch:
        raise SystemExit("--branch is required")
    resources = ["git-metadata", "integration", "runtime-sync", "canonical-runtime"]
    with acquire_agent_locks(cfg, resources, operation=f"task integrate {branch}", wait_seconds=args.wait_lock):
        git(["fetch", "origin", "--prune"], cwd=cfg.canonical_root)
        target = git(["rev-parse", f"refs/heads/{branch}"], cwd=cfg.canonical_root).stdout.strip()
        origin_master = git(["rev-parse", "refs/remotes/origin/master"], cwd=cfg.canonical_root).stdout.strip()
        if git(["merge-base", "--is-ancestor", origin_master, target], cwd=cfg.canonical_root, check=False).returncode != 0:
            raise SystemExit("origin/master is not an ancestor of the task branch; refusing non-ff integration")
        git(["push", "origin", f"{target}:refs/heads/master"], cwd=cfg.canonical_root)
        _runtime_sync_locked(argparse.Namespace(skip_bootstrap=args.skip_bootstrap), cfg)
        print(json.dumps({"integrated": True, "master": target}, indent=2))
    return 0


def runtime_sync(args: argparse.Namespace) -> int:
    cfg = AgentConfig.load()
    resources = ["git-metadata", "runtime-sync", "canonical-runtime"]
    with acquire_agent_locks(cfg, resources, operation="runtime sync", wait_seconds=args.wait_lock):
        return _runtime_sync_locked(args, cfg)


def _runtime_sync_locked(args: argparse.Namespace, cfg: AgentConfig) -> int:
    if git_top_level(cfg.canonical_root).resolve() != cfg.canonical_root.resolve():
        raise SystemExit("canonical root is not a git checkout")
    branch = git(["branch", "--show-current"], cwd=cfg.canonical_root).stdout.strip()
    if branch != "master":
        raise SystemExit(f"canonical root must be on master, found {branch}")
    if git(["status", "--porcelain"], cwd=cfg.canonical_root, check=False).stdout.strip():
        raise SystemExit("canonical root has dirty tracked/untracked state; refusing runtime sync")
    git(["fetch", "origin", "--prune"], cwd=cfg.canonical_root)
    git(["pull", "--ff-only", "origin", "master"], cwd=cfg.canonical_root)
    if not args.skip_bootstrap:
        subprocess.run(["./scripts/env/bootstrap_sr.sh", "--preset", "server2", "--doctor-only"], cwd=cfg.canonical_root, text=True, check=False)
    receipt = cfg.canonical_root / "logs" / "agent" / "runtime_sync" / f"receipt_{time.strftime('%Y%m%d_%H%M%S')}.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(detect_context(cfg.canonical_root, cfg).as_dict(), indent=2), encoding="utf-8")
    print(receipt)
    return 0


def doctor(args: argparse.Namespace) -> int:
    cfg = AgentConfig.load()
    info = detect_context(config=cfg)
    payload = info.as_dict() | {
        "python_executable": sys.executable,
        "sys_prefix": sys.prefix,
        "module_origins": module_origins(["putpocket_dataset_mining", "vllm", "lmcache"]),
        "editable_direct_urls": editable_direct_urls(),
        "path_leakage": _leakage(os.environ.get("PATH", ""), cfg),
        "pythonpath_leakage": _leakage(os.environ.get("PYTHONPATH", ""), cfg),
        "external_roots": {
            "vllm": {"path": str(cfg.external_vllm_root), "head": _head(cfg.external_vllm_root)},
            "lmcache": {"path": str(cfg.external_lmcache_root), "head": _head(cfg.external_lmcache_root)},
        },
        "production_command_permission": info.production_allowed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def worktrees_audit(args: argparse.Namespace) -> int:
    cfg = AgentConfig.load()
    rows = classify_worktrees(cfg)
    if args.markdown:
        print(render_worktree_inventory(rows))
    else:
        print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def classify_worktrees(cfg: AgentConfig) -> list[dict[str, Any]]:
    proc = git(["worktree", "list", "--porcelain"], cwd=cfg.canonical_root)
    rows = []
    item: dict[str, str] = {}
    for line in proc.stdout.splitlines() + [""]:
        if not line:
            if item:
                rows.append(_classify_worktree_item(item, cfg))
                item = {}
            continue
        key, _, value = line.partition(" ")
        item[key] = value
    return rows


def render_worktree_inventory(rows: list[dict[str, Any]]) -> str:
    lines = ["# Legacy Worktrees Inventory", "", "| Path | Branch | HEAD | Dirty | Integrated | Disposition |", "|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(f"| `{row['path']}` | `{row.get('branch','')}` | `{row.get('head','')[:12]}` | {row['dirty']} | {row['integrated_into_master']} | {row['disposition']} |")
    return "\n".join(lines) + "\n"


def render_task(task_id: str, topic: str, branch: str, worktree: Path, base: str, runtime_mode: str) -> str:
    return f"""# {task_id}

task identity: {task_id}
objective: {topic}
status: in_progress
base tip: {base}
branch: {branch}
worktree: {worktree}
runtime mode: {runtime_mode}
write scope:
  - source/docs/tests required for this task
forbidden paths:
{chr(10).join(f'  - {p}' for p in DEFAULT_FORBIDDEN_PATHS)}
fixed decisions:
  - canonical runtime checkout is /home/${{USER}}/putpocket_dataset_mining or /workspace/putpocket_dataset_mining
  - task worktrees live under /home/${{USER}}/putpocket_dataset_mining_worktrees or /workspace/putpocket_dataset_mining_worktrees
plan:
  - implement
  - validate
  - handoff
completion criteria:
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - pending
artifacts:
  - agent/tasks/{task_id}/
commits:
  - pending
final handoff link: pending
"""


def find_task_file(root: Path) -> Path | None:
    tasks = sorted((root / "agent" / "tasks").glob("*/TASK.md"))
    return tasks[-1] if tasks else None


def _classify_worktree_item(item: dict[str, str], cfg: AgentConfig) -> dict[str, Any]:
    path = Path(item.get("worktree", ""))
    branch = item.get("branch", "").replace("refs/heads/", "")
    head = item.get("HEAD", "")
    dirty = bool(git(["status", "--porcelain"], cwd=path, check=False).stdout.strip()) if path.exists() else True
    integrated = git(["merge-base", "--is-ancestor", head, "refs/remotes/origin/master"], cwd=cfg.canonical_root, check=False).returncode == 0 if head else False
    if path.resolve() == cfg.canonical_root.resolve():
        disposition = "ACTIVE_CURRENT_TASK" if branch.startswith("agent/") else "INTEGRATED_KEEP_TEMPORARILY"
    elif str(path).startswith(str(cfg.worktree_root)):
        disposition = "ACTIVE_CURRENT_TASK" if branch.startswith("agent/") else "UNKNOWN"
    elif not integrated:
        disposition = "UNINTEGRATED_REQUIRES_REVIEW"
    elif dirty:
        disposition = "INTEGRATED_KEEP_TEMPORARILY"
    else:
        disposition = "SAFE_TO_REMOVE_AFTER_BACKUP"
    return {"path": str(path), "branch": branch, "head": head, "dirty": dirty, "integrated_into_master": integrated, "unique_commits": [] if integrated else _unique_commits(head, cfg), "disposition": disposition}


def _unique_commits(head: str, cfg: AgentConfig) -> list[str]:
    if not head:
        return []
    out = git(["log", "--oneline", f"refs/remotes/origin/master..{head}"], cwd=cfg.canonical_root, check=False).stdout
    return out.splitlines()[:20]


def _lock_wait_seconds(value: float | None) -> float:
    if value is not None:
        return max(0.0, float(value))
    raw = os.environ.get("PUTPOCKET_LOCK_WAIT_SECONDS", "0")
    try:
        return max(0.0, float(raw))
    except ValueError as exc:
        raise SystemExit(f"PUTPOCKET_LOCK_WAIT_SECONDS must be numeric, got {raw!r}") from exc


def _safe_lock_name(resource: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", resource).strip("._") or "lock"


def _lock_owner_payload(config: AgentConfig, resource: str, operation: str, token: str) -> dict[str, Any]:
    checkout = git(["rev-parse", "--show-toplevel"], check=False).stdout.strip()
    branch = git(["branch", "--show-current"], check=False).stdout.strip() or "detached"
    head = git(["rev-parse", "HEAD"], check=False).stdout.strip()
    return {
        "schema_version": 1,
        "resource": resource,
        "operation": operation,
        "token": token,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown",
        "cwd": str(Path.cwd()),
        "checkout_root": checkout,
        "canonical_root": str(config.canonical_root),
        "worktree_root": str(config.worktree_root),
        "branch": branch,
        "head": head,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "started_at_epoch": time.time(),
    }


def _write_pending_lock_request(pending_root: Path, requested: dict[str, Any], blocking: dict[str, Any]) -> Path:
    payload = {
        "schema_version": 1,
        "status": "pending",
        "requested": requested,
        "blocking": blocking,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "created_at_epoch": time.time(),
    }
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    name = f"{stamp}-{os.getpid()}-{_safe_lock_name(str(requested.get('resource', 'lock')))}.json"
    path = pending_root / name
    for suffix in range(100):
        candidate = path if suffix == 0 else pending_root / f"{path.stem}-{suffix:02d}.json"
        try:
            fd = os.open(str(candidate), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return candidate
    raise SystemExit(f"unable to write pending lock request under {pending_root}")


def _format_lock_conflict(resource: str, operation: str, blocking: dict[str, Any], pending_path: Path) -> str:
    owner = {
        "resource": blocking.get("resource"),
        "operation": blocking.get("operation"),
        "pid": blocking.get("pid"),
        "hostname": blocking.get("hostname"),
        "user": blocking.get("user"),
        "branch": blocking.get("branch"),
        "checkout_root": blocking.get("checkout_root"),
        "started_at": blocking.get("started_at"),
    }
    return (
        "LOCK_HELD_PENDING_RECORDED\n"
        f"requested_resource: {resource}\n"
        f"requested_operation: {operation}\n"
        f"pending_request: {pending_path}\n"
        f"blocking_owner: {json.dumps(owner, sort_keys=True)}"
    )


def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        return {"schema_version": 1, "path": str(path), "unparseable": True, "error": str(exc)}


def _lock_payload_stale(payload: dict[str, Any]) -> bool:
    if payload.get("hostname") != socket.gethostname():
        return False
    try:
        pid = int(payload.get("pid", 0))
    except (TypeError, ValueError):
        return False
    return pid > 0 and not _pid_alive(pid)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def git_top_level(path: Path) -> Path:
    return Path(git(["-C", str(path), "rev-parse", "--show-toplevel"]).stdout.strip())


def git(args: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(args if args and args[0] == "git" else ["git", *args], cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(args)}\n{proc.stderr}")
    return proc


def _git_ref_exists(ref: str) -> bool:
    return git(["show-ref", "--verify", "--quiet", ref], check=False).returncode == 0


def _read_local_config(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    if sys.version_info >= (3, 11):
        import tomllib

        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        return {k: str(v) for k, v in payload.get("paths", payload).items()}
    return {}


def _leakage(value: str, cfg: AgentConfig) -> list[str]:
    return [p for p in value.split(os.pathsep) if p.startswith(str(cfg.worktree_root))]


def _head(path: Path) -> str:
    return git(["-C", str(path), "rev-parse", "HEAD"], check=False).stdout.strip() if path.exists() else "missing"
