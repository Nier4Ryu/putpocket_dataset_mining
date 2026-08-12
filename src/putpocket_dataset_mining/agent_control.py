from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any


TASK_RE = re.compile(r"^T(?P<date>\d{8})-(?P<seq>\d{3})__(?P<topic>[a-z0-9][a-z0-9-]*)$")
DEFAULT_FORBIDDEN_PATHS = [
    "Putpocket_env/",
    "Putpocket_env_glm52/",
    "Putpocket_env_glm52_v025/",
    "data/",
    "logs/",
    "models/",
    ".ssh/",
]


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


def task_start(args: argparse.Namespace) -> int:
    cfg = AgentConfig.load()
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
    git(["fetch", "origin", "--prune"], cwd=cfg.canonical_root)
    target = git(["rev-parse", f"refs/heads/{branch}"], cwd=cfg.canonical_root).stdout.strip()
    origin_master = git(["rev-parse", "refs/remotes/origin/master"], cwd=cfg.canonical_root).stdout.strip()
    if git(["merge-base", "--is-ancestor", origin_master, target], cwd=cfg.canonical_root, check=False).returncode != 0:
        raise SystemExit("origin/master is not an ancestor of the task branch; refusing non-ff integration")
    git(["push", "origin", f"{target}:refs/heads/master"], cwd=cfg.canonical_root)
    runtime_sync(argparse.Namespace(skip_bootstrap=args.skip_bootstrap))
    print(json.dumps({"integrated": True, "master": target}, indent=2))
    return 0


def runtime_sync(args: argparse.Namespace) -> int:
    cfg = AgentConfig.load()
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
    return [p for p in value.split(os.pathsep) if p.startswith(str(cfg.worktree_root)) or "Putpocket_env_glm52" in p]


def _head(path: Path) -> str:
    return git(["-C", str(path), "rev-parse", "HEAD"], check=False).stdout.strip() if path.exists() else "missing"
