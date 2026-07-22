from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .constants import BUILD_ENV_OVERRIDES, REPO_ROOT
from .errors import InfraError


@dataclass(frozen=True)
class ExternalRepo:
    name: str
    path: Path
    url: str
    branch: str | None
    role: str


EXTERNALS = {
    "vllm": ExternalRepo(
        name="vllm",
        path=REPO_ROOT / "externals" / "vllm",
        url="https://github.com/Nier4Ryu/vllm_mod.git",
        branch="Putpocket-v0.19.1",
        role="editable_build_dependency",
    ),
    "lmcache": ExternalRepo(
        name="lmcache",
        path=REPO_ROOT / "externals" / "lmcache",
        url="https://github.com/Nier4Ryu/LMCache_mod.git",
        branch="Putpocket-v0.4.4",
        role="editable_build_dependency",
    ),
    "cline": ExternalRepo(
        name="cline",
        path=REPO_ROOT / "externals" / "cline",
        url="https://github.com/Nier4Ryu/cline_mod.git",
        branch=None,
        role="read_only_reference_for_prompt_and_tool_format",
    ),
}


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    for name, value in BUILD_ENV_OVERRIDES.items():
        env.setdefault(name, value)
    return env


def _run_git(path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        command = " ".join(["git", "-C", str(path), *args])
        detail = result.stderr.strip() or result.stdout.strip()
        raise InfraError(f"Git command failed for {path}: {command}\n{detail}")
    return result


def _has_tracked_changes(path: Path) -> bool:
    unstaged = subprocess.run(
        ["git", "-C", str(path), "diff", "--quiet"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    staged = subprocess.run(
        ["git", "-C", str(path), "diff", "--cached", "--quiet"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    return unstaged.returncode == 1 or staged.returncode == 1


def checkout_external(name: str) -> ExternalRepo:
    repo = EXTERNALS[name]
    repo.path.parent.mkdir(parents=True, exist_ok=True)
    if repo.path.exists():
        if not (repo.path / ".git").exists():
            raise InfraError(f"External path exists but is not a git checkout: {repo.path}")
        if _has_tracked_changes(repo.path):
            raise InfraError(f"External checkout has tracked local changes; refusing to switch/update: {repo.path}")
        if repo.branch:
            _run_git(repo.path, ["fetch", "origin", repo.branch])
            current_branch = _run_git(repo.path, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
            if current_branch != repo.branch:
                local_branch = _run_git(repo.path, ["branch", "--list", repo.branch]).stdout.strip()
                if local_branch:
                    _run_git(repo.path, ["switch", repo.branch])
                else:
                    _run_git(repo.path, ["switch", "-c", repo.branch, "--track", f"origin/{repo.branch}"])
            _run_git(repo.path, ["merge", "--ff-only", f"origin/{repo.branch}"])
        else:
            _run_git(repo.path, ["fetch", "origin"])
        return repo
    cmd = ["git", "clone", repo.url, str(repo.path)]
    if repo.branch:
        cmd = ["git", "clone", "--branch", repo.branch, "--single-branch", repo.url, str(repo.path)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if result.returncode != 0:
        raise InfraError(f"Failed to clone {name}: {result.stderr.strip()}")
    return repo


def editable_install(repo: ExternalRepo, python: str = "python") -> None:
    if not repo.path.exists():
        raise InfraError(f"External path is missing: {repo.path}")
    env = build_env()
    command = [python, "-m", "pip", "install", "--no-build-isolation"]
    if env.get("PUTPOCKET_PIP_INDEX_URL"):
        command.extend(["--index-url", str(env["PUTPOCKET_PIP_INDEX_URL"])])
    if env.get("PUTPOCKET_PIP_EXTRA_INDEX_URL"):
        command.extend(["--extra-index-url", str(env["PUTPOCKET_PIP_EXTRA_INDEX_URL"])])
    if env.get("PUTPOCKET_TORCH_CONSTRAINT_FILE"):
        command.extend(["-c", str(env["PUTPOCKET_TORCH_CONSTRAINT_FILE"])])
    command.extend(["-e", str(repo.path)])
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise InfraError(f"Editable install failed for {repo.name}: {result.stderr.strip()}")


def externals_status() -> list[dict[str, str | bool]]:
    rows: list[dict[str, str | bool]] = []
    for repo in EXTERNALS.values():
        current_branch = ""
        current_commit = ""
        remote_url = ""
        if (repo.path / ".git").exists():
            current_branch = (
                subprocess.run(
                    ["git", "-C", str(repo.path), "rev-parse", "--abbrev-ref", "HEAD"],
                    text=True,
                    capture_output=True,
                ).stdout.strip()
            )
            current_commit = (
                subprocess.run(
                    ["git", "-C", str(repo.path), "rev-parse", "HEAD"],
                    text=True,
                    capture_output=True,
                ).stdout.strip()
            )
            remote_url = (
                subprocess.run(
                    ["git", "-C", str(repo.path), "remote", "get-url", "origin"],
                    text=True,
                    capture_output=True,
                ).stdout.strip()
            )
        rows.append(
            {
                "name": repo.name,
                "path": str(repo.path),
                "exists": repo.path.exists(),
                "branch": repo.branch or "",
                "current_branch": current_branch,
                "current_commit": current_commit,
                "remote_url": remote_url,
                "role": repo.role,
            }
        )
    return rows
