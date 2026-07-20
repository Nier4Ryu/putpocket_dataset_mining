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
    env.update(BUILD_ENV_OVERRIDES)
    return env


def checkout_external(name: str) -> ExternalRepo:
    repo = EXTERNALS[name]
    repo.path.parent.mkdir(parents=True, exist_ok=True)
    if repo.path.exists():
        return repo
    cmd = ["git", "clone", repo.url, str(repo.path)]
    if repo.branch:
        cmd = ["git", "clone", "--branch", repo.branch, repo.url, str(repo.path)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    if result.returncode != 0:
        raise InfraError(f"Failed to clone {name}: {result.stderr.strip()}")
    return repo


def editable_install(repo: ExternalRepo, python: str = "python") -> None:
    if not repo.path.exists():
        raise InfraError(f"External path is missing: {repo.path}")
    result = subprocess.run(
        [python, "-m", "pip", "install", "--no-build-isolation", "-e", str(repo.path)],
        cwd=str(REPO_ROOT),
        env=build_env(),
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise InfraError(f"Editable install failed for {repo.name}: {result.stderr.strip()}")


def externals_status() -> list[dict[str, str | bool]]:
    rows: list[dict[str, str | bool]] = []
    for repo in EXTERNALS.values():
        rows.append(
            {
                "name": repo.name,
                "path": str(repo.path),
                "exists": repo.path.exists(),
                "branch": repo.branch or "",
                "role": repo.role,
            }
        )
    return rows
