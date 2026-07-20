from __future__ import annotations

import os
import shutil
from pathlib import Path


def atomic_write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)


def ensure_empty_dir(path: str | Path) -> Path:
    target = Path(path)
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    return target


def copy_tree(src: str | Path, dst: str | Path) -> None:
    source = Path(src)
    target = Path(dst)
    if target.exists():
        shutil.rmtree(target)
    ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache")
    shutil.copytree(source, target, ignore=ignore)


def safe_relative_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        if len(candidate.parts) >= 2 and candidate.parts[1] == "workspace":
            candidate = Path(*candidate.parts[2:]) if len(candidate.parts) > 2 else Path(".")
        else:
            raise ValueError(f"Path must be relative to workspace: {path}")
    if candidate.parts and candidate.parts[0] == "workspace":
        candidate = Path(*candidate.parts[1:]) if len(candidate.parts) > 1 else Path(".")
    if ".." in candidate.parts:
        raise ValueError(f"Path must be relative to workspace: {path}")
    return candidate


def host_uid_gid() -> tuple[int, int]:
    return os.getuid(), os.getgid()
