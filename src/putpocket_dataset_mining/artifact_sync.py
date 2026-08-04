from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigError

SYNC_PROFILES: dict[str, dict[str, list[str]]] = {
    "analysis_minimal": {
        "include": [
            "manifest*.json",
            "eval_config.*",
            "summary.*",
            "results.jsonl",
            "trajectories/**",
            "prepared*/messages_*.json",
            "prepared*/rendered_prompt_*.txt",
            "model_responses.jsonl",
            "model_requests.jsonl",
            "verification/**/checklist.json",
            "verification/**/stdout.txt",
            "verification/**/stderr.txt",
            "judge/**",
            "metrics/**",
            "reuse_maps/**.json",
            "prefix_cache_metrics*.json",
        ],
        "exclude": ["**/tests/**", "**/*solution_code*", "**/*reference*", "**/.ssh/**", "**/*token*", "**/*secret*", "**/kv_full/**", "**/allocated_kv_pool/**"],
    },
    "analysis_with_workspaces": {
        "include": ["workspace_snapshots/**"],
        "exclude": ["workspace_snapshots/**/tests/**", "**/*solution_code*", "**/*reference*"],
    },
    "analysis_with_selected_kv": {
        "include": ["kv_selected/**", "kv_metadata/**", "reuse_maps/**", "kv_summaries/**"],
        "exclude": ["**/allocated_kv_pool/**", "**/kv_full/**"],
    },
    "verifier_input": {
        "include": ["workspace/**", "tests/**", "manifest.json", "verifier_specs/**"],
        "exclude": ["**/.ssh/**", "**/*token*", "**/*secret*"],
    },
    "verifier_output": {
        "include": ["result.json", "stdout.txt", "stderr.txt", "checklist.json", "pytest*.json"],
        "exclude": ["workspace/**/tests/**"],
    },
}


@dataclass(frozen=True)
class SyncItem:
    relative_path: str
    size: int
    sha256: str


def profile_patterns(profile: str) -> tuple[list[str], list[str]]:
    if profile not in SYNC_PROFILES:
        raise ConfigError(f"Unknown sync profile: {profile}")
    include: list[str] = []
    exclude: list[str] = []
    for name in _profile_closure(profile):
        include.extend(SYNC_PROFILES[name].get("include", []))
        exclude.extend(SYNC_PROFILES[name].get("exclude", []))
    return include, exclude


def _profile_closure(profile: str) -> list[str]:
    if profile == "analysis_with_workspaces":
        return ["analysis_minimal", "analysis_with_workspaces"]
    if profile == "analysis_with_selected_kv":
        return ["analysis_minimal", "analysis_with_selected_kv"]
    return [profile]


def build_sync_manifest(source_root: Path, profile: str) -> dict[str, Any]:
    source_root = source_root.resolve()
    include, exclude = profile_patterns(profile)
    items: list[SyncItem] = []
    for path in sorted(p for p in source_root.rglob("*") if p.is_file()):
        rel = path.relative_to(source_root).as_posix()
        if _matches(rel, include) and not _matches(rel, exclude):
            items.append(SyncItem(rel, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()))
    return {
        "schema_version": 1,
        "profile": profile,
        "source_root": str(source_root),
        "delete_enabled": False,
        "items": [item.__dict__ for item in items],
        "item_count": len(items),
    }


def copy_from_manifest(source_root: Path, destination_root: Path, manifest: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    partial = destination_root.with_name(destination_root.name + ".partial")
    copied: list[str] = []
    checksum_errors: list[str] = []
    if not dry_run:
        partial.mkdir(parents=True, exist_ok=True)
    for item in manifest.get("items", []):
        rel = str(item["relative_path"])
        _safe_rel(rel)
        src = source_root / rel
        dst = partial / rel
        actual = hashlib.sha256(src.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            checksum_errors.append(rel)
            continue
        copied.append(rel)
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    if checksum_errors:
        raise ConfigError(f"Checksum mismatch before sync: {checksum_errors[:5]}")
    if not dry_run:
        (partial / "SYNC_COMPLETE.json").write_text(json.dumps({"schema_version": 1, "profile": manifest["profile"], "item_count": len(copied)}, indent=2), encoding="utf-8")
        if destination_root.exists():
            # No delete semantics: merge the completed partial tree into destination.
            for path in sorted(p for p in partial.rglob("*") if p.is_file()):
                rel = path.relative_to(partial)
                target = destination_root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
            shutil.rmtree(partial)
        else:
            partial.rename(destination_root)
    return {"dry_run": dry_run, "copied": copied, "item_count": len(copied), "destination": str(destination_root)}


def _matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(path, pattern.replace("**/", "")) for pattern in patterns)


def _safe_rel(path: str) -> None:
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise ConfigError(f"Unsafe sync path: {path}")
