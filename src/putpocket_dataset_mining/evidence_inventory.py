"""Build and verify deterministic, remote-relative evidence inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator

from .errors import ConfigError


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = (
    REPOSITORY_ROOT
    / "configs/runpod/schemas/glm52_montblanc_evidence_inventory.schema.json"
)


def canonical_json_bytes(value: object, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + suffix
    ).encode("utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ConfigError(reason)


def safe_relative(value: str, *, reason: str) -> Path:
    path = Path(value)
    _require(
        bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
        reason,
    )
    return path


def load_spec(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "EVIDENCE_SPEC_NOT_OBJECT")
    _require(value.get("schema_version") == 1, "EVIDENCE_SPEC_VERSION_INVALID")
    _require(
        isinstance(value.get("inventory_id"), str) and value["inventory_id"],
        "EVIDENCE_SPEC_ID_INVALID",
    )
    _require(
        isinstance(value.get("source_remote_root_id"), str)
        and value["source_remote_root_id"],
        "EVIDENCE_SPEC_REMOTE_ROOT_INVALID",
    )
    groups = value.get("groups")
    exclusions = value.get("exclusions")
    _require(isinstance(groups, list) and groups, "EVIDENCE_SPEC_GROUPS_INVALID")
    _require(
        isinstance(exclusions, list) and exclusions,
        "EVIDENCE_SPEC_EXCLUSIONS_INVALID",
    )
    return value


def _group_files(root: Path, group: Mapping[str, Any]) -> list[tuple[Path, Path]]:
    local = safe_relative(str(group.get("local_path", "")), reason="EVIDENCE_GROUP_LOCAL_PATH_INVALID")
    unresolved_target = root / local
    _require(
        unresolved_target.exists() and not unresolved_target.is_symlink(),
        "EVIDENCE_GROUP_TARGET_MISSING_OR_SYMLINK",
    )
    target = unresolved_target.resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ConfigError("EVIDENCE_GROUP_LOCAL_PATH_ESCAPES_ROOT") from exc
    if target.is_file():
        return [(target, Path(target.name))]
    files: list[tuple[Path, Path]] = []
    for path in sorted(target.rglob("*")):
        _require(not path.is_symlink(), "EVIDENCE_GROUP_CONTAINS_SYMLINK")
        if path.is_file():
            files.append((path, path.relative_to(target)))
    _require(bool(files), "EVIDENCE_GROUP_EMPTY")
    return files


def build_inventory(
    root: str | Path,
    spec: Mapping[str, Any],
    *,
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    evidence_root = Path(root).resolve()
    _require(evidence_root.is_dir(), "EVIDENCE_ROOT_MISSING")
    entries: list[dict[str, Any]] = []
    local_seen: set[str] = set()
    remote_seen: set[str] = set()
    group_ids: set[str] = set()

    for raw_group in spec["groups"]:
        _require(isinstance(raw_group, Mapping), "EVIDENCE_GROUP_NOT_OBJECT")
        group_id = raw_group.get("group_id")
        content_class = raw_group.get("content_class")
        _require(
            isinstance(group_id, str) and group_id and group_id not in group_ids,
            "EVIDENCE_GROUP_ID_INVALID_OR_DUPLICATE",
        )
        _require(
            isinstance(content_class, str) and content_class,
            "EVIDENCE_GROUP_CONTENT_CLASS_INVALID",
        )
        group_ids.add(group_id)
        remote_root = safe_relative(
            str(raw_group.get("source_remote_relative", "")),
            reason="EVIDENCE_GROUP_REMOTE_PATH_INVALID",
        )
        local_root = safe_relative(
            str(raw_group.get("local_path", "")),
            reason="EVIDENCE_GROUP_LOCAL_PATH_INVALID",
        )
        target = (evidence_root / local_root).resolve()
        is_file_group = target.is_file()
        for path, suffix in _group_files(evidence_root, raw_group):
            local_relative = path.relative_to(evidence_root).as_posix()
            remote_relative = (
                remote_root if is_file_group else remote_root / suffix
            ).as_posix()
            _require(local_relative not in local_seen, "EVIDENCE_LOCAL_PATH_DUPLICATE")
            _require(remote_relative not in remote_seen, "EVIDENCE_REMOTE_PATH_DUPLICATE")
            local_seen.add(local_relative)
            remote_seen.add(remote_relative)
            entries.append(
                {
                    "group_id": group_id,
                    "content_class": content_class,
                    "local_relative_path": local_relative,
                    "source_remote_relative_path": remote_relative,
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )

    entries.sort(key=lambda item: item["local_relative_path"])
    exclusions: list[dict[str, str]] = []
    exclusion_ids: set[str] = set()
    for raw in spec["exclusions"]:
        _require(isinstance(raw, Mapping), "EVIDENCE_EXCLUSION_NOT_OBJECT")
        exclusion_id, reason = raw.get("exclusion_id"), raw.get("reason")
        _require(
            isinstance(exclusion_id, str)
            and exclusion_id
            and exclusion_id not in exclusion_ids,
            "EVIDENCE_EXCLUSION_ID_INVALID_OR_DUPLICATE",
        )
        _require(isinstance(reason, str) and reason, "EVIDENCE_EXCLUSION_REASON_INVALID")
        exclusion_ids.add(exclusion_id)
        exclusions.append({"exclusion_id": exclusion_id, "reason": reason})

    payload = {
        "schema_version": 1,
        "inventory_id": spec["inventory_id"],
        "source_remote_root_id": spec["source_remote_root_id"],
        "entry_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "groups": [
            {
                "group_id": group["group_id"],
                "content_class": group["content_class"],
                "local_path": safe_relative(
                    group["local_path"], reason="EVIDENCE_GROUP_LOCAL_PATH_INVALID"
                ).as_posix(),
                "source_remote_relative": safe_relative(
                    group["source_remote_relative"],
                    reason="EVIDENCE_GROUP_REMOTE_PATH_INVALID",
                ).as_posix(),
            }
            for group in spec["groups"]
        ],
        "entries": entries,
        "exclusions": sorted(exclusions, key=lambda item: item["exclusion_id"]),
    }
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(
            canonical_json_bytes(payload, newline=False)
        ).hexdigest(),
    }


def write_inventory(
    root: str | Path,
    document: Mapping[str, Any],
    *,
    output_path: str | Path,
    checksum_path: str | Path,
) -> None:
    evidence_root = Path(root).resolve()
    output = Path(output_path)
    checksums = Path(checksum_path)
    _require(not output.exists() and not checksums.exists(), "EVIDENCE_OUTPUT_ALREADY_EXISTS")
    output.parent.mkdir(parents=True, exist_ok=True)
    checksums.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(document))
    lines = [
        f"{entry['sha256']}  {entry['local_relative_path']}\n"
        for entry in document["payload"]["entries"]
    ]
    checksums.write_text("".join(lines), encoding="utf-8")


def verify_inventory(
    root: str | Path,
    document: Mapping[str, Any],
    *,
    schema_path: str | Path = DEFAULT_SCHEMA,
) -> dict[str, Any]:
    evidence_root = Path(root).resolve()
    payload = document.get("payload")
    observed_payload_digest = document.get("payload_sha256")
    _require(isinstance(payload, Mapping), "EVIDENCE_INVENTORY_PAYLOAD_INVALID")
    expected_payload_digest = hashlib.sha256(
        canonical_json_bytes(payload, newline=False)
    ).hexdigest()
    _require(
        observed_payload_digest == expected_payload_digest,
        "EVIDENCE_INVENTORY_PAYLOAD_DIGEST_MISMATCH",
    )
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(payload)
    group_ids = [group["group_id"] for group in payload["groups"]]
    _require(len(group_ids) == len(set(group_ids)), "EVIDENCE_GROUP_IDS_DUPLICATE")
    declared_groups = set(group_ids)
    entries = payload["entries"]
    local_paths = [entry["local_relative_path"] for entry in entries]
    remote_paths = [entry["source_remote_relative_path"] for entry in entries]
    _require(local_paths == sorted(local_paths), "EVIDENCE_ENTRIES_NOT_SORTED")
    _require(len(local_paths) == len(set(local_paths)), "EVIDENCE_LOCAL_PATH_DUPLICATE")
    _require(len(remote_paths) == len(set(remote_paths)), "EVIDENCE_REMOTE_PATH_DUPLICATE")
    _require(
        all(entry["group_id"] in declared_groups for entry in entries),
        "EVIDENCE_ENTRY_GROUP_UNDECLARED",
    )
    _require(len(entries) == payload["entry_count"], "EVIDENCE_ENTRY_COUNT_MISMATCH")
    _require(
        sum(entry["bytes"] for entry in entries) == payload["total_bytes"],
        "EVIDENCE_TOTAL_BYTES_MISMATCH",
    )
    checked = 0
    for entry in entries:
        relative = safe_relative(
            entry["local_relative_path"], reason="EVIDENCE_ENTRY_LOCAL_PATH_INVALID"
        )
        path = (evidence_root / relative).resolve()
        try:
            path.relative_to(evidence_root)
        except ValueError as exc:
            raise ConfigError("EVIDENCE_ENTRY_ESCAPES_ROOT") from exc
        _require(path.is_file() and not path.is_symlink(), "EVIDENCE_ENTRY_MISSING_OR_SYMLINK")
        _require(path.stat().st_size == entry["bytes"], "EVIDENCE_ENTRY_SIZE_MISMATCH")
        _require(file_sha256(path) == entry["sha256"], "EVIDENCE_ENTRY_DIGEST_MISMATCH")
        checked += 1
    _require(checked == payload["entry_count"], "EVIDENCE_ENTRY_COUNT_MISMATCH")
    return {
        "status": "passed",
        "entry_count": checked,
        "total_bytes": payload["total_bytes"],
        "payload_sha256": expected_payload_digest,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", required=True, type=Path)
    build.add_argument("--spec", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--checksums", required=True, type=Path)
    build.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", required=True, type=Path)
    verify.add_argument("--inventory", required=True, type=Path)
    verify.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "build":
        document = build_inventory(args.root, load_spec(args.spec), schema_path=args.schema)
        write_inventory(
            args.root,
            document,
            output_path=args.output,
            checksum_path=args.checksums,
        )
        result = {
            "status": "passed",
            "inventory": str(args.output),
            "checksums": str(args.checksums),
            "entry_count": document["payload"]["entry_count"],
            "total_bytes": document["payload"]["total_bytes"],
            "payload_sha256": document["payload_sha256"],
        }
    else:
        document = json.loads(args.inventory.read_text(encoding="utf-8"))
        result = verify_inventory(args.root, document, schema_path=args.schema)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
