#!/usr/bin/env python3
"""Statically prove task-built vLLM wheel CUDA payloads target only SM86.

This invokes ``cuobjdump`` as a file parser. It does not import torch, call a
CUDA runtime API, inspect devices, or load shared objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path


SCHEMA_VERSION = 1
ALLOWED_ARCHITECTURES = ("86",)
ARCH_PATTERN = re.compile(r"(?:sm|compute)_([0-9]+)[a-z]?", re.IGNORECASE)


class Sm86AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def architectures_from_cuobjdump(text: str) -> set[str]:
    return {match.group(1) for match in ARCH_PATTERN.finditer(text)}


def inspect_shared_object(path: Path, cuobjdump: str) -> dict[str, object]:
    outputs: list[str] = []
    for option in ("--list-elf", "--list-ptx"):
        result = subprocess.run(
            [cuobjdump, option, str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            diagnostic = "\n".join((result.stdout, result.stderr))
            if "does not contain device code" in diagnostic:
                outputs.append(diagnostic)
                continue
            raise Sm86AuditError(
                f"cuobjdump failed for {path.name} {option}: {result.stderr.strip()}"
            )
        outputs.extend((result.stdout, result.stderr))
    architectures = sorted(architectures_from_cuobjdump("\n".join(outputs)))
    foreign = sorted(set(architectures).difference(ALLOWED_ARCHITECTURES))
    if foreign:
        raise Sm86AuditError(
            f"non-SM86 device code in {path.name}: {','.join(foreign)}"
        )
    return {
        "sha256": sha256_file(path),
        "architectures": architectures,
        "device_code_present": bool(architectures),
    }


def audit_wheels(wheels: list[Path], cuobjdump: str) -> dict[str, object]:
    if not wheels:
        raise Sm86AuditError("at least one wheel is required")
    members: dict[str, dict[str, object]] = {}
    wheel_records = []
    with tempfile.TemporaryDirectory(prefix="putpocket-sm86-wheel-audit-") as temp:
        root = Path(temp)
        for wheel_index, wheel in enumerate(sorted(wheels)):
            if not wheel.is_file():
                raise Sm86AuditError(f"wheel not found: {wheel}")
            extract_root = root / str(wheel_index)
            with zipfile.ZipFile(wheel) as archive:
                shared_names = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith("vllm/") and name.endswith(".so")
                )
                for name in shared_names:
                    archive.extract(name, extract_root)
                    members[f"{wheel.name}:{name}"] = inspect_shared_object(
                        extract_root / name, cuobjdump
                    )
            wheel_records.append(
                {
                    "filename": wheel.name,
                    "sha256": sha256_file(wheel),
                    "vllm_shared_object_count": len(shared_names),
                }
            )
    found = sorted(
        {
            architecture
            for record in members.values()
            for architecture in record["architectures"]
        }
    )
    if found != list(ALLOWED_ARCHITECTURES):
        raise Sm86AuditError(
            f"expected task-built device code architecture {ALLOWED_ARCHITECTURES}, found {found}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_scope": "task-built vLLM wheel shared objects only",
        "method": "cuobjdump --list-elf and --list-ptx; no CUDA runtime",
        "allowed_architectures": list(ALLOWED_ARCHITECTURES),
        "observed_architectures": found,
        "all_task_built_cuda_payloads_sm86_only": True,
        "vendor_binary_scope_excluded": [
            "PyTorch wheels",
            "CUDA toolkit/runtime libraries",
            "NVIDIA Python dependency wheels",
        ],
        "wheels": wheel_records,
        "members": members,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheels", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cuobjdump", default=shutil.which("cuobjdump"))
    args = parser.parse_args()
    if not args.cuobjdump:
        parser.error("cuobjdump is required for the fail-closed SM86 audit")
    report = audit_wheels(args.wheels, args.cuobjdump)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
