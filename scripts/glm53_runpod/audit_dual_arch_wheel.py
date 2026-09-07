#!/usr/bin/env python3
"""Statically audit task-built GLM-5.3 CUDA payloads for SM90 and SM120.

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
REQUIRED_NATIVE_ARCHITECTURES = ("90", "120")
REQUIRED_EXTENSION_PREFIXES = (
    "vllm/vllm_flash_attn/_vllm_fa2_C",
    "vllm/vllm_flash_attn/_vllm_fa3_C",
)
# Upstream's supported W4A16 Marlin and c2x compatibility kernels deliberately
# ship lower-architecture SASS/PTX for forward compatibility on newer GPUs.
ALLOWED_COMPATIBILITY_ARCHITECTURES = ("80", "89")
ALLOWED_ARCHITECTURES = REQUIRED_NATIVE_ARCHITECTURES + ALLOWED_COMPATIBILITY_ARCHITECTURES
ARCH_PATTERN = re.compile(r"(?:sm|compute)_([0-9]+)[a-z]?", re.IGNORECASE)


class DualArchAuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def architectures_from_cuobjdump(text: str) -> set[str]:
    return {match.group(1) for match in ARCH_PATTERN.finditer(text)}


def required_extension_members(member_names: list[str]) -> dict[str, list[str]]:
    """Resolve every mandatory extension prefix without loading a shared object."""
    matches = {
        prefix: sorted(
            name for name in member_names if name.startswith(prefix) and name.endswith(".so")
        )
        for prefix in REQUIRED_EXTENSION_PREFIXES
    }
    missing = [prefix for prefix, names in matches.items() if not names]
    if missing:
        raise DualArchAuditError(
            "missing required vLLM FlashAttention extensions: " + ",".join(missing)
        )
    return matches


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
            raise DualArchAuditError(
                f"cuobjdump failed for {path.name} {option}: {result.stderr.strip()}"
            )
        outputs.extend((result.stdout, result.stderr))
    architectures = sorted(architectures_from_cuobjdump("\n".join(outputs)))
    foreign = sorted(set(architectures).difference(ALLOWED_ARCHITECTURES))
    if foreign:
        raise DualArchAuditError(
            f"out-of-scope device code in {path.name}: {','.join(foreign)}"
        )
    return {
        "sha256": sha256_file(path),
        "architectures": architectures,
        "device_code_present": bool(architectures),
    }


def audit_wheels(wheels: list[Path], cuobjdump: str) -> dict[str, object]:
    if not wheels:
        raise DualArchAuditError("at least one wheel is required")
    members: dict[str, dict[str, object]] = {}
    wheel_member_names: list[str] = []
    wheel_records = []
    with tempfile.TemporaryDirectory(prefix="putpocket-sm90-sm120-wheel-audit-") as temp:
        root = Path(temp)
        for wheel_index, wheel in enumerate(sorted(wheels)):
            if not wheel.is_file():
                raise DualArchAuditError(f"wheel not found: {wheel}")
            extract_root = root / str(wheel_index)
            with zipfile.ZipFile(wheel) as archive:
                shared_names = sorted(
                    name
                    for name in archive.namelist()
                    if name.startswith("vllm/") and name.endswith(".so")
                )
                wheel_member_names.extend(shared_names)
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
    missing = sorted(set(REQUIRED_NATIVE_ARCHITECTURES).difference(found))
    if missing:
        raise DualArchAuditError(
            f"missing required native target architectures {missing}; found {found}"
        )
    extension_members = required_extension_members(wheel_member_names)
    return {
        "schema_version": SCHEMA_VERSION,
        "audit_scope": "task-built vLLM/PutPocket wheel shared objects only",
        "method": "cuobjdump --list-elf and --list-ptx; no CUDA runtime",
        "allowed_architectures": list(ALLOWED_ARCHITECTURES),
        "required_native_architectures": list(REQUIRED_NATIVE_ARCHITECTURES),
        "required_extension_prefixes": list(REQUIRED_EXTENSION_PREFIXES),
        "required_extension_members": extension_members,
        "allowed_upstream_compatibility_architectures": list(ALLOWED_COMPATIBILITY_ARCHITECTURES),
        "observed_architectures": found,
        "out_of_scope_architectures": sorted(set(found).difference(ALLOWED_ARCHITECTURES)),
        "all_task_built_cuda_payloads_in_scope": True,
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
        parser.error("cuobjdump is required for the fail-closed SM90+SM120 audit")
    report = audit_wheels(args.wheels, args.cuobjdump)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
