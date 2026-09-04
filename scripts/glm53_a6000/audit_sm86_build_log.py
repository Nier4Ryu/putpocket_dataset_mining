#!/usr/bin/env python3
"""Fail-closed audit of CUDA compiler commands in a Docker build log."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ARCH_PATTERN = re.compile(r"(?:compute|sm)_([0-9]+)[a-z]?", re.IGNORECASE)


class BuildLogAuditError(RuntimeError):
    pass


def audit_build_log(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    text = payload.decode("utf-8", errors="replace")
    compiler_lines = [line for line in text.splitlines() if "nvcc " in line]
    architectures = sorted(
        {
            match.group(1)
            for line in compiler_lines
            for match in ARCH_PATTERN.finditer(line)
        }
    )
    if architectures not in ([], ["86"]):
        raise BuildLogAuditError(
            f"visible compiler architecture set must be empty or ['86'], got {architectures}"
        )
    always_required = (
        "PutPocket SM86-only build: FlashInfer JIT cache omitted",
        '"component":"DeepEP","included":false',
    )
    missing = [marker for marker in always_required if marker not in text]
    if missing:
        raise BuildLogAuditError(f"missing build markers: {missing}")

    fresh_compile_markers = (
        "-- CUDA target architectures: 8.6",
        "PutPocket SM86-only build: excluding non-SM86 optional external projects",
        "PutPocket SM86-only build: bundled FA2 omitted to prevent SM80 SASS/PTX",
    )
    fresh_compile_markers_present = all(
        marker in text for marker in fresh_compile_markers
    )

    wheel_audits: list[dict[str, object]] = []
    for line in text.splitlines():
        if '"all_task_built_cuda_payloads_sm86_only"' not in line:
            continue
        json_start = line.find("{")
        if json_start < 0:
            continue
        try:
            candidate = json.loads(line[json_start:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            wheel_audits.append(candidate)
    valid_wheel_audits = [
        report
        for report in wheel_audits
        if report.get("all_task_built_cuda_payloads_sm86_only") is True
        and report.get("observed_architectures") == ["86"]
        and isinstance(report.get("members"), dict)
        and not any(
            "_vllm_fa2_C" in member for member in report["members"]
        )
    ]
    if not fresh_compile_markers_present and not valid_wheel_audits:
        raise BuildLogAuditError(
            "cached build log lacks both fresh SM86 compile markers and a valid "
            "embedded SM86 wheel audit"
        )
    forbidden_execution_markers = (
        "/tmp/ep_kernels_workspace/DeepEP/build/",
        "uv pip install --system ep_kernels/dist/",
        "vllm-flash-attn/CMakeFiles/_vllm_fa2_C.dir/",
    )
    present = [marker for marker in forbidden_execution_markers if marker in text]
    if present:
        raise BuildLogAuditError(f"forbidden extension build/install markers: {present}")
    return {
        "schema_version": 1,
        "build_log": path.name,
        "build_log_sha256": hashlib.sha256(payload).hexdigest(),
        "audit_scope": "task-built compiler invocations in the successful Docker log",
        "nvcc_compiler_line_count": len(compiler_lines),
        "observed_compiler_architectures": architectures,
        "compiler_architecture_flags_visible": bool(architectures),
        "no_non_sm86_compiler_flags_observed": True,
        "sm86_configuration_marker_present": fresh_compile_markers_present,
        "configuration_evidence": (
            "fresh_compile_markers"
            if fresh_compile_markers_present
            else "cached_layer_plus_embedded_wheel_audit"
        ),
        "valid_embedded_wheel_audit_count": len(valid_wheel_audits),
        "binary_architecture_authority": "embedded sm86-cuda-audit.json",
        "deepep_built_or_installed": False,
        "bundled_fa2_built_or_installed": False,
        "non_sm86_optional_extensions_excluded": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_build_log(args.log)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
