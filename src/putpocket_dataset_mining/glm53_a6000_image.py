from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_VLLM_VERSION = "0.29.0.dev"
EXPECTED_VLLM_COMMIT = "9cd956c7e6cf54efa366b803cafa15ec6c2df827"
EXPECTED_ARCH = "8.6"
EXPECTED_CMAKE_ARCH = "86"
EXPECTED_MODEL_ARCHITECTURE = "Glm5NextForConditionalGeneration"
EXPECTED_MLA_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43]


class A6000ContractError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: str | Path) -> dict[str, Any]:
    lock_path = Path(path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    validate_lock(payload)
    return payload


def validate_lock(lock: dict[str, Any]) -> None:
    failures: list[str] = []
    runtime = lock.get("runtime", {})
    model = lock.get("model", {})
    hybrid = lock.get("hybrid_cache_contract", {})
    if lock.get("schema_version") != 1:
        failures.append("schema_version")
    if runtime.get("vllm_version") != EXPECTED_VLLM_VERSION:
        failures.append("vllm_version")
    if runtime.get("vllm_commit") != EXPECTED_VLLM_COMMIT:
        failures.append("vllm_commit")
    if runtime.get("torch_cuda_arch_list") != EXPECTED_ARCH:
        failures.append("torch_cuda_arch_list")
    if runtime.get("cmake_cuda_architectures") != EXPECTED_CMAKE_ARCH:
        failures.append("cmake_cuda_architectures")
    build_scope = runtime.get("task_built_cuda_scope", {})
    if build_scope.get("exclusive_architecture") != EXPECTED_ARCH:
        failures.append("task_built_cuda_scope.exclusive_architecture")
    if build_scope.get("deepep_included") is not False:
        failures.append("task_built_cuda_scope.deepep_included")
    if build_scope.get("non_sm86_optional_extensions_included") is not False:
        failures.append(
            "task_built_cuda_scope.non_sm86_optional_extensions_included"
        )
    if build_scope.get("bundled_fa2_included") is not False:
        failures.append("task_built_cuda_scope.bundled_fa2_included")
    vendor_scope = runtime.get("vendor_binary_scope", {})
    if vendor_scope.get("single_architecture_claim") is not False:
        failures.append("vendor_binary_scope.single_architecture_claim")
    overlay = lock.get("overlay", {})
    if overlay.get("patch_order") != [
        f"patches/vllm/{EXPECTED_VLLM_COMMIT}/glm53_a6000_sm86_only_build.patch",
        f"patches/vllm/{EXPECTED_VLLM_COMMIT}/glm53_a6000_compatibility_gate.patch",
    ]:
        failures.append("overlay.patch_order")
    if runtime.get("runtime_supported") is not False:
        failures.append("runtime_supported_must_be_false")
    if model.get("architecture") != EXPECTED_MODEL_ARCHITECTURE:
        failures.append("model_architecture")
    if hybrid.get("mla_indexer_layers") != EXPECTED_MLA_LAYERS:
        failures.append("mla_indexer_layers")
    if hybrid.get("kda_layer_count") != 34:
        failures.append("kda_layer_count")
    if hybrid.get("index_kpool") != 4:
        failures.append("index_kpool")
    semantics = lock.get("stateful_edit", {})
    expected_semantics = {
        "default_enabled": False,
        "full_target_prefill": True,
        "donor_overwrite_after_native_write": True,
        "true_partial_prefill": False,
        "speedup_claim": False,
    }
    for key, expected in expected_semantics.items():
        if semantics.get(key) != expected:
            failures.append(f"stateful_edit.{key}")
    if failures:
        raise A6000ContractError(", ".join(failures))


def static_doctor(lock_path: str | Path, root: str | Path) -> dict[str, Any]:
    lock = load_lock(lock_path)
    repo_root = Path(root)
    checks: list[dict[str, Any]] = []
    for artifact in lock["artifacts"]:
        path = repo_root / artifact["path"]
        actual = sha256_file(path) if path.is_file() else None
        checks.append(
            {
                "path": artifact["path"],
                "expected_sha256": artifact["sha256"],
                "actual_sha256": actual,
                "ok": actual == artifact["sha256"],
            }
        )
    source = lock["runtime"]["vllm_source_attestations"]
    registry = repo_root / source["support_snapshot_path"]
    registry_text = registry.read_text(encoding="utf-8") if registry.is_file() else ""
    registry_hash = sha256_file(registry) if registry.is_file() else None
    checks.append(
        {
            "path": source["support_snapshot_path"],
            "expected_sha256": source["support_snapshot_sha256"],
            "actual_sha256": registry_hash,
            "ok": registry_hash == source["support_snapshot_sha256"],
        }
    )
    image_audit_path = Path("/opt/vllm-build-evidence/sm86-cuda-audit.json")
    image_audit = None
    if repo_root == Path("/opt/putpocket"):
        if image_audit_path.is_file():
            image_audit = json.loads(image_audit_path.read_text(encoding="utf-8"))
            checks.append(
                {
                    "path": str(image_audit_path),
                    "name": "task_built_vllm_cuda_payloads_are_sm86_only",
                    "ok": image_audit.get("observed_architectures") == ["86"]
                    and image_audit.get("all_task_built_cuda_payloads_sm86_only")
                    is True,
                }
            )
        else:
            checks.append(
                {
                    "path": str(image_audit_path),
                    "name": "task_built_vllm_cuda_audit_present",
                    "ok": False,
                }
            )
        import importlib.metadata

        installed = {
            distribution.metadata["Name"].lower().replace("_", "-")
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
        forbidden_distributions = {
            "deep-ep",
            "deepep",
            "flashinfer-cubin",
            "flashinfer-jit-cache",
            "humming-kernels",
            "nvidia-cutlass-dsl",
            "nvidia-cutlass-dsl-libs-base",
            "nvidia-cutlass-dsl-libs-core",
            "nvidia-cutlass-dsl-libs-cu12",
            "quack-kernels",
            "tokenspeed-mla",
        }
        checks.append(
            {
                "name": "non_sm86_optional_distributions_absent",
                "forbidden_distributions": sorted(forbidden_distributions),
                "present_forbidden_distributions": sorted(
                    installed.intersection(forbidden_distributions)
                ),
                "ok": not installed.intersection(forbidden_distributions),
            }
        )
        try:
            vllm_files = {
                str(path) for path in (importlib.metadata.files("vllm") or ())
            }
        except importlib.metadata.PackageNotFoundError:
            vllm_files = set()
        forbidden_vllm_payload_fragments = (
            "_vllm_fa2_C",
            "_deep_gemm_C",
            "_flashkda_C",
            "_flashmla_C",
            "_flashmla_extension_C",
            "_qutlass_C",
            "_vllm_fa3_C",
            "third_party/fmha_sm100",
            "third_party/tml_fa4",
        )
        present_fragments = sorted(
            fragment
            for fragment in forbidden_vllm_payload_fragments
            if any(fragment in path for path in vllm_files)
        )
        checks.append(
            {
                "name": "non_sm86_optional_vllm_payloads_absent",
                "present_forbidden_fragments": present_fragments,
                "ok": bool(vllm_files) and not present_fragments,
            }
        )
    checks.append(
        {
            "name": "current_vllm_registers_glm5next",
            "ok": "Glm5NextForConditionalGeneration" in registry_text,
        }
    )
    checks.append(
        {
            "name": "sm86_sparse_mla_is_explicitly_unsupported",
            "ok": "capability.major == 9" in registry_text
            and "capability.major in [9, 10]" in registry_text
            and "SM86 has no official sparse MLA backend" in registry_text,
        }
    )
    ok = all(check["ok"] for check in checks)
    return {
        "schema_version": 1,
        "doctor_mode": "cpu_static_no_device_probe",
        "package_integrity_ok": ok,
        "glm53_runtime_supported": False,
        "target_compute_capability": EXPECTED_ARCH,
        "block_reason_codes": lock["runtime"]["block_reason_codes"],
        "task_built_cuda_audit": image_audit,
        "checks": checks,
    }


def runtime_doctor(lock_path: str | Path) -> dict[str, Any]:
    """Future runtime gate. Calling this probes CUDA and is forbidden in CPU-only work."""
    lock = load_lock(lock_path)
    import torch

    visible = torch.cuda.device_count()
    capabilities = [".".join(map(str, torch.cuda.get_device_capability(i))) for i in range(visible)]
    all_sm86 = bool(capabilities) and all(value == EXPECTED_ARCH for value in capabilities)
    return {
        "schema_version": 1,
        "doctor_mode": "runtime_device_probe",
        "visible_gpu_count": visible,
        "compute_capabilities": capabilities,
        "sm86_only": all_sm86,
        "glm53_runtime_supported": False,
        "block_reason_codes": lock["runtime"]["block_reason_codes"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", required=True)
    parser.add_argument("--root", default="/opt/putpocket")
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    report = runtime_doctor(args.lock) if args.runtime else static_doctor(args.lock, args.root)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    if not report.get("package_integrity_ok", True):
        return 2
    if args.runtime:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
