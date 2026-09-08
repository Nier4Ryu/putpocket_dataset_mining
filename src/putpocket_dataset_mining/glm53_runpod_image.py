"""Fail-closed image contract for the GLM-5.3 RunPod runtime.

Static validation never imports torch and is safe on a CPU-only build host.
Device inspection is confined to the explicit ``runtime-doctor`` command.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


VLLM_COMMIT = "9cd956c7e6cf54efa366b803cafa15ec6c2df827"
MODEL_ID = "Intel/GLM-5.3-Flash-W4A16-AutoRound"
MODEL_REVISION = "5eee1846f0321058ed73745f9aa16f2aaf0fc0a0"
MODEL_ARCHITECTURE = "Glm5NextForConditionalGeneration"
PROFILES = {
    "sm90": {
        "compute_capability": "9.0",
        "attention_backend": "FLASHINFER_MLA_SPARSE_SM90",
        "kv_cache_dtype": "fp8_e4m3",
        "block_size": 128,
    },
    "sm120": {
        "compute_capability": "12.0",
        "attention_backend": "FLASHINFER_MLA_SPARSE_SM120",
        "kv_cache_dtype": "fp8_ds_mla",
        "block_size": 512,
    },
}
VLLM_FLASH_ATTN_EXTENSIONS = {
    "vllm_fa2_extension": "_vllm_fa2_C",
    "vllm_fa3_extension": "_vllm_fa3_C",
}


class RunPodContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def has_elf_magic(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(4) == b"\x7fELF"


def locate_vllm_flash_attn_extensions(vllm_root: str | Path) -> dict[str, dict[str, Any]]:
    """Locate required extension files without importing vLLM, torch, or CUDA."""
    extension_root = Path(vllm_root) / "vllm_flash_attn"
    results: dict[str, dict[str, Any]] = {}
    for component, prefix in VLLM_FLASH_ATTN_EXTENSIONS.items():
        matches = sorted(extension_root.glob(f"{prefix}*.so"))
        results[component] = {
            "prefix": prefix,
            "search_path": str(extension_root / f"{prefix}*.so"),
            "path": str(matches[0]) if len(matches) == 1 else None,
            "matches": [str(path) for path in matches],
            "ok": len(matches) == 1 and has_elf_magic(matches[0]),
            "validation": "filesystem_and_elf_magic_only_no_shared_object_load",
        }
    return results


def require_vllm_flash_attn_extensions(
    components: dict[str, dict[str, Any]],
) -> None:
    missing = [name for name, result in components.items() if not result["ok"]]
    if missing:
        payload = {"missing": missing, "components": components}
        raise RunPodContractError(
            "VLLM_FLASH_ATTN_EXTENSION_MISSING: "
            + json.dumps(payload, sort_keys=True)
        )


def load_lock(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_lock(payload)
    return payload


def validate_lock(lock: dict[str, Any]) -> None:
    failures: list[str] = []
    selection = lock.get("selection", {})
    runtime = lock.get("runtime", {})
    hybrid = lock.get("hybrid_cache_contract", {})
    if lock.get("schema_version") != 1:
        failures.append("schema_version")
    expected_selection = {
        "repository": MODEL_ID,
        "revision": MODEL_REVISION,
        "architecture": MODEL_ARCHITECTURE,
        "weight_bytes": 181_472_931_028,
        "payload_bytes": 181_505_393_058,
        "quant_method": "auto-round",
        "runtime_quantization": "inc",
    }
    for key, expected in expected_selection.items():
        if selection.get(key) != expected:
            failures.append(f"selection.{key}")
    if selection.get("payload_bytes", 10**30) >= 200_000_000_000:
        failures.append("selection.payload_under_200GB")
    if runtime.get("vllm_commit") != VLLM_COMMIT:
        failures.append("runtime.vllm_commit")
    if runtime.get("torch_cuda_arch_list") != "9.0a 12.0a":
        failures.append("runtime.torch_cuda_arch_list")
    if runtime.get("cmake_cuda_architectures") != "90a;120a":
        failures.append("runtime.cmake_cuda_architectures")
    if runtime.get("profiles") != PROFILES:
        failures.append("runtime.profiles")
    expected_sm120_nope_adapter = {
        "model_query_width": 512,
        "kernel_query_width": 576,
        "zero_padding_width": 64,
        "cache_bytes_per_token": 656,
        "kernel_qk_rope_head_dim": 64,
        "kv_scale_format": "arbitrary_fp32",
        "active_topk_length_argument": "seq_lens",
        "model_index_topk": 2048,
        "kernel_topk_width": 2048,
        "kpool_tail_policy": "keep_valid_tail_drop_lowest_ranked_history",
        "attention_semantics": "unchanged_nope_zero_dot_product_tail",
    }
    if runtime.get("sm120_nope_query_adapter") != expected_sm120_nope_adapter:
        failures.append("runtime.sm120_nope_query_adapter")
    expected_cached_base_overlay = {
        "path": (
            "patches/vllm/9cd956c7e6cf54efa366b803cafa15ec6c2df827/"
            "glm53_sm120_cached_base_topk_abi.patch"
        ),
        "sha256": "44bd1e464f50f1b2bf2c7827816b8461b90c1838f9131a9272f2b28bfe522ac1",
        "preimage_sha256": "3b2ff18d2db7196f53c143acdbce146e0fd904ab4d797f0356c99a52c5823b50",
        "postimage_sha256": "c82a9697ea68a5039332e02e011afb5f07fc8a3fcc113012d552d2fc102edfb7",
    }
    if (
        lock.get("runtime_image_overlay", {}).get("cached_base_sm120_topk_abi")
        != expected_cached_base_overlay
    ):
        failures.append("runtime_image_overlay.cached_base_sm120_topk_abi")
    if runtime.get("default_gpu_count") != 4:
        failures.append("runtime.default_gpu_count")
    if runtime.get("weights_in_image") is not False:
        failures.append("runtime.weights_in_image")
    if runtime.get("silent_download_allowed") is not False:
        failures.append("runtime.silent_download_allowed")
    required_components = set(runtime.get("required_components", []))
    for component in (
        "vllm.vllm_flash_attn._vllm_fa2_C",
        "vllm.vllm_flash_attn._vllm_fa3_C",
    ):
        if component not in required_components:
            failures.append(f"runtime.required_components.{component}")
    if hybrid.get("mla_indexer_layers") != list(range(3, 45, 4)):
        failures.append("hybrid_cache_contract.mla_indexer_layers")
    if hybrid.get("kda_layer_count") != 34:
        failures.append("hybrid_cache_contract.kda_layer_count")
    if hybrid.get("index_kpool") != 4:
        failures.append("hybrid_cache_contract.index_kpool")
    stateful = lock.get("stateful_edit", {})
    for key, expected in {
        "default_enabled": False,
        "full_target_prefill": True,
        "donor_overwrite_after_native_write": True,
        "true_partial_prefill": False,
        "speedup_claim": False,
    }.items():
        if stateful.get(key) != expected:
            failures.append(f"stateful_edit.{key}")
    if failures:
        raise RunPodContractError(", ".join(failures))


def validate_model_metadata(model_path: str | Path, lock: dict[str, Any]) -> dict[str, Any]:
    root = Path(model_path)
    if not root.is_dir():
        raise RunPodContractError("MODEL_PATH_NOT_DIRECTORY")
    required = lock["selection"]["metadata_sha256"]
    checks: list[dict[str, Any]] = []
    for name, expected in sorted(required.items()):
        path = root / name
        actual = sha256_file(path) if path.is_file() else None
        checks.append({"path": name, "expected_sha256": expected, "actual_sha256": actual, "ok": actual == expected})
    config_path = root / "config.json"
    quant_path = root / "quantization_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    quant = json.loads(quant_path.read_text(encoding="utf-8")) if quant_path.is_file() else {}
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    shard_paths = [root / name for name in shard_names]
    complete_shards = len(shard_names) == lock["selection"]["weight_file_count"] and all(path.is_file() for path in shard_paths)
    weight_bytes = sum(path.stat().st_size for path in shard_paths if path.is_file())
    text = config.get("text_config", config)
    checks.extend(
        [
            {"name": "architecture", "ok": MODEL_ARCHITECTURE in config.get("architectures", [])},
            {"name": "layer_count", "ok": text.get("num_hidden_layers") == 45},
            {"name": "nope", "ok": text.get("qk_rope_head_dim") == 0 and text.get("mla_use_nope") is True},
            {"name": "index_kpool", "ok": text.get("index_kpool") == 4},
            {"name": "quantization", "ok": quant.get("quant_method") == "auto-round" and quant.get("bits") == 4},
            {"name": "weight_index_present", "ok": (root / "model.safetensors.index.json").is_file()},
            {"name": "weight_shards_complete", "count": len(shard_names), "ok": complete_shards},
            {"name": "weight_bytes", "actual": weight_bytes, "expected": lock["selection"]["weight_bytes"], "ok": weight_bytes == lock["selection"]["weight_bytes"]},
        ]
    )
    return {"ok": all(item["ok"] for item in checks), "model_path": str(root), "checks": checks}


def static_doctor(lock_path: str | Path, root: str | Path) -> dict[str, Any]:
    lock = load_lock(lock_path)
    package_root = Path(root)
    checks: list[dict[str, Any]] = []
    for artifact in lock.get("artifacts", []):
        path = package_root / artifact["path"]
        actual = sha256_file(path) if path.is_file() else None
        checks.append({"path": artifact["path"], "expected_sha256": artifact["sha256"], "actual_sha256": actual, "ok": actual == artifact["sha256"]})
    if package_root == Path("/opt/putpocket"):
        try:
            distribution = importlib.metadata.distribution("vllm")
            version = distribution.version
        except importlib.metadata.PackageNotFoundError:
            distribution = None
            version = None
        checks.append({"name": "vllm_installed", "version": version, "ok": version is not None})
        vllm_root = (
            Path(distribution.locate_file("vllm"))
            if distribution is not None
            else Path("/__missing_vllm__")
        )
        extension_components = locate_vllm_flash_attn_extensions(vllm_root)
        for name, result in extension_components.items():
            checks.append({"name": name, **result})
        installed_hook = (
            Path(distribution.locate_file("vllm/model_executor/layers/glm53_stateful_edit_accuracy_ablation.py"))
            if distribution is not None
            else Path("/__missing_vllm_hook__")
        )
        expected_hook = package_root / "instrumentation/vllm/glm53_runpod_stateful_edit_accuracy_ablation.py"
        checks.append({"name": "installed_hook_matches_package", "ok": installed_hook.is_file() and expected_hook.is_file() and sha256_file(installed_hook) == sha256_file(expected_hook)})
        installed_sm120_backend = (
            Path(distribution.locate_file("vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"))
            if distribution is not None
            else Path("/__missing_vllm_sm120_backend__")
        )
        expected_sm120_backend_hash = lock["overlay"]["postimages"][
            "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
        ]
        checks.append(
            {
                "name": "installed_sm120_nope_adapter",
                "expected_sha256": expected_sm120_backend_hash,
                "actual_sha256": (
                    sha256_file(installed_sm120_backend)
                    if installed_sm120_backend.is_file()
                    else None
                ),
                "ok": installed_sm120_backend.is_file()
                and sha256_file(installed_sm120_backend)
                == expected_sm120_backend_hash,
            }
        )
        build_evidence = Path("/opt/vllm-build-evidence/sm90-sm120-cuda-audit.json")
        evidence = json.loads(build_evidence.read_text(encoding="utf-8")) if build_evidence.is_file() else {}
        observed = set(evidence.get("observed_architectures", []))
        checks.append({"name": "task_built_dual_arch_audit", "ok": {"90", "120"}.issubset(observed) and not evidence.get("out_of_scope_architectures") and evidence.get("all_task_built_cuda_payloads_in_scope") is True})
    return {
        "schema_version": 1,
        "mode": "cpu_static_no_device_probe",
        "package_integrity_ok": all(item["ok"] for item in checks),
        "runtime_gpu_validated": False,
        "profiles": PROFILES,
        "checks": checks,
    }


def runtime_doctor(lock_path: str | Path, profile: str, model_path: str | Path) -> dict[str, Any]:
    """RunPod-only gate. This function deliberately imports torch and probes CUDA."""
    lock = load_lock(lock_path)
    if profile not in PROFILES:
        raise RunPodContractError("RUNTIME_PROFILE_INVALID")
    try:
        distribution = importlib.metadata.distribution("vllm")
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    vllm_root = (
        Path(distribution.locate_file("vllm"))
        if distribution is not None
        else Path("/__missing_vllm__")
    )
    flash_attn_extensions = locate_vllm_flash_attn_extensions(vllm_root)
    require_vllm_flash_attn_extensions(flash_attn_extensions)

    import torch
    from vllm.model_executor.layers.quantization.inc.inc import INCConfig
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if profile == "sm90":
        from vllm.utils.flashinfer import has_flashinfer_sm90_nope_mla

        sparse_backend_available = has_flashinfer_sm90_nope_mla()
        backend_registered = hasattr(AttentionBackendEnum, "FLASHINFER_MLA_SPARSE_SM90")
    else:
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        sparse_backend_available = has_flashinfer_sparse_mla_sm120()
        backend_registered = hasattr(AttentionBackendEnum, "FLASHINFER_MLA_SPARSE_SM120")

    visible = torch.cuda.device_count()
    capabilities = [".".join(map(str, torch.cuda.get_device_capability(index))) for index in range(visible)]
    expected = PROFILES[profile]["compute_capability"]
    metadata = validate_model_metadata(model_path, lock)
    components = {
        **flash_attn_extensions,
        "backend_registered": backend_registered,
        "sparse_backend_available": sparse_backend_available,
        "inc_autoround_loader": callable(getattr(INCConfig, "override_quantization_method", None)),
        "native_indexer_logits": callable(fp8_fp4_mqa_logits),
        "flashkda_extension": importlib.util.find_spec("vllm._flashkda_C") is not None,
        "deepgemm_extension": importlib.util.find_spec("vllm.third_party.deep_gemm._C") is not None,
    }
    component_ok = all(
        value["ok"] if isinstance(value, dict) else value
        for value in components.values()
    )
    return {
        "schema_version": 1,
        "mode": "runtime_device_and_model_gate",
        "profile": profile,
        "visible_gpu_count": visible,
        "compute_capabilities": capabilities,
        "expected_compute_capability": expected,
        "topology_ok": visible == lock["runtime"]["default_gpu_count"] and all(value == expected for value in capabilities),
        "model_metadata": metadata,
        "components": components,
        "ok": visible == lock["runtime"]["default_gpu_count"] and all(value == expected for value in capabilities) and metadata["ok"] and component_ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("static", "runtime"), nargs="?", default="static")
    parser.add_argument("--lock", required=True)
    parser.add_argument("--root", default="/opt/putpocket")
    parser.add_argument("--profile", default=os.getenv("PUTPOCKET_GLM53_RUNTIME_PROFILE", "sm90"))
    parser.add_argument("--model-path", default=os.getenv("MODEL_PATH", "/models/glm53-flash-w4a16-autoround"))
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        report = static_doctor(args.lock, args.root) if args.command == "static" else runtime_doctor(args.lock, args.profile, args.model_path)
    except RunPodContractError as exc:
        report = {
            "schema_version": 1,
            "mode": "runtime_or_static_fail_closed",
            "ok": False,
            "error": str(exc),
        }
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report.get("package_integrity_ok", report.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
