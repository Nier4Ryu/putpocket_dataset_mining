from __future__ import annotations

import gzip
import csv
import hashlib
import json
import math
import re
import datetime as dt
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import ConfigError


VLLM_COMMIT = "4a3447d200e5aa428d68d1a00aa00f1a19a1a729"
BUILD_SOURCE_COMMIT = "173d5125b6c8e95bd2cc4e66d5240482064a78f3"
IMMUTABLE_BUNDLE_ROOT = "/home2/jslee202403/putpocket-builds/vllm/vllm-4a3447d200e5-sm90-cu1303-py312-torch2130-patch-fc2f3734-image-3869b846"
VLLM_WHEEL_SHA256 = "3c408df63c56e2a711116449d4324fcef5f2043de1b5c3dee4d3bf561908af52"
MODEL_ID = "nvidia/GLM-5.2-NVFP4"
MODEL_REVISION = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
INSTANCE_ID = "instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
FULL_LAYERS = (0, 1, 2, *range(6, 75, 4))
DECODE_STEPS = (0, 1, 8, 32)
LOCK_PATH = Path(__file__).resolve().parents[2] / "configs/cluster/glm52_vllm_diagnostic.lock.json"
_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECRET = re.compile(
    r"(?:password|credential|secret|api[_-]?key|(?:^|_)(?:hf|access|auth)[_-]?token(?:$|_))",
    re.IGNORECASE,
)


def load_lock(path: str | Path = LOCK_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError("VLLM_DIAGNOSTIC_LOCK_NOT_OBJECT")
    return value


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    if lock.get("schema_version") != 1 or lock.get("claim") != "raw_hook_pre_acceptance_diagnostic_only_not_quality_score":
        raise ConfigError("VLLM_DIAGNOSTIC_LOCK_IDENTITY_MISMATCH")
    _reject_secrets(lock)
    vllm = _mapping(lock.get("vllm"), "vllm")
    build = _mapping(lock.get("build"), "build")
    runtime = _mapping(lock.get("runtime"), "runtime")
    layout = _mapping(lock.get("model_layout"), "model_layout")
    selection = _mapping(lock.get("selection"), "selection")
    official = _mapping(lock.get("official_evaluation"), "official_evaluation")
    exact = (
        (vllm, "commit", VLLM_COMMIT),
        (runtime, "model_id", MODEL_ID),
        (runtime, "model_revision", MODEL_REVISION),
        (runtime, "tensor_parallel", 4),
        (runtime, "quantization", "modelopt_fp4"),
        (runtime, "linear_backend", "marlin"),
        (runtime, "sparse_attention_backend", "FLASHMLA_SPARSE"),
        (runtime, "native_prefill_logits", "fp8_fp4_mqa_logits"),
        (runtime, "native_prefill_topk", "top_k_per_row_prefill"),
        (runtime, "native_decode_logits", "fp8_fp4_paged_mqa_logits"),
        (runtime, "native_decode_topk", "cooperative_topk_sm90"),
        (runtime, "max_model_len", 4096),
        (runtime, "max_num_seqs", 1),
        (runtime, "prefix_caching", False),
        (runtime, "cpu_offload_gb", 0),
        (layout, "layers", 78),
        (layout, "full_indexer_layers", list(FULL_LAYERS)),
        (layout, "shared_indexer_count", 57),
        (layout, "index_topk", 2048),
        (selection, "instance_id", INSTANCE_ID),
        (selection, "serialized_prompt_token_count", 2071),
        (official, "selection_size", 1),
        (official, "score_eligible", False),
        (official, "full_selection_reachable", False),
        (build, "torch_cuda_arch_list", "9.0"),
        (build, "project_source_commit", BUILD_SOURCE_COMMIT),
        (build, "immutable_bundle_root", IMMUTABLE_BUNDLE_ROOT),
        (build, "vllm_wheel_sha256", VLLM_WHEEL_SHA256),
        (build, "cmake_cuda_architectures", "90"),
        (build, "vllm_target_device", "cuda"),
        (build, "vllm_use_precompiled", False),
        (build, "run_wheel_check", False),
        (build, "upstream_release_wheel_limit_mb", 500),
        (build, "wheel_size_exception_scope", "intentional_sm90_cuda13_source_build_only"),
        (build, "general_h200_compilation_allowed", False),
        (build, "h200_runtime_jit_scope", "native_first_use_deepgemm_dsa_only"),
        (build, "pinned_source_runtime_jit_required", True),
        (build, "runtime_jit_cache_reuse", False),
    )
    for container, key, expected in exact:
        if container.get(key) != expected:
            raise ConfigError(f"VLLM_DIAGNOSTIC_LOCK_MISMATCH:{key}")
    for key in ("speculative_mtp", "disaggregation"):
        if runtime.get(key) is not False:
            raise ConfigError(f"VLLM_RUNTIME_FORBIDDEN:{key}")
    for value in (build["project_source_commit"], vllm["commit"], runtime["model_revision"], selection["dataset_revision"], official["harness_commit"]):
        if not _SHA40.fullmatch(str(value)):
            raise ConfigError("IMMUTABLE_40_CHARACTER_REVISION_REQUIRED")
    if not _SHA256.fullmatch(str(build["vllm_wheel_sha256"])):
        raise ConfigError("IMMUTABLE_VLLM_WHEEL_SHA256_REQUIRED")
    expected_bundle_root = Path(str(build["shared_root"])) / str(build["bundle_key"])
    if not expected_bundle_root.is_absolute() or str(expected_bundle_root) != build["immutable_bundle_root"]:
        raise ConfigError("IMMUTABLE_BUNDLE_ROOT_MISMATCH")
    for key in (
        "patch_target_sha256",
        "patch_target_post_sha256",
        "build_patch_target_sha256",
        "build_patch_target_post_sha256",
        "patch_sha256",
        "instrumentation_sha256",
        "compiler_audit_sha256",
    ):
        if not _SHA256.fullmatch(str(vllm.get(key, ""))):
            raise ConfigError(f"VLLM_SOURCE_DIGEST_REQUIRED:{key}")
    if "@sha256:" not in str(build.get("base_image", "")):
        raise ConfigError("IMMUTABLE_BUILD_IMAGE_REQUIRED")
    if vllm.get("source_file_digests", {}).get(vllm["build_patch_target"]) != vllm["build_patch_target_sha256"]:
        raise ConfigError("VLLM_BUILD_PATCH_PREIMAGE_LOCK_MISMATCH")
    mapping = {int(k): int(v) for k, v in _mapping(layout.get("shared_layer_mapping"), "shared_layer_mapping").items()}
    if len(mapping) != 57 or set(mapping) & set(FULL_LAYERS):
        raise ConfigError("INDEXER_SHARED_MAPPING_INVALID")
    canonical = json.dumps({str(k): mapping[k] for k in sorted(mapping)}, separators=(",", ":"), sort_keys=True).encode()
    if hashlib.sha256(canonical).hexdigest() != layout.get("shared_layer_mapping_sha256"):
        raise ConfigError("INDEXER_SHARED_MAPPING_DIGEST_MISMATCH")
    return {
        "schema_version": 1,
        "status": "passed",
        "vllm_commit": VLLM_COMMIT,
        "build_source_commit": BUILD_SOURCE_COMMIT,
        "bundle_key": build["bundle_key"],
        "immutable_bundle_root": IMMUTABLE_BUNDLE_ROOT,
        "vllm_wheel_sha256": VLLM_WHEEL_SHA256,
        "model_revision": MODEL_REVISION,
        "instance_id": INSTANCE_ID,
        "full_layers": list(FULL_LAYERS),
    }


def validate_source_tree(project_root: str | Path, source_root: str | Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    validate_lock(lock)
    project = Path(project_root)
    source = Path(source_root)
    vllm = _mapping(lock["vllm"], "vllm")
    expected = {
        vllm["patch_target"]: vllm["patch_target_sha256"],
        vllm["build_patch_target"]: vllm["build_patch_target_sha256"],
        **vllm["source_file_digests"],
    }
    for relative, digest in expected.items():
        path = source / relative
        if not path.is_file() or file_sha256(path) != digest:
            raise ConfigError(f"VLLM_SOURCE_CONTEXT_DIGEST_MISMATCH:{relative}")
    patch = project / vllm["patch_path"]
    instrument = project / vllm["instrumentation_source"]
    compiler_audit = project / vllm["compiler_audit_source"]
    if file_sha256(patch) != vllm["patch_sha256"]:
        raise ConfigError("VLLM_PATCH_DIGEST_MISMATCH")
    if file_sha256(instrument) != vllm["instrumentation_sha256"]:
        raise ConfigError("VLLM_INSTRUMENTATION_DIGEST_MISMATCH")
    if file_sha256(compiler_audit) != vllm["compiler_audit_sha256"]:
        raise ConfigError("VLLM_COMPILER_AUDIT_DIGEST_MISMATCH")
    return {"schema_version": 1, "status": "passed", "source_files": sorted(expected)}


def validate_patched_tree(source_root: str | Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    source = Path(source_root)
    vllm = _mapping(lock["vllm"], "vllm")
    if file_sha256(source / vllm["patch_target"]) != vllm["patch_target_post_sha256"]:
        raise ConfigError("VLLM_PATCHED_TARGET_DIGEST_MISMATCH")
    if file_sha256(source / vllm["build_patch_target"]) != vllm["build_patch_target_post_sha256"]:
        raise ConfigError("VLLM_PATCHED_BUILD_TARGET_DIGEST_MISMATCH")
    if file_sha256(source / vllm["instrumentation_destination"]) != vllm["instrumentation_sha256"]:
        raise ConfigError("VLLM_PATCHED_INSTRUMENTATION_DIGEST_MISMATCH")
    return {"schema_version": 1, "status": "passed"}


def validate_build_manifest(manifest: Mapping[str, Any], bundle_root: str | Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    validate_lock(lock)
    build = lock["build"]
    if manifest.get("schema_version") != 1 or manifest.get("status") != "SUCCESS":
        raise ConfigError("BUILD_MANIFEST_NOT_SUCCESSFUL")
    exact = {
        "project_commit": build["project_source_commit"],
        "vllm_commit": VLLM_COMMIT,
        "bundle_key": build["bundle_key"],
        "patch_sha256": lock["vllm"]["patch_sha256"],
        "patch_target_post_sha256": lock["vllm"]["patch_target_post_sha256"],
        "build_patch_target_post_sha256": lock["vllm"]["build_patch_target_post_sha256"],
        "instrumentation_sha256": lock["vllm"]["instrumentation_sha256"],
        "compiler_audit_sha256": lock["vllm"]["compiler_audit_sha256"],
        "base_image": build["base_image"],
        "python": build["python"],
        "torch": build["torch"],
        "cuda": build["cuda"],
        "torch_cuda_arch_list": "9.0",
        "cmake_cuda_architectures": "90",
        "vllm_target_device": "cuda",
        "vllm_use_precompiled": False,
        "general_h200_compilation_allowed": False,
        "h200_runtime_jit_scope": "native_first_use_deepgemm_dsa_only",
        "pinned_source_runtime_jit_required": True,
        "runtime_jit_cache_reuse": False,
    }
    for key, expected in exact.items():
        if manifest.get(key) != expected:
            raise ConfigError(f"BUILD_MANIFEST_TARGET_MISMATCH:{key}")
    if manifest.get("prebuilt_vllm_wheel_used") is not False or manifest.get("built_from_scratch") is not True:
        raise ConfigError("PREBUILT_VLLM_SUBSTITUTION_FORBIDDEN")
    wheel_policy = _mapping(manifest.get("wheel_release_policy"), "wheel_release_policy")
    wheel_policy_exact = {
        "schema_version": 1,
        "run_wheel_check": False,
        "upstream_release_wheel_limit_mb": build["upstream_release_wheel_limit_mb"],
        "exception_scope": build["wheel_size_exception_scope"],
    }
    for key, expected in wheel_policy_exact.items():
        if wheel_policy.get(key) != expected:
            raise ConfigError(f"BUILD_MANIFEST_WHEEL_POLICY_MISMATCH:{key}")
    if manifest.get("runtime_gate") != "ALLOW_NATIVE_FIRST_USE_JIT_WITH_RUN_LOCAL_AUDIT":
        raise ConfigError("BUILD_MANIFEST_RUNTIME_JIT_GATE_MISSING")
    if "sm_90" not in manifest.get("compiled_arch_evidence", []):
        raise ConfigError("BUILD_MANIFEST_SM90_EVIDENCE_MISSING")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(manifest.get("runtime_image_id", ""))):
        raise ConfigError("BUILD_MANIFEST_RUNTIME_IMAGE_ID_INVALID")
    for label in ("build_environment", "runtime_environment"):
        identity = _mapping(manifest.get(label), label)
        if identity.get("python_major_minor") != "3.12" or identity.get("torch_base") != "2.13.0" or identity.get("torch_cuda") != "13.0":
            raise ConfigError(f"BUILD_MANIFEST_ENVIRONMENT_IDENTITY_MISMATCH:{label}")
        packages = identity.get("resolved_packages")
        if not isinstance(packages, list) or not packages or packages != sorted(packages):
            raise ConfigError(f"BUILD_MANIFEST_RESOLVED_PACKAGES_INVALID:{label}")
    runtime_identity = _mapping(manifest["runtime_environment"], "runtime_environment")
    try:
        transformers_version = tuple(int(value) for value in str(runtime_identity["transformers"]).split(".")[:2])
    except (KeyError, ValueError) as exc:
        raise ConfigError("BUILD_MANIFEST_TRANSFORMERS_VERSION_INVALID") from exc
    if transformers_version < (5, 3) or not str(runtime_identity.get("vllm", "")):
        raise ConfigError("BUILD_MANIFEST_RUNTIME_PACKAGE_IDENTITY_MISMATCH")
    root = Path(bundle_root)
    files = _mapping(manifest.get("files"), "files")
    required = {"runtime_image_tar", "vllm_wheel", "source_bundle"}
    if set(files) != required:
        raise ConfigError("BUILD_MANIFEST_FILE_SET_MISMATCH")
    provenance = _mapping(manifest.get("provenance_files"), "provenance_files")
    required_provenance = {
        "source_preflight.json", "source_post_patch.json", "build-wheel-image.log",
        "compiled_arches.txt", "wheel_artifact.json", "build_environment.json", "build_nvcc.txt",
        "build-runtime-image.log", "runtime_environment.json", "runtime_nvcc.txt",
    }
    if set(provenance) != required_provenance:
        raise ConfigError("BUILD_MANIFEST_PROVENANCE_FILE_SET_MISMATCH")
    wheel_entry = _mapping(files["vllm_wheel"], "files.vllm_wheel")
    if wheel_entry.get("sha256") != build["vllm_wheel_sha256"]:
        raise ConfigError("IMMUTABLE_VLLM_WHEEL_SHA256_MISMATCH")
    if (
        wheel_policy.get("wheel_path") != wheel_entry.get("path")
        or wheel_policy.get("wheel_bytes") != wheel_entry.get("bytes")
        or wheel_policy.get("wheel_sha256") != wheel_entry.get("sha256")
    ):
        raise ConfigError("BUILD_MANIFEST_WHEEL_ARTIFACT_MISMATCH")
    checksum_lines: list[str] = []
    for name, item in [*files.items(), *provenance.items()]:
        entry = _mapping(item, f"files.{name}")
        relative = str(entry.get("path", ""))
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ConfigError("BUILD_MANIFEST_UNSAFE_PATH")
        path = root / relative
        if not path.is_file() or file_sha256(path) != entry.get("sha256") or path.stat().st_size != entry.get("bytes"):
            raise ConfigError(f"BUILD_BUNDLE_DIGEST_MISMATCH:{name}")
        checksum_lines.append(f"{entry['sha256']}  {relative}\n")
    try:
        wheel_artifact = json.loads((root / provenance["wheel_artifact.json"]["path"]).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, TypeError) as exc:
        raise ConfigError("BUILD_MANIFEST_WHEEL_ARTIFACT_INVALID") from exc
    if wheel_artifact != dict(wheel_policy):
        raise ConfigError("BUILD_MANIFEST_WHEEL_ARTIFACT_MISMATCH")
    if (root / "SHA256SUMS").read_text(encoding="utf-8") != "".join(sorted(checksum_lines, key=lambda line: line.split("  ", 1)[1])):
        raise ConfigError("BUILD_BUNDLE_SHA256SUMS_MISMATCH")
    if not (root / "SUCCESS").is_file():
        raise ConfigError("BUILD_BUNDLE_SUCCESS_MARKER_MISSING")
    return {
        "schema_version": 1,
        "status": "passed",
        "bundle_key": build["bundle_key"],
        "immutable_build_source_commit": manifest["project_commit"],
        "expected_immutable_bundle_root": build["immutable_bundle_root"],
        "vllm_wheel_sha256": wheel_entry["sha256"],
    }


def validate_source_identities(
    *,
    pinned_build_source_commit: str,
    expected_build_source_commit: str,
    runtime_source_commit: str,
    observed_runtime_source_commit: str,
    wrapper_source_commit: str,
    allow_runtime_source_split: bool,
) -> dict[str, Any]:
    identities = {
        "pinned_build_source_commit": pinned_build_source_commit,
        "expected_build_source_commit": expected_build_source_commit,
        "runtime_source_commit": runtime_source_commit,
        "observed_runtime_source_commit": observed_runtime_source_commit,
        "wrapper_source_commit": wrapper_source_commit,
    }
    if not all(_SHA40.fullmatch(value) for value in identities.values()):
        raise ConfigError("SOURCE_PROVENANCE_FULL_SHA_REQUIRED")
    if expected_build_source_commit != pinned_build_source_commit:
        raise ConfigError("IMMUTABLE_BUILD_SOURCE_COMMIT_MISMATCH")
    if observed_runtime_source_commit != runtime_source_commit:
        raise ConfigError("RUNTIME_SOURCE_COMMIT_MISMATCH")
    if wrapper_source_commit != runtime_source_commit:
        raise ConfigError("WRAPPER_SOURCE_COMMIT_MISMATCH")
    split = runtime_source_commit != expected_build_source_commit
    if split and allow_runtime_source_split is not True:
        raise ConfigError("RUNTIME_BUILD_SOURCE_SPLIT_NOT_EXPLICITLY_AUTHORIZED")
    if not split and allow_runtime_source_split is True:
        raise ConfigError("RUNTIME_BUILD_SOURCE_SPLIT_AUTHORIZATION_UNNECESSARY")
    return {
        "schema_version": 1,
        "status": "passed",
        "immutable_build_source_commit": expected_build_source_commit,
        "runtime_source_commit": runtime_source_commit,
        "observed_runtime_source_commit": observed_runtime_source_commit,
        "wrapper_source_commit": wrapper_source_commit,
        "runtime_wrapper_source_commit": runtime_source_commit,
        "source_split": split,
        "source_split_explicitly_authorized": split and allow_runtime_source_split,
        "source_split_authorization": "explicit_renderer_flag" if split else "not_required",
    }


def validate_source_provenance(
    manifest: Mapping[str, Any],
    bundle_root: str | Path,
    lock: Mapping[str, Any],
    *,
    expected_build_source_commit: str,
    runtime_source_commit: str,
    observed_runtime_source_commit: str,
    wrapper_source_commit: str,
    allow_runtime_source_split: bool,
) -> dict[str, Any]:
    validate_lock(lock)
    pinned_build = str(lock["build"]["project_source_commit"])
    provenance = validate_source_identities(
        pinned_build_source_commit=pinned_build,
        expected_build_source_commit=expected_build_source_commit,
        runtime_source_commit=runtime_source_commit,
        observed_runtime_source_commit=observed_runtime_source_commit,
        wrapper_source_commit=wrapper_source_commit,
        allow_runtime_source_split=allow_runtime_source_split,
    )
    bundle = validate_build_manifest(manifest, bundle_root, lock)
    if manifest.get("project_commit") != expected_build_source_commit:
        raise ConfigError("IMMUTABLE_BUILD_SOURCE_COMMIT_MISMATCH")
    return {
        **provenance,
        "bundle_key": bundle["bundle_key"],
        "expected_immutable_bundle_root": bundle["expected_immutable_bundle_root"],
        "vllm_wheel_sha256": bundle["vllm_wheel_sha256"],
    }


def validate_model_config(config: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    runtime, layout = lock["runtime"], lock["model_layout"]
    architectures = config.get("architectures")
    if architectures != [runtime["architecture"]] or config.get("model_type") != runtime["model_type"]:
        raise ConfigError("MODEL_ARCHITECTURE_MISMATCH")
    layers = config.get("num_hidden_layers", config.get("n_layer"))
    if layers != layout["layers"] or config.get("index_topk") != layout["index_topk"]:
        raise ConfigError("MODEL_LAYER_OR_TOPK_MISMATCH")
    representation: str
    pattern = config.get("index_topk_pattern")
    indexer_types = config.get("indexer_types")
    if isinstance(pattern, str) and len(pattern) == 78:
        observed = [index for index, value in enumerate(pattern) if value.upper() == "F"]
        shared = sum(value.upper() == "S" for value in pattern)
        representation = "index_topk_pattern"
    elif isinstance(indexer_types, list) and len(indexer_types) == 78:
        observed = [index for index, value in enumerate(indexer_types) if str(value).lower() == "full"]
        shared = sum(str(value).lower() == "shared" for value in indexer_types)
        representation = "indexer_types"
    else:
        raise ConfigError("MODEL_INDEX_PATTERN_UNREPRESENTED")
    if observed != list(FULL_LAYERS) or shared != 57:
        raise ConfigError("MODEL_INDEX_PATTERN_MISMATCH")
    quant = config.get("quantization_config", {})
    if not isinstance(quant, Mapping) or str(quant.get("quant_method", "")).lower() not in {"modelopt", "modelopt_fp4"}:
        raise ConfigError("MODEL_QUANTIZATION_CONFIG_MISMATCH")
    if str(quant.get("quant_algo", "")).upper() not in {"NVFP4", "W4A16_NVFP4"}:
        raise ConfigError("MODEL_NVFP4_ALGORITHM_MISMATCH")
    return {"schema_version": 1, "status": "passed", "full_layers": observed, "normalized_from": representation}


def validate_inventory_csv(csv_text: str, listing: str) -> dict[str, Any]:
    rows = list(csv.DictReader(csv_text.splitlines()))
    if len(rows) != 4:
        raise ConfigError(f"GPU_INVENTORY_COUNT_MISMATCH:{len(rows)}")
    uuids: list[str] = []
    for row in rows:
        uuid, name = row.get("uuid", "").strip(), row.get("name", "").strip()
        try:
            total = int(float(row.get("memory_total_mib", "")))
            free = int(float(row.get("memory_free_mib", "")))
        except ValueError as exc:
            raise ConfigError("GPU_INVENTORY_MEMORY_INVALID") from exc
        mig, capability = row.get("mig_mode", "").strip().lower(), row.get("compute_capability", "").strip()
        if not uuid.startswith("GPU-") or uuid in uuids:
            raise ConfigError("GPU_UUID_INVALID_OR_DUPLICATE")
        if "H200" not in name.upper():
            raise ConfigError("GPU_TYPE_MISMATCH")
        if not 140_000 <= total <= 146_000 or not 0 < free <= total:
            raise ConfigError("GPU_FULL_141GB_MEMORY_CLASS_MISMATCH")
        if mig not in {"disabled", "n/a", "not supported"} or re.search(r"\bMIG\s+[0-9]+g\.", listing, re.I):
            raise ConfigError("MIG_ENABLED")
        if capability not in {"9", "9.0"}:
            raise ConfigError("GPU_SM90_REQUIRED")
        uuids.append(uuid)
    return {"schema_version": 1, "status": "passed", "gpu_count": 4, "gpu_uuids": uuids, "mig": "disabled", "memory_class": "full_141gb", "compute_capability": "9.0"}


def validate_trace_equivalence(off: Mapping[str, Any], on: Mapping[str, Any], *, off_ns: int, on_ns: int) -> dict[str, Any]:
    off_value = _completion(off)
    on_value = _completion(on)
    if off_value["token_ids"] != on_value["token_ids"]:
        raise ConfigError("TRACE_OUTPUT_TOKEN_ID_MISMATCH")
    return {
        "schema_version": 1,
        "status": "passed",
        "cache_isolation": "prefix_caching_disabled_and_reset_prefix_cache_before_each_serial_request",
        "same_live_server": True,
        "output_token_count": len(off_value["token_ids"]),
        "output_token_ids_sha256": off_value["sha256"],
        "off_duration_ns": off_ns,
        "on_duration_ns": on_ns,
        "instrumentation_overhead_ns": on_ns - off_ns,
        "off": off_value,
        "on": on_value,
    }


def build_runtime_jit_manifest(
    cache_root: str | Path,
    audit_log: str | Path,
    *,
    started_utc: str,
    completed_utc: str,
    build_source_commit: str,
    runtime_source_commit: str,
    observed_runtime_source_commit: str,
    wrapper_source_commit: str,
    allow_runtime_source_split: bool,
    runtime_image_id: str,
    build_manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    source_provenance = validate_source_provenance(
        build_manifest,
        Path(str(build_manifest["_bundle_root"])),
        lock,
        expected_build_source_commit=build_source_commit,
        runtime_source_commit=runtime_source_commit,
        observed_runtime_source_commit=observed_runtime_source_commit,
        wrapper_source_commit=wrapper_source_commit,
        allow_runtime_source_split=allow_runtime_source_split,
    )
    try:
        started = dt.datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
        completed = dt.datetime.fromisoformat(completed_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConfigError("RUNTIME_JIT_TIMESTAMP_INVALID") from exc
    if (
        completed < started
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", runtime_image_id)
    ):
        raise ConfigError("RUNTIME_JIT_PROVENANCE_IDENTITY_INVALID")
    root = Path(cache_root)
    if not root.is_absolute() or not root.is_dir():
        raise ConfigError("RUNTIME_JIT_CACHE_ROOT_INVALID")
    audit_path = Path(audit_log)
    if not audit_path.is_file():
        raise ConfigError("RUNTIME_JIT_COMPILER_AUDIT_MISSING")
    records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records or not any(record.get("tool") == "nvcc" for record in records):
        raise ConfigError("RUNTIME_JIT_NVCC_COMMAND_NOT_PROVEN")
    allowed_tools = {"nvcc", "ptxas", "cc", "gcc", "c++", "g++"}
    forbidden = re.compile(r"(?:setup\.py|bdist_wheel|pip\s+install|git\s+(?:clone|fetch)|cmake\s+--build|/project/(?:src|scripts))", re.I)
    for record in records:
        if record.get("schema_version") != 1 or record.get("tool") not in allowed_tools:
            raise ConfigError("RUNTIME_JIT_COMPILER_TOOL_NOT_ALLOWLISTED")
        command = " ".join(str(value) for value in record.get("argv", []))
        try:
            event_time = dt.datetime.fromisoformat(str(record.get("timestamp_utc", "")).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ConfigError("RUNTIME_JIT_AUDIT_TIMESTAMP_INVALID") from exc
        if event_time < started or event_time > completed:
            raise ConfigError("RUNTIME_JIT_AUDIT_OUTSIDE_RECORDED_WINDOW")
        if forbidden.search(command):
            raise ConfigError("GENERAL_PROJECT_COMPILATION_DETECTED_ON_H200")
        if not any(marker in command.lower() for marker in ("deep_gemm", "deepgemm", "flashinfer", "triton")):
            raise ConfigError("NON_NATIVE_RUNTIME_JIT_COMMAND_DETECTED")
    files: list[dict[str, Any]] = []
    components: set[str] = set()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        lowered = relative.lower()
        if "deep_gemm" in lowered or "deepgemm" in lowered:
            components.add("deep_gemm_native_dsa")
        elif "flashinfer" in lowered:
            components.add("flashinfer_native")
        elif "triton" in lowered:
            components.add("triton_native")
        elif "torchinductor" in lowered:
            components.add("torchinductor_native")
        elif lowered.startswith("cuda/"):
            components.add("cuda_driver_cache")
        elif lowered.startswith(("vllm/", "tmp/")):
            components.add("vllm_startup_native")
        else:
            raise ConfigError(f"RUNTIME_JIT_CACHE_COMPONENT_NOT_ALLOWLISTED:{relative}")
        files.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    if not files or "deep_gemm_native_dsa" not in components:
        raise ConfigError("NATIVE_DEEPGEMM_JIT_CACHE_NOT_PROVEN")
    return {
        "schema_version": 1,
        "status": "passed",
        "scope": "native_first_use_deepgemm_dsa_only",
        "cache_reuse": False,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "project_commit": runtime_source_commit,
        "build_source_commit": build_source_commit,
        "runtime_source_commit": runtime_source_commit,
        "wrapper_source_commit": wrapper_source_commit,
        "source_split": source_provenance["source_split"],
        "source_provenance": source_provenance,
        "vllm_commit": lock["vllm"]["commit"],
        "patch_sha256": lock["vllm"]["patch_sha256"],
        "build_image": lock["build"]["base_image"],
        "runtime_image_id": runtime_image_id,
        "torch": lock["build"]["torch"],
        "cuda": lock["build"]["cuda"],
        "sm": "90",
        "components": sorted(components),
        "compiler_commands": records,
        "audit_log_sha256": file_sha256(audit_path),
        "files": files,
    }


def validate_capture_records(records: Iterable[Mapping[str, Any]], *, output_token_count: int, lock: Mapping[str, Any]) -> dict[str, Any]:
    values = list(records)
    existing = [step for step in DECODE_STEPS if step < max(0, output_token_count - 1)]
    expected = {(rank, layer, "prefill_last_query") for rank in range(4) for layer in FULL_LAYERS}
    expected |= {(rank, layer, f"decode_{step}") for rank in range(4) for layer in FULL_LAYERS for step in existing}
    observed: set[tuple[int, int, str]] = set()
    for record in values:
        _reject_secrets(record)
        record_payload = dict(record)
        recorded_sha = record_payload.pop("record_sha256", None)
        encoded = (json.dumps(record_payload, separators=(",", ":"), sort_keys=True) + "\n").encode()
        if recorded_sha != hashlib.sha256(encoded).hexdigest():
            raise ConfigError("CAPTURE_RECORD_DIGEST_MISMATCH")
        if any(key in record for key in ("prompt", "messages", "problem_statement", "raw_prompt")):
            raise ConfigError("CAPTURE_RAW_PROMPT_FORBIDDEN")
        rank, layer, sample = record.get("rank"), record.get("layer"), record.get("sample_point")
        key = (rank, layer, sample)
        if key in observed:
            raise ConfigError("CAPTURE_DUPLICATE_COVERAGE_CELL")
        observed.add(key)
        if rank not in range(4) or layer not in FULL_LAYERS or key not in expected:
            raise ConfigError("CAPTURE_UNEXPECTED_COVERAGE_CELL")
        if record.get("trace_mode") != "ON" or record.get("instance_id") != INSTANCE_ID or record.get("topk") != 2048:
            raise ConfigError("CAPTURE_IDENTITY_MISMATCH")
        if record.get("full_indexer_layers") != list(FULL_LAYERS):
            raise ConfigError("CAPTURE_FULL_LAYER_LAYOUT_MISMATCH")
        if record.get("shared_layer_mapping") != lock["model_layout"]["shared_layer_mapping"]:
            raise ConfigError("CAPTURE_SHARED_LAYER_MAPPING_MISMATCH")
        if record.get("shared_layer_mapping_sha256") != lock["model_layout"]["shared_layer_mapping_sha256"]:
            raise ConfigError("CAPTURE_SHARED_LAYER_MAPPING_DIGEST_MISMATCH")
        revisions = record.get("revisions")
        required_revisions = {
            "model": MODEL_REVISION,
            "vllm": VLLM_COMMIT,
            "patch_sha256": lock["vllm"]["patch_sha256"],
            "build_image": lock["build"]["base_image"],
            "bundle_key": lock["build"]["bundle_key"],
        }
        if not isinstance(revisions, Mapping) or any(revisions.get(name) != value for name, value in required_revisions.items()):
            raise ConfigError("CAPTURE_REVISION_PROVENANCE_MISMATCH")
        if not _SHA40.fullmatch(str(revisions.get("project", ""))) or not str(revisions.get("runtime_image_id", "")).startswith("sha256:"):
            raise ConfigError("CAPTURE_RUNTIME_PROVENANCE_INVALID")
        if record.get("native_topk_backend") not in {"top_k_per_row_prefill", "cooperative_topk_sm90"}:
            raise ConfigError("CAPTURE_NATIVE_BACKEND_AMBIGUOUS")
        raw, ids, scores = record.get("raw_scores"), record.get("selected_ids"), record.get("selected_scores")
        context = record.get("context_length")
        if not isinstance(raw, list) or len(raw) != context or not _finite(raw):
            raise ConfigError("CAPTURE_RAW_VECTOR_INVALID")
        if not isinstance(ids, list) or len(ids) != 2048 or len(set(ids)) != 2048:
            raise ConfigError("CAPTURE_SELECTED_IDS_INVALID")
        if not isinstance(scores, list) or len(scores) != 2048 or not _finite(scores):
            raise ConfigError("CAPTURE_SELECTED_SCORES_INVALID")
        if any(not isinstance(index, int) or index < 0 or index >= context for index in ids):
            raise ConfigError("CAPTURE_CAUSAL_BOUNDS_INVALID")
        if any(float(raw[index]) != float(score) for index, score in zip(ids, scores, strict=True)):
            raise ConfigError("CAPTURE_SELECTED_SCORE_ALIGNMENT_INVALID")
        selected = set(ids)
        unselected = set(range(context)) - selected
        if unselected and min(float(raw[index]) for index in selected) < max(float(raw[index]) for index in unselected):
            raise ConfigError("CAPTURE_TOPK_RAW_CONSISTENCY_INVALID")
    missing = expected - observed
    if missing:
        raise ConfigError(f"CAPTURE_COVERAGE_INCOMPLETE:{len(missing)}")
    return {
        "schema_version": 1,
        "status": "passed",
        "record_count": len(values),
        "sample_points": ["prefill_last_query", *(f"decode_{step}" for step in existing)],
        "full_layer_count": 21,
        "rank_count": 4,
        "coverage_cells": [
            {"rank": rank, "layer": layer, "sample_point": sample}
            for rank, layer, sample in sorted(observed)
        ],
        "shared_layer_mapping": lock["model_layout"]["shared_layer_mapping"],
    }


def compress_jsonl(raw_paths: Sequence[Path], output_root: Path) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    files = []
    for raw in sorted(raw_paths):
        target = output_root / f"{raw.name}.gz"
        payload = gzip.compress(raw.read_bytes(), compresslevel=9, mtime=0)
        partial = target.with_suffix(target.suffix + ".partial")
        partial.write_bytes(payload)
        partial.replace(target)
        files.append({"path": target.name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
    return files


def _completion(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ConfigError("COMPLETION_CARDINALITY_MISMATCH")
    choice = choices[0]
    text, ids = choice.get("text"), choice.get("token_ids")
    if not isinstance(text, str) or not text.strip() or not isinstance(ids, list) or not ids:
        raise ConfigError("COMPLETION_EMPTY_OR_TOKEN_IDS_MISSING")
    if not all(isinstance(value, int) and value >= 0 for value in ids):
        raise ConfigError("COMPLETION_TOKEN_IDS_INVALID")
    if re.search(r"(?:^|\W)(?:nan|[+-]?inf(?:inity)?)(?:$|\W)", text, re.I):
        raise ConfigError("COMPLETION_NONFINITE_TEXT")
    normalized = " ".join(text.split())
    terms = re.findall(r"\w+|[^\w\s]", normalized.lower())
    if len(terms) >= 16 and max(terms.count(term) for term in set(terms)) / len(terms) > 0.5:
        raise ConfigError("COMPLETION_REPETITION_GARBAGE")
    payload = json.dumps(ids, separators=(",", ":")).encode("ascii")
    return {
        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "normalized_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
        "token_ids": ids,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "finish_reason": choice.get("finish_reason"),
    }


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _finite(values: Sequence[Any]) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _SECRET.search(str(key)):
                raise ConfigError(f"SECRET_FIELD_FORBIDDEN:{key}")
            _reject_secrets(child)
    elif isinstance(value, list):
        for child in value:
            _reject_secrets(child)
