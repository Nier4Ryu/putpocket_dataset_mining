from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

from .errors import ConfigError


MODEL_ID = "nvidia/GLM-5.2-NVFP4"
MODEL_REF = "main"
EXPECTED_ARCHITECTURE = "GlmMoeDsaForCausalLM"
EXPECTED_MODEL_TYPE = "glm_moe_dsa"
EXPECTED_LAYERS = 78
EXPECTED_FULL_INDEXERS = 21
EXPECTED_SHARED_INDEXERS = 57
EXPECTED_INDEX_TOPK = 2048
MIN_H200_MEMORY_MIB = 140_000
MAX_H200_MEMORY_MIB = 146_000
SOURCE_LOCK = Path(__file__).resolve().parents[2] / "configs" / "cluster" / "glm52_sglang_gate_sources.lock.json"
SITE_PROFILE = Path(__file__).resolve().parents[2] / "configs" / "cluster" / "sites" / "herdr_h200_sglang_gate.json"

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_DIRECTIVE = re.compile(r"^[A-Za-z0-9_.:@+,&|!%=/:-]+$")
_SECRET_KEYS = re.compile(r"(?:token|secret|password|credential|api[_-]?key)", re.IGNORECASE)


@dataclass(frozen=True)
class InventorySummary:
    uuids: tuple[str, ...]
    names: tuple[str, ...]
    total_memory_mib: tuple[int, ...]
    free_memory_mib: tuple[int, ...]
    compute_capabilities: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "passed",
            "gpu_count": len(self.uuids),
            "gpu_uuids": list(self.uuids),
            "gpu_names": list(self.names),
            "total_memory_mib": list(self.total_memory_mib),
            "free_memory_mib": list(self.free_memory_mib),
            "compute_capabilities": list(self.compute_capabilities),
            "mig": "disabled",
            "physical_full_gpu": True,
        }


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"Expected a JSON object in {path}")
    return value


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(target)


def validate_source_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    _reject_secret_fields(lock)
    if lock.get("schema_version") != 1:
        raise ConfigError("SGLang gate source lock schema_version must be 1")
    source = _mapping(lock.get("sglang_source"), "sglang_source")
    image = _mapping(lock.get("runtime_image"), "runtime_image")
    model = _mapping(lock.get("model"), "model")
    dependencies = _mapping(lock.get("dependency_contract"), "dependency_contract")
    if source.get("repository") != "https://github.com/sgl-project/sglang.git":
        raise ConfigError("SGLang source must use the official repository")
    if not _SHA40.fullmatch(str(source.get("commit", ""))):
        raise ConfigError("SGLang source commit must be a full Git SHA")
    if image.get("human_tag") != "lmsysorg/sglang:latest":
        raise ConfigError("SGLang runtime human tag must name the official image")
    if not _SHA256.fullmatch(str(image.get("linux_amd64_digest", ""))):
        raise ConfigError("SGLang runtime image must have an immutable linux/amd64 digest")
    if model.get("id") != MODEL_ID or model.get("requested_revision") != MODEL_REF:
        raise ConfigError("Model source contract changed")
    if model.get("resolved_revision_policy") != "runtime_full_40_char_hf_commit":
        raise ConfigError("Model revision must be resolved to a full HF commit at runtime")
    minimum = str(dependencies.get("transformers_minimum", ""))
    if _version_tuple(minimum) < (5, 3):
        raise ConfigError("transformers minimum must be at least 5.3")
    return {
        "schema_version": 1,
        "status": "passed",
        "sglang_source_commit": source["commit"],
        "runtime_image": f"{image['repository']}@{image['linux_amd64_digest']}",
        "runtime_image_human_tag": image["human_tag"],
        "model_id": MODEL_ID,
        "model_requested_revision": MODEL_REF,
    }


def validate_inventory_rows(rows: Iterable[Mapping[str, Any]], *, mig_listing: str = "") -> InventorySummary:
    materialized = list(rows)
    if len(materialized) != 4:
        raise ConfigError(f"ALLOCATION_GPU_COUNT_MISMATCH: expected 4 GPUs, observed {len(materialized)}")
    uuids: list[str] = []
    names: list[str] = []
    totals: list[int] = []
    frees: list[int] = []
    capabilities: list[str] = []
    for index, row in enumerate(materialized):
        uuid = str(row.get("uuid", "")).strip()
        name = str(row.get("name", "")).strip()
        mig = str(row.get("mig_mode", "")).strip().lower()
        capability = str(row.get("compute_capability", "")).strip()
        try:
            total = int(float(str(row.get("memory_total_mib", ""))))
            free = int(float(str(row.get("memory_free_mib", ""))))
        except ValueError as exc:
            raise ConfigError(f"GPU_INVENTORY_INVALID: non-numeric memory for GPU {index}") from exc
        if not uuid.startswith("GPU-") or uuid in uuids:
            raise ConfigError("GPU_INVENTORY_INVALID: physical GPU UUIDs must be unique GPU-* values")
        if "H200" not in name.upper():
            raise ConfigError(f"GPU_TYPE_MISMATCH: expected H200, observed {name!r}")
        if not (MIN_H200_MEMORY_MIB <= total <= MAX_H200_MEMORY_MIB):
            raise ConfigError(f"GPU_MEMORY_CLASS_MISMATCH: {total} MiB is not full 141GB-class H200 memory")
        if free <= 0 or free > total:
            raise ConfigError("GPU_INVENTORY_INVALID: free memory must be positive and no greater than total")
        if mig not in {"disabled", "n/a", "not supported"}:
            raise ConfigError(f"MIG_ENABLED: GPU {uuid} reports {row.get('mig_mode')!r}")
        if capability not in {"9.0", "9"}:
            raise ConfigError(f"GPU_ARCH_MISMATCH: H200 must report SM90, observed {capability!r}")
        uuids.append(uuid)
        names.append(name)
        totals.append(total)
        frees.append(free)
        capabilities.append("9.0")
    if re.search(r"\bMIG\s+[0-9]+g\.", mig_listing, re.IGNORECASE):
        raise ConfigError("MIG_ENABLED: nvidia-smi -L exposed a MIG device")
    return InventorySummary(tuple(uuids), tuple(names), tuple(totals), tuple(frees), tuple(capabilities))


def validate_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    architectures = config.get("architectures")
    if architectures != [EXPECTED_ARCHITECTURE]:
        raise ConfigError(f"MODEL_ARCHITECTURE_MISMATCH: expected {[EXPECTED_ARCHITECTURE]!r}")
    if config.get("model_type") != EXPECTED_MODEL_TYPE:
        raise ConfigError(f"MODEL_TYPE_MISMATCH: expected {EXPECTED_MODEL_TYPE}")
    if config.get("num_hidden_layers") != EXPECTED_LAYERS:
        raise ConfigError(f"LAYER_LAYOUT_MISMATCH: expected {EXPECTED_LAYERS} layers")
    indexers = config.get("indexer_types")
    if not isinstance(indexers, list) or len(indexers) != EXPECTED_LAYERS:
        raise ConfigError("INDEXER_LAYOUT_MISMATCH: indexer_types must explicitly contain 78 entries")
    full = sum(value == "full" for value in indexers)
    shared = sum(value == "shared" for value in indexers)
    if full != EXPECTED_FULL_INDEXERS or shared != EXPECTED_SHARED_INDEXERS or full + shared != len(indexers):
        raise ConfigError(f"INDEXER_LAYOUT_MISMATCH: expected 21 full/57 shared, observed {full}/{shared}")
    if config.get("index_topk") != EXPECTED_INDEX_TOPK:
        raise ConfigError(f"DSA_TOPK_MISMATCH: expected {EXPECTED_INDEX_TOPK}")
    quant = _mapping(config.get("quantization_config"), "quantization_config")
    if str(quant.get("quant_method", "")).lower() != "modelopt":
        raise ConfigError("QUANTIZATION_MISMATCH: expected ModelOpt")
    if str(quant.get("quant_algo", "")).upper() != "NVFP4":
        raise ConfigError("QUANTIZATION_MISMATCH: expected NVFP4")
    if quant.get("group_size") != 16:
        raise ConfigError("QUANTIZATION_MISMATCH: NVFP4 group_size must be 16")
    return {
        "schema_version": 1,
        "status": "passed",
        "architecture": EXPECTED_ARCHITECTURE,
        "model_type": EXPECTED_MODEL_TYPE,
        "layers": EXPECTED_LAYERS,
        "indexer_layout": {"full": full, "shared": shared},
        "index_topk": EXPECTED_INDEX_TOPK,
        "quantization": {"method": "modelopt", "algorithm": "NVFP4", "group_size": 16},
    }


def validate_capability_report(report: Mapping[str, Any]) -> dict[str, Any]:
    version = str(report.get("transformers_version", ""))
    if _version_tuple(version) < (5, 3):
        raise ConfigError(f"TRANSFORMERS_TOO_OLD: observed {version!r}, require >=5.3")
    required_imports = ("torch", "transformers", "sglang", "modelopt", "flashinfer", "flash_mla", "sgl_kernel")
    imports = _mapping(report.get("imports"), "imports")
    missing_imports = [name for name in required_imports if imports.get(name) is not True]
    if missing_imports:
        raise ConfigError(f"BACKEND_IMPORT_MISSING: {','.join(missing_imports)}")
    nccl_version = report.get("torch_nccl_version")
    if not report.get("torch_cuda_version") or nccl_version is None or nccl_version == "unavailable":
        raise ConfigError("GPU_RUNTIME_VERSION_UNAVAILABLE: container CUDA/NCCL versions must be recordable")
    required_symbols = (
        "ModelOptFp4Config",
        "ModelOptFp4LinearMethod",
        "ModelOptNvFp4FusedMoEMethod",
        "prepare_nvfp4_layer_for_marlin",
        "prepare_moe_nvfp4_layer_for_marlin",
        "marlin_w4a16",
        "hopper_marlin_selection",
        "glm_moe_dsa_runtime",
        "flashmla_sparse",
        "fa3",
        "sgl-kernel",
    )
    symbols = _mapping(report.get("symbols"), "symbols")
    missing_symbols = [name for name in required_symbols if symbols.get(name) is not True]
    if missing_symbols:
        raise ConfigError(f"BACKEND_CAPABILITY_MISSING: {','.join(missing_symbols)}")
    defaults = _mapping(report.get("server_defaults"), "server_defaults")
    expected_defaults = {
        "cpu_offload_gb": 0,
        "disaggregation_mode": None,
        "speculative_algorithm": None,
        "weight_cache_mode": "off",
    }
    for key, expected in expected_defaults.items():
        if defaults.get(key) != expected:
            raise ConfigError(f"UNSAFE_RUNTIME_DEFAULT: {key}={defaults.get(key)!r}, expected {expected!r}")
    controls = _mapping(report.get("server_controls"), "server_controls")
    expected_controls = {
        "quantization": "modelopt_fp4",
        "fp4_gemm_backend": "marlin",
        "moe_runner_backend": "marlin",
        "dsa_prefill_backend": "flashmla_sparse",
        "dsa_decode_backend": "fa3",
        "dsa_topk_backend": "sgl-kernel",
    }
    for key, expected in expected_controls.items():
        values = controls.get(key)
        if not isinstance(values, list) or expected not in values:
            raise ConfigError(f"BACKEND_CONTROL_MISSING: {key} cannot select {expected}")
    return {
        "schema_version": 1,
        "status": "passed",
        "transformers_version": version,
        "torch_cuda_version": report["torch_cuda_version"],
        "torch_nccl_version": report["torch_nccl_version"],
        "imports": required_imports,
        "backend_contract": expected_controls,
        "disabled_defaults": expected_defaults,
    }


def validate_server_info(info: Mapping[str, Any]) -> dict[str, Any]:
    _reject_nonfinite_values(info, "server_info")
    flat = _flatten_by_key(info)
    required: dict[str, Any] = {
        "tp_size": 4,
        "quantization": "modelopt_fp4",
        "fp4_gemm_backend": "marlin",
        "moe_runner_backend": "marlin",
        "dsa_prefill_backend": "flashmla_sparse",
        "dsa_decode_backend": "fa3",
        "dsa_topk_backend": "sgl-kernel",
        "context_length": 4096,
        "max_running_requests": 1,
        "cpu_offload_gb": 0,
        "weight_cache_mode": "off",
    }
    aliases = {"tp_size": ("tp_size", "tensor_parallel_size")}
    for expected_key, expected_value in required.items():
        candidates = aliases.get(expected_key, (expected_key,))
        observed = [value for candidate in candidates for value in flat.get(candidate, [])]
        if expected_value not in observed:
            raise ConfigError(f"RUNTIME_BACKEND_AMBIGUOUS: {expected_key} did not resolve to {expected_value!r}")
    for key in ("disaggregation_mode", "speculative_algorithm"):
        observed = flat.get(key, [])
        if observed and any(value not in {None, "null", "none", "None"} for value in observed):
            raise ConfigError(f"UNSAFE_RUNTIME_MODE: {key} must be disabled")
    return {"schema_version": 1, "status": "passed", "effective": required}


def validate_runtime_log(text: str) -> dict[str, Any]:
    required_groups = {
        "architecture": ("glm_moe_dsa", "GlmMoeDsaForCausalLM"),
        "quantization": ("modelopt_fp4",),
        "marlin": ("marlin",),
        "prefill": ("flashmla_sparse",),
        "decode": ("fa3",),
        "topk": ("sgl-kernel",),
    }
    lowered = text.lower()
    if re.search(r"(?:^|[^a-z0-9_])(?:nan|[+-]?inf(?:inity)?)(?:$|[^a-z0-9_])", lowered):
        raise ConfigError("RUNTIME_NONFINITE: runtime log contains a NaN or Inf value")
    missing = [label for label, choices in required_groups.items() if not any(value.lower() in lowered for value in choices)]
    if missing:
        raise ConfigError(f"RUNTIME_BACKEND_AMBIGUOUS: missing runtime log evidence for {','.join(missing)}")
    forbidden = {
        "OFFLOAD_DETECTED": (r"cpu offload[^\n]*(?:enabled|[1-9][0-9]*(?:\.[0-9]+)?)", r"nvme offload", r"weight cache[^\n]*(?:server|client)"),
        "SILENT_FALLBACK": (r"fall(?:ing)? back", r"fallback to", r"unsupported.+using"),
        "DENSE_ATTENTION_DETECTED": (r"dense attention", r"disable[^\n]*dsa"),
    }
    for failure, patterns in forbidden.items():
        if any(re.search(pattern, lowered) for pattern in patterns):
            raise ConfigError(f"{failure}: forbidden runtime log evidence")
    return {"schema_version": 1, "status": "passed", "evidence": sorted(required_groups)}


def validate_sentinel_response(response: Mapping[str, Any]) -> dict[str, Any]:
    _reject_nonfinite_values(response, "sentinel_response")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ConfigError("SENTINEL_INVALID: expected exactly one completion choice")
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        raise ConfigError("SENTINEL_INVALID: completion message is missing")
    raw = message.get("content")
    if not isinstance(raw, str):
        raise ConfigError("SENTINEL_INVALID: completion content is not text")
    normalized = " ".join(raw.split())
    if not normalized:
        raise ConfigError("SENTINEL_EMPTY: normalized output is empty")
    if re.search(r"(?:^|\W)(?:nan|[+-]?inf(?:inity)?)(?:$|\W)", normalized, re.IGNORECASE):
        raise ConfigError("SENTINEL_NONFINITE: output contains NaN or Inf")
    tokens = re.findall(r"\w+|[^\w\s]", normalized.lower())
    if len(tokens) >= 12:
        dominant = max(tokens.count(token) for token in set(tokens))
        if dominant / len(tokens) > 0.50:
            raise ConfigError("SENTINEL_REPETITION: one token dominates the output")
        for width in (2, 3, 4):
            chunks = [tuple(tokens[index : index + width]) for index in range(0, len(tokens) - width + 1, width)]
            if len(chunks) >= 4 and len(set(chunks)) <= len(chunks) // 2:
                raise ConfigError("SENTINEL_REPETITION: repeated token blocks dominate the output")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    raw_digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "status": "passed",
        "raw_output": raw,
        "raw_sha256": raw_digest,
        "normalized_output": normalized,
        "normalized_sha256": digest,
    }


def summarize_hbm(samples: Iterable[Mapping[str, Any]], expected_uuids: Iterable[str]) -> dict[str, Any]:
    expected = tuple(expected_uuids)
    grouped: dict[str, list[Mapping[str, Any]]] = {uuid: [] for uuid in expected}
    for sample in samples:
        uuid = str(sample.get("uuid", ""))
        if uuid in grouped:
            grouped[uuid].append(sample)
    if len(expected) != 4 or set(grouped) != set(expected) or any(not values for values in grouped.values()):
        raise ConfigError("HBM_EVIDENCE_INCOMPLETE: samples must cover the four allocated GPU UUIDs")
    devices = []
    for uuid in expected:
        values = grouped[uuid]
        totals = {int(float(str(value["memory_total_mib"]))) for value in values}
        if len(totals) != 1:
            raise ConfigError("HBM_EVIDENCE_INVALID: total memory changed during sampling")
        total = totals.pop()
        peak = max(int(float(str(value["memory_used_mib"]))) for value in values)
        min_free = min(int(float(str(value["memory_free_mib"]))) for value in values)
        headroom = min(total - peak, min_free)
        if peak <= 0:
            raise ConfigError("MODEL_NOT_RESIDENT: no positive HBM use was measured")
        if headroom <= 0:
            raise ConfigError("HBM_HEADROOM_NONPOSITIVE: all-resident load left no positive headroom")
        devices.append({"uuid": uuid, "total_mib": total, "peak_used_mib": peak, "minimum_free_mib": min_free, "headroom_mib": headroom})
    return {
        "schema_version": 1,
        "status": "passed",
        "all_resident": True,
        "offload": False,
        "devices": devices,
        "minimum_headroom_mib": min(device["headroom_mib"] for device in devices),
    }


def parse_inventory_csv(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            raise ConfigError("GPU_INVENTORY_INVALID: expected seven CSV columns")
        rows.append(
            {
                "index": fields[0],
                "uuid": fields[1],
                "name": fields[2],
                "memory_total_mib": fields[3],
                "memory_free_mib": fields[4],
                "mig_mode": fields[5],
                "compute_capability": fields[6],
            }
        )
    return rows


def parse_hbm_csv(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("timestamp,"):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            raise ConfigError("HBM_EVIDENCE_INVALID: expected five CSV columns")
        rows.append(
            {
                "timestamp": fields[0],
                "uuid": fields[1],
                "memory_total_mib": fields[2],
                "memory_used_mib": fields[3],
                "memory_free_mib": fields[4],
            }
        )
    return rows


def classify_startup_failure(log_text: str) -> str:
    lowered = log_text.lower()
    if ("out of memory" in lowered or "cuda oom" in lowered) and any(term in lowered for term in ("marlin", "repack", "modelopt")):
        return "MARLIN_REPACK_OOM"
    if "out of memory" in lowered or "cuda oom" in lowered:
        return "MODEL_LOAD_OOM"
    if any(term in lowered for term in ("flashmla_sparse", "dsa", "marlin", "modelopt_fp4")):
        return "REQUIRED_BACKEND_STARTUP_FAILED"
    return "MODEL_LOAD_FAILED"


def validate_checkpoint_layout(model_root: str | Path) -> dict[str, Any]:
    root = Path(model_root)
    if not root.is_absolute() or not root.is_dir():
        raise ConfigError("CHECKPOINT_LAYOUT_MISSING: model root must be an existing absolute directory")
    validate_model_config(load_json(root / "config.json"))
    indexes = list(root.glob("*.safetensors.index.json"))
    if len(indexes) != 1:
        raise ConfigError("CHECKPOINT_LAYOUT_MISSING: expected exactly one safetensors index")
    index = load_json(indexes[0])
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ConfigError("CHECKPOINT_LAYOUT_INVALID: safetensors index has no weight_map")
    shard_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors") for name in shard_names):
        raise ConfigError("CHECKPOINT_LAYOUT_INVALID: shard names must be safe safetensors basenames")
    shards = [root / str(name) for name in shard_names]
    if any(not shard.is_file() or shard.stat().st_size <= 0 for shard in shards):
        raise ConfigError("CHECKPOINT_LAYOUT_INCOMPLETE: an indexed safetensors shard is missing or empty")
    return {
        "schema_version": 1,
        "status": "passed",
        "index_file": indexes[0].name,
        "tensor_name_count": len(weight_map),
        "shard_count": len(shards),
        "total_checkpoint_bytes": sum(shard.stat().st_size for shard in shards),
        "hash_policy": "no_full_tensor_hash",
    }


def validate_checkpoint_marker(model_root: str | Path, revision: str) -> dict[str, Any]:
    if not _SHA40.fullmatch(revision):
        raise ConfigError("MODEL_REVISION_UNRESOLVED")
    root = Path(model_root)
    marker = load_json(root / ".putpocket_checkpoint_ready.json")
    if marker.get("status") != "ready" or marker.get("model_id") != MODEL_ID or marker.get("revision") != revision:
        raise ConfigError("CHECKPOINT_MARKER_MISMATCH: cached checkpoint identity does not match the resolved model")
    layout = validate_checkpoint_layout(root)
    recorded = marker.get("layout")
    if not isinstance(recorded, Mapping) or recorded.get("shard_count") != layout["shard_count"] or recorded.get("total_checkpoint_bytes") != layout["total_checkpoint_bytes"]:
        raise ConfigError("CHECKPOINT_MARKER_MISMATCH: cached checkpoint layout changed after validation")
    return layout


def validate_public_project(url: str, commit: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "github.com" or parsed.username or parsed.password:
        raise ConfigError("Project URL must be a public credential-free GitHub HTTPS URL")
    if parsed.query or parsed.fragment or not parsed.path.endswith(".git"):
        raise ConfigError("Project URL must end in .git and contain no query or fragment")
    if not _SHA40.fullmatch(commit):
        raise ConfigError("Project commit must be a full Git SHA")


def safe_directive(value: Any, field: str) -> str:
    rendered = str(value)
    if not _SAFE_DIRECTIVE.fullmatch(rendered):
        raise ConfigError(f"Unsafe Slurm value for {field}: {rendered!r}")
    return rendered


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field} must be a mapping")
    return value


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if not parts:
        return ()
    return tuple(int(part or 0) for part in parts.groups())


def _reject_secret_fields(value: Any, path: str = "lock") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _SECRET_KEYS.search(str(key)):
                raise ConfigError(f"Secret-like field is forbidden at {path}.{key}")
            _reject_secret_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_fields(child, f"{path}[{index}]")


def _flatten_by_key(value: Any, output: dict[str, list[Any]] | None = None) -> dict[str, list[Any]]:
    result = output if output is not None else {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.setdefault(str(key), []).append(child)
            _flatten_by_key(child, result)
    elif isinstance(value, list):
        for child in value:
            _flatten_by_key(child, result)
    return result


def _reject_nonfinite_values(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_nonfinite_values(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite_values(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ConfigError(f"RUNTIME_NONFINITE: {path} contains a NaN or Inf value")
