from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import ConfigError


MODEL_ID = "nvidia/GLM-5.2-NVFP4"
MODEL_REVISION = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
SGLANG_COMMIT = "83d7d453306977dd3aad4402c921c8a6b66d9a9d"
SGLANG_IMAGE = "lmsysorg/sglang@sha256:3be8803490a8b899a44f7ab2e22d8f6a1fb877cab52faeb400769a1555317db4"
DATASET_ID = "ScaleAI/SWE-bench_Pro"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
HARNESS_COMMIT = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
MINI_SWE_COMMIT = "d74716a3c8104a113f77cc9ab94cf407ecdcf1e9"
INSTANCE_ID = "instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
FULL_LAYERS = (0, 1, 2, *range(6, 75, 4))
DECODE_SAMPLES = (0, 1, 8, 32)
TOPK = 2048
PROMPT_TOKEN_COUNT = 2071
MAX_NEW_TOKENS = 512
LOCK_PATH = Path(__file__).resolve().parents[2] / "configs" / "cluster" / "glm52_dsa_diagnostic.lock.json"

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECRET = re.compile(
    r"(?:secret|password|credential|api[_-]?key|(?:^|_)(?:hf|access|auth)[_-]?token(?:$|_))",
    re.IGNORECASE,
)


def load_lock(path: str | Path = LOCK_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError("GLM DSA diagnostic lock must be a JSON object")
    return value


def validate_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    if lock.get("schema_version") != 1 or lock.get("claim") != "diagnostic_only_not_quality_score":
        raise ConfigError("Diagnostic lock must be schema v1 and diagnostic-only")
    _reject_secret_fields(lock)
    model = _mapping(lock.get("model"), "model")
    sglang = _mapping(lock.get("sglang"), "sglang")
    runtime = _mapping(lock.get("runtime"), "runtime")
    benchmark = _mapping(lock.get("swebench_pro"), "swebench_pro")
    selection = _mapping(lock.get("selection"), "selection")
    exact = (
        (model, "id", MODEL_ID),
        (model, "resolved_revision", MODEL_REVISION),
        (model, "full_indexer_layers", list(FULL_LAYERS)),
        (model, "shared_indexer_count", 57),
        (model, "index_topk", TOPK),
        (sglang, "commit", SGLANG_COMMIT),
        (sglang, "image", SGLANG_IMAGE),
        (runtime, "tensor_parallel", 4),
        (runtime, "quantization", "modelopt_fp4"),
        (runtime, "fp4_gemm_backend", "marlin"),
        (runtime, "moe_runner_backend", "marlin"),
        (runtime, "dsa_prefill_backend", "flashmla_sparse"),
        (runtime, "dsa_decode_backend", "fa3"),
        (runtime, "dsa_topk_backend", "sgl-kernel"),
        (runtime, "context_length", 4096),
        (runtime, "max_running_requests", 1),
        (runtime, "max_new_tokens", MAX_NEW_TOKENS),
        (runtime, "seed", 0),
        (benchmark, "dataset", DATASET_ID),
        (benchmark, "dataset_revision", DATASET_REVISION),
        (benchmark, "harness_commit", HARNESS_COMMIT),
        (benchmark, "mini_swe_agent_commit", MINI_SWE_COMMIT),
        (benchmark, "score_eligible", False),
        (benchmark, "full_selection_reachable", False),
        (selection, "instance_id", INSTANCE_ID),
        (selection, "serialized_prompt_token_count", PROMPT_TOKEN_COUNT),
    )
    for container, key, expected in exact:
        if container.get(key) != expected:
            raise ConfigError(f"Diagnostic lock mismatch for {key}: expected {expected!r}")
    for key in ("offload", "speculative_mtp", "disaggregation"):
        if runtime.get(key) is not False:
            raise ConfigError(f"Diagnostic runtime must disable {key}")
    if runtime.get("disable_radix_cache") is not True:
        raise ConfigError("Diagnostic runtime must disable radix cache for cross-run isolation")
    if runtime.get("cuda_graph_prefill") != "disabled" or runtime.get("cuda_graph_decode") != "disabled":
        raise ConfigError("Diagnostic hooks require request-time eager prefill/decode dispatch")
    if PROMPT_TOKEN_COUNT <= TOPK or PROMPT_TOKEN_COUNT + MAX_NEW_TOKENS > int(runtime["context_length"]):
        raise ConfigError("Pinned prompt must exercise native logits and fit the bounded context")
    for value in (
        model["resolved_revision"],
        sglang["commit"],
        benchmark["dataset_revision"],
        benchmark["harness_commit"],
        benchmark["swe_agent_commit"],
        benchmark["mini_swe_agent_commit"],
    ):
        if not _SHA40.fullmatch(str(value)):
            raise ConfigError("Every source/model/dataset identity must be a full 40-character SHA")
    digest_fields = (
        "patch_target_sha256",
        "patch_target_post_sha256",
        "patch_sha256",
        "instrumentation_sha256",
    )
    for key in digest_fields:
        if not _SHA256.fullmatch(str(sglang.get(key, ""))):
            raise ConfigError(f"Diagnostic source lock requires SHA-256 for {key}")
    mapping = full_shared_mapping()
    if len(mapping) != 78 or sum(source == layer for layer, source in mapping.items()) != 21:
        raise ConfigError("Internal 21 full/57 shared indexer mapping is invalid")
    return {
        "schema_version": 1,
        "status": "passed",
        "model_revision": MODEL_REVISION,
        "sglang_commit": SGLANG_COMMIT,
        "instance_id": INSTANCE_ID,
        "prompt_token_count": PROMPT_TOKEN_COUNT,
        "full_layers": list(FULL_LAYERS),
        "shared_mapping": {str(layer): source for layer, source in mapping.items() if layer != source},
    }


def full_shared_mapping() -> dict[int, int]:
    mapping: dict[int, int] = {}
    source = 0
    for layer in range(78):
        if layer in FULL_LAYERS:
            source = layer
        mapping[layer] = source
    return mapping


def canonical_row_sha256(row: Mapping[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_selected_row(row: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    selection = _mapping(lock.get("selection"), "selection")
    if row.get("instance_id") != selection.get("instance_id"):
        raise ConfigError("SELECTED_INSTANCE_MISMATCH: runtime row is not the hard-pinned instance")
    if row.get("repo") != selection.get("repo") or row.get("dockerhub_tag") != selection.get("dockerhub_tag"):
        raise ConfigError("SELECTED_INSTANCE_CONTENT_MISMATCH: repo/dockerhub_tag changed")
    row_digest = canonical_row_sha256(row)
    if row_digest != selection.get("row_sha256"):
        raise ConfigError("SELECTED_INSTANCE_ROW_DIGEST_MISMATCH")
    problem = row.get("problem_statement")
    if not isinstance(problem, str) or hashlib.sha256(problem.encode("utf-8")).hexdigest() != selection.get(
        "problem_statement_sha256"
    ):
        raise ConfigError("SELECTED_INSTANCE_PROBLEM_DIGEST_MISMATCH")
    return {
        "schema_version": 1,
        "status": "passed",
        "instance_id": INSTANCE_ID,
        "row_sha256": row_digest,
        "problem_statement_sha256": selection["problem_statement_sha256"],
        "dockerhub_tag": selection["dockerhub_tag"],
        "score_eligible": False,
    }


def validate_serialized_prompt(
    serialized: str,
    token_ids: Sequence[int],
    lock: Mapping[str, Any],
    *,
    tokenizer_file_digests: Mapping[str, str],
    tokenizer_class: str,
) -> dict[str, Any]:
    selection = _mapping(lock.get("selection"), "selection")
    tokenizer = _mapping(selection.get("tokenizer"), "selection.tokenizer")
    expected_files = {
        "tokenizer.json": tokenizer["tokenizer_json_sha256"],
        "tokenizer_config.json": tokenizer["tokenizer_config_sha256"],
        "chat_template.jinja": tokenizer["chat_template_sha256"],
    }
    if dict(tokenizer_file_digests) != expected_files:
        raise ConfigError("TOKENIZER_METADATA_DIGEST_MISMATCH")
    if tokenizer_class != tokenizer.get("class"):
        raise ConfigError("TOKENIZER_CLASS_MISMATCH")
    encoded = serialized.encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if digest != selection.get("serialized_prompt_sha256"):
        raise ConfigError("SERIALIZED_PROMPT_DIGEST_MISMATCH")
    if len(encoded) != selection.get("serialized_prompt_utf8_bytes"):
        raise ConfigError("SERIALIZED_PROMPT_BYTE_COUNT_MISMATCH")
    if len(token_ids) != selection.get("serialized_prompt_token_count"):
        raise ConfigError("SERIALIZED_PROMPT_TOKEN_COUNT_MISMATCH")
    if not all(isinstance(value, int) and value >= 0 for value in token_ids):
        raise ConfigError("SERIALIZED_PROMPT_TOKEN_IDS_INVALID")
    return {
        "schema_version": 1,
        "status": "passed",
        "instance_id": INSTANCE_ID,
        "prompt_sha256": digest,
        "prompt_utf8_bytes": len(encoded),
        "prompt_token_count": len(token_ids),
        "tokenizer_model_revision": MODEL_REVISION,
        "tokenizer_class": tokenizer_class,
        "raw_prompt_persisted": False,
    }


def validate_patch_inputs(repository_root: str | Path, source_root: str | Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    project = Path(repository_root)
    source = Path(source_root)
    sglang = _mapping(lock.get("sglang"), "sglang")
    checks = {
        str(sglang["patch_target"]): str(sglang["patch_target_sha256"]),
        **{str(path): str(digest) for path, digest in _mapping(sglang.get("source_file_digests"), "source_file_digests").items()},
    }
    for relative, expected in checks.items():
        target = source / relative
        if not target.is_file() or _file_sha256(target) != expected:
            raise ConfigError(f"SGLANG_PATCH_CONTEXT_DIGEST_MISMATCH:{relative}")
    patch = project / str(sglang["patch_path"])
    instrumentation = project / str(sglang["instrumentation_source"])
    if _file_sha256(patch) != sglang["patch_sha256"]:
        raise ConfigError("SGLANG_PATCH_DIGEST_MISMATCH")
    if _file_sha256(instrumentation) != sglang["instrumentation_sha256"]:
        raise ConfigError("SGLANG_INSTRUMENTATION_DIGEST_MISMATCH")
    return {"schema_version": 1, "status": "passed", "checked_source_files": sorted(checks)}


def extract_completion(response: Mapping[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], Mapping):
        raise ConfigError("DIAGNOSTIC_COMPLETION_INVALID: expected exactly one choice")
    choice = choices[0]
    raw = choice.get("text")
    ids = choice.get("token_ids")
    if not isinstance(raw, str) or not raw.strip():
        raise ConfigError("DIAGNOSTIC_COMPLETION_EMPTY")
    if not isinstance(ids, list) or not ids or not all(isinstance(value, int) and value >= 0 for value in ids):
        raise ConfigError("DIAGNOSTIC_OUTPUT_TOKEN_IDS_MISSING")
    if re.search(r"(?:^|\W)(?:nan|[+-]?inf(?:inity)?)(?:$|\W)", raw, re.IGNORECASE):
        raise ConfigError("DIAGNOSTIC_COMPLETION_NONFINITE_TEXT")
    normalized = " ".join(raw.split())
    words = re.findall(r"\w+|[^\w\s]", normalized.lower())
    if len(words) >= 16 and max(words.count(word) for word in set(words)) / len(words) > 0.5:
        raise ConfigError("DIAGNOSTIC_COMPLETION_REPETITION_GARBAGE")
    ids_payload = json.dumps(ids, separators=(",", ":")).encode("ascii")
    return {
        "raw_output": raw,
        "normalized_output": normalized,
        "output_token_ids": ids,
        "output_token_ids_sha256": hashlib.sha256(ids_payload).hexdigest(),
        "raw_output_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "normalized_output_sha256": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "finish_reason": choice.get("finish_reason"),
    }


def validate_trace_equivalence(off_response: Mapping[str, Any], on_response: Mapping[str, Any]) -> dict[str, Any]:
    off = extract_completion(off_response)
    on = extract_completion(on_response)
    if off["output_token_ids"] != on["output_token_ids"]:
        raise ConfigError("TRACE_OUTPUT_TOKEN_ID_MISMATCH")
    if off["output_token_ids_sha256"] != on["output_token_ids_sha256"]:
        raise ConfigError("TRACE_OUTPUT_HASH_MISMATCH")
    return {
        "schema_version": 1,
        "status": "passed",
        "same_live_server_required": True,
        "cache_isolation": "disable_radix_cache_plus_successful_flush_cache_before_each_run",
        "seed": 0,
        "temperature": 0.0,
        "max_new_tokens": MAX_NEW_TOKENS,
        "output_token_count": len(off["output_token_ids"]),
        "output_token_ids_sha256": off["output_token_ids_sha256"],
        "off": off,
        "on": on,
    }


def expected_sample_points(output_token_count: int) -> tuple[tuple[str, int | None], ...]:
    points: list[tuple[str, int | None]] = [("prefill_last_query", None)]
    decode_forward_count = max(0, output_token_count - 1)
    points.extend(("decode", step) for step in DECODE_SAMPLES if step < decode_forward_count)
    return tuple(points)


def validate_capture_record(record: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    _reject_secret_fields(record)
    if any(key in record for key in ("prompt", "raw_prompt", "messages", "problem_statement")):
        raise ConfigError("CAPTURE_RAW_PROMPT_FORBIDDEN")
    required = {
        "schema_version",
        "record_type",
        "run_id",
        "instance_id",
        "trace_mode",
        "phase",
        "layer",
        "full_indexer_layer",
        "shared_layers",
        "query_position",
        "decode_step",
        "context_length",
        "rank",
        "backend_identities",
        "dtype",
        "device",
        "native_logits_shape",
        "topk",
        "source_token_coordinate_semantics",
        "native_transform_kind",
        "native_selected_logical_token_ids",
        "native_selected_scores",
        "native_pre_topk_raw_score_vector",
        "native_forced_token_mask",
        "revisions",
    }
    if set(record) != required:
        raise ConfigError(f"CAPTURE_SCHEMA_FIELDS_MISMATCH:{sorted(required - set(record))}")
    layer = record.get("layer")
    if record.get("schema_version") != 1 or record.get("record_type") != "native_glm52_dsa_indexer_scores":
        raise ConfigError("CAPTURE_SCHEMA_VERSION_MISMATCH")
    if layer not in FULL_LAYERS or record.get("full_indexer_layer") != layer:
        raise ConfigError("CAPTURE_FULL_INDEXER_LAYER_MISMATCH")
    if record.get("instance_id") != INSTANCE_ID or record.get("trace_mode") != "ON":
        raise ConfigError("CAPTURE_RUN_IDENTITY_MISMATCH")
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", str(record.get("run_id", ""))):
        raise ConfigError("CAPTURE_RUN_ID_INVALID")
    rank = record.get("rank")
    if not isinstance(rank, int) or not 0 <= rank < 4:
        raise ConfigError("CAPTURE_TP_RANK_MISMATCH")
    context = record.get("context_length")
    raw = record.get("native_pre_topk_raw_score_vector")
    ids = record.get("native_selected_logical_token_ids")
    scores = record.get("native_selected_scores")
    if not isinstance(context, int) or not TOPK < context <= 4096:
        raise ConfigError("CAPTURE_CONTEXT_LENGTH_INVALID")
    if not isinstance(raw, list) or len(raw) != context or not _all_finite(raw):
        raise ConfigError("CAPTURE_RAW_SCORE_VECTOR_INVALID")
    if not isinstance(ids, list) or len(ids) != TOPK or len(set(ids)) != TOPK:
        raise ConfigError("CAPTURE_SELECTED_IDS_INVALID")
    if not all(isinstance(value, int) and 0 <= value < context for value in ids):
        raise ConfigError("CAPTURE_SELECTED_ID_CAUSAL_BOUNDS_INVALID")
    if not isinstance(scores, list) or len(scores) != TOPK or not _all_finite(scores):
        raise ConfigError("CAPTURE_SELECTED_SCORES_INVALID")
    if any(not math.isclose(float(score), float(raw[index]), rel_tol=0.0, abs_tol=0.0) for index, score in zip(ids, scores, strict=True)):
        raise ConfigError("CAPTURE_SELECTED_ID_SCORE_ALIGNMENT_INVALID")
    mask = _mapping(record.get("native_forced_token_mask"), "native_forced_token_mask")
    init_count = mask.get("num_init_tokens")
    local_count = mask.get("num_local_tokens")
    if (
        not isinstance(init_count, int)
        or not isinstance(local_count, int)
        or init_count < 0
        or local_count < 0
        or init_count + local_count > TOPK
        or mask.get("applied_by_sglang_after_raw_capture") is not True
    ):
        raise ConfigError("CAPTURE_FORCED_TOKEN_MASK_INVALID")
    forced = set(range(init_count)) | set(range(max(0, context - local_count), context))
    selected = set(ids)
    if not forced.issubset(selected):
        raise ConfigError("CAPTURE_NATIVE_TOPK_FORCED_TOKEN_MISSING")
    selected_unforced = selected - forced
    unselected = set(range(context)) - selected
    if selected_unforced and unselected:
        minimum_selected = min(float(raw[index]) for index in selected_unforced)
        maximum_unselected = max(float(raw[index]) for index in unselected)
        if minimum_selected < maximum_unselected:
            raise ConfigError("CAPTURE_NATIVE_TOPK_CONSISTENCY_INVALID")
    expected_shared = [candidate for candidate, source in full_shared_mapping().items() if source == layer and candidate != layer]
    if record.get("shared_layers") != expected_shared:
        raise ConfigError("CAPTURE_SHARED_LAYER_MAPPING_DAMAGED")
    phase, step = record.get("phase"), record.get("decode_step")
    if (phase, step) not in (("prefill_last_query", None), *(("decode", value) for value in DECODE_SAMPLES)):
        raise ConfigError("CAPTURE_SAMPLE_POINT_UNBOUNDED")
    if record.get("query_position") != context - 1 or record.get("topk") != TOPK:
        raise ConfigError("CAPTURE_QUERY_OR_TOPK_MISMATCH")
    shape = record.get("native_logits_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(value, int) and value > 0 for value in shape)
        or shape[1] < context
    ):
        raise ConfigError("CAPTURE_NATIVE_LOGITS_SHAPE_INVALID")
    if record.get("native_transform_kind") not in {"paged", "ragged"}:
        raise ConfigError("CAPTURE_NATIVE_TRANSFORM_KIND_INVALID")
    if record.get("source_token_coordinate_semantics") != "zero_based_logical_causal_position_within_exact_request":
        raise ConfigError("CAPTURE_SOURCE_COORDINATE_SEMANTICS_INVALID")
    if not str(record.get("dtype", "")).startswith("torch.") or not str(record.get("device", "")).startswith("cuda:"):
        raise ConfigError("CAPTURE_DTYPE_OR_DEVICE_IDENTITY_INVALID")
    identities = _mapping(record.get("backend_identities"), "backend_identities")
    expected_backends = {
        "quantization": "modelopt_fp4",
        "fp4_gemm": "marlin_w4a16",
        "dsa_prefill": "flashmla_sparse",
        "dsa_decode": "fa3",
        "dsa_topk": "sgl-kernel",
    }
    if any(identities.get(key) != value for key, value in expected_backends.items()):
        raise ConfigError("CAPTURE_BACKEND_IDENTITY_MISMATCH")
    revisions = _mapping(record.get("revisions"), "revisions")
    image_digest = SGLANG_IMAGE.split("@", 1)[1]
    if (
        revisions.get("model") != MODEL_REVISION
        or revisions.get("sglang") != SGLANG_COMMIT
        or revisions.get("image") != image_digest
        or not _SHA40.fullmatch(str(revisions.get("project", "")))
    ):
        raise ConfigError("CAPTURE_REVISION_MISMATCH")
    return {"layer": layer, "rank": rank, "phase": phase, "decode_step": step, "context_length": context}


def validate_diagnostic_server_isolation(info: Mapping[str, Any]) -> dict[str, Any]:
    """Require request-scoped tracing to run eagerly with reusable caches disabled."""

    observed: dict[str, list[Any]] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                observed.setdefault(str(key), []).append(child)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(info)
    expected = {
        "disable_radix_cache": True,
        "cuda_graph_backend_prefill": "disabled",
        "cuda_graph_backend_decode": "disabled",
    }
    for key, required in expected.items():
        if required not in observed.get(key, []):
            raise ConfigError(f"TRACE_ISOLATION_RUNTIME_AMBIGUOUS:{key}={required!r}")
    return {"schema_version": 1, "status": "passed", "effective": expected}


def validate_capture_coverage(
    records: Iterable[Mapping[str, Any]], lock: Mapping[str, Any], *, output_token_count: int
) -> dict[str, Any]:
    materialized = list(records)
    points = expected_sample_points(output_token_count)
    expected = {(layer, rank, phase, step) for layer in FULL_LAYERS for rank in range(4) for phase, step in points}
    observed: set[tuple[int, int, str, int | None]] = set()
    contexts: dict[str, set[int]] = {}
    run_ids: set[str] = set()
    for record in materialized:
        summary = validate_capture_record(record, lock)
        run_ids.add(str(record["run_id"]))
        key = (summary["layer"], summary["rank"], summary["phase"], summary["decode_step"])
        if key in observed:
            raise ConfigError("CAPTURE_DUPLICATE_COVERAGE_CELL")
        observed.add(key)
        sample = summary["phase"] if summary["decode_step"] is None else f"decode_{summary['decode_step']}"
        contexts.setdefault(sample, set()).add(summary["context_length"])
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        raise ConfigError(f"CAPTURE_COVERAGE_INCOMPLETE:{missing[:4]}")
    if extra:
        raise ConfigError(f"CAPTURE_COVERAGE_UNBOUNDED:{extra[:4]}")
    if len(run_ids) != 1:
        raise ConfigError("CAPTURE_RUN_ID_COVERAGE_MISMATCH")
    return {
        "schema_version": 1,
        "status": "passed",
        "record_count": len(materialized),
        "full_layer_count": len(FULL_LAYERS),
        "rank_count": 4,
        "run_id": next(iter(run_ids)),
        "sample_points": [phase if step is None else f"decode_{step}" for phase, step in points],
        "coverage_cells": len(observed),
        "contexts": {key: sorted(value) for key, value in contexts.items()},
        "shared_layer_to_full_layer": {
            str(layer): source for layer, source in full_shared_mapping().items() if layer != source
        },
    }


def compress_capture_records(
    raw_paths: Sequence[Path], output_root: Path, *, prefer_zstd: bool = True
) -> tuple[list[dict[str, Any]], list[Path], str]:
    output_root.mkdir(parents=True, exist_ok=True)
    zstd = shutil.which("zstd") if prefer_zstd else None
    records: list[dict[str, Any]] = []
    compressed: list[Path] = []
    algorithm = "zstd-cli-19" if zstd else "python-gzip-level9-mtime0"
    for raw_path in sorted(raw_paths):
        raw_bytes = raw_path.read_bytes()
        value = json.loads(raw_bytes)
        if not isinstance(value, dict):
            raise ConfigError(f"Capture record is not an object: {raw_path.name}")
        records.append(value)
        if zstd:
            target = output_root / f"{raw_path.name}.zst"
            result = subprocess.run(
                [zstd, "--no-progress", "--threads=1", "-19", "--stdout", str(raw_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if result.returncode != 0:
                raise ConfigError(f"zstd compression failed for {raw_path.name}")
            payload = result.stdout
        else:
            target = output_root / f"{raw_path.name}.gz"
            payload = gzip.compress(raw_bytes, compresslevel=9, mtime=0)
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(payload)
        partial.replace(target)
        compressed.append(target)
    return records, compressed, algorithm


def build_artifact_manifest(
    compressed_paths: Sequence[Path], *, compression: str, coverage: Mapping[str, Any], run_id: str
) -> dict[str, Any]:
    files = [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": _file_sha256(path)}
        for path in sorted(compressed_paths)
    ]
    return {
        "schema_version": 1,
        "status": "passed",
        "run_id": run_id,
        "instance_id": INSTANCE_ID,
        "compression": compression,
        "compressed_byte_hash_policy": "sha256_of_exact_compressed_bytes",
        "files": files,
        "file_count": len(files),
        "compressed_bytes": sum(item["bytes"] for item in files),
        "coverage": dict(coverage),
        "raw_prompt_included": False,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _all_finite(values: Sequence[Any]) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _reject_secret_fields(value: Any, prefix: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            current = f"{prefix}.{key}" if prefix else str(key)
            if _SECRET.search(str(key)):
                raise ConfigError(f"Secret-bearing field is forbidden: {current}")
            _reject_secret_fields(child, current)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_fields(child, f"{prefix}[{index}]")
