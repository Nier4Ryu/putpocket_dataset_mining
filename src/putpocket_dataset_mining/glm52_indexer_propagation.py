from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .constants import REPO_ROOT
from .errors import ConfigError
from .glm52_attention_indexer import canonical_json_bytes, file_sha256


EPISODE_SCHEMA = REPO_ROOT / "configs/runpod/schemas/glm52_indexer_matrix_episode.schema.json"
REPORT_SCHEMA = REPO_ROOT / "configs/runpod/schemas/glm52_indexer_multihop_report.schema.json"
MATRIX_ROW_SCHEMA = REPO_ROOT / "configs/runpod/schemas/glm52_indexer_matrix_row.schema.json"
TOKEN_ROW_SCHEMA = REPO_ROOT / "configs/runpod/schemas/glm52_indexer_multihop_token_row.schema.json"
RECURRENCE_ID = "putpocket_raw_indexer_strict_causal_multihop_v1"
MAX_PROPAGATION_LEVEL = 16
NATIVE_SCORE_ORIGIN = (
    "kernel_native_fp8_fp4_mqa_logits_before_top_k_per_row_prefill"
)
NATIVE_HEAD_AGGREGATION = (
    "native_learned_weighted_sum_across_32_indexer_heads"
)
_SCHEMA_VALIDATORS: dict[Path, jsonschema.Draft202012Validator] = {}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ConfigError(reason)


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _token_digest(tokens: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps(list(tokens), separators=(",", ":")).encode("ascii")
    ).hexdigest()


def _validate_schema(value: Mapping[str, Any], schema_path: str | Path) -> None:
    resolved = Path(schema_path).resolve()
    validator = _SCHEMA_VALIDATORS.get(resolved)
    if validator is None:
        validator = jsonschema.Draft202012Validator(_load_object(resolved))
        _SCHEMA_VALIDATORS[resolved] = validator
    try:
        validator.validate(value)
    except jsonschema.ValidationError as exc:
        raise ConfigError(
            f"SCHEMA_VALIDATION_FAILED:{Path(schema_path).name}:{exc.json_path}"
        ) from exc


def _normalized_ranges(
    ranges: Sequence[Sequence[int]], *, label: str
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for value in ranges:
        _require(
            len(value) == 2
            and all(isinstance(item, int) for item in value)
            and 0 <= value[0] < value[1],
            f"{label}_RANGE_INVALID",
        )
        result.append((value[0], value[1]))
    _require(bool(result), f"{label}_RANGES_EMPTY")
    result.sort()
    _require(
        all(left[1] <= right[0] for left, right in zip(result, result[1:])),
        f"{label}_RANGES_OVERLAP",
    )
    return tuple(result)


def validate_seed_ranges(
    q1_ranges: Sequence[Sequence[int]],
    q2_ranges: Sequence[Sequence[int]],
    window: Sequence[int],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    _require(
        len(window) == 2
        and all(isinstance(item, int) for item in window)
        and 0 <= window[0] < window[1],
        "PROPAGATION_WINDOW_INVALID",
    )
    q1 = _normalized_ranges(q1_ranges, label="Q1")
    q2 = _normalized_ranges(q2_ranges, label="Q2")
    combined = sorted((*q1, *q2))
    _require(
        all(left[1] <= right[0] for left, right in zip(combined, combined[1:])),
        "Q1_Q2_RANGES_OVERLAP",
    )
    _require(
        all(window[0] <= start < end <= window[1] for start, end in combined),
        "SEED_RANGE_OUTSIDE_PROPAGATION_WINDOW",
    )
    return q1, q2


def load_matrix_episode_manifest(
    path: str | Path,
    *,
    verify_source: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    value = _load_object(manifest_path)
    _validate_schema(value, EPISODE_SCHEMA)
    prompt = value["prompt"]
    tokens = prompt["token_ids"]
    _require(prompt["token_count"] == len(tokens), "MATRIX_EPISODE_TOKEN_COUNT_MISMATCH")
    _require(
        prompt["token_ids_sha256"] == _token_digest(tokens),
        "MATRIX_EPISODE_TOKEN_DIGEST_MISMATCH",
    )
    window = value["propagation_window"]
    q1, q2 = validate_seed_ranges(
        value["segments"]["q1_ranges"], value["segments"]["q2_ranges"], window
    )
    _require(window[1] <= len(tokens), "MATRIX_EPISODE_WINDOW_OUTSIDE_PROMPT")
    width = window[1] - window[0]
    edge_count = width * (width - 1) // 2 * len(value["capture"]["layers"])
    _require(
        width <= value["capture"]["hard_max_window_tokens"]
        and edge_count <= value["capture"]["hard_max_total_edges_per_rank"],
        "MATRIX_EPISODE_COST_CAP_EXCEEDED",
    )
    _require(
        value["capture"]["layers"] == sorted(set(value["capture"]["layers"])),
        "MATRIX_EPISODE_LAYERS_NOT_CANONICAL",
    )
    if verify_source:
        source = value["source_frozen_episode_manifest"]
        relative_source = Path(source["path"])
        _require(not relative_source.is_absolute(), "SOURCE_FROZEN_EPISODE_PATH_UNSAFE")
        source_path = (manifest_path.parent / relative_source).resolve()
        _require(
            source_path.is_relative_to(manifest_path.parent),
            "SOURCE_FROZEN_EPISODE_PATH_UNSAFE",
        )
        _require(source_path.is_file(), "SOURCE_FROZEN_EPISODE_MISSING")
        _require(
            file_sha256(source_path) == source["sha256"],
            "SOURCE_FROZEN_EPISODE_DIGEST_MISMATCH",
        )
        source_value = _load_object(source_path)
        for field in ("scenario_id", "instance_id"):
            if field in source_value:
                _require(
                    source_value[field] == value[field],
                    f"SOURCE_FROZEN_EPISODE_{field.upper()}_MISMATCH",
                )
        source_tokenization = source_value.get("tokenization")
        if isinstance(source_tokenization, Mapping):
            expected = source_tokenization.get("first_post_edit_request_token_ids_sha256")
            if expected is not None:
                _require(
                    expected == prompt["token_ids_sha256"],
                    "SOURCE_FROZEN_EPISODE_PROMPT_DIGEST_MISMATCH",
                )
            expected_q2 = source_tokenization.get(
                "q2_token_range_in_first_post_edit_request"
            )
            if expected_q2 is not None:
                _require(
                    q2 == (tuple(expected_q2),),
                    "SOURCE_FROZEN_EPISODE_Q2_RANGE_MISMATCH",
                )
    return value


def _validated_record(line: str, *, path: Path, line_number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"MATRIX_CAPTURE_JSON_INVALID:{path.name}:{line_number}") from exc
    _require(isinstance(value, dict), "MATRIX_CAPTURE_RECORD_NOT_OBJECT")
    raw_scores = value.get("raw_scores")
    _require(
        isinstance(raw_scores, list)
        and all(
            isinstance(item, (int, float)) and math.isfinite(float(item))
            for item in raw_scores
        ),
        "MATRIX_CAPTURE_RAW_VECTOR_NONFINITE",
    )
    observed = value.get("record_sha256")
    payload = dict(value)
    payload.pop("record_sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    _require(observed == expected, "MATRIX_CAPTURE_RECORD_DIGEST_MISMATCH")
    _validate_schema(value, MATRIX_ROW_SCHEMA)
    return value


@dataclass(frozen=True)
class MatrixCapture:
    window: tuple[int, int]
    token_ids: tuple[int, ...]
    layers: tuple[int, ...]
    matrices: Mapping[int, tuple[tuple[float, ...], ...]]
    input_files: tuple[Mapping[str, Any], ...]
    tp_max_abs_difference_observed: float


def load_indexer_matrix_capture(
    capture_root: str | Path,
    episode: Mapping[str, Any],
    *,
    layers: Sequence[int],
    window: Sequence[int],
) -> MatrixCapture:
    root = Path(capture_root)
    files = sorted(root.glob("matrix-rank-*-chunk-*.jsonl"))
    _require(bool(files), "MATRIX_CAPTURE_FILES_MISSING")
    requested_layers = tuple(sorted(set(int(item) for item in layers)))
    _require(
        bool(requested_layers)
        and list(requested_layers) == list(layers)
        and set(requested_layers).issubset(episode["capture"]["layers"]),
        "MATRIX_CAPTURE_LAYERS_INVALID",
    )
    requested_window = tuple(int(item) for item in window)
    _require(
        len(requested_window) == 2
        and list(requested_window) == episode["propagation_window"],
        "MATRIX_CAPTURE_WINDOW_MANIFEST_MISMATCH",
    )
    start, end = requested_window
    tokens = tuple(int(item) for item in episode["prompt"]["token_ids"])
    tp_size = int(episode["capture"]["tensor_parallel_size"])
    expected_ranks = set(range(tp_size))
    chunk_size = int(episode["capture"]["row_chunk_size"])
    grouped: dict[tuple[int, int], dict[int, dict[str, Any]]] = {}
    evidence: list[dict[str, Any]] = []
    for path in files:
        match = re.fullmatch(r"matrix-rank-(\d{2})-chunk-(\d{4})\.jsonl", path.name)
        _require(match is not None, "MATRIX_CAPTURE_FILENAME_INVALID")
        filename_rank = int(match.group(1))
        filename_chunk = int(match.group(2))
        encoded = path.read_bytes()
        try:
            lines = [line for line in encoded.decode("utf-8").splitlines() if line]
        except UnicodeDecodeError as exc:
            raise ConfigError(f"MATRIX_CAPTURE_UTF8_INVALID:{path.name}") from exc
        _require(bool(lines), f"MATRIX_CAPTURE_FILE_EMPTY:{path.name}")
        for line_number, line in enumerate(lines, 1):
            record = _validated_record(line, path=path, line_number=line_number)
            _require(
                record.get("record_kind")
                == "indexer_native_strict_causal_matrix_row"
                and record.get("capture_mode") == "strict_causal_indexer_matrix"
                and record.get("score_origin") == NATIVE_SCORE_ORIGIN
                and record.get("kernel_native") is True
                and record.get("pre_top_k") is True
                and record.get("normalized") is False
                and record.get("padding_or_masked_values_included") is False
                and record.get("head_aggregation_at_capture")
                == NATIVE_HEAD_AGGREGATION,
                "MATRIX_CAPTURE_SCORE_SEMANTICS_INVALID",
            )
            layer = record.get("layer")
            query = record.get("query_position")
            rank = record.get("rank")
            if layer not in requested_layers:
                continue
            _require(
                isinstance(query, int)
                and start <= query < end
                and rank in expected_ranks,
                "MATRIX_CAPTURE_CELL_OUT_OF_RANGE",
            )
            _require(
                rank == filename_rank
                and record.get("tensor_parallel_size") == tp_size
                and record.get("tp_semantics")
                == "replicated_native_aggregate_consensus_required_offline"
                and record.get("indexer_head_count") == 32
                and record.get("indexer_head_dim") == 128
                and record.get("scale")
                == {"softmax_scale": 128**-0.5, "indexer_head_scale": 32**-0.5},
                "MATRIX_CAPTURE_TP_OR_SCALE_PROVENANCE_INVALID",
            )
            _require(
                record.get("propagation_window") == [start, end]
                and record.get("native_valid_start") == 0
                and record.get("native_valid_end_exclusive") == query + 1
                and record.get("mask") == "strict_causal_key_position_lt_query_position",
                "MATRIX_CAPTURE_CAUSAL_BOUNDARY_INVALID",
            )
            chunk_index = (query - start) // chunk_size
            chunk_start = start + chunk_index * chunk_size
            chunk_end = min(end, chunk_start + chunk_size)
            _require(
                record.get("row_chunk_index") == chunk_index
                and filename_chunk == chunk_index
                and record.get("row_chunk_range") == [chunk_start, chunk_end],
                "MATRIX_CAPTURE_CHUNK_METADATA_INVALID",
            )
            expected_positions = list(range(start, query))
            _require(
                record.get("key_positions") == expected_positions,
                "MATRIX_CAPTURE_ROW_INCOMPLETE_OR_NONCAUSAL",
            )
            _require(
                record.get("query_token_id") == tokens[query]
                and record.get("key_token_ids")
                == [tokens[position] for position in expected_positions],
                "MATRIX_CAPTURE_TOKEN_MISMATCH",
            )
            raw = record.get("raw_scores")
            _require(
                isinstance(raw, list)
                and len(raw) == len(expected_positions)
                and all(
                    isinstance(item, (int, float)) and math.isfinite(float(item))
                    for item in raw
                ),
                "MATRIX_CAPTURE_RAW_VECTOR_INVALID",
            )
            cell = grouped.setdefault((int(layer), query), {})
            _require(rank not in cell, "MATRIX_CAPTURE_DUPLICATE_RANK_ROW")
            cell[int(rank)] = record
        evidence.append(
            {
                "path": path.name,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    expected_cells = {
        (layer, query) for layer in requested_layers for query in range(start, end)
    }
    _require(set(grouped) == expected_cells, "MATRIX_CAPTURE_INTERMEDIATE_ROWS_MISSING")
    matrices: dict[int, tuple[tuple[float, ...], ...]] = {}
    max_difference = 0.0
    width = end - start
    tolerance = float(episode["capture"]["indexer_tp_max_abs_difference"])
    for layer in requested_layers:
        rows: list[tuple[float, ...]] = []
        for query in range(start, end):
            replicas = grouped[(layer, query)]
            _require(set(replicas) == expected_ranks, "MATRIX_CAPTURE_TP_COVERAGE_INCOMPLETE")
            reference = [float(item) for item in replicas[0]["raw_scores"]]
            for rank in sorted(expected_ranks - {0}):
                other = [float(item) for item in replicas[rank]["raw_scores"]]
                difference = max(
                    (abs(left - right) for left, right in zip(reference, other, strict=True)),
                    default=0.0,
                )
                max_difference = max(max_difference, difference)
                _require(difference <= tolerance, "MATRIX_CAPTURE_TP_REPLICA_DISAGREEMENT")
            rows.append(tuple(reference + [0.0] * (width - len(reference))))
        matrices[layer] = tuple(rows)
    return MatrixCapture(
        window=(start, end),
        token_ids=tokens[start:end],
        layers=requested_layers,
        matrices=matrices,
        input_files=tuple(sorted(evidence, key=lambda item: str(item["path"]))),
        tp_max_abs_difference_observed=max_difference,
    )


def _row_times_matrix(
    vector: Sequence[float], matrix: Sequence[Sequence[float]]
) -> list[float]:
    width = len(vector)
    _require(len(matrix) == width and all(len(row) == width for row in matrix), "MATRIX_SHAPE_INVALID")
    result = [0.0] * width
    for query_index, weight in enumerate(vector):
        _require(math.isfinite(weight), "PROPAGATION_INPUT_NONFINITE")
        if weight == 0.0:
            continue
        for key_index in range(query_index):
            product = weight * float(matrix[query_index][key_index])
            _require(math.isfinite(product), "PROPAGATION_FLOAT64_OVERFLOW")
            result[key_index] += product
            _require(math.isfinite(result[key_index]), "PROPAGATION_FLOAT64_OVERFLOW")
    return result


def _checked_sum(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        total += float(value)
        _require(math.isfinite(total), "PROPAGATION_FLOAT64_OVERFLOW")
    return total


def compute_level_scores(
    capture: MatrixCapture,
    *,
    q1_ranges: Sequence[Sequence[int]],
    q2_ranges: Sequence[Sequence[int]],
    max_level: int,
) -> dict[str, Any]:
    _require(
        sys.float_info.mant_dig == 53 and sys.float_info.radix == 2,
        "FLOAT64_RUNTIME_UNSUPPORTED",
    )
    _require(
        isinstance(max_level, int) and 1 <= max_level <= MAX_PROPAGATION_LEVEL,
        "PROPAGATION_LEVEL_OUT_OF_RANGE",
    )
    q1, q2 = validate_seed_ranges(q1_ranges, q2_ranges, capture.window)
    start, end = capture.window
    q1_positions = {position for left, right in q1 for position in range(left, right)}
    q2_positions = {position for left, right in q2 for position in range(left, right)}
    seed = [
        1.0 if position in q1_positions or position in q2_positions else 0.0
        for position in range(start, end)
    ]
    _require(any(seed), "PROPAGATION_SEED_EMPTY")
    layer_results: dict[int, dict[str, list[list[float]] | list[float]]] = {}
    for layer in capture.layers:
        current = seed
        cumulative = [0.0] * len(seed)
        hops: list[list[float]] = []
        cumulative_by_level: list[list[float]] = []
        for _level in range(1, max_level + 1):
            current = _row_times_matrix(current, capture.matrices[layer])
            hops.append(current)
            cumulative = [left + right for left, right in zip(cumulative, current, strict=True)]
            _require(all(math.isfinite(item) for item in cumulative), "PROPAGATION_FLOAT64_OVERFLOW")
            cumulative_by_level.append(cumulative)
        layer_results[layer] = {
            "hop_contributions": hops,
            "cumulative_by_level": cumulative_by_level,
            "cumulative": cumulative,
        }
    return {
        "recurrence_id": RECURRENCE_ID,
        "max_level": max_level,
        "window": [start, end],
        "q1_ranges": [list(item) for item in q1],
        "q2_ranges": [list(item) for item in q2],
        "seed": seed,
        "layer_results": layer_results,
    }


def write_multihop_report(
    *,
    episode_path: str | Path,
    capture: MatrixCapture,
    scores: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    episode = load_matrix_episode_manifest(episode_path)
    _require(
        capture.token_ids
        == tuple(episode["prompt"]["token_ids"][capture.window[0] : capture.window[1]])
        and list(capture.layers) == list(scores["layer_results"]),
        "PROPAGATION_CAPTURE_EPISODE_OR_LAYER_MISMATCH",
    )
    _require(
        scores["q1_ranges"] == episode["segments"]["q1_ranges"]
        and scores["q2_ranges"] == episode["segments"]["q2_ranges"],
        "PROPAGATION_RANGES_FROZEN_EPISODE_MISMATCH",
    )
    start, end = capture.window
    q1_positions = {
        position
        for left, right in scores["q1_ranges"]
        for position in range(left, right)
    }
    q2_positions = {
        position
        for left, right in scores["q2_ranges"]
        for position in range(left, right)
    }
    token_path = output / "indexer-multihop-token-scores.jsonl"
    token_bytes = bytearray()
    for index, position in enumerate(range(start, end)):
        memberships = []
        if position in q1_positions:
            memberships.append("q1")
        if position in q2_positions:
            memberships.append("q2")
        per_layer = []
        layer_sum_hops = [0.0] * scores["max_level"]
        layer_sum_cumulative = 0.0
        for layer in capture.layers:
            result = scores["layer_results"][layer]
            hops = [values[index] for values in result["hop_contributions"]]
            cumulative = result["cumulative"][index]
            per_layer.append(
                {
                    "layer": layer,
                    "hop_contributions": hops,
                    "cumulative_score": cumulative,
                }
            )
            layer_sum_hops = [
                left + right for left, right in zip(layer_sum_hops, hops, strict=True)
            ]
            layer_sum_cumulative += cumulative
            _require(
                all(math.isfinite(item) for item in layer_sum_hops)
                and math.isfinite(layer_sum_cumulative),
                "PROPAGATION_FLOAT64_OVERFLOW",
            )
        row = {
            "schema_version": 1,
            "position": position,
            "token_id": capture.token_ids[index],
            "segment_membership": memberships,
            "seed_value": scores["seed"][index],
            "per_layer": per_layer,
            "layer_sum": {
                "hop_contributions": layer_sum_hops,
                "cumulative_score": layer_sum_cumulative,
            },
        }
        row["record_sha256"] = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
        _validate_schema(row, TOKEN_ROW_SCHEMA)
        token_bytes.extend(canonical_json_bytes(row))
    token_path.write_bytes(bytes(token_bytes))
    per_layer_summary = []
    for layer in capture.layers:
        result = scores["layer_results"][layer]
        per_layer_summary.append(
            {
                "layer": layer,
                "hop_total_by_level": [_checked_sum(values) for values in result["hop_contributions"]],
                "cumulative_total": _checked_sum(result["cumulative"]),
            }
        )
    payload = {
        "status": "passed",
        "recurrence": {
            "id": RECURRENCE_ID,
            "level_1": "c_l,1 = u A_l",
            "additional_level": "c_l,n+1 = c_l,n A_l",
            "reported": "s_l,L = sum(c_l,n for n=1..L)",
            "cumulative_interpretation": True,
            "cross_layer_operation": "exact_unnormalized_sum_only",
        },
        "float_policy": "CPython_IEEE754_binary64_no_normalization_no_clipping_no_rectification_no_softmax_fail_nonfinite",
        "episode": {
            "path": Path(episode_path).name,
            "sha256": file_sha256(episode_path),
            "scenario_id": episode["scenario_id"],
            "instance_id": episode["instance_id"],
            "benchmark_provenance": episode["benchmark_provenance"],
            "source_frozen_episode_manifest": episode[
                "source_frozen_episode_manifest"
            ],
        },
        "inputs": list(capture.input_files),
        "window": list(capture.window),
        "layers": list(capture.layers),
        "q1_ranges": scores["q1_ranges"],
        "q2_ranges": scores["q2_ranges"],
        "max_level": scores["max_level"],
        "tp_max_abs_difference_observed": capture.tp_max_abs_difference_observed,
        "per_layer_summary": per_layer_summary,
        "token_level_artifact": {
            "path": token_path.name,
            "record_count": end - start,
            "sha256": file_sha256(token_path),
            "schema_path": "configs/runpod/schemas/glm52_indexer_multihop_token_row.schema.json",
            "schema_sha256": file_sha256(TOKEN_ROW_SCHEMA),
        },
        "scientific_result_policy": "offline_score_only_no_threshold_no_inference_decision",
    }
    report = {
        "schema_version": 1,
        "report_kind": "glm52_raw_indexer_multihop_offline",
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }
    _validate_schema(report, REPORT_SCHEMA)
    report_path = output / "indexer-multihop-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def score_matrix_capture(
    *,
    episode_path: str | Path,
    capture_root: str | Path,
    output_root: str | Path,
    q1_ranges: Sequence[Sequence[int]],
    q2_ranges: Sequence[Sequence[int]],
    layers: Sequence[int],
    window: Sequence[int],
    max_level: int,
) -> dict[str, Any]:
    initial_episode_sha256 = file_sha256(episode_path)
    episode = load_matrix_episode_manifest(episode_path)
    q1, q2 = validate_seed_ranges(q1_ranges, q2_ranges, window)
    _require(
        [list(item) for item in q1] == episode["segments"]["q1_ranges"]
        and [list(item) for item in q2] == episode["segments"]["q2_ranges"],
        "PROPAGATION_RANGES_FROZEN_EPISODE_MISMATCH",
    )
    capture = load_indexer_matrix_capture(
        capture_root, episode, layers=layers, window=window
    )
    scores = compute_level_scores(
        capture, q1_ranges=q1, q2_ranges=q2, max_level=max_level
    )
    _require(
        file_sha256(episode_path) == initial_episode_sha256,
        "MATRIX_EPISODE_MUTATED_DURING_SCORING",
    )
    return write_multihop_report(
        episode_path=episode_path,
        capture=capture,
        scores=scores,
        output_root=output_root,
    )
