"""Offline rank-normalized propagation over accepted GLM indexer matrices.

This module deliberately does not alter the legacy signed recurrence or any
vLLM inference decision.  It converts each strictly causal raw-score row into
a nonnegative probability row, averages those rows across layers, and then
propagates probability mass backwards along the original q -> k orientation.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import re
import subprocess
from array import array
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .constants import REPO_ROOT
from .errors import ConfigError
from .glm52_attention_indexer import canonical_json_bytes, file_sha256
from .glm52_indexer_propagation import (
    MAX_PROPAGATION_LEVEL,
    MatrixCapture,
    load_indexer_matrix_capture,
    load_matrix_episode_manifest,
    validate_seed_ranges,
)


RECURRENCE_ID = "putpocket_rank_normalized_indexer_multihop_v1"
LEGACY_RECURRENCE_ID = "putpocket_raw_indexer_strict_causal_multihop_v1"
PRIMARY_VARIANT_ID = "rank_dcg_k64"
DEFAULT_TRANSITION_K = (16, 64, 256)
DEFAULT_EVALUATION_K = (16, 64, 256)
DEFAULT_SOFTMAX_TEMPERATURE = 1.0
MASS_TOLERANCE = 1e-10
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPORT_SCHEMA = (
    REPO_ROOT
    / "configs/runpod/schemas/glm52_rank_normalized_multihop_report.schema.json"
)
TOKEN_ROW_SCHEMA = (
    REPO_ROOT
    / "configs/runpod/schemas/glm52_rank_normalized_multihop_token_row.schema.json"
)
ATTESTATION_SCHEMA = (
    REPO_ROOT
    / "configs/runpod/schemas/glm52_rank_normalized_input_attestation.schema.json"
)
_SCHEMA_VALIDATORS: dict[Path, jsonschema.Draft202012Validator] = {}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ConfigError(reason)


def _validate_schema(value: Mapping[str, Any], schema_path: str | Path) -> None:
    resolved = Path(schema_path).resolve()
    validator = _SCHEMA_VALIDATORS.get(resolved)
    if validator is None:
        schema = json.loads(resolved.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
        _SCHEMA_VALIDATORS[resolved] = validator
    try:
        validator.validate(value)
    except jsonschema.ValidationError as exc:
        raise ConfigError(
            f"SCHEMA_VALIDATION_FAILED:{resolved.name}:{exc.json_path}"
        ) from exc


def _safe_relative(value: str, *, reason: str) -> Path:
    path = Path(value)
    _require(
        bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
        reason,
    )
    return path


def _resolve_file(root: Path, value: str, *, reason: str) -> tuple[Path, str]:
    relative = _safe_relative(value, reason=reason)
    path = (root / relative).resolve()
    try:
        normalized = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ConfigError(reason) from exc
    _require(path.is_file() and not path.is_symlink(), reason)
    return path, normalized


def _resolve_directory(
    root: Path, value: str, *, reason: str
) -> tuple[Path, str]:
    relative = _safe_relative(value, reason=reason)
    path = (root / relative).resolve()
    try:
        normalized = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ConfigError(reason) from exc
    _require(path.is_dir() and not path.is_symlink(), reason)
    return path, normalized


def load_checksum_manifest(
    evidence_root: str | Path, relative_path: str
) -> tuple[Path, str, dict[str, str]]:
    root = Path(evidence_root).resolve()
    path, normalized = _resolve_file(
        root, relative_path, reason="RANK_PROPAGATION_CHECKSUM_MANIFEST_INVALID"
    )
    entries: dict[str, str] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(None, 1)
        _require(
            len(fields) == 2 and SHA256_RE.fullmatch(fields[0]) is not None,
            f"RANK_PROPAGATION_CHECKSUM_LINE_INVALID:{line_number}",
        )
        name = _safe_relative(
            fields[1].lstrip("* "),
            reason=f"RANK_PROPAGATION_CHECKSUM_PATH_INVALID:{line_number}",
        ).as_posix()
        _require(name not in entries, "RANK_PROPAGATION_CHECKSUM_PATH_DUPLICATE")
        entries[name] = fields[0]
    _require(bool(entries), "RANK_PROPAGATION_CHECKSUM_MANIFEST_EMPTY")
    return path, normalized, entries


def _attest_file(
    path: Path, relative: str, manifest: Mapping[str, str]
) -> dict[str, Any]:
    observed = file_sha256(path)
    _require(
        manifest.get(relative) == observed,
        f"RANK_PROPAGATION_INPUT_NOT_ATTESTED_OR_CHANGED:{relative}",
    )
    return {"path": relative, "bytes": path.stat().st_size, "sha256": observed}


def rank_dcg_row(raw_scores: Sequence[float], transition_k: int) -> array:
    """Map one raw row to a deterministic top-K DCG probability row."""

    _require(
        isinstance(transition_k, int) and transition_k > 0,
        "RANK_TRANSITION_K_INVALID",
    )
    values = [float(value) for value in raw_scores]
    _require(
        all(math.isfinite(value) for value in values),
        "RANK_TRANSITION_RAW_SCORE_NONFINITE",
    )
    probabilities = array("d", [0.0]) * len(values)
    if not values:
        return probabilities
    retained = min(transition_k, len(values))
    order = heapq.nsmallest(
        retained,
        range(len(values)),
        key=lambda index: (-values[index], index),
    )
    weights = [1.0 / math.log2(rank + 1.0) for rank in range(1, retained + 1)]
    denominator = math.fsum(weights)
    _require(
        math.isfinite(denominator) and denominator > 0.0,
        "RANK_TRANSITION_NORMALIZATION_INVALID",
    )
    for index, weight in zip(order, weights, strict=True):
        probabilities[index] = weight / denominator
    _require(
        abs(math.fsum(probabilities) - 1.0) <= MASS_TOLERANCE,
        "RANK_TRANSITION_ROW_MASS_INVALID",
    )
    return probabilities


def zscore_softmax_row(
    raw_scores: Sequence[float], *, temperature: float = DEFAULT_SOFTMAX_TEMPERATURE
) -> array:
    """Map one raw row to a population-z-score softmax probability row."""

    _require(
        isinstance(temperature, (int, float))
        and math.isfinite(float(temperature))
        and float(temperature) > 0.0,
        "SOFTMAX_TRANSITION_TEMPERATURE_INVALID",
    )
    values = [float(value) for value in raw_scores]
    _require(
        all(math.isfinite(value) for value in values),
        "SOFTMAX_TRANSITION_RAW_SCORE_NONFINITE",
    )
    if not values:
        return array("d")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    _require(math.isfinite(variance) and variance >= 0.0, "SOFTMAX_VARIANCE_INVALID")
    if variance == 0.0:
        logits = [0.0] * len(values)
    else:
        scale = math.sqrt(variance) * float(temperature)
        logits = [(value - mean) / scale for value in values]
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    denominator = math.fsum(exponentials)
    _require(
        math.isfinite(denominator) and denominator > 0.0,
        "SOFTMAX_TRANSITION_NORMALIZATION_INVALID",
    )
    probabilities = array("d", (value / denominator for value in exponentials))
    _require(
        abs(math.fsum(probabilities) - 1.0) <= MASS_TOLERANCE,
        "SOFTMAX_TRANSITION_ROW_MASS_INVALID",
    )
    return probabilities


def build_layer_averaged_transition(
    capture: MatrixCapture,
    *,
    kind: str,
    transition_k: int | None = None,
    temperature: float | None = None,
) -> tuple[array, ...]:
    """Build T[q,k] by normalizing each layer row before uniform averaging."""

    _require(bool(capture.layers), "RANK_PROPAGATION_LAYERS_EMPTY")
    width = capture.window[1] - capture.window[0]
    _require(
        width == len(capture.token_ids)
        and all(len(capture.matrices[layer]) == width for layer in capture.layers),
        "RANK_PROPAGATION_CAPTURE_SHAPE_INVALID",
    )
    rows: list[array] = []
    layer_scale = 1.0 / len(capture.layers)
    for query_index in range(width):
        averaged = array("d", [0.0]) * query_index
        for layer in capture.layers:
            raw = capture.matrices[layer][query_index][:query_index]
            if kind == "rank_dcg":
                _require(transition_k is not None, "RANK_TRANSITION_K_MISSING")
                normalized = rank_dcg_row(raw, transition_k)
            elif kind == "zscore_softmax":
                _require(temperature is not None, "SOFTMAX_TEMPERATURE_MISSING")
                normalized = zscore_softmax_row(raw, temperature=temperature)
            else:
                raise ConfigError("RANK_TRANSITION_KIND_UNSUPPORTED")
            for key_index, probability in enumerate(normalized):
                averaged[key_index] += probability * layer_scale
        expected_mass = 0.0 if query_index == 0 else 1.0
        _require(
            abs(math.fsum(averaged) - expected_mass) <= MASS_TOLERANCE,
            "LAYER_AVERAGED_TRANSITION_ROW_MASS_INVALID",
        )
        rows.append(averaged)
    return tuple(rows)


def normalize_distribution(
    values: Sequence[float], *, reason: str
) -> tuple[list[float], float]:
    vector = [float(value) for value in values]
    _require(
        bool(vector)
        and all(math.isfinite(value) and value >= 0.0 for value in vector),
        f"{reason}_NONFINITE_OR_NEGATIVE",
    )
    mass = math.fsum(vector)
    _require(math.isfinite(mass) and mass > 0.0, f"{reason}_ZERO_OR_NONFINITE_MASS")
    normalized = [value / mass for value in vector]
    _require(
        abs(math.fsum(normalized) - 1.0) <= MASS_TOLERANCE,
        f"{reason}_NORMALIZED_MASS_INVALID",
    )
    return normalized, mass


def propagate_distribution(
    distribution: Sequence[float], transition: Sequence[Sequence[float]]
) -> tuple[list[float], float]:
    """Compute h_next[k] = sum_q h[q] T[q,k], then normalize."""

    width = len(distribution)
    _require(len(transition) == width, "RANK_PROPAGATION_TRANSITION_SHAPE_INVALID")
    result = [0.0] * width
    for query_index, source_mass in enumerate(distribution):
        mass = float(source_mass)
        _require(
            math.isfinite(mass) and mass >= 0.0,
            "RANK_PROPAGATION_SOURCE_MASS_INVALID",
        )
        row = transition[query_index]
        _require(len(row) == query_index, "RANK_PROPAGATION_ROW_SHAPE_INVALID")
        if mass == 0.0:
            continue
        for key_index, probability in enumerate(row):
            result[key_index] += mass * float(probability)
    return normalize_distribution(result, reason="RANK_PROPAGATION_HOP")


def deterministic_order(distribution: Sequence[float]) -> list[int]:
    values = [float(value) for value in distribution]
    _require(
        bool(values)
        and all(math.isfinite(value) and value >= 0.0 for value in values),
        "RANK_ORDER_DISTRIBUTION_INVALID",
    )
    return sorted(range(len(values)), key=lambda index: (-values[index], index))


def distribution_summary(distribution: Sequence[float]) -> dict[str, float]:
    values = [float(value) for value in distribution]
    mass = math.fsum(values)
    _require(
        all(math.isfinite(value) and value >= 0.0 for value in values)
        and abs(mass - 1.0) <= MASS_TOLERANCE,
        "RANK_DISTRIBUTION_MASS_INVALID",
    )
    entropy = -math.fsum(value * math.log(value) for value in values if value > 0.0)
    effective_support = math.exp(entropy)
    _require(
        math.isfinite(entropy) and math.isfinite(effective_support),
        "RANK_DISTRIBUTION_SUMMARY_NONFINITE",
    )
    return {
        "mass": mass,
        "entropy_nats": entropy,
        "effective_support": effective_support,
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    _require(len(left) == len(right) and len(left) >= 2, "RANK_PEARSON_INPUT_INVALID")
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_norm = math.sqrt(math.fsum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(math.fsum((y - right_mean) ** 2 for y in right))
    _require(left_norm > 0.0 and right_norm > 0.0, "RANK_PEARSON_VARIANCE_ZERO")
    value = numerator / (left_norm * right_norm)
    _require(math.isfinite(value), "RANK_PEARSON_NONFINITE")
    return value


def rank_stability(
    current: Sequence[float],
    baseline: Sequence[float],
    *,
    evaluation_k: Sequence[int],
) -> dict[str, Any]:
    _require(len(current) == len(baseline), "RANK_STABILITY_LENGTH_MISMATCH")
    width = len(current)
    current_order = deterministic_order(current)
    baseline_order = deterministic_order(baseline)
    current_rank = {index: rank for rank, index in enumerate(current_order, 1)}
    baseline_rank = {index: rank for rank, index in enumerate(baseline_order, 1)}
    results = []
    for requested_k in evaluation_k:
        _require(
            isinstance(requested_k, int) and 1 <= requested_k <= width,
            "RANK_STABILITY_K_INVALID",
        )
        current_top = current_order[:requested_k]
        baseline_top = baseline_order[:requested_k]
        current_set, baseline_set = set(current_top), set(baseline_top)
        overlap = len(current_set & baseline_set)
        union = current_set | baseline_set
        relevance = {
            index: 1.0 / math.log2(baseline_rank[index] + 1.0)
            for index in range(width)
        }
        dcg = math.fsum(
            relevance[index] / math.log2(rank + 1.0)
            for rank, index in enumerate(current_top, 1)
        )
        idcg = math.fsum(
            relevance[index] / math.log2(rank + 1.0)
            for rank, index in enumerate(baseline_top, 1)
        )
        ordered_union = sorted(union)
        secondary_spearman = (
            1.0
            if len(ordered_union) == 1
            else _pearson(
                [float(current_rank[index]) for index in ordered_union],
                [float(baseline_rank[index]) for index in ordered_union],
            )
        )
        results.append(
            {
                "k": requested_k,
                "overlap_count": overlap,
                "retention": overlap / requested_k,
                "jaccard": overlap / len(union),
                "ndcg_against_baseline_rank": dcg / idcg,
                "secondary_spearman_on_topk_union": secondary_spearman,
                "topk_union_size": len(union),
            }
        )
    return {
        "deterministic_tie_break": "higher_probability_then_lower_absolute_position",
        "secondary_rank_scope": "union_of_current_and_baseline_top_k_using_global_deterministic_ranks",
        "by_k": results,
    }


def _variant_specs(
    transition_k: Sequence[int], softmax_temperature: float
) -> list[dict[str, Any]]:
    values = tuple(int(value) for value in transition_k)
    _require(
        values == tuple(sorted(set(values))) and bool(values),
        "RANK_TRANSITION_K_SWEEP_INVALID",
    )
    specs = [
        {
            "variant_id": f"rank_dcg_k{value}",
            "kind": "rank_dcg",
            "transition_k": value,
            "temperature": None,
        }
        for value in values
    ]
    specs.append(
        {
            "variant_id": "zscore_softmax_t1",
            "kind": "zscore_softmax",
            "transition_k": None,
            "temperature": float(softmax_temperature),
        }
    )
    return specs


def compute_rank_normalized_scores(
    capture: MatrixCapture,
    *,
    q1_ranges: Sequence[Sequence[int]],
    q2_ranges: Sequence[Sequence[int]],
    max_level: int,
    transition_k: Sequence[int] = DEFAULT_TRANSITION_K,
    evaluation_k: Sequence[int] = DEFAULT_EVALUATION_K,
    softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
) -> dict[str, Any]:
    _require(
        isinstance(max_level, int) and 1 <= max_level <= MAX_PROPAGATION_LEVEL,
        "RANK_PROPAGATION_LEVEL_OUT_OF_RANGE",
    )
    q1, q2 = validate_seed_ranges(q1_ranges, q2_ranges, capture.window)
    start, end = capture.window
    width = end - start
    evaluation = tuple(int(value) for value in evaluation_k)
    _require(
        evaluation == tuple(sorted(set(evaluation)))
        and bool(evaluation)
        and all(1 <= value <= width for value in evaluation),
        "RANK_EVALUATION_K_INVALID",
    )
    q1_positions = {position for left, right in q1 for position in range(left, right)}
    q2_positions = {position for left, right in q2 for position in range(left, right)}
    seed_indices = [
        position - start
        for position in range(start, end)
        if position in q1_positions or position in q2_positions
    ]
    _require(bool(seed_indices), "RANK_PROPAGATION_SEED_EMPTY")
    seed = [0.0] * width
    for index in seed_indices:
        seed[index] = 1.0 / len(seed_indices)
    _require(abs(math.fsum(seed) - 1.0) <= MASS_TOLERANCE, "RANK_SEED_MASS_INVALID")

    variants: list[dict[str, Any]] = []
    for spec in _variant_specs(transition_k, softmax_temperature):
        transition = build_layer_averaged_transition(
            capture,
            kind=spec["kind"],
            transition_k=spec["transition_k"],
            temperature=spec["temperature"],
        )
        current = seed
        running = [0.0] * width
        hop_distributions: list[list[float]] = []
        cumulative_distributions: list[list[float]] = []
        pre_normalization_mass: list[float] = []
        for level in range(1, max_level + 1):
            current, observed_mass = propagate_distribution(current, transition)
            hop_distributions.append(current)
            pre_normalization_mass.append(observed_mass)
            for index, value in enumerate(current):
                running[index] += value
            cumulative, cumulative_mass = normalize_distribution(
                running, reason="RANK_PROPAGATION_CUMULATIVE"
            )
            _require(
                abs(cumulative_mass - level) <= MASS_TOLERANCE * max(level, 1),
                "RANK_PROPAGATION_EQUAL_HOP_MIXTURE_MASS_INVALID",
            )
            cumulative_distributions.append(cumulative)

        hop_rank_by_level = []
        cumulative_rank_by_level = []
        level_results = []
        for level_index in range(max_level):
            hop = hop_distributions[level_index]
            cumulative = cumulative_distributions[level_index]
            hop_order = deterministic_order(hop)
            cumulative_order = deterministic_order(cumulative)
            hop_rank_by_level.append(
                [rank for rank, _ in sorted(enumerate(hop_order, 1), key=lambda item: item[1])]
            )
            cumulative_rank_by_level.append(
                [
                    rank
                    for rank, _ in sorted(
                        enumerate(cumulative_order, 1), key=lambda item: item[1]
                    )
                ]
            )
            hop_previous = (
                None
                if level_index == 0
                else rank_stability(
                    hop, hop_distributions[level_index - 1], evaluation_k=evaluation
                )
            )
            cumulative_previous = (
                None
                if level_index == 0
                else rank_stability(
                    cumulative,
                    cumulative_distributions[level_index - 1],
                    evaluation_k=evaluation,
                )
            )
            level_results.append(
                {
                    "level": level_index + 1,
                    "pre_normalization_outgoing_mass": pre_normalization_mass[
                        level_index
                    ],
                    "terminal_mass_discarded_before_normalization": 1.0
                    - pre_normalization_mass[level_index],
                    "hop": {
                        **distribution_summary(hop),
                        "versus_level_1": rank_stability(
                            hop, hop_distributions[0], evaluation_k=evaluation
                        ),
                        "versus_previous_level": hop_previous,
                    },
                    "cumulative_equal_hop_mixture": {
                        **distribution_summary(cumulative),
                        "versus_level_1": rank_stability(
                            cumulative,
                            cumulative_distributions[0],
                            evaluation_k=evaluation,
                        ),
                        "versus_previous_level": cumulative_previous,
                    },
                }
            )
        variants.append(
            {
                **spec,
                "hop_distributions": hop_distributions,
                "cumulative_distributions": cumulative_distributions,
                "hop_rank_by_level": hop_rank_by_level,
                "cumulative_rank_by_level": cumulative_rank_by_level,
                "level_results": level_results,
            }
        )
    return {
        "recurrence_id": RECURRENCE_ID,
        "window": [start, end],
        "layers": list(capture.layers),
        "q1_ranges": [list(item) for item in q1],
        "q2_ranges": [list(item) for item in q2],
        "seed": seed,
        "seed_count": len(seed_indices),
        "max_level": max_level,
        "transition_k": list(int(value) for value in transition_k),
        "evaluation_k": list(evaluation),
        "softmax_temperature": float(softmax_temperature),
        "variants": variants,
    }


def _project_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    value = result.stdout.strip()
    _require(result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) is not None, "PROJECT_COMMIT_UNAVAILABLE")
    return value


def _artifact_document(path: Path, relative: str) -> dict[str, Any]:
    return {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def write_rank_normalized_report(
    *,
    episode: Mapping[str, Any],
    episode_attestation: Mapping[str, Any],
    source_episode_attestation: Mapping[str, Any],
    capture: MatrixCapture,
    capture_attestations: Sequence[Mapping[str, Any]],
    control_attestations: Sequence[Mapping[str, Any]],
    legacy_report_attestation: Mapping[str, Any],
    legacy_token_attestation: Mapping[str, Any],
    checksum_manifest_attestation: Mapping[str, Any],
    scores: Mapping[str, Any],
    output_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_root).resolve()
    _require(not output.exists(), "RANK_PROPAGATION_OUTPUT_ALREADY_EXISTS")
    output.mkdir(parents=True)
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

    token_path = output / "rank-normalized-multihop-token-scores.jsonl"
    token_bytes = bytearray()
    for index, position in enumerate(range(start, end)):
        memberships = []
        if position in q1_positions:
            memberships.append("q1")
        if position in q2_positions:
            memberships.append("q2")
        variants = []
        for variant in scores["variants"]:
            levels = []
            for level_index in range(scores["max_level"]):
                levels.append(
                    {
                        "level": level_index + 1,
                        "hop_probability": variant["hop_distributions"][level_index][index],
                        "hop_rank": variant["hop_rank_by_level"][level_index][index],
                        "cumulative_probability": variant["cumulative_distributions"][level_index][index],
                        "cumulative_rank": variant["cumulative_rank_by_level"][level_index][index],
                    }
                )
            variants.append({"variant_id": variant["variant_id"], "levels": levels})
        row = {
            "schema_version": 1,
            "position": position,
            "token_id": capture.token_ids[index],
            "segment_membership": memberships,
            "seed_probability": scores["seed"][index],
            "variants": variants,
        }
        row["record_sha256"] = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
        _validate_schema(row, TOKEN_ROW_SCHEMA)
        token_bytes.extend(canonical_json_bytes(row))
    token_path.write_bytes(bytes(token_bytes))

    attestation_payload = {
        "schema_version": 1,
        "artifact_kind": "glm52_rank_normalized_multihop_input_attestation",
        "project_commit": _project_commit(),
        "checksum_manifest": checksum_manifest_attestation,
        "episode_inputs": [episode_attestation, source_episode_attestation],
        "capture_control_inputs": list(control_attestations),
        "matrix_inputs": list(capture_attestations),
        "matrix_input_count": len(capture_attestations),
        "legacy_signed_inputs": [legacy_report_attestation, legacy_token_attestation],
        "all_inputs_checksum_attested": True,
    }
    _validate_schema(attestation_payload, ATTESTATION_SCHEMA)
    attestation_path = output / "rank-normalized-input-attestation.json"
    attestation_path.write_text(
        json.dumps(attestation_payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )

    variant_results = []
    for variant in scores["variants"]:
        variant_results.append(
            {
                "variant_id": variant["variant_id"],
                "kind": variant["kind"],
                "transition_k": variant["transition_k"],
                "temperature": variant["temperature"],
                "level_results": variant["level_results"],
            }
        )
    payload = {
        "status": "passed",
        "project_commit": attestation_payload["project_commit"],
        "recurrence": {
            "id": RECURRENCE_ID,
            "legacy_signed_id": LEGACY_RECURRENCE_ID,
            "orientation": "row_vector_backward_causal_h_next[k]=sum_q(h[q]*T[q,k])_for_k_lt_q",
            "rank_row": "descending_native_raw_score_tie_lower_absolute_key_position_top_K_weight_1/log2(rank+1)_then_row_normalize",
            "softmax_row": "population_zscore_then_temperature_1_softmax_over_every_valid_k_lt_q",
            "layer_aggregation": "uniform_mean_of_independently_normalized_layer_rows",
            "seed": "uniform_distribution_over_all_and_only_frozen_Q1_plus_Q2_token_rows",
            "level_1": "h_1=normalize(u*T)",
            "additional_level": "h_n_plus_1=normalize(h_n*T)",
            "terminal_policy": "mass_arriving_at_rows_without_earlier_keys_is_reported_then_remaining_outgoing_mass_is_renormalized",
            "cumulative": "g_L=(1/L)*sum(h_n_for_n_1_through_L)_equal_hop_mixture",
            "nonnegative": True,
            "normalization_tolerance": MASS_TOLERANCE,
        },
        "transition_config": {
            "primary_variant_id": PRIMARY_VARIANT_ID,
            "rank_transition_k_sweep": scores["transition_k"],
            "softmax_temperature": scores["softmax_temperature"],
            "evaluation_top_k": scores["evaluation_k"],
            "max_level": scores["max_level"],
        },
        "metric_policy": {
            "primary": "top_k_overlap_retention_jaccard_and_ndcg_against_baseline_rank",
            "baselines": ["level_1", "previous_level"],
            "distribution_views": ["hop_only", "cumulative_equal_hop_mixture"],
            "secondary": "Spearman_Pearson_correlation_of_global_deterministic_ranks_on_explicit_top_k_union",
            "entropy": "natural_log_Shannon_entropy_and_exp_entropy_effective_support",
        },
        "episode": {
            "scenario_id": episode["scenario_id"],
            "instance_id": episode["instance_id"],
            "benchmark_provenance": episode["benchmark_provenance"],
            "window": scores["window"],
            "q1_ranges": scores["q1_ranges"],
            "q2_ranges": scores["q2_ranges"],
            "seed_count": scores["seed_count"],
            "layers": scores["layers"],
        },
        "capture": {
            "score_origin": "kernel_native_fp8_fp4_mqa_logits_before_top_k_per_row_prefill",
            "strict_causal": True,
            "tp_max_abs_difference_observed": capture.tp_max_abs_difference_observed,
            "input_attestation_artifact": _artifact_document(
                attestation_path, attestation_path.name
            ),
        },
        "legacy_signed_evidence": {
            "preserved_immutable": True,
            "scientific_status": "legacy_auditable_but_not_selector_relevance_metric_due_to_signed_cancellation_and_scale_explosion",
            "report": legacy_report_attestation,
            "token_rows": legacy_token_attestation,
        },
        "variant_results": variant_results,
        "token_level_artifact": {
            **_artifact_document(token_path, token_path.name),
            "record_count": end - start,
            "schema_path": "configs/runpod/schemas/glm52_rank_normalized_multihop_token_row.schema.json",
            "schema_sha256": file_sha256(TOKEN_ROW_SCHEMA),
        },
        "scientific_result_policy": "offline_rank_relevance_analysis_only_no_threshold_no_inference_decision_no_stateful_quality_claim",
    }
    report = {
        "schema_version": 1,
        "report_kind": "glm52_rank_normalized_indexer_multihop_offline",
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }
    _validate_schema(report, REPORT_SCHEMA)
    report_path = output / "rank-normalized-multihop-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checksum_paths = [attestation_path, report_path, token_path]
    (output / "REPORT_SHA256SUMS").write_text(
        "".join(
            f"{file_sha256(path)}  {path.name}\n"
            for path in sorted(checksum_paths, key=lambda item: item.name)
        ),
        encoding="utf-8",
    )
    return report


def score_rank_normalized_evidence(
    *,
    evidence_root: str | Path,
    checksum_manifest: str,
    episode_path: str,
    capture_root: str,
    matrix_run_path: str,
    matrix_config_path: str,
    legacy_report_path: str,
    legacy_token_path: str,
    output_root: str | Path,
    max_level: int = 6,
    transition_k: Sequence[int] = DEFAULT_TRANSITION_K,
    evaluation_k: Sequence[int] = DEFAULT_EVALUATION_K,
    softmax_temperature: float = DEFAULT_SOFTMAX_TEMPERATURE,
) -> dict[str, Any]:
    root = Path(evidence_root).resolve()
    _require(root.is_dir(), "RANK_PROPAGATION_EVIDENCE_ROOT_MISSING")
    manifest_path, manifest_relative, manifest = load_checksum_manifest(
        root, checksum_manifest
    )
    episode_file, episode_relative = _resolve_file(
        root, episode_path, reason="RANK_PROPAGATION_EPISODE_PATH_INVALID"
    )
    capture_directory, capture_relative = _resolve_directory(
        root, capture_root, reason="RANK_PROPAGATION_CAPTURE_ROOT_INVALID"
    )
    matrix_run_file, matrix_run_relative = _resolve_file(
        root, matrix_run_path, reason="RANK_PROPAGATION_MATRIX_RUN_PATH_INVALID"
    )
    matrix_config_file, matrix_config_relative = _resolve_file(
        root, matrix_config_path, reason="RANK_PROPAGATION_MATRIX_CONFIG_PATH_INVALID"
    )
    legacy_report_file, legacy_report_relative = _resolve_file(
        root, legacy_report_path, reason="RANK_PROPAGATION_LEGACY_REPORT_PATH_INVALID"
    )
    legacy_token_file, legacy_token_relative = _resolve_file(
        root, legacy_token_path, reason="RANK_PROPAGATION_LEGACY_TOKEN_PATH_INVALID"
    )
    episode = load_matrix_episode_manifest(episode_file)
    source_relative_from_episode = _safe_relative(
        episode["source_frozen_episode_manifest"]["path"],
        reason="RANK_PROPAGATION_SOURCE_EPISODE_PATH_INVALID",
    )
    source_file = (episode_file.parent / source_relative_from_episode).resolve()
    source_relative = source_file.relative_to(root).as_posix()
    _require(source_file.is_file() and not source_file.is_symlink(), "RANK_PROPAGATION_SOURCE_EPISODE_PATH_INVALID")

    central = [
        _attest_file(episode_file, episode_relative, manifest),
        _attest_file(source_file, source_relative, manifest),
        _attest_file(matrix_run_file, matrix_run_relative, manifest),
        _attest_file(matrix_config_file, matrix_config_relative, manifest),
        _attest_file(legacy_report_file, legacy_report_relative, manifest),
        _attest_file(legacy_token_file, legacy_token_relative, manifest),
    ]
    matrix_run = json.loads(matrix_run_file.read_text(encoding="utf-8"))
    matrix_config = json.loads(matrix_config_file.read_text(encoding="utf-8"))
    _require(
        matrix_run.get("status") == "captured"
        and matrix_run.get("episode_sha256") == file_sha256(episode_file)
        and matrix_run.get("instrumentation_config_sha256")
        == file_sha256(matrix_config_file)
        and matrix_config.get("capture_mode") == "strict_causal_indexer_matrix"
        and matrix_config.get("propagation_window") == episode["propagation_window"]
        and matrix_config.get("layers") == episode["capture"]["layers"],
        "RANK_PROPAGATION_CAPTURE_CONTROL_MISMATCH",
    )
    legacy = json.loads(legacy_report_file.read_text(encoding="utf-8"))
    _require(
        legacy.get("report_kind") == "glm52_raw_indexer_multihop_offline"
        and legacy.get("payload", {}).get("status") == "passed"
        and legacy.get("payload", {}).get("recurrence", {}).get("id")
        == LEGACY_RECURRENCE_ID
        and legacy.get("payload", {}).get("token_level_artifact", {}).get("sha256")
        == file_sha256(legacy_token_file),
        "RANK_PROPAGATION_LEGACY_EVIDENCE_INVALID",
    )

    capture = load_indexer_matrix_capture(
        capture_directory,
        episode,
        layers=episode["capture"]["layers"],
        window=episode["propagation_window"],
    )
    capture_attestations = []
    for item in capture.input_files:
        relative = (Path(capture_relative) / str(item["path"])).as_posix()
        path = root / relative
        attested = _attest_file(path, relative, manifest)
        _require(
            attested["bytes"] == item["bytes"]
            and attested["sha256"] == item["sha256"],
            "RANK_PROPAGATION_LOADER_ATTESTATION_MISMATCH",
        )
        capture_attestations.append(attested)
    scores = compute_rank_normalized_scores(
        capture,
        q1_ranges=episode["segments"]["q1_ranges"],
        q2_ranges=episode["segments"]["q2_ranges"],
        max_level=max_level,
        transition_k=transition_k,
        evaluation_k=evaluation_k,
        softmax_temperature=softmax_temperature,
    )
    return write_rank_normalized_report(
        episode=episode,
        episode_attestation=central[0],
        source_episode_attestation=central[1],
        capture=capture,
        capture_attestations=capture_attestations,
        control_attestations=central[2:4],
        legacy_report_attestation=central[4],
        legacy_token_attestation=central[5],
        checksum_manifest_attestation=_artifact_document(
            manifest_path, manifest_relative
        ),
        scores=scores,
        output_root=output_root,
    )


def _parse_ints(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--checksum-manifest", required=True)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--matrix-run", required=True)
    parser.add_argument("--matrix-config", required=True)
    parser.add_argument("--legacy-report", required=True)
    parser.add_argument("--legacy-token-rows", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--max-level", type=int, default=6)
    parser.add_argument("--transition-k", type=_parse_ints, default=list(DEFAULT_TRANSITION_K))
    parser.add_argument("--evaluation-k", type=_parse_ints, default=list(DEFAULT_EVALUATION_K))
    parser.add_argument("--softmax-temperature", type=float, default=DEFAULT_SOFTMAX_TEMPERATURE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = score_rank_normalized_evidence(
        evidence_root=args.evidence_root,
        checksum_manifest=args.checksum_manifest,
        episode_path=args.episode,
        capture_root=args.capture_root,
        matrix_run_path=args.matrix_run,
        matrix_config_path=args.matrix_config,
        legacy_report_path=args.legacy_report,
        legacy_token_path=args.legacy_token_rows,
        output_root=args.output_root,
        max_level=args.max_level,
        transition_k=args.transition_k,
        evaluation_k=args.evaluation_k,
        softmax_temperature=args.softmax_temperature,
    )
    print(
        json.dumps(
            {
                "status": report["payload"]["status"],
                "report_kind": report["report_kind"],
                "payload_sha256": report["payload_sha256"],
                "output_root": str(Path(args.output_root).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
