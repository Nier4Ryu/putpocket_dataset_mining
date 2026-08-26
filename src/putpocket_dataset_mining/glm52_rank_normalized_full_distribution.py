"""Full-universe analysis for rank-normalized GLM multihop evidence.

This module is an offline postprocessor. It validates the immutable
rank-normalized artifact, retains every token position, and computes scalar
concentration summaries plus identity-aligned Jensen-Shannon distances. It is
not imported by vLLM and does not alter inference or selector decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .constants import REPO_ROOT
from .errors import ConfigError
from .glm52_attention_indexer import canonical_json_bytes, file_sha256


ANALYSIS_ID = "putpocket_rank_normalized_full_distribution_v1"
PRIMARY_VARIANT_ID = "rank_dcg_k64"
MASS_TOLERANCE = 1e-10
JS_TOLERANCE = 1e-12
DEFAULT_LOG_PROBABILITY_FLOOR = 1e-12
REPORT_SCHEMA = (
    REPO_ROOT
    / "configs/runpod/schemas/glm52_rank_normalized_multihop_report.schema.json"
)
TOKEN_SCHEMA = (
    REPO_ROOT
    / "configs/runpod/schemas/glm52_rank_normalized_multihop_token_row.schema.json"
)
ATTESTATION_SCHEMA = (
    REPO_ROOT
    / "configs/runpod/schemas/glm52_rank_normalized_full_distribution_input_attestation.schema.json"
)
SUMMARY_SCHEMA = (
    REPO_ROOT
    / "configs/runpod/schemas/glm52_rank_normalized_full_distribution_summary.schema.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCHEMA_VALIDATORS: dict[Path, jsonschema.Draft202012Validator] = {}


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ConfigError(reason)


def _schema_validator(path: str | Path) -> jsonschema.Draft202012Validator:
    resolved = Path(path).resolve()
    validator = _SCHEMA_VALIDATORS.get(resolved)
    if validator is None:
        schema = json.loads(resolved.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
        _SCHEMA_VALIDATORS[resolved] = validator
    return validator


def validate_schema(value: Mapping[str, Any], path: str | Path) -> None:
    try:
        _schema_validator(path).validate(value)
    except jsonschema.ValidationError as exc:
        raise ConfigError(
            f"FULL_DISTRIBUTION_SCHEMA_INVALID:{Path(path).name}:{exc.json_path}"
        ) from exc


def _safe_relative(value: str, *, reason: str) -> Path:
    path = Path(value)
    _require(
        bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
        reason,
    )
    return path


def _file_item(path: Path, relative: str | None = None) -> dict[str, Any]:
    return {
        "path": relative if relative is not None else path.name,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
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
    _require(
        result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", value) is not None,
        "FULL_DISTRIBUTION_PROJECT_COMMIT_UNAVAILABLE",
    )
    return value


def load_final_checksum_manifest(root: Path) -> tuple[Path, list[dict[str, Any]]]:
    manifest = root / "FINAL_SHA256SUMS"
    _require(
        manifest.is_file() and not manifest.is_symlink(),
        "FULL_DISTRIBUTION_FINAL_MANIFEST_MISSING",
    )
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        fields = line.split(None, 1)
        _require(
            len(fields) == 2 and SHA256_RE.fullmatch(fields[0]) is not None,
            f"FULL_DISTRIBUTION_FINAL_MANIFEST_LINE_INVALID:{line_number}",
        )
        relative = _safe_relative(
            fields[1].lstrip("* "),
            reason=f"FULL_DISTRIBUTION_FINAL_MANIFEST_PATH_INVALID:{line_number}",
        ).as_posix()
        _require(
            relative not in seen and relative != manifest.name,
            "FULL_DISTRIBUTION_FINAL_MANIFEST_PATH_DUPLICATE_OR_SELF",
        )
        seen.add(relative)
        path = (root / relative).resolve()
        try:
            normalized = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ConfigError(
                "FULL_DISTRIBUTION_FINAL_MANIFEST_PATH_ESCAPE"
            ) from exc
        _require(
            normalized == relative and path.is_file() and not path.is_symlink(),
            f"FULL_DISTRIBUTION_FINAL_MANIFEST_FILE_INVALID:{relative}",
        )
        observed = file_sha256(path)
        _require(
            observed == fields[0],
            f"FULL_DISTRIBUTION_FINAL_MANIFEST_DIGEST_MISMATCH:{relative}",
        )
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": observed}
        )
    _require(bool(entries), "FULL_DISTRIBUTION_FINAL_MANIFEST_EMPTY")
    required = {
        "rank-normalized-input-attestation.json",
        "rank-normalized-multihop-report.json",
        "rank-normalized-multihop-token-scores.jsonl",
        "rank-normalized-plot-summary.json",
    }
    _require(
        required.issubset(seen),
        "FULL_DISTRIBUTION_REQUIRED_SOURCE_NOT_ATTESTED",
    )
    return manifest, entries


@dataclass(frozen=True)
class FullDistributionSource:
    root: Path
    report: Mapping[str, Any]
    positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    segment_membership: tuple[tuple[str, ...], ...]
    levels: tuple[int, ...]
    hop: tuple[tuple[float, ...], ...]
    cumulative: tuple[tuple[float, ...], ...]
    source_manifest: Mapping[str, Any]
    source_entries: tuple[Mapping[str, Any], ...]
    token_identity_sha256: str


def validate_probability_matrix(
    matrix: Sequence[Sequence[float]], *, token_count: int, reason: str
) -> tuple[tuple[float, ...], ...]:
    _require(bool(matrix), f"{reason}_LEVELS_EMPTY")
    normalized: list[tuple[float, ...]] = []
    for level_index, row in enumerate(matrix, 1):
        values = tuple(float(value) for value in row)
        _require(
            len(values) == token_count,
            f"{reason}_TOKEN_COUNT_MISMATCH:{level_index}",
        )
        _require(
            all(math.isfinite(value) and value >= 0.0 for value in values),
            f"{reason}_NONFINITE_OR_NEGATIVE:{level_index}",
        )
        _require(
            abs(math.fsum(values) - 1.0) <= MASS_TOLERANCE,
            f"{reason}_MASS_INVALID:{level_index}",
        )
        normalized.append(values)
    return tuple(normalized)


def _expected_membership(position: int, episode: Mapping[str, Any]) -> tuple[str, ...]:
    labels = []
    for label in ("q1", "q2"):
        if any(
            int(left) <= position < int(right)
            for left, right in episode[f"{label}_ranges"]
        ):
            labels.append(label)
    return tuple(labels)


def load_full_distribution_source(source_root: str | Path) -> FullDistributionSource:
    root = Path(source_root).resolve()
    _require(
        root.is_dir() and not root.is_symlink(),
        "FULL_DISTRIBUTION_SOURCE_ROOT_INVALID",
    )
    manifest, entries = load_final_checksum_manifest(root)
    entry_map = {item["path"]: item for item in entries}

    report_path = root / "rank-normalized-multihop-report.json"
    token_path = root / "rank-normalized-multihop-token-scores.jsonl"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_schema(report, REPORT_SCHEMA)
    _require(
        report["payload_sha256"]
        == hashlib.sha256(canonical_json_bytes(report["payload"])).hexdigest(),
        "FULL_DISTRIBUTION_REPORT_PAYLOAD_DIGEST_MISMATCH",
    )
    payload = report["payload"]
    _require(
        payload["recurrence"]["id"]
        == "putpocket_rank_normalized_indexer_multihop_v1"
        and payload["transition_config"]["primary_variant_id"]
        == PRIMARY_VARIANT_ID,
        "FULL_DISTRIBUTION_SOURCE_CONTRACT_INVALID",
    )
    episode = payload["episode"]
    start, end = (int(value) for value in episode["window"])
    token_count = end - start
    max_level = int(payload["transition_config"]["max_level"])
    levels = tuple(range(1, max_level + 1))

    positions: list[int] = []
    token_ids: list[int] = []
    memberships: list[tuple[str, ...]] = []
    hop_by_position: list[tuple[float, ...]] = []
    cumulative_by_position: list[tuple[float, ...]] = []
    token_validator = _schema_validator(TOKEN_SCHEMA)
    with token_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                token_validator.validate(row)
            except jsonschema.ValidationError as exc:
                raise ConfigError(
                    f"FULL_DISTRIBUTION_TOKEN_SCHEMA_INVALID:{exc.json_path}"
                ) from exc
            observed_record = row.pop("record_sha256")
            expected_record = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
            row["record_sha256"] = observed_record
            _require(
                observed_record == expected_record,
                "FULL_DISTRIBUTION_TOKEN_RECORD_DIGEST_MISMATCH",
            )
            variants = [
                item for item in row["variants"] if item["variant_id"] == PRIMARY_VARIANT_ID
            ]
            _require(
                len(variants) == 1,
                "FULL_DISTRIBUTION_PRIMARY_VARIANT_MISSING_OR_DUPLICATE",
            )
            variant_levels = variants[0]["levels"]
            _require(
                [item["level"] for item in variant_levels] == list(levels),
                "FULL_DISTRIBUTION_LEVEL_SEQUENCE_INVALID",
            )
            position = int(row["position"])
            membership = tuple(row["segment_membership"])
            _require(
                membership == _expected_membership(position, episode),
                "FULL_DISTRIBUTION_SEGMENT_MEMBERSHIP_MISMATCH",
            )
            positions.append(position)
            token_ids.append(int(row["token_id"]))
            memberships.append(membership)
            hop_by_position.append(
                tuple(float(item["hop_probability"]) for item in variant_levels)
            )
            cumulative_by_position.append(
                tuple(float(item["cumulative_probability"]) for item in variant_levels)
            )

    _require(
        positions == list(range(start, end))
        and len(positions) == token_count
        and len(positions) == payload["token_level_artifact"]["record_count"]
        and file_sha256(token_path) == payload["token_level_artifact"]["sha256"]
        and entry_map[token_path.name]["sha256"] == file_sha256(token_path),
        "FULL_DISTRIBUTION_TOKEN_UNIVERSE_INVALID",
    )
    hop = validate_probability_matrix(
        tuple(
            tuple(hop_by_position[position][level] for position in range(token_count))
            for level in range(max_level)
        ),
        token_count=token_count,
        reason="FULL_DISTRIBUTION_HOP",
    )
    cumulative = validate_probability_matrix(
        tuple(
            tuple(
                cumulative_by_position[position][level]
                for position in range(token_count)
            )
            for level in range(max_level)
        ),
        token_count=token_count,
        reason="FULL_DISTRIBUTION_CUMULATIVE",
    )
    identity = [
        {
            "position": position,
            "token_id": token_id,
            "segment_membership": list(membership),
        }
        for position, token_id, membership in zip(
            positions, token_ids, memberships, strict=True
        )
    ]
    return FullDistributionSource(
        root=root,
        report=report,
        positions=tuple(positions),
        token_ids=tuple(token_ids),
        segment_membership=tuple(memberships),
        levels=levels,
        hop=hop,
        cumulative=cumulative,
        source_manifest=_file_item(manifest),
        source_entries=tuple(entries),
        token_identity_sha256=hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
    )


def jensen_shannon_distance(
    left: Sequence[float], right: Sequence[float]
) -> float:
    """Return sqrt(base-2 Jensen-Shannon divergence), bounded in [0, 1]."""

    _require(
        len(left) == len(right) and bool(left),
        "FULL_DISTRIBUTION_JS_LENGTH_INVALID",
    )
    p = validate_probability_matrix(
        [left], token_count=len(left), reason="FULL_DISTRIBUTION_JS_LEFT"
    )[0]
    q = validate_probability_matrix(
        [right], token_count=len(right), reason="FULL_DISTRIBUTION_JS_RIGHT"
    )[0]
    divergence = 0.0
    for p_value, q_value in zip(p, q, strict=True):
        mixture = 0.5 * (p_value + q_value)
        if p_value > 0.0:
            divergence += 0.5 * p_value * math.log2(p_value / mixture)
        if q_value > 0.0:
            divergence += 0.5 * q_value * math.log2(q_value / mixture)
    _require(
        math.isfinite(divergence)
        and -JS_TOLERANCE <= divergence <= 1.0 + JS_TOLERANCE,
        "FULL_DISTRIBUTION_JS_DIVERGENCE_OUT_OF_BOUNDS",
    )
    distance = math.sqrt(min(1.0, max(0.0, divergence)))
    _require(
        math.isfinite(distance) and 0.0 <= distance <= 1.0,
        "FULL_DISTRIBUTION_JS_DISTANCE_OUT_OF_BOUNDS",
    )
    return distance


def pairwise_js_distance_matrix(
    distributions: Sequence[Sequence[float]],
) -> list[list[float]]:
    _require(bool(distributions), "FULL_DISTRIBUTION_JS_LEVELS_EMPTY")
    count = len(distributions)
    result = [[0.0] * count for _ in range(count)]
    for left_index in range(count):
        for right_index in range(left_index + 1, count):
            distance = jensen_shannon_distance(
                distributions[left_index], distributions[right_index]
            )
            result[left_index][right_index] = distance
            result[right_index][left_index] = distance
    _require(
        all(result[index][index] == 0.0 for index in range(count))
        and all(
            abs(result[left][right] - result[right][left]) <= JS_TOLERANCE
            for left in range(count)
            for right in range(count)
        )
        and all(
            0.0 <= value <= 1.0 for row in result for value in row
        ),
        "FULL_DISTRIBUTION_JS_MATRIX_INVARIANT_FAILED",
    )
    return result


def distribution_scalars(distribution: Sequence[float]) -> dict[str, Any]:
    values = validate_probability_matrix(
        [distribution],
        token_count=len(distribution),
        reason="FULL_DISTRIBUTION_SCALAR",
    )[0]
    positives = [value for value in values if value > 0.0]
    entropy = -math.fsum(value * math.log(value) for value in positives)
    ascending = sorted(values)
    width = len(values)
    gini = (
        2.0
        * math.fsum((index + 1) * value for index, value in enumerate(ascending))
        / width
        - (width + 1.0) / width
    )
    result = {
        "mass": math.fsum(values),
        "entropy_nats": entropy,
        "effective_support": math.exp(entropy),
        "nonzero_support": len(positives),
        "zero_count": width - len(positives),
        "maximum_probability": max(values),
        "minimum_positive_probability": min(positives),
        "herfindahl_concentration": math.fsum(value * value for value in values),
        "gini_concentration": gini,
        "sorted_point_count": width,
        "sorted_cumulative_final_mass": math.fsum(sorted(values, reverse=True)),
    }
    _require(
        all(
            math.isfinite(value)
            for value in result.values()
            if isinstance(value, float)
        )
        and -JS_TOLERANCE <= gini <= 1.0 + JS_TOLERANCE,
        "FULL_DISTRIBUTION_SCALAR_NONFINITE_OR_INVALID",
    )
    return result


def build_full_distribution_analysis(source: FullDistributionSource) -> dict[str, Any]:
    views = {"hop": source.hop, "cumulative_equal_hop_mixture": source.cumulative}
    scalars = {
        view: [
            {"level": level, **distribution_scalars(distribution)}
            for level, distribution in zip(source.levels, distributions, strict=True)
        ]
        for view, distributions in views.items()
    }
    matrices = {
        view: pairwise_js_distance_matrix(distributions)
        for view, distributions in views.items()
    }
    return {
        "levels": list(source.levels),
        "token_count": len(source.positions),
        "scalar_summaries": scalars,
        "jensen_shannon_distance": matrices,
    }


def build_input_attestation(source: FullDistributionSource) -> dict[str, Any]:
    payload = source.report["payload"]
    value = {
        "schema_version": 1,
        "artifact_kind": "glm52_rank_normalized_full_distribution_input_attestation",
        "analysis_project_commit": _project_commit(),
        "source_artifact_label": source.root.name,
        "source_project_commit": payload["project_commit"],
        "source_report_payload_sha256": source.report["payload_sha256"],
        "source_final_checksum_manifest": source.source_manifest,
        "source_manifest_entries": list(source.source_entries),
        "source_manifest_entry_count": len(source.source_entries),
        "all_source_entries_verified": True,
        "primary_variant_id": PRIMARY_VARIANT_ID,
        "token_count": len(source.positions),
        "levels": list(source.levels),
        "token_identity_sha256": source.token_identity_sha256,
    }
    validate_schema(value, ATTESTATION_SCHEMA)
    return value


def build_summary_report(
    *,
    source: FullDistributionSource,
    analysis: Mapping[str, Any],
    attestation_artifact: Mapping[str, Any],
    implementation: Sequence[Mapping[str, Any]],
    figures: Sequence[Mapping[str, Any]],
    log_probability_floor: float,
) -> dict[str, Any]:
    _require(
        math.isfinite(log_probability_floor)
        and 0.0 < log_probability_floor < 1.0,
        "FULL_DISTRIBUTION_LOG_FLOOR_INVALID",
    )
    episode = source.report["payload"]["episode"]
    payload = {
        "status": "passed",
        "analysis_id": ANALYSIS_ID,
        "analysis_project_commit": _project_commit(),
        "source_project_commit": source.report["payload"]["project_commit"],
        "source_report_payload_sha256": source.report["payload_sha256"],
        "input_attestation_artifact": dict(attestation_artifact),
        "implementation": list(implementation),
        "primary_variant_id": PRIMARY_VARIANT_ID,
        "token_universe": {
            "window": list(episode["window"]),
            "token_count": analysis["token_count"],
            "positions_complete_and_ordered": True,
            "token_identity_sha256": source.token_identity_sha256,
            "q1_ranges": episode["q1_ranges"],
            "q2_ranges": episode["q2_ranges"],
            "all_tokens_included": True,
            "top_k_truncation": False,
        },
        "levels": analysis["levels"],
        "formulas": {
            "position_heatmap": f"log10(max(probability,{log_probability_floor:.0e}))",
            "position_heatmap_fixed_limits": [
                math.log10(log_probability_floor),
                0.0,
            ],
            "sorted_curve": "sort_all_token_probabilities_descending_tie_lower_absolute_position; retain_N_points_and_full_cumulative_mass",
            "js_divergence": "JSD_2(P,Q)=0.5*KL_2(P||M)+0.5*KL_2(Q||M),M=(P+Q)/2",
            "js_distance": "sqrt(JSD_2(P,Q)); identity_aligned_over_all_token_positions; range_[0,1]",
            "entropy": "-sum_i(p_i*ln(p_i))",
            "effective_support": "exp(entropy_nats)",
            "gini": "standard_probability_vector_Gini_over_all_tokens",
        },
        "scalar_summaries": analysis["scalar_summaries"],
        "jensen_shannon_distance": analysis["jensen_shannon_distance"],
        "figures": list(figures),
        "claim_boundary": [
            "full_token_universe_offline_analysis",
            "identity_preserved_only_in_position_heatmaps_and_js",
            "sorted_curves_compare_shape_not_token_identity",
            "no_top_k_truncation",
            "no_signed_raw_recurrence",
            "no_inference_or_selector_integration",
            "no_stateful_cache_or_task_quality_claim",
            "no_gpu_or_runpod_rerun",
        ],
    }
    report = {
        "schema_version": 1,
        "report_kind": "glm52_rank_normalized_full_distribution_supplement",
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }
    validate_schema(report, SUMMARY_SCHEMA)
    return report
