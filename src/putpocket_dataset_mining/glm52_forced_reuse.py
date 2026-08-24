"""Freeze a stability-ranked selector for the GLM forced-reuse ablation.

This is deliberately an offline evidence transform.  It reads only native DSA
capture JSONL files and never reads completions, benchmark scores, or evaluator
outcomes.  The output is therefore suitable for attestation before the unsafe
ratio sweep starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROMPT_TOKENS = 2071
EDIT_POSITION = 114
OLD_TOKEN_ID = 17526
NEW_TOKEN_ID = 11660
DOWNSTREAM_START = 115
DOWNSTREAM_END = 2071
RATIOS = tuple(range(0, 101, 10))
FULL_LAYERS = (0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74)
SAMPLE_POINTS = ("prefill_last_query", "decode_0", "decode_1", "decode_8", "decode_32")


class SelectorError(RuntimeError):
    """Native evidence cannot safely produce the frozen selector."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SelectorError(reason)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(values: Sequence[Any]) -> bool:
    return all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in values)


def _rank(values: Sequence[float]) -> list[float]:
    """Average ranks for ties, ascending; sufficient for Spearman."""
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + end - 1) / 2.0
        for index in order[cursor:end]:
            result[index] = average
        cursor = end
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    _require(len(left) == len(right) and len(left) > 1, "CORRELATION_VECTOR_INVALID")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    numerator = math.fsum(a * b for a, b in zip(left_delta, right_delta, strict=True))
    denominator = math.sqrt(math.fsum(value * value for value in left_delta) * math.fsum(value * value for value in right_delta))
    return numerator / denominator if denominator else (1.0 if left == right else 0.0)


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    _require(bool(sorted_values), "PERCENTILE_VECTOR_EMPTY")
    coordinate = (len(sorted_values) - 1) * percentile
    lower = int(math.floor(coordinate))
    upper = int(math.ceil(coordinate))
    if lower == upper:
        return float(sorted_values[lower])
    weight = coordinate - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(sorted_values[upper]) * weight


def _descending_ranks(values: Sequence[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: (-float(values[index]), index))
    result = [0] * len(values)
    for rank, index in enumerate(order):
        result[index] = rank
    return result


def _record_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    return int(record["rank"]), int(record["layer"]), str(record["sample_point"])


def _load_records(root: Path) -> tuple[dict[tuple[int, int, str], dict[str, Any]], list[dict[str, Any]]]:
    _require(root.is_absolute() and root.is_dir(), "CAPTURE_ROOT_INVALID")
    blocked = sorted(root.glob("BLOCKED*"))
    _require(not blocked, "CAPTURE_ROOT_HAS_BLOCKED_MARKER")
    paths = sorted(root.glob("captures.rank-*.jsonl"))
    _require(bool(paths), "CAPTURE_JSONL_MISSING")
    records: dict[tuple[int, int, str], dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    for path in paths:
        files.append({"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)})
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                record = json.loads(line)
                recorded = record.pop("record_sha256", None)
                encoded = (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode()
                _require(recorded == hashlib.sha256(encoded).hexdigest(), f"CAPTURE_RECORD_DIGEST_INVALID:{path.name}:{line_number}")
                key = _record_key(record)
                _require(key not in records, "CAPTURE_DUPLICATE_CELL")
                _require(key[1] in FULL_LAYERS and key[2] in SAMPLE_POINTS, "CAPTURE_CELL_UNEXPECTED")
                raw = record.get("raw_scores")
                selected = record.get("selected_ids")
                _require(isinstance(raw, list) and len(raw) >= PROMPT_TOKENS and _finite(raw), "CAPTURE_RAW_INVALID")
                _require(isinstance(selected, list) and len(selected) == 2048 and len(set(selected)) == 2048, "CAPTURE_SELECTED_INVALID")
                _require(all(isinstance(value, int) and 0 <= value < len(raw) for value in selected), "CAPTURE_SELECTED_BOUNDS_INVALID")
                record["record_sha256"] = recorded
                records[key] = record
    ranks = sorted({key[0] for key in records})
    expected = {(rank, layer, sample) for rank in ranks for layer in FULL_LAYERS for sample in SAMPLE_POINTS}
    _require(set(records) == expected and len(ranks) >= 2, "CAPTURE_COVERAGE_INCOMPLETE")
    return records, files


def _prompt_ids(root: Path, explicit: Path | None) -> tuple[list[int], Path]:
    candidates = ([explicit] if explicit is not None else []) + [root / "prompt-token-ids.json", root / "prompt_token_ids.json", root / "artifacts" / "prompt-token-ids.json"]
    for path in candidates:
        if path is None or not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("prompt"), list):
            values = payload["prompt"]
        else:
            values = payload.get("prompt_token_ids", payload.get("token_ids", payload)) if isinstance(payload, dict) else payload
        _require(isinstance(values, list) and len(values) == PROMPT_TOKENS and all(isinstance(value, int) for value in values), "PROMPT_TOKEN_ARTIFACT_INVALID")
        return values, path.resolve()
    raise SelectorError("PROMPT_TOKEN_ID_ATTESTATION_MISSING")


def _assert_tp_raw_consensus(records: Mapping[tuple[int, int, str], Mapping[str, Any]]) -> dict[str, Any]:
    by_group: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for (_, layer, sample), record in records.items():
        by_group[(layer, sample)].append(record)
    for key, group in by_group.items():
        reference = group[0]["raw_scores"]
        _require(all(record["raw_scores"] == reference for record in group[1:]), f"TP_RAW_SCORE_CONSENSUS_FAILED:{key}")
    return {"rank_count": len({key[0] for key in records}), "group_count": len(by_group), "raw_score_exact_consensus": True}


def build_selector(
    baseline_root: Path,
    edited_root: Path,
    output: Path,
    *,
    baseline_prompt_path: Path | None = None,
    edited_prompt_path: Path | None = None,
) -> dict[str, Any]:
    baseline, baseline_files = _load_records(baseline_root)
    edited, edited_files = _load_records(edited_root)
    _require(set(baseline) == set(edited), "BASELINE_EDIT_COVERAGE_MISMATCH")
    baseline_ids, baseline_prompt_source = _prompt_ids(baseline_root, baseline_prompt_path)
    edited_ids, edited_prompt_source = _prompt_ids(edited_root, edited_prompt_path)
    differences = [index for index, (left, right) in enumerate(zip(baseline_ids, edited_ids, strict=True)) if left != right]
    _require(differences == [EDIT_POSITION], "PROMPT_EDIT_NOT_EXACTLY_TOKEN_114")
    _require(baseline_ids[EDIT_POSITION] == OLD_TOKEN_ID and edited_ids[EDIT_POSITION] == NEW_TOKEN_ID, "PROMPT_EDIT_TOKEN_IDS_INVALID")
    baseline_digest = hashlib.sha256(json.dumps(baseline_ids, separators=(",", ":")).encode()).hexdigest()
    edited_digest = hashlib.sha256(json.dumps(edited_ids, separators=(",", ":")).encode()).hexdigest()
    baseline_tp = _assert_tp_raw_consensus(baseline)
    edited_tp = _assert_tp_raw_consensus(edited)

    positions = tuple(range(DOWNSTREAM_START, DOWNSTREAM_END))
    aggregate: dict[int, dict[str, list[float] | int]] = {
        position: {"normalized_delta": [], "rank_shift": [], "cutoff_margin_retention": [], "membership_stable": 0, "membership_both": 0}
        for position in positions
    }
    group_metrics = []
    for key in sorted(baseline):
        before = [float(value) for value in baseline[key]["raw_scores"][:PROMPT_TOKENS]]
        after = [float(value) for value in edited[key]["raw_scores"][:PROMPT_TOKENS]]
        ordered = sorted(before)
        iqr = max(_percentile(ordered, 0.75) - _percentile(ordered, 0.25), 1e-12)
        before_ranks = _descending_ranks(before)
        after_ranks = _descending_ranks(after)
        before_selected = {int(value) for value in baseline[key]["selected_ids"] if int(value) < PROMPT_TOKENS}
        after_selected = {int(value) for value in edited[key]["selected_ids"] if int(value) < PROMPT_TOKENS}
        union = before_selected | after_selected
        intersection = before_selected & after_selected
        before_cutoff = min((before[index] for index in before_selected), default=min(before))
        after_cutoff = min((after[index] for index in after_selected), default=min(after))
        rmse = math.sqrt(statistics.fmean((left - right) ** 2 for left, right in zip(before, after, strict=True))) / iqr
        group_metrics.append({
            "rank": key[0], "layer": key[1], "sample_point": key[2],
            "pearson": _pearson(before, after),
            "spearman": _pearson(_rank(before), _rank(after)),
            "topk_prompt_jaccard": len(intersection) / len(union) if union else 1.0,
            "iqr_normalized_rmse": rmse,
        })
        for position in positions:
            entry = aggregate[position]
            entry["normalized_delta"].append(abs(before[position] - after[position]) / iqr)  # type: ignore[union-attr]
            entry["rank_shift"].append(float(abs(before_ranks[position] - after_ranks[position])))  # type: ignore[union-attr]
            entry["cutoff_margin_retention"].append(min((before[position] - before_cutoff) / iqr, (after[position] - after_cutoff) / iqr))  # type: ignore[union-attr]
            entry["membership_stable"] = int(entry["membership_stable"]) + int((position in before_selected) == (position in after_selected))
            entry["membership_both"] = int(entry["membership_both"]) + int(position in before_selected and position in after_selected)

    token_metrics: dict[int, dict[str, float | int]] = {}
    for position, values in aggregate.items():
        deltas = sorted(values["normalized_delta"])  # type: ignore[arg-type]
        shifts = sorted(values["rank_shift"])  # type: ignore[arg-type]
        margins = sorted(values["cutoff_margin_retention"])  # type: ignore[arg-type]
        token_metrics[position] = {
            "membership_stable_count": int(values["membership_stable"]),
            "membership_both_count": int(values["membership_both"]),
            "max_iqr_normalized_delta": max(deltas),
            "p95_iqr_normalized_delta": _percentile(deltas, 0.95),
            "mean_iqr_normalized_delta": statistics.fmean(deltas),
            "max_rank_shift": int(max(shifts)),
            "p95_rank_shift": _percentile(shifts, 0.95),
            "min_cutoff_margin_retention": min(margins),
        }

    # Attested before outcomes: lexicographic, per-layer/query normalized, and
    # stability/margin based.  The final position term is only a deterministic
    # tie-break; no raw global threshold is used.
    ranking = sorted(
        positions,
        key=lambda position: (
            -int(token_metrics[position]["membership_stable_count"]),
            -int(token_metrics[position]["membership_both_count"]),
            float(token_metrics[position]["max_iqr_normalized_delta"]),
            float(token_metrics[position]["p95_iqr_normalized_delta"]),
            float(token_metrics[position]["max_rank_shift"]),
            float(token_metrics[position]["p95_rank_shift"]),
            -float(token_metrics[position]["min_cutoff_margin_retention"]),
            position,
        ),
    )
    positions_by_ratio = {
        str(ratio): sorted(ranking[: len(positions) * ratio // 100]) for ratio in RATIOS
    }
    group_summary = {
        "count": len(group_metrics),
        "pearson_min": min(value["pearson"] for value in group_metrics),
        "spearman_min": min(value["spearman"] for value in group_metrics),
        "topk_prompt_jaccard_min": min(value["topk_prompt_jaccard"] for value in group_metrics),
        "iqr_normalized_rmse_max": max(value["iqr_normalized_rmse"] for value in group_metrics),
    }
    payload = {
        "schema_version": 1,
        "status": "attested_before_benchmark_outcomes",
        "unsafe_research_ablation": True,
        "scenario": {
            "name": "sr_cc_1_equal_token_system_edit_114",
            "prompt_token_count": PROMPT_TOKENS,
            "edit_position": EDIT_POSITION,
            "old_token_id": OLD_TOKEN_ID,
            "new_token_id": NEW_TOKEN_ID,
            "downstream_range": [DOWNSTREAM_START, DOWNSTREAM_END],
            "same_length": True,
            "rope_positions_unchanged": True,
            "insertion_deletion_supported": False,
        },
        "selection_definition": {
            "requested_ratio_denominator": DOWNSTREAM_END - DOWNSTREAM_START,
            "algorithm": "lexicographic_worst_case_layer_query_stability_v1",
            "uses_global_raw_score_threshold": False,
            "fields_in_order": [
                "membership_stable_count_desc", "membership_both_count_desc",
                "max_iqr_normalized_delta_asc", "p95_iqr_normalized_delta_asc",
                "max_rank_shift_asc", "p95_rank_shift_asc",
                "min_cutoff_margin_retention_desc", "absolute_position_asc_tie_break",
            ],
        },
        "source_evidence": {
            "baseline_root": str(baseline_root),
            "edited_root": str(edited_root),
            "baseline_prompt_token_ids_sha256": baseline_digest,
            "edited_prompt_token_ids_sha256": edited_digest,
            "baseline_prompt_token_ids_path": str(baseline_prompt_source),
            "edited_prompt_token_ids_path": str(edited_prompt_source),
            "baseline_files": baseline_files,
            "edited_files": edited_files,
            "baseline_tp_consensus": baseline_tp,
            "edited_tp_consensus": edited_tp,
            "group_metric_summary": group_summary,
        },
        "ranking": ranking,
        "positions_by_ratio": positions_by_ratio,
        "token_metrics": {str(position): token_metrics[position] for position in positions},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--edited-root", type=Path, required=True)
    parser.add_argument("--baseline-prompt-token-ids", type=Path)
    parser.add_argument("--edited-prompt-token-ids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    build_selector(
        arguments.baseline_root.resolve(),
        arguments.edited_root.resolve(),
        arguments.output.resolve(),
        baseline_prompt_path=arguments.baseline_prompt_token_ids.resolve() if arguments.baseline_prompt_token_ids else None,
        edited_prompt_path=arguments.edited_prompt_token_ids.resolve() if arguments.edited_prompt_token_ids else None,
    )
    print(json.dumps({"status": "passed", "output": str(arguments.output.resolve()), "sha256": _sha256(arguments.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
