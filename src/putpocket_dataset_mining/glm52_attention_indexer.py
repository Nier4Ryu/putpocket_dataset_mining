from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .errors import ConfigError


def canonical_json_bytes(value: Any, *, newline: bool = True) -> bytes:
    suffix = "\n" if newline else ""
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + suffix
    ).encode("utf-8")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ConfigError(reason)


def _finite(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    _require(result and all(math.isfinite(value) for value in result), "SCORE_VECTOR_NONFINITE_OR_EMPTY")
    return result


def mean_vector(rows: Sequence[Sequence[float]]) -> list[float]:
    _require(bool(rows), "HEAD_MATRIX_EMPTY")
    width = len(rows[0])
    _require(width > 0 and all(len(row) == width for row in rows), "HEAD_MATRIX_RAGGED")
    return [statistics.fmean(float(row[index]) for row in rows) for index in range(width)]


def softmax(values: Sequence[float]) -> list[float]:
    vector = _finite(values)
    maximum = max(vector)
    exponentials = [math.exp(value - maximum) for value in vector]
    denominator = sum(exponentials)
    _require(math.isfinite(denominator) and denominator > 0, "SOFTMAX_NORMALIZATION_INVALID")
    return [value / denominator for value in exponentials]


def mean_head_softmax(rows: Sequence[Sequence[float]]) -> list[float]:
    return mean_vector([softmax(row) for row in rows])


def zscore(values: Sequence[float]) -> list[float]:
    vector = _finite(values)
    mean = statistics.fmean(vector)
    variance = statistics.fmean((value - mean) ** 2 for value in vector)
    if variance == 0:
        return [0.0 for _ in vector]
    scale = math.sqrt(variance)
    return [(value - mean) / scale for value in vector]


def pearson(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = _finite(left), _finite(right)
    _require(len(x) == len(y), "PEARSON_LENGTH_MISMATCH")
    xz, yz = zscore(x), zscore(y)
    if not any(xz) or not any(yz):
        return 0.0
    return statistics.fmean(a * b for a, b in zip(xz, yz, strict=True))


def _average_ranks(values: Sequence[float]) -> list[float]:
    vector = _finite(values)
    ordered = sorted(range(len(vector)), key=lambda index: (vector[index], index))
    result = [0.0] * len(vector)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and vector[ordered[end]] == vector[ordered[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for item in ordered[cursor:end]:
            result[item] = rank
        cursor = end
    return result


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return pearson(_average_ranks(left), _average_ranks(right))


def cosine_after_zscore(left: Sequence[float], right: Sequence[float]) -> float:
    x, y = zscore(left), zscore(right)
    _require(len(x) == len(y), "COSINE_LENGTH_MISMATCH")
    left_norm = math.sqrt(sum(value * value for value in x))
    right_norm = math.sqrt(sum(value * value for value in y))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(x, y, strict=True)) / (left_norm * right_norm)


def js_divergence(left: Sequence[float], right: Sequence[float]) -> float:
    p, q = _finite(left), _finite(right)
    _require(len(p) == len(q), "JS_LENGTH_MISMATCH")
    _require(abs(sum(p) - 1.0) <= 1e-8 and abs(sum(q) - 1.0) <= 1e-8, "JS_INPUT_NOT_NORMALIZED")
    midpoint = [(a + b) / 2.0 for a, b in zip(p, q, strict=True)]

    def kl(source: Sequence[float]) -> float:
        return sum(value * math.log(value / mean) for value, mean in zip(source, midpoint, strict=True) if value > 0)

    return (kl(p) + kl(q)) / 2.0


def descending_order(values: Sequence[float]) -> list[int]:
    vector = _finite(values)
    return sorted(range(len(vector)), key=lambda index: (-vector[index], index))


def topk_metrics(
    main_relevance: Sequence[float], indexer_scores: Sequence[float], k: int
) -> dict[str, float | int]:
    relevance = _finite(main_relevance)
    predicted = _finite(indexer_scores)
    _require(len(relevance) == len(predicted) and 0 < k <= len(relevance), "TOPK_INPUT_INVALID")
    main_top = descending_order(relevance)[:k]
    predicted_top = descending_order(predicted)[:k]
    overlap = len(set(main_top) & set(predicted_top))

    def dcg(order: Sequence[int]) -> float:
        return sum(
            relevance[index] / math.log2(rank + 2)
            for rank, index in enumerate(order)
        )

    ideal = dcg(main_top)
    return {
        "k": k,
        "overlap_count": overlap,
        "overlap_fraction": overlap / k,
        "recall": overlap / k,
        "ndcg": dcg(predicted_top) / ideal if ideal > 0 else 0.0,
    }


def raw_rank_topk_metrics(
    main_scores: Sequence[float], indexer_scores: Sequence[float], k: int
) -> dict[str, float | int | str]:
    """Compare raw descending orders without routing through lossy softmax."""

    main = _finite(main_scores)
    predicted = _finite(indexer_scores)
    _require(len(main) == len(predicted) and 0 < k <= len(main), "TOPK_INPUT_INVALID")
    main_order = descending_order(main)
    predicted_order = descending_order(predicted)
    main_top = main_order[:k]
    predicted_top = predicted_order[:k]
    overlap = len(set(main_top) & set(predicted_top))
    relevance = [0.0] * len(main)
    for rank, index in enumerate(main_order):
        relevance[index] = float(len(main) - rank)

    def dcg(order: Sequence[int]) -> float:
        return sum(
            relevance[index] / math.log2(rank + 2)
            for rank, index in enumerate(order)
        )

    ideal = dcg(main_top)
    return {
        "k": k,
        "overlap_count": overlap,
        "overlap_fraction": overlap / k,
        "recall": overlap / k,
        "ndcg": dcg(predicted_top) / ideal if ideal > 0 else 0.0,
        "ranking_basis": "raw_descending_pre_softmax",
        "ndcg_relevance": "main_raw_descending_rank_n_to_1",
    }


def probability_diagnostics(values: Sequence[float]) -> dict[str, float | int]:
    probability = _finite(values)
    _require(
        abs(sum(probability) - 1.0) <= 1e-8,
        "PROBABILITY_DIAGNOSTIC_INPUT_NOT_NORMALIZED",
    )
    return {
        "positive_support_count": sum(value > 0.0 for value in probability),
        "underflow_zero_count": sum(value == 0.0 for value in probability),
        "max_probability": max(probability),
        "effective_support_inverse_simpson": 1.0 / sum(
            value * value for value in probability
        ),
    }


def raw_statistics(values: Sequence[float]) -> dict[str, float]:
    vector = sorted(_finite(values))
    return {
        "min": vector[0],
        "max": vector[-1],
        "mean": statistics.fmean(vector),
        "std_population": math.sqrt(
            statistics.fmean((item - statistics.fmean(vector)) ** 2 for item in vector)
        ),
        "p05": percentile(vector, 0.05),
        "p50": percentile(vector, 0.50),
        "p95": percentile(vector, 0.95),
    }


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    _require(bool(sorted_values) and 0 <= fraction <= 1, "PERCENTILE_INPUT_INVALID")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower]) * (1 - weight) + float(sorted_values[upper]) * weight


def compare_aligned_scores(
    main_logits_by_head: Sequence[Sequence[float]],
    indexer_raw_logits: Sequence[float],
    *,
    k_values: Sequence[int],
) -> dict[str, Any]:
    main_rows = [_finite(row) for row in main_logits_by_head]
    indexer = _finite(indexer_raw_logits)
    _require(main_rows and all(len(row) == len(indexer) for row in main_rows), "ALIGNED_SCORE_LENGTH_MISMATCH")
    main_raw = mean_vector(main_rows)
    main_probability = mean_head_softmax(main_rows)
    indexer_probability = softmax(indexer)
    normalized_k = sorted({min(int(k), len(indexer)) for k in k_values if int(k) > 0})
    _require(bool(normalized_k), "TOPK_VALUES_EMPTY")
    return {
        "valid_token_count": len(indexer),
        "main_head_count": len(main_rows),
        "aggregation": {
            "main_raw": "arithmetic_mean_of_per-head_pre-softmax_logits",
            "main_distribution": "arithmetic_mean_of_per-head_softmax_distributions",
            "indexer_raw": "native_learned_weighted_sum_across_indexer_heads",
            "indexer_distribution": "softmax_of_native_aggregated_raw_indexer_logits",
            "cosine_normalization": "independent_population_zscore_then_cosine",
        },
        "raw_statistics": {
            "main": raw_statistics(main_raw),
            "indexer": raw_statistics(indexer),
        },
        "pearson_raw": pearson(main_raw, indexer),
        "spearman_raw": spearman(main_raw, indexer),
        "cosine_zscore": cosine_after_zscore(main_raw, indexer),
        "js_divergence_normalized": js_divergence(main_probability, indexer_probability),
        "topk": [topk_metrics(main_probability, indexer, k) for k in normalized_k],
        "vectors": {
            "main_raw_mean": main_raw,
            "main_probability": main_probability,
            "indexer_raw": indexer,
            "indexer_probability": indexer_probability,
        },
    }


def compare_query_sums(
    main_raw_sum: Sequence[float],
    indexer_raw_sum: Sequence[float],
    *,
    k_values: Sequence[int],
) -> dict[str, Any]:
    """Compare like-for-like sums over the same declared query-token rows."""

    main = _finite(main_raw_sum)
    indexer = _finite(indexer_raw_sum)
    _require(len(main) == len(indexer), "QUERY_SUM_LENGTH_MISMATCH")
    main_native_probability = softmax(main)
    indexer_native_probability = softmax(indexer)
    main_probability = softmax(zscore(main))
    indexer_probability = softmax(zscore(indexer))
    normalized_k = sorted({min(int(k), len(indexer)) for k in k_values if int(k) > 0})
    _require(bool(normalized_k), "TOPK_VALUES_EMPTY")
    return {
        "valid_candidate_token_count": len(main),
        "aggregation": {
            "main_per_query": "arithmetic_mean_across_all_64_main_attention_heads",
            "main_across_queries": "signed_sum_over_every_q1_q2_content_token_query_row",
            "indexer_per_query": "native_learned_weighted_sum_across_32_indexer_heads",
            "indexer_across_queries": "signed_sum_over_the_same_q1_q2_content_token_query_rows",
            "main_distribution": "softmax_of_independently_population_zscored_query_summed_main_raw_logits",
            "indexer_distribution": "softmax_of_independently_population_zscored_query_summed_native_raw_indexer_scores",
            "cosine_normalization": "independent_population_zscore_then_cosine",
            "topk_ranking": "raw_descending_pre_softmax",
            "ndcg_relevance": "main_raw_descending_rank_n_to_1",
            "native_softmax": "diagnostic_only_not_used_for_topk_or_primary_js",
        },
        "raw_statistics": {"main": raw_statistics(main), "indexer": raw_statistics(indexer)},
        "pearson_raw": pearson(main, indexer),
        "spearman_raw": spearman(main, indexer),
        "cosine_zscore": cosine_after_zscore(main, indexer),
        "js_divergence_normalized": js_divergence(main_probability, indexer_probability),
        "js_divergence_native_softmax_diagnostic": js_divergence(
            main_native_probability, indexer_native_probability
        ),
        "normalization_diagnostics": {
            "primary_distribution": "independent_population_zscore_then_softmax",
            "main_native_softmax": probability_diagnostics(main_native_probability),
            "indexer_native_softmax": probability_diagnostics(indexer_native_probability),
        },
        "topk": [raw_rank_topk_metrics(main, indexer, k) for k in normalized_k],
        "vectors": {
            "main_raw_query_sum": main,
            "main_probability": main_probability,
            "indexer_raw_query_sum": indexer,
            "indexer_probability": indexer_probability,
            "main_native_probability": main_native_probability,
            "indexer_native_probability": indexer_native_probability,
        },
    }


def _validate_record_digest(record: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(record)
    observed = value.pop("record_sha256", None)
    expected = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    _require(observed == expected, "CAPTURE_RECORD_DIGEST_MISMATCH")
    return dict(record)


def load_capture_records(root: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    capture_root = Path(root)
    files = sorted(capture_root.glob("capture-rank-*.jsonl"))
    _require(bool(files), "CAPTURE_FILES_MISSING")
    records: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for path in files:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        _require(bool(lines), f"CAPTURE_FILE_EMPTY:{path.name}")
        for line_number, line in enumerate(lines, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"CAPTURE_JSON_INVALID:{path.name}:{line_number}") from exc
            _require(isinstance(value, dict), "CAPTURE_RECORD_NOT_OBJECT")
            records.append(_validate_record_digest(value))
        evidence.append({"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    return records, evidence


def analyze_capture(
    capture_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    *,
    doctor_payload_sha256: str,
) -> dict[str, Any]:
    records, source_files = load_capture_records(capture_root)
    expected_ranks = set(range(int(config["runtime"]["tensor_parallel_size"])))
    expected_layers = set(config["capture"]["layers"])
    expected_queries = set(config["capture"]["query_positions"])
    grouped: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    identities: set[tuple[str, str, str]] = set()
    for record in records:
        kind = record.get("record_kind")
        _require(kind in {"main_attention_reference", "indexer_native_pre_topk"}, "CAPTURE_RECORD_KIND_INVALID")
        layer, query, rank = record.get("layer"), record.get("query_position"), record.get("rank")
        _require(layer in expected_layers and query in expected_queries and rank in expected_ranks, "CAPTURE_COVERAGE_CELL_INVALID")
        grouped[(int(layer), int(query))][str(kind)].append(record)
        identities.add((str(record.get("diagnostic_id")), str(record.get("instance_id")), str(record.get("scenario_id"))))
    _require(len(identities) == 1, "CAPTURE_IDENTITY_DISAGREEMENT")
    expected_cells = {(layer, query) for layer in expected_layers for query in expected_queries}
    _require(set(grouped) == expected_cells, "CAPTURE_CELL_COVERAGE_INCOMPLETE")

    comparisons: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for layer, query in sorted(grouped):
        group = grouped[(layer, query)]
        main_records = sorted(group["main_attention_reference"], key=lambda item: item["rank"])
        indexer_records = sorted(group["indexer_native_pre_topk"], key=lambda item: item["rank"])
        _require(
            {item["rank"] for item in main_records} == expected_ranks
            and {item["rank"] for item in indexer_records} == expected_ranks,
            "CAPTURE_TP_COVERAGE_INCOMPLETE",
        )
        reference_positions = main_records[0]["candidate_positions"]
        reference_tokens = main_records[0]["candidate_token_ids"]
        for item in [*main_records, *indexer_records]:
            _require(
                item["candidate_positions"] == reference_positions
                and item["candidate_token_ids"] == reference_tokens,
                "CAPTURE_TOKEN_ALIGNMENT_MISMATCH",
            )
        indexer_reference = _finite(indexer_records[0]["raw_logits"])
        max_tp_difference = max(
            abs(left - right)
            for item in indexer_records[1:]
            for left, right in zip(indexer_reference, _finite(item["raw_logits"]), strict=True)
        ) if len(indexer_records) > 1 else 0.0
        _require(
            max_tp_difference <= float(config["capture"]["indexer_tp_max_abs_difference"]),
            "INDEXER_TP_REPLICA_DISAGREEMENT",
        )
        main_heads = [
            head
            for item in main_records
            for head in item["raw_logits_by_local_head"]
        ]
        _require(
            len(main_heads) == config["model_layout"]["main_attention_heads"],
            "MAIN_GLOBAL_HEAD_COVERAGE_INVALID",
        )
        metrics = compare_aligned_scores(
            main_heads,
            indexer_reference,
            k_values=config["analysis"]["topk_values"],
        )
        vectors = metrics.pop("vectors")
        cell = {
            "layer": layer,
            "query_position": query,
            "query_token_id": main_records[0]["query_token_id"],
            "candidate_start_position": reference_positions[0],
            "candidate_end_position_inclusive": reference_positions[-1],
            "main_score_origin": main_records[0]["score_origin"],
            "indexer_score_origin": indexer_records[0]["score_origin"],
            "indexer_tp_max_abs_difference": max_tp_difference,
            **metrics,
        }
        comparisons.append(cell)
        main_ranks = _average_ranks(vectors["main_raw_mean"])
        indexer_ranks = _average_ranks(vectors["indexer_raw"])
        for index, position in enumerate(reference_positions):
            token_rows.append(
                {
                    "schema_version": 1,
                    "layer": layer,
                    "query_position": query,
                    "candidate_position": position,
                    "candidate_token_id": reference_tokens[index],
                    "main_raw_mean": vectors["main_raw_mean"][index],
                    "main_probability": vectors["main_probability"][index],
                    "main_rank_ascending": main_ranks[index],
                    "indexer_raw": vectors["indexer_raw"][index],
                    "indexer_probability": vectors["indexer_probability"][index],
                    "indexer_rank_ascending": indexer_ranks[index],
                }
            )

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    token_path = output / "aligned-token-scores.jsonl"
    token_path.write_bytes(b"".join(canonical_json_bytes(row) for row in token_rows))
    scalar_fields = (
        "pearson_raw",
        "spearman_raw",
        "cosine_zscore",
        "js_divergence_normalized",
    )
    aggregate = {}
    for field in scalar_fields:
        values = sorted(float(item[field]) for item in comparisons)
        aggregate[field] = {
            "min": values[0],
            "p25": percentile(values, 0.25),
            "median": percentile(values, 0.50),
            "p75": percentile(values, 0.75),
            "max": values[-1],
            "mean": statistics.fmean(values),
        }
    diagnostic_id, instance_id, scenario_id = next(iter(identities))
    payload = {
        "schema_version": 1,
        "report_id": config["test_id"],
        "status": "passed",
        "scientific_result_policy": "report_only_no_similarity_threshold_failure",
        "diagnostic_id": diagnostic_id,
        "doctor_payload_sha256": doctor_payload_sha256,
        "benchmark_provenance": config["benchmark_provenance"],
        "instance_id": instance_id,
        "scenario_id": scenario_id,
        "probe_boundary": config["probe_boundary"],
        "formula_and_origin": config["formula_and_origin"],
        "capture_files": source_files,
        "token_level_artifact": {
            "path": token_path.name,
            "record_count": len(token_rows),
            "bytes": token_path.stat().st_size,
            "sha256": file_sha256(token_path),
        },
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "aggregate_quantiles": aggregate,
        "hypothesis_interpretation": {
            "supporting_pattern": "consistently positive high rank/correlation and cosine, low normalized JS divergence, and strong top-k recall/NDCG across preregistered layers and queries",
            "refuting_pattern": "weak or negative rank/correlation, high normalized JS divergence, and poor top-k recall/NDCG across layers/queries",
            "threshold_preregistered": False,
            "dissimilarity_is_test_failure": False,
        },
    }
    report = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload, newline=False)).hexdigest(),
    }
    report_path = output / "attention-indexer-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def analyze_query_sum_capture(
    capture_root: str | Path,
    output_root: str | Path,
    config: Mapping[str, Any],
    probe: Mapping[str, Any],
    *,
    doctor_payload_sha256: str,
) -> dict[str, Any]:
    """Validate all Q1/Q2 rows and compare their aligned signed query sums."""

    records, source_files = load_capture_records(capture_root)
    declared = config["query_sum_capture"]
    expected_ranks = set(range(int(config["runtime"]["tensor_parallel_size"])))
    expected_layers = set(declared["layers"])
    q1_ranges = probe["segments"]["q1_ranges"]
    q2_ranges = probe["segments"]["q2_ranges"]
    expected_queries = {
        position
        for start, end in [*q1_ranges, *q2_ranges]
        for position in range(start, end)
    }
    grouped: dict[tuple[int, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    identities: set[tuple[str, str, str]] = set()
    for record in records:
        kind = record.get("record_kind")
        _require(kind in {"main_attention_reference", "indexer_native_pre_topk"}, "CAPTURE_RECORD_KIND_INVALID")
        layer, query, rank = record.get("layer"), record.get("query_position"), record.get("rank")
        _require(layer in expected_layers and query in expected_queries and rank in expected_ranks, "CAPTURE_COVERAGE_CELL_INVALID")
        _require(record.get("probe_kind") == "benchmark_derived_two_query_q1_q2_score_probe", "QUERY_SUM_PROBE_KIND_INVALID")
        _require(record.get("q2_in_probe") is True, "QUERY_SUM_Q2_MISSING")
        grouped[(int(layer), int(query))][str(kind)].append(record)
        identities.add((str(record.get("diagnostic_id")), str(record.get("instance_id")), str(record.get("scenario_id"))))
    _require(len(identities) == 1, "CAPTURE_IDENTITY_DISAGREEMENT")
    expected_cells = {(layer, query) for layer in expected_layers for query in expected_queries}
    _require(set(grouped) == expected_cells, "CAPTURE_CELL_COVERAGE_INCOMPLETE")

    prompt_ids = probe["prompt"]["token_ids"]
    window_start, window_end = probe["propagation_window"]
    per_query: list[dict[str, Any]] = []
    layer_vectors: dict[int, dict[int, list[float]]] = {
        layer: {position: [0.0, 0.0, 0.0] for position in range(window_start, window_end)}
        for layer in expected_layers
    }
    for layer, query in sorted(grouped):
        group = grouped[(layer, query)]
        main_records = sorted(group["main_attention_reference"], key=lambda item: item["rank"])
        indexer_records = sorted(group["indexer_native_pre_topk"], key=lambda item: item["rank"])
        _require(
            {item["rank"] for item in main_records} == expected_ranks
            and {item["rank"] for item in indexer_records} == expected_ranks,
            "CAPTURE_TP_COVERAGE_INCOMPLETE",
        )
        expected_positions = list(range(window_start, query))
        expected_tokens = [prompt_ids[position] for position in expected_positions]
        for item in [*main_records, *indexer_records]:
            _require(
                item["query_token_id"] == prompt_ids[query]
                and item["candidate_positions"] == expected_positions
                and item["candidate_token_ids"] == expected_tokens,
                "CAPTURE_TOKEN_ALIGNMENT_MISMATCH",
            )
        indexer_reference = _finite(indexer_records[0]["raw_logits"])
        max_tp_difference = max(
            abs(left - right)
            for item in indexer_records[1:]
            for left, right in zip(indexer_reference, _finite(item["raw_logits"]), strict=True)
        ) if len(indexer_records) > 1 else 0.0
        _require(max_tp_difference <= float(declared["indexer_tp_max_abs_difference"]), "INDEXER_TP_REPLICA_DISAGREEMENT")
        main_heads = [head for item in main_records for head in item["raw_logits_by_local_head"]]
        _require(len(main_heads) == config["model_layout"]["main_attention_heads"], "MAIN_GLOBAL_HEAD_COVERAGE_INVALID")
        metrics = compare_aligned_scores(main_heads, indexer_reference, k_values=config["analysis"]["topk_values"])
        vectors = metrics.pop("vectors")
        per_query.append({
            "layer": layer,
            "query_position": query,
            "query_token_id": prompt_ids[query],
            "segment_membership": "q1" if any(start <= query < end for start, end in q1_ranges) else "q2",
            "indexer_tp_max_abs_difference": max_tp_difference,
            **metrics,
        })
        for index, position in enumerate(expected_positions):
            accumulator = layer_vectors[layer][position]
            accumulator[0] += vectors["main_raw_mean"][index]
            accumulator[1] += vectors["indexer_raw"][index]
            accumulator[2] += 1.0

    layer_summaries: list[dict[str, Any]] = []
    token_rows: list[dict[str, Any]] = []
    for layer in sorted(layer_vectors):
        included = [position for position, values in layer_vectors[layer].items() if values[2] > 0]
        main_sum = [layer_vectors[layer][position][0] for position in included]
        indexer_sum = [layer_vectors[layer][position][1] for position in included]
        metrics = compare_query_sums(main_sum, indexer_sum, k_values=config["analysis"]["topk_values"])
        vectors = metrics.pop("vectors")
        layer_summaries.append({"layer": layer, **metrics})
        main_ranks = _average_ranks(vectors["main_raw_query_sum"])
        indexer_ranks = _average_ranks(vectors["indexer_raw_query_sum"])
        for index, position in enumerate(included):
            token_rows.append({
                "schema_version": 2,
                "layer": layer,
                "candidate_position": position,
                "candidate_token_id": prompt_ids[position],
                "seed_membership": (
                    "q1" if any(start <= position < end for start, end in q1_ranges)
                    else "q2" if any(start <= position < end for start, end in q2_ranges)
                    else "none"
                ),
                "receiving_query_row_count": int(layer_vectors[layer][position][2]),
                "main_raw_query_sum": vectors["main_raw_query_sum"][index],
                "main_probability": vectors["main_probability"][index],
                "main_rank_ascending": main_ranks[index],
                "indexer_raw_query_sum": vectors["indexer_raw_query_sum"][index],
                "indexer_probability": vectors["indexer_probability"][index],
                "indexer_rank_ascending": indexer_ranks[index],
            })

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    token_path = output / "query-summed-token-scores.jsonl"
    token_path.write_bytes(b"".join(canonical_json_bytes(row) for row in token_rows))
    per_query_path = output / "per-query-comparisons.json"
    per_query_path.write_bytes(canonical_json_bytes(per_query))
    scalar_fields = (
        "pearson_raw",
        "spearman_raw",
        "cosine_zscore",
        "js_divergence_normalized",
        "js_divergence_native_softmax_diagnostic",
    )
    aggregate = {
        field: {
            "min": min(float(item[field]) for item in layer_summaries),
            "median": percentile(sorted(float(item[field]) for item in layer_summaries), 0.5),
            "max": max(float(item[field]) for item in layer_summaries),
            "mean": statistics.fmean(float(item[field]) for item in layer_summaries),
        }
        for field in scalar_fields
    }
    diagnostic_id, instance_id, scenario_id = next(iter(identities))
    payload = {
        "schema_version": 2,
        "report_id": "glm52-all-q1-q2-query-sum-compare-v2",
        "status": "passed",
        "scientific_result_policy": "report_only_no_similarity_threshold_failure",
        "diagnostic_id": diagnostic_id,
        "doctor_payload_sha256": doctor_payload_sha256,
        "benchmark_provenance": config["benchmark_provenance"],
        "instance_id": instance_id,
        "scenario_id": scenario_id,
        "probe_boundary": probe["claim_boundary"],
        "prompt_token_ids_sha256": probe["prompt"]["token_ids_sha256"],
        "q1_ranges": q1_ranges,
        "q2_ranges": q2_ranges,
        "query_token_count": len(expected_queries),
        "candidate_window": [window_start, window_end],
        "capture_files": source_files,
        "per_query_artifact": {"path": per_query_path.name, "record_count": len(per_query), "bytes": per_query_path.stat().st_size, "sha256": file_sha256(per_query_path)},
        "token_level_artifact": {"path": token_path.name, "record_count": len(token_rows), "bytes": token_path.stat().st_size, "sha256": file_sha256(token_path)},
        "layers": layer_summaries,
        "aggregate_quantiles": aggregate,
        "hypothesis_interpretation": {
            "supporting_pattern": "consistently positive high rank/correlation and cosine, low normalized JS divergence, and strong top-k metrics across preregistered layers",
            "refuting_pattern": "weak or negative rank/correlation, high normalized JS divergence, and poor top-k metrics across layers",
            "primary_distribution_normalization": "independent_population_zscore_then_softmax",
            "topk_ranking": "raw_descending_pre_softmax",
            "native_softmax_saturation_is_diagnostic_only": True,
            "threshold_preregistered": False,
            "dissimilarity_is_test_failure": False,
        },
    }
    report = {"payload": payload, "payload_sha256": hashlib.sha256(canonical_json_bytes(payload, newline=False)).hexdigest()}
    (output / "query-sum-attention-indexer-report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_report_digest(report: Mapping[str, Any]) -> str:
    payload = report.get("payload")
    observed = report.get("payload_sha256")
    _require(isinstance(payload, Mapping), "REPORT_PAYLOAD_INVALID")
    expected = hashlib.sha256(canonical_json_bytes(payload, newline=False)).hexdigest()
    _require(observed == expected, "REPORT_PAYLOAD_DIGEST_MISMATCH")
    return expected
