from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.glm52_attention_indexer import canonical_json_bytes, file_sha256
from putpocket_dataset_mining.glm52_indexer_propagation import (
    MatrixCapture,
    compute_level_scores,
    load_indexer_matrix_capture,
    load_matrix_episode_manifest,
    score_matrix_capture,
    validate_seed_ranges,
)
from putpocket_dataset_mining.glm52_runpod_cli import main as runpod_cli_main
from putpocket_dataset_mining.glm52_runpod import capture_matrix_episode


INSTANCE = (
    "instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-"
    "vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
)
SCENARIO = "glm52-sys-policy-equal-replacement-v1"
TOKENS = [10, 11, 12, 13, 14]
ROWS = [[], [1.0], [2.0, 3.0], [4.0, -1.0, 2.0], [1.0, 5.0, -2.0, 3.0]]


def _token_digest(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode("ascii")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _episode(tmp_path: Path, *, layers: list[int] | None = None) -> tuple[Path, dict[str, object]]:
    source = {
        "scenario_id": SCENARIO,
        "instance_id": INSTANCE,
        "tokenization": {
            "first_post_edit_request_token_ids_sha256": _token_digest(TOKENS),
            "q2_token_range_in_first_post_edit_request": [4, 5],
        },
    }
    source_path = tmp_path / "source-episode.json"
    _write_json(source_path, source)
    value: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "frozen_indexer_matrix_episode",
        "status": "frozen_before_outcomes",
        "episode_id": "synthetic-frozen-episode",
        "scenario_id": SCENARIO,
        "instance_id": INSTANCE,
        "benchmark_provenance": {
            "dataset": "ScaleAI/SWE-bench_Pro",
            "dataset_revision": "7ab5114912baf22bb098818e604c02fe7ad2c11f",
            "split": "test",
            "instance_id": INSTANCE,
            "native_components": ["problem", "repository", "evaluator"],
            "project_authored_components": ["A1 execution", "Q2 freeze", "SYS edit"],
            "outcome_data_used": False,
        },
        "source_frozen_episode_manifest": {
            "path": source_path.name,
            "sha256": file_sha256(source_path),
        },
        "prompt": {
            "kind": "frozen_first_post_edit_request_or_frozen_ordinary_prefill_probe",
            "token_ids": TOKENS,
            "token_count": len(TOKENS),
            "token_ids_sha256": _token_digest(TOKENS),
            "serializer_id": "synthetic-chat-template",
            "tokenizer_revision": "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa",
            "q2_included": True,
        },
        "segments": {
            "q1_ranges": [[3, 4]],
            "q2_ranges": [[4, 5]],
            "q2_semantics": "frozen_pre_edit_tool_observation_in_first_post_edit_request",
        },
        "propagation_window": [0, 5],
        "capture": {
            "mode": "strict_causal_indexer_matrix",
            "layers": layers or [0],
            "tensor_parallel_size": 4,
            "row_chunk_size": 2,
            "indexer_tp_max_abs_difference": 0.0,
            "hard_max_window_tokens": 512,
            "hard_max_total_edges_per_rank": 523264,
            "cost_warning": "O(layers * window_tokens^2) capture cost",
        },
        "outcome_independent": True,
        "authorship_boundary": "benchmark supplies problem/repository/evaluator; PutPocket authors trajectory and edit",
    }
    path = tmp_path / "frozen-matrix-episode.json"
    _write_json(path, value)
    return path, value


def _record(*, layer: int, query: int, rank: int, values: list[float]) -> dict[str, object]:
    chunk = query // 2
    record: dict[str, object] = {
        "schema_version": 1,
        "diagnostic_id": "synthetic-matrix",
        "instance_id": INSTANCE,
        "scenario_id": SCENARIO,
        "probe_kind": "frozen_episode_strict_causal_indexer_matrix",
        "q2_in_probe": True,
        "rank": rank,
        "tensor_parallel_size": 4,
        "layer": layer,
        "query_position": query,
        "query_token_id": TOKENS[query],
        "dtype": "float32_capture_from_bfloat16_or_quantized_runtime_state",
        "record_kind": "indexer_native_strict_causal_matrix_row",
        "capture_mode": "strict_causal_indexer_matrix",
        "score_origin": "kernel_native_fp8_fp4_mqa_logits_before_top_k_per_row_prefill",
        "kernel_native": True,
        "pre_top_k": True,
        "normalized": False,
        "reference_recomputed": False,
        "padding_or_masked_values_included": False,
        "formula": "native weighted indexer sum",
        "scale": {"softmax_scale": 128**-0.5, "indexer_head_scale": 32**-0.5},
        "mask": "strict_causal_key_position_lt_query_position",
        "head_aggregation_at_capture": "native_learned_weighted_sum_across_32_indexer_heads",
        "tp_semantics": "replicated_native_aggregate_consensus_required_offline",
        "indexer_head_count": 32,
        "indexer_head_dim": 128,
        "propagation_window": [0, 5],
        "row_chunk_index": chunk,
        "row_chunk_range": [chunk * 2, min(5, chunk * 2 + 2)],
        "key_positions": list(range(query)),
        "key_token_ids": TOKENS[:query],
        "raw_scores": values,
        "native_valid_start": 0,
        "native_valid_end_exclusive": query + 1,
    }
    record["record_sha256"] = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
    return record


def _capture(tmp_path: Path, *, layers: list[int] | None = None) -> Path:
    selected = layers or [0]
    root = tmp_path / "capture"
    root.mkdir(parents=True)
    for rank in range(4):
        for chunk in range(3):
            records = []
            for layer in selected:
                scale = 1.0 if layer == 0 else -2.0
                for query, row in enumerate(ROWS):
                    if query // 2 == chunk:
                        records.append(
                            _record(
                                layer=layer,
                                query=query,
                                rank=rank,
                                values=[scale * value for value in row],
                            )
                        )
            path = root / f"matrix-rank-{rank:02d}-chunk-{chunk:04d}.jsonl"
            path.write_bytes(b"".join(canonical_json_bytes(record) for record in records))
    return root


def _doctor(path: Path) -> None:
    payload = {
        "schema_version": 1,
        "test_id": "glm52-runpod-doctor-v1",
        "status": "passed",
        "read_only": True,
        "project_commit": "1" * 40,
        "vllm_commit": "4a3447d200e5aa428d68d1a00aa00f1a19a1a729",
        "model_revision": "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa",
        "checks": [
            {"check_id": f"synthetic-{index}", "status": "passed", "evidence": {"ok": True}}
            for index in range(10)
        ],
    }
    _write_json(path, {
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload, newline=False)).hexdigest(),
    })


def _rewrite_record(path: Path, predicate: object, mutation: object) -> None:
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for record in records:
        if predicate(record):  # type: ignore[operator]
            mutation(record)  # type: ignore[operator]
            record.pop("record_sha256", None)
            record["record_sha256"] = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
            break
    path.write_bytes(b"".join(canonical_json_bytes(record) for record in records))


def test_hand_computed_levels_one_two_three_and_negative_scores(tmp_path: Path) -> None:
    episode_path, _ = _episode(tmp_path)
    capture = load_indexer_matrix_capture(
        _capture(tmp_path), load_matrix_episode_manifest(episode_path), layers=[0], window=[0, 5]
    )
    result = compute_level_scores(
        capture, q1_ranges=[[3, 4]], q2_ranges=[[4, 5]], max_level=3
    )
    layer = result["layer_results"][0]
    assert layer["hop_contributions"] == [
        [5.0, 4.0, 0.0, 3.0, 0.0],
        [16.0, -3.0, 6.0, 0.0, 0.0],
        [9.0, 18.0, 0.0, 0.0, 0.0],
    ]
    assert layer["cumulative"] == [30.0, 19.0, 6.0, 3.0, 0.0]
    assert result["seed"] == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_multiple_seed_ranges_layers_exact_sum_and_query_membership(tmp_path: Path) -> None:
    episode_path, _ = _episode(tmp_path, layers=[0, 22])
    capture_root = _capture(tmp_path, layers=[0, 22])
    report = score_matrix_capture(
        episode_path=episode_path,
        capture_root=capture_root,
        output_root=tmp_path / "report",
        q1_ranges=[[3, 4]],
        q2_ranges=[[4, 5]],
        layers=[0, 22],
        window=[0, 5],
        max_level=3,
    )
    rows = [json.loads(line) for line in (tmp_path / "report/indexer-multihop-token-scores.jsonl").read_text().splitlines()]
    assert rows[3]["segment_membership"] == ["q1"]
    assert rows[4]["segment_membership"] == ["q2"]
    assert rows[3]["seed_value"] == rows[4]["seed_value"] == 1.0
    # For A'=-2A, cumulative is -2*c1 + 4*c2 - 8*c3.
    expected_layer_22_position_0 = -2 * 5 + 4 * 16 - 8 * 9
    assert rows[0]["per_layer"][1]["cumulative_score"] == expected_layer_22_position_0
    assert rows[0]["layer_sum"]["cumulative_score"] == 30 + expected_layer_22_position_0
    assert report["payload"]["recurrence"]["cumulative_interpretation"] is True
    assert report["payload"]["recurrence"]["cross_layer_operation"] == "exact_unnormalized_sum_only"


def test_deterministic_output_digest(tmp_path: Path) -> None:
    episode_path, _ = _episode(tmp_path)
    capture_root = _capture(tmp_path)
    kwargs = dict(
        episode_path=episode_path,
        capture_root=capture_root,
        q1_ranges=[[3, 4]], q2_ranges=[[4, 5]], layers=[0], window=[0, 5], max_level=3,
    )
    first = score_matrix_capture(output_root=tmp_path / "one", **kwargs)
    second = score_matrix_capture(output_root=tmp_path / "two", **kwargs)
    assert first["payload_sha256"] == second["payload_sha256"]
    assert file_sha256(tmp_path / "one/indexer-multihop-token-scores.jsonl") == file_sha256(
        tmp_path / "two/indexer-multihop-token-scores.jsonl"
    )
    assert first["payload"]["inputs"]
    assert all(len(item["sha256"]) == 64 for item in first["payload"]["inputs"])


def test_cli_scores_explicit_ranges_layers_window_and_level(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    episode_path, _ = _episode(tmp_path)
    root = _capture(tmp_path)
    assert runpod_cli_main([
        "score-matrix", "--episode-manifest", str(episode_path),
        "--capture-root", str(root), "--output-root", str(tmp_path / "cli-report"),
        "--q1-range", "3:4", "--q2-range", "4:5", "--window", "0:5",
        "--layers", "0", "--max-level", "3",
    ]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["report_kind"] == "glm52_raw_indexer_multihop_offline"


def test_matrix_capture_dry_run_is_bounded_and_default_separate(tmp_path: Path) -> None:
    episode_path, _ = _episode(tmp_path, layers=[0, 22, 46, 74])
    doctor = tmp_path / "doctor.json"
    _doctor(doctor)
    output = tmp_path / "matrix-plan"
    plan = capture_matrix_episode(
        doctor_report=doctor,
        episode_path=episode_path,
        model_root=tmp_path / "model-not-loaded-in-dry-run",
        output_root=output,
        dry_run=True,
    )
    assert plan["status"] == "dry_run"
    assert plan["capture_mode"] == "strict_causal_indexer_matrix"
    assert plan["edge_values_per_rank"] == 40
    assert plan["cost_class"] == "O(layers * window_tokens^2)"
    config = json.loads((output / "matrix-instrumentation-config.json").read_text())
    assert config["q2_in_probe"] is True
    assert "query_positions" not in config
    (output / "matrix-rank-00-chunk-0000.jsonl").write_text("stale\n")
    with pytest.raises(ConfigError, match="MATRIX_CAPTURE_OUTPUT_NOT_EMPTY"):
        capture_matrix_episode(
            doctor_report=doctor, episode_path=episode_path,
            model_root=tmp_path / "model", output_root=output, dry_run=True,
        )


@pytest.mark.parametrize(
    ("case", "mutation", "error"),
    [
        ("duplicate_key", lambda row: row.update(key_positions=[0, 0]), "SCHEMA_VALIDATION_FAILED|ROW_INCOMPLETE"),
        ("noncausal", lambda row: row.update(key_positions=[0, 1]), "ROW_INCOMPLETE_OR_NONCAUSAL"),
        ("token", lambda row: row.update(key_token_ids=[99]), "TOKEN_MISMATCH"),
        ("post_topk", lambda row: row.update(pre_top_k=False), "SCHEMA_VALIDATION_FAILED|SCORE_SEMANTICS"),
        ("normalized", lambda row: row.update(normalized=True), "SCHEMA_VALIDATION_FAILED|SCORE_SEMANTICS"),
        ("masked", lambda row: row.update(padding_or_masked_values_included=True), "SCHEMA_VALIDATION_FAILED|SCORE_SEMANTICS"),
    ],
)
def test_invalid_matrix_rows_fail_closed(
    tmp_path: Path, case: str, mutation: object, error: str
) -> None:
    episode_path, _ = _episode(tmp_path)
    root = _capture(tmp_path)
    path = root / "matrix-rank-00-chunk-0000.jsonl"
    _rewrite_record(path, lambda row: row["query_position"] == 1, mutation)
    with pytest.raises(ConfigError, match=error):
        load_indexer_matrix_capture(
            root, load_matrix_episode_manifest(episode_path), layers=[0], window=[0, 5]
        )


def test_missing_intermediate_duplicate_row_and_tp_disagreement_fail(tmp_path: Path) -> None:
    episode_path, _ = _episode(tmp_path)
    episode = load_matrix_episode_manifest(episode_path)
    missing = _capture(tmp_path)
    path = missing / "matrix-rank-00-chunk-0001.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_bytes(b"".join(canonical_json_bytes(row) for row in records if row["query_position"] != 2))
    with pytest.raises(ConfigError, match="INTERMEDIATE_ROWS_MISSING|TP_COVERAGE_INCOMPLETE"):
        load_indexer_matrix_capture(missing, episode, layers=[0], window=[0, 5])

    duplicate_root = tmp_path / "duplicate"
    duplicate_root.mkdir()
    for source in sorted(missing.glob("*.jsonl")):
        (duplicate_root / source.name).write_bytes(source.read_bytes())
    target = duplicate_root / "matrix-rank-00-chunk-0000.jsonl"
    target.write_bytes(target.read_bytes() + target.read_bytes().splitlines(keepends=True)[0])
    with pytest.raises(ConfigError, match="DUPLICATE_RANK_ROW"):
        load_indexer_matrix_capture(duplicate_root, episode, layers=[0], window=[0, 5])

    disagreement = _capture(tmp_path / "tp")
    _rewrite_record(
        disagreement / "matrix-rank-03-chunk-0001.jsonl",
        lambda row: row["query_position"] == 3,
        lambda row: row.update(raw_scores=[4.0, -1.0, 2.5]),
    )
    with pytest.raises(ConfigError, match="TP_REPLICA_DISAGREEMENT"):
        load_indexer_matrix_capture(disagreement, episode, layers=[0], window=[0, 5])


def test_ranges_episode_mismatch_level_bound_nonfinite_and_overflow(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Q1_Q2_RANGES_OVERLAP"):
        validate_seed_ranges([[2, 4]], [[3, 5]], [0, 5])
    with pytest.raises(ConfigError, match="OUTSIDE_PROPAGATION_WINDOW"):
        validate_seed_ranges([[0, 1]], [[5, 6]], [0, 5])
    q1, q2 = validate_seed_ranges([[1, 2], [3, 4]], [[5, 6], [7, 8]], [0, 9])
    assert q1 == ((1, 2), (3, 4)) and q2 == ((5, 6), (7, 8))
    episode_path, _ = _episode(tmp_path)
    root = _capture(tmp_path)
    with pytest.raises(ConfigError, match="FROZEN_EPISODE_MISMATCH"):
        score_matrix_capture(
            episode_path=episode_path, capture_root=root, output_root=tmp_path / "bad",
            q1_ranges=[[2, 3]], q2_ranges=[[4, 5]], layers=[0], window=[0, 5], max_level=1,
        )
    capture = load_indexer_matrix_capture(
        root, load_matrix_episode_manifest(episode_path), layers=[0], window=[0, 5]
    )
    with pytest.raises(ConfigError, match="LEVEL_OUT_OF_RANGE"):
        compute_level_scores(capture, q1_ranges=[[3, 4]], q2_ranges=[[4, 5]], max_level=17)

    path = root / "matrix-rank-00-chunk-0000.jsonl"
    raw = path.read_text().replace('"raw_scores":[1.0]', '"raw_scores":[NaN]')
    path.write_text(raw)
    with pytest.raises((ConfigError, ValueError), match="NONFINITE|[Oo]ut of range"):
        load_indexer_matrix_capture(
            root, load_matrix_episode_manifest(episode_path), layers=[0], window=[0, 5]
        )

    huge_matrix = ((0.0, 0.0, 0.0), (1e308, 0.0, 0.0), (0.0, 1e308, 0.0))
    huge = MatrixCapture(
        window=(0, 3), token_ids=(1, 2, 3), layers=(0,), matrices={0: huge_matrix},
        input_files=({"path": "synthetic", "bytes": 1, "sha256": "a" * 64},),
        tp_max_abs_difference_observed=0.0,
    )
    with pytest.raises(ConfigError, match="FLOAT64_OVERFLOW"):
        compute_level_scores(huge, q1_ranges=[[2, 3]], q2_ranges=[[1, 2]], max_level=2)


def test_source_episode_digest_and_q2_mismatch_fail(tmp_path: Path) -> None:
    episode_path, value = _episode(tmp_path)
    source = tmp_path / "source-episode.json"
    source.write_text(source.read_text() + " ")
    with pytest.raises(ConfigError, match="SOURCE_FROZEN_EPISODE_DIGEST_MISMATCH"):
        load_matrix_episode_manifest(episode_path)
    _write_json(source, {
        "scenario_id": SCENARIO, "instance_id": INSTANCE,
        "tokenization": {
            "first_post_edit_request_token_ids_sha256": _token_digest(TOKENS),
            "q2_token_range_in_first_post_edit_request": [3, 5],
        },
    })
    value["source_frozen_episode_manifest"] = {"path": source.name, "sha256": file_sha256(source)}
    _write_json(episode_path, value)
    with pytest.raises(ConfigError, match="Q2_RANGE_MISMATCH"):
        load_matrix_episode_manifest(episode_path)


def test_manifest_cost_cap_fails_before_capture(tmp_path: Path) -> None:
    episode_path, value = _episode(tmp_path)
    value["capture"]["hard_max_window_tokens"] = 4  # type: ignore[index]
    _write_json(episode_path, value)
    with pytest.raises(ConfigError, match="COST_CAP_EXCEEDED"):
        load_matrix_episode_manifest(episode_path, verify_source=False)


def test_sampled_capture_source_contract_remains_distinct() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "instrumentation/vllm/glm52_attention_indexer_scores.py"
    ).read_text(encoding="utf-8")
    assert 'SAMPLED_MODE = "sampled_attention_indexer_comparison"' in source
    assert 'MATRIX_MODE = "strict_causal_indexer_matrix"' in source
    assert '"mask": "native_cu_seqlen_ks_ke_causal_inclusive"' in source
    assert '"mask": "strict_causal_key_position_lt_query_position"' in source
    assert 'logits[row, window_start:query_position]' in source
    assert 'logits[row, first:valid_end]' in source
