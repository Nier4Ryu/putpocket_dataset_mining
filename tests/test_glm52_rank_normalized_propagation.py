from __future__ import annotations

import math
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.glm52_indexer_propagation import MatrixCapture
from putpocket_dataset_mining.glm52_rank_normalized_propagation import (
    build_layer_averaged_transition,
    compute_rank_normalized_scores,
    deterministic_order,
    distribution_summary,
    normalize_distribution,
    propagate_distribution,
    rank_dcg_row,
    rank_stability,
    zscore_softmax_row,
    write_rank_normalized_report,
)


def _capture(*, two_layers: bool = False) -> MatrixCapture:
    # Rows are q -> every strictly earlier k. Higher raw score is more relevant.
    layer0 = (
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (9.0, 0.0, 0.0, 0.0, 0.0),
        (-2.0, 8.0, 0.0, 0.0, 0.0),
        (-3.0, -1.0, 7.0, 0.0, 0.0),
        (-4.0, -2.0, 0.0, 6.0, 0.0),
    )
    matrices = {0: layer0}
    layers = (0,)
    if two_layers:
        layer2 = (
            layer0[0],
            layer0[1],
            layer0[2],
            (-3.0, 10.0, 7.0, 0.0, 0.0),
            layer0[4],
        )
        matrices[2] = layer2
        layers = (0, 2)
    return MatrixCapture(
        window=(0, 5),
        token_ids=(10, 11, 12, 13, 14),
        layers=layers,
        matrices=matrices,
        input_files=({"path": "synthetic", "bytes": 1, "sha256": "a" * 64},),
        tp_max_abs_difference_observed=0.0,
    )


def test_rank_dcg_is_descending_nonnegative_normalized_and_tie_deterministic() -> None:
    row = rank_dcg_row([-5.0, 100.0, 0.0], 2)
    first, second = 1.0, 1.0 / math.log2(3.0)
    assert list(row) == pytest.approx([0.0, first / (first + second), second / (first + second)])
    assert math.fsum(row) == pytest.approx(1.0)
    assert all(value >= 0.0 for value in row)
    tied = rank_dcg_row([1.0, 1.0, 0.0], 1)
    assert list(tied) == [1.0, 0.0, 0.0]
    # Rank order is invariant to sign and scale as long as descending order is unchanged.
    assert list(rank_dcg_row([-30.0, -10.0, -20.0], 2)) == pytest.approx(
        list(rank_dcg_row([0.0, 2.0, 1.0], 2))
    )


def test_zscore_softmax_is_nonnegative_normalized_and_affine_invariant() -> None:
    original = zscore_softmax_row([-2.0, 1.0, 7.0], temperature=1.0)
    shifted_scaled = zscore_softmax_row([6.0, 12.0, 24.0], temperature=1.0)
    assert list(original) == pytest.approx(list(shifted_scaled))
    assert math.fsum(original) == pytest.approx(1.0)
    assert all(value > 0.0 for value in original)
    assert list(zscore_softmax_row([3.0, 3.0], temperature=1.0)) == [0.5, 0.5]


def test_layer_rows_are_normalized_before_uniform_average() -> None:
    transition = build_layer_averaged_transition(
        _capture(two_layers=True), kind="rank_dcg", transition_k=1
    )
    # Layer 0 selects k=2 and layer 2 selects k=1 for q=3.
    assert list(transition[3]) == [0.0, 0.5, 0.5]
    assert math.fsum(transition[3]) == pytest.approx(1.0)
    assert len(transition[0]) == 0


def test_backward_causal_orientation_and_per_hop_normalization() -> None:
    transition = build_layer_averaged_transition(
        _capture(), kind="rank_dcg", transition_k=1
    )
    seed = [0.0, 0.0, 0.0, 0.0, 1.0]
    hop1, mass1 = propagate_distribution(seed, transition)
    hop2, mass2 = propagate_distribution(hop1, transition)
    hop3, mass3 = propagate_distribution(hop2, transition)
    assert deterministic_order(hop1)[0] == 3
    assert deterministic_order(hop2)[0] == 2
    assert deterministic_order(hop3)[0] == 1
    assert [mass1, mass2, mass3] == pytest.approx([1.0, 1.0, 1.0])
    assert [math.fsum(hop1), math.fsum(hop2), math.fsum(hop3)] == pytest.approx(
        [1.0, 1.0, 1.0]
    )


def test_q1_q2_seed_and_equal_hop_cumulative_are_auditable() -> None:
    scores = compute_rank_normalized_scores(
        _capture(),
        q1_ranges=[[3, 4]],
        q2_ranges=[[4, 5]],
        max_level=3,
        transition_k=[1, 2, 3],
        evaluation_k=[1, 2, 3],
    )
    assert scores["seed"] == [0.0, 0.0, 0.0, 0.5, 0.5]
    primary = next(item for item in scores["variants"] if item["variant_id"] == "rank_dcg_k2")
    for level, (hop, cumulative) in enumerate(
        zip(
            primary["hop_distributions"],
            primary["cumulative_distributions"],
            strict=True,
        ),
        1,
    ):
        assert math.fsum(hop) == pytest.approx(1.0)
        assert math.fsum(cumulative) == pytest.approx(1.0)
        expected = [
            math.fsum(
                primary["hop_distributions"][hop_index][position]
                for hop_index in range(level)
            )
            / level
            for position in range(5)
        ]
        assert cumulative == pytest.approx(expected)


def test_topk_metrics_use_explicit_union_and_deterministic_global_ranks() -> None:
    baseline = [0.5, 0.3, 0.15, 0.05]
    current = [0.45, 0.1, 0.4, 0.05]
    result = rank_stability(current, baseline, evaluation_k=[2])
    metric = result["by_k"][0]
    assert metric["overlap_count"] == 1
    assert metric["retention"] == 0.5
    assert metric["jaccard"] == pytest.approx(1.0 / 3.0)
    assert 0.0 < metric["ndcg_against_baseline_rank"] < 1.0
    assert metric["topk_union_size"] == 3
    assert "top_k" in result["secondary_rank_scope"]
    assert "union" in result["secondary_rank_scope"]


def test_entropy_effective_support_and_fail_closed_boundaries() -> None:
    summary = distribution_summary([0.5, 0.5])
    assert summary["entropy_nats"] == pytest.approx(math.log(2.0))
    assert summary["effective_support"] == pytest.approx(2.0)
    with pytest.raises(ConfigError, match="RANK_TRANSITION_K_INVALID"):
        rank_dcg_row([1.0], 0)
    with pytest.raises(ConfigError, match="TEMPERATURE_INVALID"):
        zscore_softmax_row([1.0], temperature=0.0)
    with pytest.raises(ConfigError, match="NONFINITE_OR_NEGATIVE"):
        normalize_distribution([1.0, -1.0], reason="TEST")
    with pytest.raises(ConfigError, match="LEVEL_OUT_OF_RANGE"):
        compute_rank_normalized_scores(
            _capture(),
            q1_ranges=[[3, 4]],
            q2_ranges=[[4, 5]],
            max_level=17,
            transition_k=[1],
            evaluation_k=[1],
        )


def test_legacy_signed_source_and_schemas_remain_separate() -> None:
    root = Path(__file__).resolve().parents[1]
    legacy = (root / "src/putpocket_dataset_mining/glm52_indexer_propagation.py").read_text(
        encoding="utf-8"
    )
    corrected = (
        root / "src/putpocket_dataset_mining/glm52_rank_normalized_propagation.py"
    ).read_text(encoding="utf-8")
    assert 'RECURRENCE_ID = "putpocket_raw_indexer_strict_causal_multihop_v1"' in legacy
    assert 'RECURRENCE_ID = "putpocket_rank_normalized_indexer_multihop_v1"' in corrected
    assert "def compute_level_scores" in legacy
    assert "def compute_rank_normalized_scores" in corrected
    assert "from vllm" not in corrected
    assert "PUTPOCKET_VLLM" not in corrected


def test_report_and_token_artifacts_validate_with_default_scientific_config(
    tmp_path: Path,
) -> None:
    width = 300
    matrix = tuple(
        tuple([float(index) for index in range(query)] + [0.0] * (width - query))
        for query in range(width)
    )
    capture = MatrixCapture(
        window=(0, width),
        token_ids=tuple(range(1000, 1000 + width)),
        layers=(0,),
        matrices={0: matrix},
        input_files=({"path": "matrix.jsonl", "bytes": 1, "sha256": "a" * 64},),
        tp_max_abs_difference_observed=0.0,
    )
    scores = compute_rank_normalized_scores(
        capture,
        q1_ranges=[[297, 299]],
        q2_ranges=[[299, 300]],
        max_level=2,
    )
    file_item = {"path": "input.json", "bytes": 1, "sha256": "b" * 64}
    report = write_rank_normalized_report(
        episode={
            "scenario_id": "synthetic-scenario",
            "instance_id": "synthetic-instance",
            "benchmark_provenance": {"outcome_data_used": False},
        },
        episode_attestation=file_item,
        source_episode_attestation={**file_item, "path": "source.json"},
        capture=capture,
        capture_attestations=[{**file_item, "path": "matrix.jsonl"}],
        control_attestations=[
            {**file_item, "path": "run.json"},
            {**file_item, "path": "config.json"},
        ],
        legacy_report_attestation={**file_item, "path": "legacy-report.json"},
        legacy_token_attestation={**file_item, "path": "legacy-token.jsonl"},
        checksum_manifest_attestation={**file_item, "path": "SHA256SUMS"},
        scores=scores,
        output_root=tmp_path / "artifact",
    )
    assert report["payload"]["status"] == "passed"
    assert report["payload"]["transition_config"]["primary_variant_id"] == "rank_dcg_k64"
    assert [item["variant_id"] for item in report["payload"]["variant_results"]] == [
        "rank_dcg_k16",
        "rank_dcg_k64",
        "rank_dcg_k256",
        "zscore_softmax_t1",
    ]
    token_rows = [
        json.loads(line)
        for line in (tmp_path / "artifact/rank-normalized-multihop-token-scores.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(token_rows) == width
    assert token_rows[299]["segment_membership"] == ["q2"]
    assert math.fsum(row["seed_probability"] for row in token_rows) == pytest.approx(1.0)
    pytest.importorskip("matplotlib")
    repository_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repository_root / "src")
    plotted = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts/analysis/plot_glm52_rank_normalized_multihop.py"),
            "--artifact-root",
            str(tmp_path / "artifact"),
        ],
        cwd=repository_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert plotted.returncode == 0, plotted.stderr
    assert len(list((tmp_path / "artifact").glob("rank_normalized_*.png"))) == 3
    assert (tmp_path / "artifact/primary_k64_level_token_rank_heatmap.png").is_file()
    assert (tmp_path / "artifact/FINAL_SHA256SUMS").is_file()
    with pytest.raises(ConfigError, match="OUTPUT_ALREADY_EXISTS"):
        write_rank_normalized_report(
            episode={
                "scenario_id": "synthetic-scenario",
                "instance_id": "synthetic-instance",
                "benchmark_provenance": {},
            },
            episode_attestation=file_item,
            source_episode_attestation=file_item,
            capture=capture,
            capture_attestations=[file_item],
            control_attestations=[file_item, file_item],
            legacy_report_attestation=file_item,
            legacy_token_attestation=file_item,
            checksum_manifest_attestation=file_item,
            scores=scores,
            output_root=tmp_path / "artifact",
        )
