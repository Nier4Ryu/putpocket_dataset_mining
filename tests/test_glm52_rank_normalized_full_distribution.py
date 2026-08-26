from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.glm52_attention_indexer import file_sha256
from putpocket_dataset_mining.glm52_indexer_propagation import MatrixCapture
from putpocket_dataset_mining.glm52_rank_normalized_full_distribution import (
    ANALYSIS_ID,
    build_full_distribution_analysis,
    distribution_scalars,
    jensen_shannon_distance,
    load_full_distribution_source,
    pairwise_js_distance_matrix,
    validate_probability_matrix,
)
from putpocket_dataset_mining.glm52_rank_normalized_propagation import (
    compute_rank_normalized_scores,
    write_rank_normalized_report,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_source(tmp_path: Path, *, width: int = 300) -> Path:
    matrix = tuple(
        tuple(
            [float((query * 17 + key * 11) % 31) for key in range(query)]
            + [0.0] * (width - query)
        )
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
        q1_ranges=[[270, 290]],
        q2_ranges=[[295, 300]],
        max_level=3,
    )
    item = {"path": "input.json", "bytes": 1, "sha256": "b" * 64}
    source = tmp_path / "rank-normalized-multihop-v1"
    write_rank_normalized_report(
        episode={
            "scenario_id": "synthetic",
            "instance_id": "synthetic-instance",
            "benchmark_provenance": {"outcome_data_used": False},
        },
        episode_attestation=item,
        source_episode_attestation={**item, "path": "source.json"},
        capture=capture,
        capture_attestations=[{**item, "path": "matrix.jsonl"}],
        control_attestations=[
            {**item, "path": "run.json"},
            {**item, "path": "config.json"},
        ],
        legacy_report_attestation={**item, "path": "legacy-report.json"},
        legacy_token_attestation={**item, "path": "legacy-token.jsonl"},
        checksum_manifest_attestation={**item, "path": "SHA256SUMS"},
        scores=scores,
        output_root=source,
    )
    (source / "rank-normalized-plot-summary.json").write_text(
        json.dumps({"synthetic": True}) + "\n", encoding="utf-8"
    )
    files = sorted(
        [path for path in source.iterdir() if path.is_file()],
        key=lambda path: path.name,
    )
    (source / "FINAL_SHA256SUMS").write_text(
        "".join(f"{file_sha256(path)}  {path.name}\n" for path in files),
        encoding="utf-8",
    )
    return source


def test_js_distance_is_symmetric_zero_diagonal_and_base2_bounded() -> None:
    left = [0.75, 0.25, 0.0]
    right = [0.25, 0.25, 0.5]
    forward = jensen_shannon_distance(left, right)
    reverse = jensen_shannon_distance(right, left)
    assert forward == pytest.approx(reverse)
    assert 0.0 < forward < 1.0
    assert jensen_shannon_distance(left, left) == 0.0
    assert jensen_shannon_distance([1.0, 0.0], [0.0, 1.0]) == 1.0

    matrix = pairwise_js_distance_matrix([left, right, [0.0, 0.0, 1.0]])
    assert len(matrix) == 3
    for row in range(3):
        assert matrix[row][row] == 0.0
        for column in range(3):
            assert matrix[row][column] == pytest.approx(matrix[column][row])
            assert 0.0 <= matrix[row][column] <= 1.0


def test_probability_validation_and_scalars_use_complete_universe() -> None:
    values = [0.5, 0.25, 0.25, 0.0]
    validated = validate_probability_matrix(
        [values], token_count=4, reason="TEST_FULL"
    )
    assert validated == (tuple(values),)
    scalars = distribution_scalars(values)
    assert scalars["sorted_point_count"] == 4
    assert scalars["nonzero_support"] == 3
    assert scalars["zero_count"] == 1
    assert scalars["sorted_cumulative_final_mass"] == pytest.approx(1.0)
    assert scalars["effective_support"] == pytest.approx(math.exp(scalars["entropy_nats"]))
    with pytest.raises(ConfigError, match="MASS_INVALID"):
        validate_probability_matrix([[0.4, 0.4]], token_count=2, reason="TEST")
    with pytest.raises(ConfigError, match="NONFINITE_OR_NEGATIVE"):
        validate_probability_matrix([[1.1, -0.1]], token_count=2, reason="TEST")


def test_loader_retains_positions_tokens_segments_and_every_level(tmp_path: Path) -> None:
    source = load_full_distribution_source(_synthetic_source(tmp_path))
    assert source.positions == tuple(range(300))
    assert source.token_ids == tuple(range(1000, 1300))
    assert source.levels == (1, 2, 3)
    assert source.segment_membership[269] == ()
    assert source.segment_membership[270] == ("q1",)
    assert source.segment_membership[295] == ("q2",)
    assert len(source.hop) == len(source.cumulative) == 3
    assert all(len(row) == 300 for row in source.hop + source.cumulative)
    assert all(math.fsum(row) == pytest.approx(1.0) for row in source.hop)
    analysis = build_full_distribution_analysis(source)
    assert analysis["token_count"] == 300
    assert analysis["levels"] == [1, 2, 3]
    assert all(
        item["sorted_point_count"] == 300
        for view in analysis["scalar_summaries"].values()
        for item in view
    )


def test_loader_rejects_any_post_attestation_source_mutation(tmp_path: Path) -> None:
    source = _synthetic_source(tmp_path)
    plot_summary = source / "rank-normalized-plot-summary.json"
    plot_summary.write_text('{"synthetic":false}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="DIGEST_MISMATCH"):
        load_full_distribution_source(source)


def test_full_plot_cli_emits_exact_complete_universe_artifact(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    source = _synthetic_source(tmp_path)
    output = tmp_path / "rank-normalized-full-distribution-v1"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
        sys.executable,
        str(ROOT / "scripts/analysis/plot_glm52_rank_normalized_full_distribution.py"),
        "--source-artifact-root",
        str(source),
        "--output-root",
        str(output),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["status"] == "passed"
    assert response["token_count"] == 300
    assert response["levels"] == [1, 2, 3]
    assert len(list(output.glob("*.png"))) == 6
    assert len(list(output.glob("*.pdf"))) == 6
    assert (output / "full-distribution-input-attestation.json").is_file()
    summary = json.loads((output / "full-distribution-summary.json").read_text())
    assert summary["payload"]["analysis_id"] == ANALYSIS_ID
    assert summary["payload"]["token_universe"]["token_count"] == 300
    assert summary["payload"]["token_universe"]["all_tokens_included"] is True
    assert summary["payload"]["token_universe"]["top_k_truncation"] is False
    assert len(summary["payload"]["figures"]) == 12
    for view in ("hop", "cumulative_equal_hop_mixture"):
        matrix = summary["payload"]["jensen_shannon_distance"][view]
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)
        assert all(matrix[index][index] == 0.0 for index in range(3))
    entries = {}
    for line in (output / "FINAL_SHA256SUMS").read_text().splitlines():
        digest, name = line.split(None, 1)
        entries[name] = digest
    assert len(entries) == 14
    for name, digest in entries.items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest

    repeated = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "FULL_DISTRIBUTION_OUTPUT_ALREADY_EXISTS" in repeated.stderr


def test_full_distribution_code_is_offline_and_not_signed_recurrence() -> None:
    module = (
        ROOT
        / "src/putpocket_dataset_mining/glm52_rank_normalized_full_distribution.py"
    ).read_text(encoding="utf-8")
    script = (
        ROOT / "scripts/analysis/plot_glm52_rank_normalized_full_distribution.py"
    ).read_text(encoding="utf-8")
    assert "from vllm" not in module + script
    assert "PUTPOCKET_VLLM" not in module + script
    assert "raw_indexer_strict_causal_multihop" not in module + script
    assert "top_k_truncation" in module
    assert "jensen_shannon_distance" in module
