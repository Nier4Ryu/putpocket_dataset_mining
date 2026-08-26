from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
PLOT_SCRIPT = ROOT / "scripts/analysis/plot_glm52_score_evidence.py"
PLOT_SCHEMA = ROOT / "configs/runpod/schemas/glm52_score_plot_summary.schema.json"
TRANSFER_SCRIPT = ROOT / "scripts/runpod/transfer_framed_file.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _synthetic_evidence(root: Path) -> tuple[str, str, str]:
    query_root = Path("accepted/query-v2")
    multihop_root = Path("accepted/multihop")
    layer_metrics = []
    query_rows = []
    for layer, main, indexer in (
        (0, [1.0, -2.0, 3.0, 0.5], [0.1, -0.4, 0.8, 0.2]),
        (2, [-1.0, 4.0, 2.0, -0.5], [-0.3, 0.9, 0.5, -0.1]),
    ):
        layer_metrics.append(
            {
                "layer": layer,
                "pearson_raw": 0.9,
                "spearman_raw": 0.8,
                "cosine_zscore": 0.9,
                "js_divergence_normalized": 0.1,
            }
        )
        for position, (main_value, indexer_value) in enumerate(
            zip(main, indexer, strict=True)
        ):
            query_rows.append(
                {
                    "layer": layer,
                    "candidate_position": position,
                    "main_raw_query_sum": main_value,
                    "indexer_raw_query_sum": indexer_value,
                }
            )
    _write_json(
        root / query_root / "query-sum-attention-indexer-report.json",
        {"payload": {"schema_version": 2, "status": "passed", "layers": layer_metrics}},
    )
    _write_jsonl(root / query_root / "query-summed-token-scores.jsonl", query_rows)

    _write_json(
        root / multihop_root / "indexer-multihop-report.json",
        {"payload": {"status": "passed", "layers": [0, 2], "max_level": 3}},
    )
    multihop_rows = []
    for position in range(4):
        first = [position + 1.0, -(position + 1.0) / 2.0, position + 0.25]
        second = [-(position * 0.25 + 0.5), position + 2.0, -(position + 0.75)]
        layer_sum = [left + right for left, right in zip(first, second, strict=True)]
        multihop_rows.append(
            {
                "position": position,
                "per_layer": [
                    {"layer": 0, "hop_contributions": first},
                    {"layer": 2, "hop_contributions": second},
                ],
                "layer_sum": {"hop_contributions": layer_sum},
            }
        )
    _write_jsonl(
        root / multihop_root / "indexer-multihop-token-scores.jsonl",
        multihop_rows,
    )

    inputs = sorted(path for path in root.rglob("*") if path.is_file())
    manifest = root / "PLOT_INPUT_SHA256SUMS"
    manifest.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in inputs),
        encoding="utf-8",
    )
    return query_root.as_posix(), multihop_root.as_posix(), manifest.name


def test_plotter_renders_only_checksum_attested_relative_inputs(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("numpy")
    query_root, multihop_root, manifest = _synthetic_evidence(tmp_path)
    output = tmp_path / "plots"
    command = [
        sys.executable,
        str(PLOT_SCRIPT),
        "--evidence-root",
        str(tmp_path),
        "--query-report-root",
        query_root,
        "--multihop-root",
        multihop_root,
        "--checksum-manifest",
        manifest,
        "--output-root",
        str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    summary = json.loads((output / "plot-summary.json").read_text(encoding="utf-8"))
    schema = json.loads(PLOT_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(summary)
    assert len(summary["outputs"]) == 10
    assert all(not Path(item["path"]).is_absolute() for item in summary["inputs"])
    assert {path.suffix for path in output.glob("*.*")} >= {".png", ".pdf"}
    for item in summary["outputs"]:
        assert _sha256(output / item["path"]) == item["sha256"]

    tampered = tmp_path / query_root / "query-summed-token-scores.jsonl"
    tampered.write_text(tampered.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    failed = subprocess.run(
        [*command[:-1], str(tmp_path / "tampered-plots")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "PLOT_INPUT_DIGEST_MISMATCH" in failed.stderr


def test_transfer_helper_is_generic_fail_fast_and_dry_runnable(tmp_path: Path) -> None:
    subprocess.run(["bash", "-n", str(TRANSFER_SCRIPT)], check=True)
    content = TRANSFER_SCRIPT.read_text(encoding="utf-8")
    assert "set -euo pipefail" in content
    assert "runpod.io" not in content
    assert "/home/dyryu" not in content
    assert "rm " not in content
    result = subprocess.run(
        [
            "bash",
            str(TRANSFER_SCRIPT),
            "--ssh-target",
            "user@example.invalid",
            "--identity",
            str(tmp_path / "not-opened-in-dry-run"),
            "--remote-file",
            "/workspace/task/evidence.tar.gz",
            "--destination",
            str(tmp_path / "evidence.tar.gz"),
            "--expected-sha256",
            "a" * 64,
            "--expected-bytes",
            "2097153",
            "--chunk-mib",
            "1",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["chunk_count"] == 3
