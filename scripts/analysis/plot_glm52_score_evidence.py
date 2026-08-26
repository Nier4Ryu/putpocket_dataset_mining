#!/usr/bin/env python3
"""Render GLM score plots only from checksum-attested evidence files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from jsonschema import Draft202012Validator


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = (
    REPOSITORY_ROOT
    / "configs/runpod/schemas/glm52_score_plot_summary.schema.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL_INPUT_EMPTY_OR_INVALID:{path}")
    return rows


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"INPUT_PATH_NOT_SAFE_RELATIVE:{value}")
    return path


def resolve_input(root: Path, relative: str) -> tuple[Path, str]:
    rel = safe_relative_path(relative)
    path = (root / rel).resolve()
    try:
        normalized = path.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"INPUT_PATH_ESCAPES_EVIDENCE_ROOT:{relative}") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"INPUT_FILE_MISSING_OR_SYMLINK:{normalized}")
    return path, normalized


def load_checksum_manifest(root: Path, relative: str) -> tuple[Path, dict[str, str]]:
    path, _ = resolve_input(root, relative)
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not SHA256_RE.fullmatch(parts[0]):
            raise ValueError(f"CHECKSUM_MANIFEST_LINE_INVALID:{line_number}")
        name = safe_relative_path(parts[1].lstrip("* ")).as_posix()
        if name in entries:
            raise ValueError(f"CHECKSUM_MANIFEST_DUPLICATE:{name}")
        entries[name] = parts[0]
    if not entries:
        raise ValueError("CHECKSUM_MANIFEST_EMPTY")
    return path, entries


def zscore(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("ZSCORE_INPUT_INVALID")
    std = float(values.std())
    return np.zeros_like(values) if std == 0.0 else (values - values.mean()) / std


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        result[order[cursor:end]] = (cursor + end - 1) / 2.0
        cursor = end
    return result


def similarity(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("SIMILARITY_INPUT_INVALID")
    left_z, right_z = zscore(left), zscore(right)
    denominator = float(np.linalg.norm(left_z) * np.linalg.norm(right_z))
    cosine = float(np.dot(left_z, right_z) / denominator) if denominator else 0.0
    pearson = float(np.corrcoef(left, right)[0, 1])
    spearman = float(np.corrcoef(average_ranks(left), average_ranks(right))[0, 1])
    if not all(math.isfinite(value) for value in (cosine, pearson, spearman)):
        raise ValueError("SIMILARITY_NONFINITE")
    return {"pearson": pearson, "spearman": spearman, "cosine_zscore": cosine}


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("PLOT_INPUT_NONFINITE_OR_EMPTY")
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "sum": float(values.sum(dtype=np.float64)),
        "l1_mass": float(np.abs(values).sum(dtype=np.float64)),
        "positive_mass": float(values[values > 0].sum(dtype=np.float64)),
        "negative_mass": float(values[values < 0].sum(dtype=np.float64)),
    }


def symlog_ticks(limit: float, linthresh: float) -> list[float]:
    middle = math.sqrt(limit * linthresh)
    return [-limit, -middle, 0.0, middle, limit]


def save_figure(fig: Any, output_root: Path, stem: str) -> list[Path]:
    paths = [output_root / f"{stem}.png", output_root / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(
            path,
            dpi=180 if path.suffix == ".png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return paths


def validate_attested_inputs(
    evidence_root: Path,
    manifest_entries: dict[str, str],
    relative_paths: list[str],
) -> list[Path]:
    paths: list[Path] = []
    for relative in relative_paths:
        path, normalized = resolve_input(evidence_root, relative)
        expected = manifest_entries.get(normalized)
        if expected is None:
            raise ValueError(f"PLOT_INPUT_NOT_IN_CHECKSUM_MANIFEST:{normalized}")
        observed = sha256(path)
        if observed != expected:
            raise ValueError(f"PLOT_INPUT_DIGEST_MISMATCH:{normalized}")
        paths.append(path)
    return paths


def render(args: argparse.Namespace) -> dict[str, Any]:
    evidence_root = args.evidence_root.resolve()
    if not evidence_root.is_dir():
        raise ValueError("EVIDENCE_ROOT_MISSING")
    manifest_path, manifest_entries = load_checksum_manifest(
        evidence_root, args.checksum_manifest
    )
    query_root = safe_relative_path(args.query_report_root)
    multihop_root = safe_relative_path(args.multihop_root)
    relative_inputs = [
        (query_root / "query-sum-attention-indexer-report.json").as_posix(),
        (query_root / "query-summed-token-scores.jsonl").as_posix(),
        (multihop_root / "indexer-multihop-report.json").as_posix(),
        (multihop_root / "indexer-multihop-token-scores.jsonl").as_posix(),
    ]
    input_paths = validate_attested_inputs(
        evidence_root, manifest_entries, relative_inputs
    )
    query_report_path, query_tokens_path, multihop_report_path, multihop_tokens_path = input_paths

    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("PLOT_OUTPUT_NOT_EMPTY")
    output.mkdir(parents=True, exist_ok=True)

    query_document = load_json(query_report_path)
    multihop_document = load_json(multihop_report_path)
    query_report = query_document.get("payload", {})
    multihop_report = multihop_document.get("payload", {})
    if (
        query_report.get("schema_version") != 2
        or query_report.get("status") != "passed"
        or multihop_report.get("status") != "passed"
    ):
        raise ValueError("SOURCE_REPORT_NOT_ACCEPTED")
    query_rows = load_jsonl(query_tokens_path)
    multihop_rows = load_jsonl(multihop_tokens_path)
    layers = [int(item["layer"]) for item in query_report["layers"]]
    if layers != multihop_report.get("layers") or len(set(layers)) != len(layers):
        raise ValueError("SOURCE_LAYER_MISMATCH")

    metric_names = ["pearson_raw", "spearman_raw", "cosine_zscore", "one_minus_js"]
    metric_matrix = np.array(
        [
            [
                item["pearson_raw"],
                item["spearman_raw"],
                item["cosine_zscore"],
                1.0 - item["js_divergence_normalized"],
            ]
            for item in query_report["layers"]
        ],
        dtype=np.float64,
    )
    if not np.isfinite(metric_matrix).all():
        raise ValueError("QUERY_METRIC_NONFINITE")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    image = ax.imshow(metric_matrix, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(metric_names)), [name.replace("_", "\n") for name in metric_names])
    ax.set_yticks(range(len(layers)), [f"layer {layer}" for layer in layers])
    for row in range(metric_matrix.shape[0]):
        for column in range(metric_matrix.shape[1]):
            ax.text(column, row, f"{metric_matrix[row, column]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("All-Q1/Q2 query-sum attention/indexer similarity")
    fig.colorbar(image, ax=ax, label="similarity (1-JS for JS column)")
    output_paths = save_figure(fig, output, "attention_indexer_similarity_heatmap")

    grouped_query: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layers}
    for row in query_rows:
        layer = int(row["layer"])
        if layer not in grouped_query:
            raise ValueError("QUERY_TOKEN_LAYER_UNDECLARED")
        grouped_query[layer].append(row)
    fig, axes = plt.subplots(len(layers), 1, figsize=(13, 2.6 * len(layers)), sharex=True)
    axes = np.atleast_1d(axes)
    query_numeric: dict[str, dict[str, Any]] = {}
    for ax, layer in zip(axes, layers, strict=True):
        rows = sorted(grouped_query[layer], key=lambda item: item["candidate_position"])
        positions = np.array([row["candidate_position"] for row in rows], dtype=np.int64)
        if positions.size == 0 or len(set(positions.tolist())) != positions.size:
            raise ValueError("QUERY_TOKEN_POSITIONS_INVALID")
        main_values = np.array([row["main_raw_query_sum"] for row in rows], dtype=np.float64)
        indexer_values = np.array([row["indexer_raw_query_sum"] for row in rows], dtype=np.float64)
        ax.plot(positions, zscore(main_values), linewidth=0.8, label="main reference query sum (z-score)")
        ax.plot(positions, zscore(indexer_values), linewidth=0.8, label="native indexer query sum (z-score)")
        ax.set_ylabel(f"L{layer}\nz-score")
        ax.grid(alpha=0.2)
        query_numeric[str(layer)] = {
            "main_raw": finite_summary(main_values),
            "indexer_raw": finite_summary(indexer_values),
            "computed_similarity": similarity(main_values, indexer_values),
        }
    axes[0].legend(loc="upper left", ncol=2, fontsize=8)
    axes[-1].set_xlabel("absolute candidate token position")
    fig.suptitle("Aligned per-token all-Q1/Q2 query-summed scores (display-only z-score)")
    output_paths += save_figure(fig, output, "query_summed_aligned_token_scores_zscore")

    multihop_rows = sorted(multihop_rows, key=lambda row: int(row["position"]))
    positions = np.array([row["position"] for row in multihop_rows], dtype=np.int64)
    if positions.size == 0 or len(set(positions.tolist())) != positions.size:
        raise ValueError("MULTIHOP_TOKEN_POSITIONS_INVALID")
    max_level = int(multihop_report["max_level"])
    layer_sum_hops = np.array(
        [row["layer_sum"]["hop_contributions"] for row in multihop_rows],
        dtype=np.float64,
    ).T
    if layer_sum_hops.shape != (max_level, positions.size):
        raise ValueError("MULTIHOP_LEVEL_SHAPE_INVALID")
    layer_sum_cumulative = np.cumsum(layer_sum_hops, axis=0, dtype=np.float64)
    if not np.isfinite(layer_sum_cumulative).all():
        raise ValueError("MULTIHOP_LAYER_SUM_NONFINITE")

    max_abs = float(np.abs(layer_sum_cumulative).max())
    nonzero_abs = np.abs(layer_sum_cumulative[np.nonzero(layer_sum_cumulative)])
    linthresh = max(
        float(np.quantile(nonzero_abs, 0.05)) if nonzero_abs.size else 1.0,
        np.finfo(float).tiny,
    )
    fig, ax = plt.subplots(figsize=(14, 5))
    image = ax.imshow(
        layer_sum_cumulative,
        aspect="auto",
        origin="lower",
        extent=[positions[0], positions[-1] + 1, 0.5, max_level + 0.5],
        cmap="coolwarm",
        norm=SymLogNorm(linthresh=linthresh, vmin=-max_abs, vmax=max_abs, base=10),
    )
    ax.set_yticks(range(1, max_level + 1))
    ax.set_xlabel("absolute token position")
    ax.set_ylabel("cumulative level L")
    ax.set_title("Signed unnormalized cumulative layer-sum score by level (symmetric log color)")
    fig.colorbar(image, ax=ax, label="exact signed layer sum", ticks=symlog_ticks(max_abs, linthresh), format="%.1e")
    output_paths += save_figure(fig, output, "multihop_cumulative_layersum_symlog")

    normalized = np.vstack([zscore(row) for row in layer_sum_cumulative])
    norm_limit = max(3.0, float(np.quantile(np.abs(normalized), 0.995)))
    fig, ax = plt.subplots(figsize=(14, 5))
    image = ax.imshow(
        normalized,
        aspect="auto",
        origin="lower",
        extent=[positions[0], positions[-1] + 1, 0.5, max_level + 0.5],
        cmap="coolwarm",
        vmin=-norm_limit,
        vmax=norm_limit,
    )
    ax.set_yticks(range(1, max_level + 1))
    ax.set_xlabel("absolute token position")
    ax.set_ylabel("cumulative level L")
    ax.set_title("Cumulative layer-sum distribution movement (row-wise z-score for display)")
    fig.colorbar(image, ax=ax, label="within-level z-score")
    output_paths += save_figure(fig, output, "multihop_cumulative_layersum_zscore")

    per_layer_arrays: dict[int, np.ndarray] = {}
    for layer_index, layer in enumerate(layers):
        hops = np.array(
            [row["per_layer"][layer_index]["hop_contributions"] for row in multihop_rows],
            dtype=np.float64,
        ).T
        if hops.shape != (max_level, positions.size):
            raise ValueError("MULTIHOP_PER_LAYER_SHAPE_INVALID")
        per_layer_arrays[layer] = np.cumsum(hops, axis=0, dtype=np.float64)
    fig, axes = plt.subplots(len(layers), 1, figsize=(14, 3.1 * len(layers)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, layer in zip(axes, layers, strict=True):
        values = per_layer_arrays[layer]
        limit = float(np.abs(values).max())
        nonzero = np.abs(values[np.nonzero(values)])
        threshold = max(
            float(np.quantile(nonzero, 0.05)) if nonzero.size else 1.0,
            np.finfo(float).tiny,
        )
        image = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            extent=[positions[0], positions[-1] + 1, 0.5, max_level + 0.5],
            cmap="coolwarm",
            norm=SymLogNorm(linthresh=threshold, vmin=-limit, vmax=limit, base=10),
        )
        ax.set_ylabel(f"layer {layer}\nlevel")
        ax.set_yticks(range(1, max_level + 1))
        fig.colorbar(image, ax=ax, fraction=0.02, pad=0.01, ticks=symlog_ticks(limit, threshold), format="%.1e")
    axes[-1].set_xlabel("absolute token position")
    fig.suptitle("Per-layer signed unnormalized cumulative scores (symmetric log color)")
    output_paths += save_figure(fig, output, "multihop_per_layer_cumulative_symlog")

    multihop_numeric: dict[str, Any] = {"layer_sum": {}, "per_layer": {}}
    for level_index in range(max_level):
        values = layer_sum_cumulative[level_index]
        top_indices = np.argsort(np.abs(values))[-10:][::-1]
        record: dict[str, Any] = finite_summary(values)
        record["top_absolute_positions"] = [
            {"position": int(positions[index]), "score": float(values[index])}
            for index in top_indices
        ]
        if level_index:
            record["similarity_to_level_1"] = similarity(layer_sum_cumulative[0], values)
            record["similarity_to_previous_level"] = similarity(layer_sum_cumulative[level_index - 1], values)
        multihop_numeric["layer_sum"][str(level_index + 1)] = record
    for layer in layers:
        multihop_numeric["per_layer"][str(layer)] = {
            str(level_index + 1): finite_summary(per_layer_arrays[layer][level_index])
            for level_index in range(max_level)
        }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "source_policy": "plots_generated_only_from_checksum_attested_transferred_runpod_artifacts",
        "implementation": {
            "path": "scripts/analysis/plot_glm52_score_evidence.py",
            "sha256": sha256(Path(__file__)),
        },
        "input_checksum_manifest": {
            "path": safe_relative_path(args.checksum_manifest).as_posix(),
            "sha256": sha256(manifest_path),
        },
        "display_transform_policy": {
            "query_comparison": "independent within-layer population z-score for display; raw summaries retained",
            "raw_multihop": "symmetric logarithmic color only; signed raw values unchanged",
            "normalized_multihop": "independent within-level population z-score for display only",
        },
        "inputs": [
            {"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size}
            for relative, path in zip(relative_inputs, input_paths, strict=True)
        ],
        "layers": layers,
        "max_level": max_level,
        "query_sum_metrics_from_report": query_report["layers"],
        "query_sum_numeric_checks": query_numeric,
        "multihop_numeric_summary": multihop_numeric,
        "outputs": [],
        "claim_boundary": [
            "observation_only_not_proof_of_distillation",
            "no_executed_A1_no_stateful_or_quality_claim",
            "main_attention_is_reference_recomputed",
            "NVFP4_weight_only_Marlin_on_H200",
            "sampled_layers_and_bounded_window",
        ],
    }
    for path in output_paths:
        summary["outputs"].append(
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    schema = load_json(args.schema)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(summary)
    summary_path = output / "plot-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checksum_paths = sorted([*output_paths, summary_path], key=lambda path: path.name)
    (output / "PLOT_SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return {
        "status": "passed",
        "plot_summary": str(summary_path),
        "plots": len(output_paths),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--query-report-root", required=True)
    parser.add_argument("--multihop-root", required=True)
    parser.add_argument("--checksum-manifest", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = render(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
