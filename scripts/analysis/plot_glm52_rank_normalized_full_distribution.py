#!/usr/bin/env python3
"""Render complete-token GLM rank-normalized distribution comparisons."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from putpocket_dataset_mining.glm52_attention_indexer import file_sha256
from putpocket_dataset_mining.glm52_rank_normalized_full_distribution import (
    DEFAULT_LOG_PROBABILITY_FLOOR,
    PRIMARY_VARIANT_ID,
    build_full_distribution_analysis,
    build_input_attestation,
    build_summary_report,
    load_full_distribution_source,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
FIGURE_STEMS = (
    "primary_k64_hop_full_position_log_probability_heatmap",
    "primary_k64_cumulative_full_position_log_probability_heatmap",
    "primary_k64_hop_all_token_sorted_distribution",
    "primary_k64_cumulative_all_token_sorted_distribution",
    "primary_k64_hop_full_js_distance_matrix",
    "primary_k64_cumulative_full_js_distance_matrix",
)


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _file_item(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _save(fig: Any, output: Path, stem: str) -> list[Path]:
    paths = [output / f"{stem}.png", output / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(
            path,
            dpi=190 if path.suffix == ".png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)
    return paths


def _segment_legend_and_boundaries(
    ax: Any, q1_ranges: Sequence[Sequence[int]], q2_ranges: Sequence[Sequence[int]]
) -> None:
    colors = {"Q1": "tab:blue", "Q2": "tab:orange"}
    for label, ranges in (("Q1", q1_ranges), ("Q2", q2_ranges)):
        for left, right in ranges:
            ax.axvspan(left, right, color=colors[label], alpha=0.075, linewidth=0)
            ax.axvline(left, color=colors[label], alpha=0.65, linewidth=0.8)
            ax.axvline(right, color=colors[label], alpha=0.65, linewidth=0.8)
    ax.legend(
        handles=[
            Patch(facecolor=colors["Q1"], alpha=0.18, label="Q1 range"),
            Patch(facecolor=colors["Q2"], alpha=0.18, label="Q2 range"),
        ],
        loc="upper right",
        fontsize=8,
    )


def _position_heatmap(
    *,
    distributions: Sequence[Sequence[float]],
    positions: Sequence[int],
    levels: Sequence[int],
    q1_ranges: Sequence[Sequence[int]],
    q2_ranges: Sequence[Sequence[int]],
    view_label: str,
    output: Path,
    stem: str,
    floor: float,
) -> list[Path]:
    matrix = np.asarray(distributions, dtype=np.float64)
    transformed = np.log10(np.maximum(matrix, floor))
    start, end = positions[0], positions[-1] + 1
    fig, ax = plt.subplots(figsize=(17, 5.2))
    image = ax.imshow(
        transformed,
        aspect="auto",
        origin="lower",
        extent=(start, end, levels[0] - 0.5, levels[-1] + 0.5),
        interpolation="nearest",
        cmap="magma",
        vmin=math.log10(floor),
        vmax=0.0,
    )
    _segment_legend_and_boundaries(ax, q1_ranges, q2_ranges)
    ax.set_yticks(levels)
    ax.set_ylabel("propagation level")
    ax.set_xlabel("absolute token position (all tokens, identity preserved)")
    ax.set_title(
        f"Primary {PRIMARY_VARIANT_ID}: {view_label} full-position probability"
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.012)
    colorbar.set_label(f"log10(max(p, {floor:.0e})); fixed [{math.log10(floor):.0f}, 0]")
    return _save(fig, output, stem)


def _sorted_distribution_figure(
    *,
    distributions: Sequence[Sequence[float]],
    positions: Sequence[int],
    levels: Sequence[int],
    view_label: str,
    output: Path,
    stem: str,
    floor: float,
) -> list[Path]:
    width = len(positions)
    ranks = np.arange(1, width + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(levels)))
    for color, level, distribution in zip(
        colors, levels, distributions, strict=True
    ):
        order = sorted(
            range(width), key=lambda index: (-float(distribution[index]), positions[index])
        )
        sorted_values = np.asarray(
            [float(distribution[index]) for index in order], dtype=np.float64
        )
        axes[0].loglog(
            ranks,
            np.maximum(sorted_values, floor),
            color=color,
            linewidth=1.5,
            label=f"L{level}",
        )
        axes[1].semilogx(
            ranks,
            np.cumsum(sorted_values),
            color=color,
            linewidth=1.5,
            label=f"L{level}",
        )
    axes[0].set_title("all-token rank–probability curve")
    axes[0].set_xlabel(f"descending rank, 1..{width}")
    axes[0].set_ylabel(f"probability (display floor {floor:.0e})")
    axes[1].set_title("all-token cumulative-mass curve")
    axes[1].set_xlabel(f"descending rank, 1..{width}")
    axes[1].set_ylabel("cumulative probability mass")
    axes[1].set_ylim(-0.01, 1.01)
    for ax in axes:
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(ncol=2, fontsize=8)
    fig.suptitle(
        f"Primary {PRIMARY_VARIANT_ID}: {view_label}; complete {width}-token universe"
    )
    return _save(fig, output, stem)


def _js_matrix_figure(
    *,
    matrix: Sequence[Sequence[float]],
    levels: Sequence[int],
    view_label: str,
    output: Path,
    stem: str,
) -> list[Path]:
    values = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="cividis")
    labels = [f"L{level}" for level in levels]
    ax.set_xticks(range(len(levels)), labels)
    ax.set_yticks(range(len(levels)), labels)
    ax.set_xlabel("level")
    ax.set_ylabel("level")
    ax.set_title(
        f"Primary {PRIMARY_VARIANT_ID}: {view_label}\n"
        "full-token Jensen–Shannon distance (base 2)"
    )
    for row in range(len(levels)):
        for column in range(len(levels)):
            value = values[row, column]
            ax.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color="white" if value > 0.48 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=ax, label="sqrt(JSD₂), range [0,1]")
    return _save(fig, output, stem)


def render(
    source_artifact_root: Path,
    output_root: Path,
    *,
    log_probability_floor: float = DEFAULT_LOG_PROBABILITY_FLOOR,
) -> dict[str, Any]:
    _require(
        math.isfinite(log_probability_floor)
        and 0.0 < log_probability_floor < 1.0,
        "FULL_DISTRIBUTION_LOG_FLOOR_INVALID",
    )
    output = output_root.resolve()
    _require(not output.exists(), "FULL_DISTRIBUTION_OUTPUT_ALREADY_EXISTS")
    _require(output.parent.is_dir(), "FULL_DISTRIBUTION_OUTPUT_PARENT_MISSING")
    source = load_full_distribution_source(source_artifact_root)
    analysis = build_full_distribution_analysis(source)
    q1_ranges = source.report["payload"]["episode"]["q1_ranges"]
    q2_ranges = source.report["payload"]["episode"]["q2_ranges"]

    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.tmp-", dir=output.parent
    ) as temporary:
        staging = Path(temporary)
        figure_paths: list[Path] = []
        figure_paths += _position_heatmap(
            distributions=source.hop,
            positions=source.positions,
            levels=source.levels,
            q1_ranges=q1_ranges,
            q2_ranges=q2_ranges,
            view_label="hop-only",
            output=staging,
            stem=FIGURE_STEMS[0],
            floor=log_probability_floor,
        )
        figure_paths += _position_heatmap(
            distributions=source.cumulative,
            positions=source.positions,
            levels=source.levels,
            q1_ranges=q1_ranges,
            q2_ranges=q2_ranges,
            view_label="equal-hop cumulative",
            output=staging,
            stem=FIGURE_STEMS[1],
            floor=log_probability_floor,
        )
        figure_paths += _sorted_distribution_figure(
            distributions=source.hop,
            positions=source.positions,
            levels=source.levels,
            view_label="hop-only",
            output=staging,
            stem=FIGURE_STEMS[2],
            floor=log_probability_floor,
        )
        figure_paths += _sorted_distribution_figure(
            distributions=source.cumulative,
            positions=source.positions,
            levels=source.levels,
            view_label="equal-hop cumulative",
            output=staging,
            stem=FIGURE_STEMS[3],
            floor=log_probability_floor,
        )
        figure_paths += _js_matrix_figure(
            matrix=analysis["jensen_shannon_distance"]["hop"],
            levels=source.levels,
            view_label="hop-only",
            output=staging,
            stem=FIGURE_STEMS[4],
        )
        figure_paths += _js_matrix_figure(
            matrix=analysis["jensen_shannon_distance"][
                "cumulative_equal_hop_mixture"
            ],
            levels=source.levels,
            view_label="equal-hop cumulative",
            output=staging,
            stem=FIGURE_STEMS[5],
        )
        _require(
            len(figure_paths) == 12
            and {path.stem for path in figure_paths} == set(FIGURE_STEMS),
            "FULL_DISTRIBUTION_FIGURE_SET_INVALID",
        )

        attestation = build_input_attestation(source)
        attestation_path = staging / "full-distribution-input-attestation.json"
        attestation_path.write_text(
            json.dumps(attestation, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        implementation = [
            {
                "path": "src/putpocket_dataset_mining/glm52_rank_normalized_full_distribution.py",
                "sha256": file_sha256(
                    REPOSITORY_ROOT
                    / "src/putpocket_dataset_mining/glm52_rank_normalized_full_distribution.py"
                ),
            },
            {
                "path": "scripts/analysis/plot_glm52_rank_normalized_full_distribution.py",
                "sha256": file_sha256(Path(__file__)),
            },
        ]
        report = build_summary_report(
            source=source,
            analysis=analysis,
            attestation_artifact=_file_item(attestation_path),
            implementation=implementation,
            figures=[_file_item(path) for path in sorted(figure_paths)],
            log_probability_floor=log_probability_floor,
        )
        summary_path = staging / "full-distribution-summary.json"
        summary_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        final_files = sorted(
            [path for path in staging.iterdir() if path.is_file()],
            key=lambda path: path.name,
        )
        (staging / "FINAL_SHA256SUMS").write_text(
            "".join(
                f"{file_sha256(path)}  {path.name}\n" for path in final_files
            ),
            encoding="utf-8",
        )
        os.rename(staging, output)

    return {
        "status": "passed",
        "output_root": str(output),
        "figure_files": 12,
        "token_count": len(source.positions),
        "levels": list(source.levels),
        "summary_payload_sha256": report["payload_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-artifact-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--log-probability-floor",
        type=float,
        default=DEFAULT_LOG_PROBABILITY_FLOOR,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(
        json.dumps(
            render(
                args.source_artifact_root,
                args.output_root,
                log_probability_floor=args.log_probability_floor,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
