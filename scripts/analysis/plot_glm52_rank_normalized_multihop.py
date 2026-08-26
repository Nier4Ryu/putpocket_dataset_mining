#!/usr/bin/env python3
"""Plot a schema-valid GLM rank-normalized multihop artifact in place."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jsonschema import Draft202012Validator

from putpocket_dataset_mining.glm52_attention_indexer import canonical_json_bytes


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_SCHEMA = (
    REPOSITORY_ROOT
    / "configs/runpod/schemas/glm52_rank_normalized_multihop_report.schema.json"
)
TOKEN_SCHEMA = (
    REPOSITORY_ROOT
    / "configs/runpod/schemas/glm52_rank_normalized_multihop_token_row.schema.json"
)
PLOT_SCHEMA = (
    REPOSITORY_ROOT
    / "configs/runpod/schemas/glm52_rank_normalized_plot_summary.schema.json"
)
EXPECTED_SOURCE_FILES = {
    "REPORT_SHA256SUMS",
    "rank-normalized-input-attestation.json",
    "rank-normalized-multihop-report.json",
    "rank-normalized-multihop-token-scores.jsonl",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _load_schema(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(value)
    return value


def _verify_source_checksums(root: Path) -> None:
    manifest = root / "REPORT_SHA256SUMS"
    entries: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(None, 1)
        name = name.lstrip("* ")
        _require(name not in entries, "PLOT_SOURCE_CHECKSUM_DUPLICATE")
        entries[name] = digest
    _require(
        set(entries) == EXPECTED_SOURCE_FILES - {"REPORT_SHA256SUMS"},
        "PLOT_SOURCE_CHECKSUM_SET_INVALID",
    )
    for name, expected in entries.items():
        path = root / name
        _require(path.is_file() and not path.is_symlink(), "PLOT_SOURCE_MISSING")
        _require(sha256(path) == expected, "PLOT_SOURCE_DIGEST_MISMATCH")


def _load_sources(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    existing = {path.name for path in root.iterdir()}
    _require(existing == EXPECTED_SOURCE_FILES, "PLOT_ARTIFACT_ROOT_NOT_PRISTINE")
    _verify_source_checksums(root)
    report_path = root / "rank-normalized-multihop-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    Draft202012Validator(_load_schema(REPORT_SCHEMA)).validate(report)
    expected_payload = hashlib.sha256(canonical_json_bytes(report["payload"])).hexdigest()
    _require(report["payload_sha256"] == expected_payload, "PLOT_REPORT_PAYLOAD_DIGEST_MISMATCH")
    rows = []
    validator = Draft202012Validator(_load_schema(TOKEN_SCHEMA))
    token_path = root / "rank-normalized-multihop-token-scores.jsonl"
    with token_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            validator.validate(row)
            observed = row.pop("record_sha256")
            expected = hashlib.sha256(canonical_json_bytes(row)).hexdigest()
            row["record_sha256"] = observed
            _require(observed == expected, "PLOT_TOKEN_ROW_DIGEST_MISMATCH")
            rows.append(row)
    token_artifact = report["payload"]["token_level_artifact"]
    _require(
        len(rows) == token_artifact["record_count"]
        and sha256(token_path) == token_artifact["sha256"],
        "PLOT_TOKEN_ARTIFACT_MISMATCH",
    )
    positions = [row["position"] for row in rows]
    _require(
        positions == list(range(positions[0], positions[0] + len(positions))),
        "PLOT_TOKEN_POSITIONS_NOT_CONTIGUOUS",
    )
    return report, rows


def _metric(level: Mapping[str, Any], view: str, baseline: str, k: int) -> Mapping[str, Any] | None:
    distribution = level[view]
    stability = distribution[baseline]
    if stability is None:
        return None
    return next(item for item in stability["by_k"] if item["k"] == k)


def _save(fig: Any, root: Path, stem: str) -> list[Path]:
    paths = [root / f"{stem}.png", root / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, dpi=180 if path.suffix == ".png" else None, bbox_inches="tight")
    plt.close(fig)
    return paths


def _variant_token_data(rows: list[dict[str, Any]]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    result: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        for variant in row["variants"]:
            result.setdefault(variant["variant_id"], {})[row["position"]] = variant["levels"]
    return result


def render(artifact_root: Path) -> dict[str, Any]:
    root = artifact_root.resolve()
    _require(root.is_dir(), "PLOT_ARTIFACT_ROOT_MISSING")
    report, token_rows = _load_sources(root)
    payload = report["payload"]
    variant_results = payload["variant_results"]
    variant_ids = [item["variant_id"] for item in variant_results]
    max_level = payload["transition_config"]["max_level"]
    primary = payload["transition_config"]["primary_variant_id"]
    _require(primary in variant_ids, "PLOT_PRIMARY_VARIANT_MISSING")
    colors = plt.cm.tab10(np.linspace(0, 1, len(variant_ids)))
    output_paths: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    view_specs = [
        ("hop", "hop-only"),
        ("cumulative_equal_hop_mixture", "equal-hop cumulative"),
    ]
    baseline_specs = [
        ("versus_level_1", "versus level 1"),
        ("versus_previous_level", "versus previous level"),
    ]
    for row_index, (view, view_label) in enumerate(view_specs):
        for column_index, (baseline, baseline_label) in enumerate(baseline_specs):
            ax = axes[row_index, column_index]
            for color, variant in zip(colors, variant_results, strict=True):
                levels, values = [], []
                for level in variant["level_results"]:
                    metric = _metric(level, view, baseline, 64)
                    if metric is not None:
                        levels.append(level["level"])
                        values.append(metric["retention"])
                ax.plot(levels, values, marker="o", label=variant["variant_id"], color=color)
            ax.set_title(f"{view_label}: {baseline_label}")
            ax.set_ylim(-0.02, 1.02)
            ax.grid(alpha=0.25)
            ax.set_xlabel("level")
            ax.set_ylabel("top-64 retention")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Rank-normalized multihop top-k stability")
    output_paths += _save(fig, root, "rank_normalized_topk_stability")

    token_data = _variant_token_data(token_rows)
    primary_data = token_data[primary]
    union: set[int] = set()
    for position, levels in primary_data.items():
        if any(
            level[rank_field] <= 64
            for level in levels
            for rank_field in ("hop_rank", "cumulative_rank")
        ):
            union.add(position)
    _require(bool(union), "PLOT_PRIMARY_TOP_UNION_EMPTY")
    ordered_union = sorted(
        union,
        key=lambda position: (
            primary_data[position][0]["hop_rank"],
            min(
                min(level["hop_rank"], level["cumulative_rank"])
                for level in primary_data[position]
            ),
            position,
        ),
    )
    width = len(token_rows)
    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(ordered_union) * 0.065), 7), sharex=True)
    for ax, (rank_field, title) in zip(
        axes,
        (
            ("hop_rank", "hop-only normalized rank score"),
            ("cumulative_rank", "equal-hop cumulative normalized rank score"),
        ),
        strict=True,
    ):
        matrix = np.array(
            [
                [
                    1.0 - (primary_data[position][level_index][rank_field] - 1) / (width - 1)
                    for position in ordered_union
                ]
                for level_index in range(max_level)
            ],
            dtype=np.float64,
        )
        image = ax.imshow(matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_yticks(range(max_level), range(1, max_level + 1))
        ax.set_ylabel("level")
        ax.set_title(title)
        fig.colorbar(image, ax=ax, label="1 - (global rank-1)/(N-1)")
    axes[-1].set_xticks(
        range(len(ordered_union)),
        [str(position) for position in ordered_union],
        rotation=90,
        fontsize=5,
    )
    axes[-1].set_xlabel("absolute token position; deterministic union ordering")
    fig.suptitle(f"Primary {primary}: stable union of level top-64 tokens")
    output_paths += _save(fig, root, "primary_k64_level_token_rank_heatmap")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for row_index, (view, view_label) in enumerate(view_specs):
        for column_index, (field, field_label) in enumerate(
            (("entropy_nats", "entropy (nats)"), ("effective_support", "effective support"))
        ):
            ax = axes[row_index, column_index]
            for color, variant in zip(colors, variant_results, strict=True):
                levels = [item["level"] for item in variant["level_results"]]
                values = [item[view][field] for item in variant["level_results"]]
                ax.plot(levels, values, marker="o", label=variant["variant_id"], color=color)
            ax.set_title(f"{view_label}: {field_label}")
            ax.set_xlabel("level")
            ax.set_ylabel(field_label)
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Normalized distribution concentration by level")
    output_paths += _save(fig, root, "rank_normalized_entropy_effective_support")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (view, view_label) in zip(axes, view_specs, strict=True):
        matrix = np.array(
            [
                [
                    _metric(level, view, "versus_level_1", 64)["jaccard"]
                    for level in variant["level_results"]
                ]
                for variant in variant_results
            ],
            dtype=np.float64,
        )
        image = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="magma")
        ax.set_xticks(range(max_level), range(1, max_level + 1))
        ax.set_yticks(range(len(variant_ids)), variant_ids)
        ax.set_xlabel("level")
        ax.set_title(f"{view_label}: top-64 Jaccard vs L1")
        for variant_index in range(len(variant_ids)):
            for level_index in range(max_level):
                ax.text(level_index, variant_index, f"{matrix[variant_index, level_index]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix[variant_index, level_index] < 0.55 else "black")
        fig.colorbar(image, ax=ax, label="Jaccard")
    fig.suptitle("Transition sensitivity: DCG K sweep and z-score softmax")
    output_paths += _save(fig, root, "rank_normalized_transition_sensitivity")

    primary_result = next(item for item in variant_results if item["variant_id"] == primary)
    primary_table = []
    for level in primary_result["level_results"]:
        row: dict[str, Any] = {
            "level": level["level"],
            "hop_mass": level["hop"]["mass"],
            "cumulative_mass": level["cumulative_equal_hop_mixture"]["mass"],
        }
        for view, prefix in (
            ("hop", "hop"),
            ("cumulative_equal_hop_mixture", "cumulative"),
        ):
            for baseline, suffix in (
                ("versus_level_1", "vs_l1"),
                ("versus_previous_level", "vs_previous"),
            ):
                metric = _metric(level, view, baseline, 64)
                row[f"{prefix}_{suffix}_retention"] = None if metric is None else metric["retention"]
                row[f"{prefix}_{suffix}_jaccard"] = None if metric is None else metric["jaccard"]
        primary_table.append(row)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "glm52_rank_normalized_multihop_plot_summary",
        "project_commit": payload["project_commit"],
        "implementation": {
            "path": "scripts/analysis/plot_glm52_rank_normalized_multihop.py",
            "sha256": sha256(Path(__file__)),
        },
        "inputs": [
            {"path": "rank-normalized-multihop-report.json", "bytes": (root / "rank-normalized-multihop-report.json").stat().st_size, "sha256": sha256(root / "rank-normalized-multihop-report.json")},
            {"path": "rank-normalized-multihop-token-scores.jsonl", "bytes": (root / "rank-normalized-multihop-token-scores.jsonl").stat().st_size, "sha256": sha256(root / "rank-normalized-multihop-token-scores.jsonl")},
        ],
        "recurrence_id": payload["recurrence"]["id"],
        "primary_variant_id": primary,
        "max_level": max_level,
        "stable_top_union": {"evaluation_k": 64, "positions": ordered_union},
        "primary_k64_table": primary_table,
        "display_policy": {
            "stability": "exact report top-64 retention and Jaccard",
            "token_heatmap": "global deterministic rank converted to 1-(rank-1)/(N-1); probabilities unchanged in token artifact",
            "entropy": "exact normalized-distribution entropy and effective support",
        },
        "outputs": [],
        "claim_boundary": [
            "offline_rank_relevance_only",
            "legacy_signed_outputs_preserved",
            "no_inference_selector_integration",
            "no_stateful_cache_or_task_quality_claim",
            "no_gpu_rerun",
        ],
    }
    for path in output_paths:
        summary["outputs"].append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)})
    Draft202012Validator(_load_schema(PLOT_SCHEMA)).validate(summary)
    summary_path = root / "rank-normalized-plot-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    final_files = sorted(
        [path for path in root.iterdir() if path.is_file() and path.name != "FINAL_SHA256SUMS"],
        key=lambda path: path.name,
    )
    (root / "FINAL_SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in final_files),
        encoding="utf-8",
    )
    return {"status": "passed", "plots": len(output_paths), "plot_summary": str(summary_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    print(json.dumps(render(parse_args().artifact_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
