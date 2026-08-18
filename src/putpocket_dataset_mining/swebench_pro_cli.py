from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from .cluster_safety import require_slurm_allocation, safe_absolute_path
from .errors import ConfigError
from .swebench_pro import (
    AGENT_OVERLAY,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    DOCKERHUB_NAMESPACE,
    HARNESS_SHA,
    MINI_SWE_AGENT_SHA,
    MODEL_ID,
    NON_SCORE_ELIGIBLE_SMOKE_ONLY,
    SWE_AGENT_SHA,
    build_runtime_agent_config,
    finalize_official_results,
    load_selection,
    run_restartable_stage,
    select_rows,
    validate_agent_overlay,
    validate_official_image_mapping,
    validate_source_lock,
    validate_smoke_only_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _validate(args)
        if args.command == "render":
            return _render(args)
        if args.command == "render-wrap":
            return _render_wrap(args)
        if args.command == "prepare":
            return _prepare(args)
        if args.command == "agent-config":
            return _agent_config(args)
        if args.command == "gather":
            return _gather(args)
        if args.command == "stage":
            return _stage(args)
        if args.command == "finalize":
            return _finalize(args)
        if args.command == "assert-smoke-report":
            return _assert_smoke_report(args)
        if args.command == "provenance":
            return _provenance(args)
    except (ConfigError, OSError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"schema_version": 1, "status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    parser.error("a command is required")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="putpocket-swebench-pro")
    commands = parser.add_subparsers(dest="command")
    validate = commands.add_parser("validate", help="Validate pinned phase-2 source, selection, and agent contracts")
    validate.add_argument(
        "--smoke-only",
        action="store_true",
        help="Validate only the one-instance smoke contract; never load the full selection",
    )

    render = commands.add_parser("render", help="Render the exact-four-H200 Slurm job without submitting")
    render.add_argument("--site", required=True)
    render.add_argument("--project-url", required=True)
    render.add_argument("--project-commit", required=True)
    render_mode = render.add_mutually_exclusive_group()
    render_mode.add_argument("--preflight-only", action="store_true")
    render_mode.add_argument(
        "--smoke-only",
        action="store_true",
        help="Render the non-score-eligible one-instance path with no full-selection transition",
    )
    render.add_argument("--output", default=None)

    render_wrap = commands.add_parser(
        "render-wrap", help="Render a compact smoke-only sbatch --wrap command without submitting"
    )
    render_wrap.add_argument("--site", required=True)
    render_wrap.add_argument("--project-url", required=True)
    render_wrap.add_argument("--project-commit", required=True)

    prepare = commands.add_parser("prepare", help="Allocation-only pinned dataset selection materialization")
    prepare.add_argument("--selection", choices=["smoke", "full"], required=True)
    prepare.add_argument("--harness-root", required=True)
    prepare.add_argument("--output-root", required=True)

    agent = commands.add_parser("agent-config", help="Render the pinned mini-swe-agent local-vLLM config")
    agent.add_argument("--harness-root", required=True)
    agent.add_argument("--runtime", choices=["docker", "singularity"], required=True)
    agent.add_argument("--output", required=True)

    gather = commands.add_parser("gather", help="Materialize mini-swe-agent predictions and call the official gather helper")
    gather.add_argument("--harness-root", required=True)
    gather.add_argument("--inference-root", required=True)
    gather.add_argument("--prefix", required=True)
    gather.add_argument("--output", required=True)

    stage = commands.add_parser("stage", help="Run or resume one allocation-guarded benchmark stage")
    stage.add_argument("--stage", choices=["prepare", "inference", "gather", "evaluate", "finalize"], required=True)
    stage.add_argument("--artifact-root", required=True)
    stage.add_argument("--fingerprint", required=True)
    stage.add_argument("stage_command", nargs=argparse.REMAINDER)

    finalize = commands.add_parser("finalize", help="Summarize unchanged official scorer results")
    finalize.add_argument("--selection", choices=["smoke", "full"], required=True)
    finalize.add_argument("--selection-manifest", required=True)
    finalize.add_argument("--eval-results", required=True)
    finalize.add_argument("--output", required=True)

    smoke_report = commands.add_parser(
        "assert-smoke-report", help="Fail closed unless a completed report remains one-instance and non-score-eligible"
    )
    smoke_report.add_argument("--report", required=True)

    provenance = commands.add_parser("provenance", help="Write allowlisted baseline provenance")
    provenance.add_argument("--project-root", required=True)
    provenance.add_argument("--harness-root", required=True)
    provenance.add_argument("--model-revision", required=True)
    provenance.add_argument(
        "--parallel-profile",
        choices=["glm52_nvfp4_tp1_pcp4_ep", "glm52_nvfp4_tp2_pcp2_ep"],
        required=True,
    )
    provenance.add_argument("--container-runtime", choices=["docker"], required=True)
    provenance.add_argument("--runtime-manifest", required=True)
    provenance.add_argument("--artifact-root", required=True)
    provenance.add_argument("--output", required=True)
    return parser


def _validate(args: argparse.Namespace) -> int:
    validate_source_lock()
    overlay = validate_agent_overlay()
    modes = ["smoke"] if args.smoke_only else ["smoke", "full"]
    selections = [load_selection(mode) for mode in modes]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "harness_commit": HARNESS_SHA,
                "dataset_revision": DATASET_REVISION,
                "model_id": overlay["model"]["id"],
                "claim_boundary": NON_SCORE_ELIGIBLE_SMOKE_ONLY if args.smoke_only else "full_contract_available",
                "selections": [item.selection_id for item in selections],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _render(args: argparse.Namespace) -> int:
    from .swebench_pro_slurm import load_baseline_site, render_baseline_job

    rendered = render_baseline_job(
        site=load_baseline_site(args.site),
        project_url=args.project_url,
        project_commit=args.project_commit,
        preflight_only=args.preflight_only,
        smoke_only=args.smoke_only,
    )
    if args.output:
        output = Path(args.output)
        if output.suffix != ".sbatch":
            raise ConfigError("Rendered Slurm output must use the .sbatch suffix")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(json.dumps({"status": "rendered", "submitted": False, "output": str(output)}, indent=2))
    else:
        print(rendered, end="")
    return 0


def _render_wrap(args: argparse.Namespace) -> int:
    from .swebench_pro_slurm import load_baseline_site, render_compact_smoke_submission

    rendered = render_compact_smoke_submission(
        site=load_baseline_site(args.site),
        project_url=args.project_url,
        project_commit=args.project_commit,
    )
    print(rendered)
    return 0


def _prepare(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    selection = load_selection(args.selection)
    harness_root = safe_absolute_path(args.harness_root, "harness_root")
    output_root = safe_absolute_path(args.output_root, "output_root")
    _validate_harness_checkout(harness_root)
    from datasets import Dataset, load_dataset

    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, split=DATASET_SPLIT)
    rows = select_rows(list(dataset), selection)
    validate_official_image_mapping(harness_root, rows)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_path = output_root / "raw_samples.jsonl"
    _write_jsonl(raw_path, rows)
    mini_root = output_root / "mini_dataset" / "data"
    mini_root.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(mini_root / "test-00000-of-00001.parquet")
    manifest = {
        "schema_version": 1,
        "selection_id": selection.selection_id,
        "mode": selection.mode,
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "split": DATASET_SPLIT,
        "count": len(rows),
        "instance_ids": [row["instance_id"] for row in rows],
        "dockerhub_namespace": DOCKERHUB_NAMESPACE,
        "image_source": "dataset.dockerhub_tag",
        "raw_samples": str(raw_path),
        "mini_dataset": str(output_root / "mini_dataset"),
    }
    _write_json(output_root / "selection_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _agent_config(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    harness_root = safe_absolute_path(args.harness_root, "harness_root")
    _validate_harness_checkout(harness_root)
    overlay = validate_agent_overlay(AGENT_OVERLAY)
    relative = overlay["scaffold"]["config"]
    base_path = harness_root / "mini-swe-agent" / relative
    if not base_path.is_file():
        raise ConfigError(f"Pinned mini-swe-agent scaffold config is missing: {relative}")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ConfigError("Pinned mini-swe-agent scaffold config root must be a mapping")
    rendered = build_runtime_agent_config(base, overlay, args.runtime)
    output = safe_absolute_path(args.output, "output")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(yaml.safe_dump(rendered, sort_keys=False), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": "rendered", "runtime": args.runtime, "output": str(output)}, indent=2))
    return 0


def _gather(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    harness_root = safe_absolute_path(args.harness_root, "harness_root")
    inference_root = safe_absolute_path(args.inference_root, "inference_root")
    output = safe_absolute_path(args.output, "output")
    _validate_harness_checkout(harness_root)
    if not args.prefix or not all(ch.isalnum() or ch in "_.-" for ch in args.prefix):
        raise ConfigError("Prediction prefix contains unsafe characters")
    predictions_path = inference_root / "preds.json"
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, dict):
        raise ConfigError("mini-swe-agent preds.json must be an instance mapping")
    pred_root = inference_root / "official_pred_inputs"
    for instance_id, prediction in predictions.items():
        if not isinstance(instance_id, str) or not instance_id.startswith("instance_") or not isinstance(prediction, dict):
            raise ConfigError("mini-swe-agent prediction has an invalid instance entry")
        target = pred_root / instance_id / f"{instance_id}.pred"
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(target, prediction)
    helper = harness_root / "helper_code" / "gather_patches.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(helper), "--directory", str(pred_root), "--prefix", args.prefix, "--output", str(output)],
        check=False,
    )
    if result.returncode != 0:
        raise ConfigError(f"Official gather_patches.py failed with return code {result.returncode}")
    return 0


def _stage(args: argparse.Namespace) -> int:
    command = list(args.stage_command)
    if command and command[0] == "--":
        command = command[1:]
    result = run_restartable_stage(
        stage=args.stage,
        command=command,
        artifact_root=args.artifact_root,
        fingerprint=args.fingerprint,
    )
    print(
        json.dumps(
            {"schema_version": 1, "stage": result.stage, "status": result.status, "returncode": result.returncode},
            indent=2,
        )
    )
    return result.returncode


def _finalize(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    selection = load_selection(args.selection)
    manifest = json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))
    results = json.loads(Path(args.eval_results).read_text(encoding="utf-8"))
    if manifest.get("dataset_revision") != DATASET_REVISION or manifest.get("selection_id") != selection.selection_id:
        raise ConfigError("Selection manifest does not match the pinned dataset/selection")
    if not isinstance(results, dict):
        raise ConfigError("Official eval_results.json must contain an instance-to-boolean mapping")
    report = finalize_official_results(
        selection=selection,
        expected_instance_ids=manifest.get("instance_ids", []),
        eval_results=results,
    )
    _write_json(safe_absolute_path(args.output, "output"), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "complete" else 3


def _assert_smoke_report(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    report_path = safe_absolute_path(args.report, "report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ConfigError("Smoke acceptance report must be a JSON object")
    claim = validate_smoke_only_report(report)
    print(json.dumps(claim, indent=2, sort_keys=True))
    return 0


def _provenance(args: argparse.Namespace) -> int:
    allocation = require_slurm_allocation()
    project_root = safe_absolute_path(args.project_root, "project_root")
    harness_root = safe_absolute_path(args.harness_root, "harness_root")
    artifact_root = safe_absolute_path(args.artifact_root, "artifact_root")
    runtime_manifest = safe_absolute_path(args.runtime_manifest, "runtime_manifest")
    if not runtime_manifest.is_file():
        raise ConfigError("Phase-1 runtime hardware manifest is missing")
    project_sha = _git_head(project_root)
    _validate_harness_checkout(harness_root)
    versions: dict[str, str] = {}
    for package in ("torch", "vllm", "flashinfer-python", "mini-swe-agent", "datasets", "docker"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    payload = {
        "schema_version": 1,
        "project_git_sha": project_sha,
        "harness": {"repository": "https://github.com/scaleapi/SWE-bench_Pro-os.git", "commit": HARNESS_SHA},
        "submodules": {"SWE-agent": SWE_AGENT_SHA, "mini-swe-agent": MINI_SWE_AGENT_SHA},
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": DATASET_SPLIT},
        "model": {"id": MODEL_ID, "revision": args.model_revision, "behavior": "unmodified_baseline"},
        "parallel_profile_used": args.parallel_profile,
        "slurm": allocation,
        "container_runtime": args.container_runtime,
        "package_versions": versions,
        "runtime_hardware_manifest": str(runtime_manifest),
        "artifact_root": str(artifact_root),
        "secret_policy": "allowlisted_fields_only_no_environment_dump",
        "checkpoint_hash_policy": "metadata_only_no_full_tensor_hash",
    }
    _write_json(safe_absolute_path(args.output, "output"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _validate_harness_checkout(root: Path) -> None:
    if _git_head(root) != HARNESS_SHA:
        raise ConfigError("Official SWE-bench Pro harness checkout is not at the pinned commit")
    if _git_head(root / "SWE-agent") != SWE_AGENT_SHA:
        raise ConfigError("Official SWE-agent submodule is not at the pinned commit")
    if _git_head(root / "mini-swe-agent") != MINI_SWE_AGENT_SHA:
        raise ConfigError("Official mini-swe-agent submodule is not at the pinned commit")


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True, check=False, timeout=30
    )
    if result.returncode != 0:
        raise ConfigError(f"Git checkout unavailable: {root}")
    return result.stdout.strip()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
