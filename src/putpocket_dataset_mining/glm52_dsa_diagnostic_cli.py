from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .cluster_safety import require_slurm_allocation, safe_absolute_path
from .errors import ConfigError
from .glm52_dsa_diagnostic import (
    DATASET_ID,
    DATASET_REVISION,
    HARNESS_COMMIT,
    INSTANCE_ID,
    LOCK_PATH,
    MINI_SWE_COMMIT,
    MODEL_ID,
    MODEL_REVISION,
    SGLANG_COMMIT,
    SGLANG_IMAGE,
    build_artifact_manifest,
    compress_capture_records,
    extract_completion,
    load_lock,
    validate_capture_coverage,
    validate_diagnostic_server_isolation,
    validate_lock,
    validate_patch_inputs,
    validate_selected_row,
    validate_serialized_prompt,
    validate_trace_equivalence,
)
from .glm52_sglang_gate import (
    load_json,
    parse_hbm_csv,
    summarize_hbm,
    validate_model_config,
    validate_runtime_log,
    validate_server_info,
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        handlers = {
            "validate-lock": _validate_lock,
            "render-wrap": _render_wrap,
            "validate-patch": _validate_patch,
            "prepare": _prepare,
            "control": _control,
            "trace-equivalence": _trace_equivalence,
            "finalize-captures": _finalize_captures,
            "extract-action": _extract_action,
            "make-prediction": _make_prediction,
            "finalize-diagnostic": _finalize_diagnostic,
        }
        if args.command in handlers:
            return handlers[args.command](args)
    except (ConfigError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema_version": 1, "status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    parser.error("a command is required")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="putpocket-glm52-dsa-diagnostic")
    commands = parser.add_subparsers(dest="command")
    check = commands.add_parser("validate-lock")
    check.add_argument("--lock", default=str(LOCK_PATH))

    render = commands.add_parser("render-wrap")
    render.add_argument("--site", required=True)
    render.add_argument("--project-url", required=True)
    render.add_argument("--project-commit", required=True)

    patch = commands.add_parser("validate-patch")
    patch.add_argument("--lock", default=str(LOCK_PATH))
    patch.add_argument("--repository-root", required=True)
    patch.add_argument("--source-root", required=True)
    patch.add_argument("--output", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--lock", default=str(LOCK_PATH))
    prepare.add_argument("--model-root", required=True)
    prepare.add_argument("--harness-root", required=True)
    prepare.add_argument("--ephemeral-root", required=True)
    prepare.add_argument("--artifact-root", required=True)

    control = commands.add_parser("control")
    control.add_argument("--lock", default=str(LOCK_PATH))
    control.add_argument("--mode", choices=["OFF", "ON"], required=True)
    control.add_argument("--run-id", required=True)
    control.add_argument("--output", required=True)

    equivalence = commands.add_parser("trace-equivalence")
    equivalence.add_argument("--off-response", required=True)
    equivalence.add_argument("--on-response", required=True)
    equivalence.add_argument("--off-duration-ns", type=int, required=True)
    equivalence.add_argument("--on-duration-ns", type=int, required=True)
    equivalence.add_argument("--output", required=True)

    captures = commands.add_parser("finalize-captures")
    captures.add_argument("--lock", default=str(LOCK_PATH))
    captures.add_argument("--raw-root", required=True)
    captures.add_argument("--output-root", required=True)
    captures.add_argument("--trace-report", required=True)
    captures.add_argument("--run-id", required=True)

    action = commands.add_parser("extract-action")
    action.add_argument("--response", required=True)
    action.add_argument("--action-output", required=True)
    action.add_argument("--metadata-output", required=True)

    prediction = commands.add_parser("make-prediction")
    prediction.add_argument("--patch", required=True)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--official-pred-root", required=True)

    final = commands.add_parser("finalize-diagnostic")
    final.add_argument("--inventory", required=True)
    final.add_argument("--model-config", required=True)
    final.add_argument("--server-info", required=True)
    final.add_argument("--server-log", required=True)
    final.add_argument("--model-revision", required=True)
    final.add_argument("--trace-report", required=True)
    final.add_argument("--capture-manifest", required=True)
    final.add_argument("--eval-results", required=True)
    final.add_argument("--load-hbm", required=True)
    final.add_argument("--off-hbm", required=True)
    final.add_argument("--on-hbm", required=True)
    final.add_argument("--project-commit", required=True)
    final.add_argument("--output", required=True)
    return parser


def _validate_lock(args: argparse.Namespace) -> int:
    report = validate_lock(load_lock(args.lock))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _render_wrap(args: argparse.Namespace) -> int:
    from .glm52_dsa_diagnostic_slurm import render_compact_diagnostic_submission
    from .glm52_sglang_gate_slurm import load_gate_site

    command = render_compact_diagnostic_submission(
        site=load_gate_site(args.site),
        project_url=args.project_url,
        project_commit=args.project_commit,
    )
    print(command)
    return 0


def _validate_patch(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    report = validate_patch_inputs(args.repository_root, args.source_root, load_lock(args.lock))
    _write_json(safe_absolute_path(args.output, "output"), report)
    return 0


def _prepare(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    lock = load_lock(args.lock)
    validate_lock(lock)
    model_root = safe_absolute_path(args.model_root, "model_root")
    harness_root = safe_absolute_path(args.harness_root, "harness_root")
    ephemeral_root = safe_absolute_path(args.ephemeral_root, "ephemeral_root")
    artifact_root = safe_absolute_path(args.artifact_root, "artifact_root")
    _validate_harness(harness_root, lock)
    from datasets import load_dataset
    from jinja2 import StrictUndefined, Template
    import yaml
    from transformers import AutoTokenizer

    dataset = load_dataset(
        lock["swebench_pro"]["dataset"],
        revision=lock["swebench_pro"]["dataset_revision"],
        split=lock["swebench_pro"]["split"],
    )
    matches = [dict(row) for row in dataset if row.get("instance_id") == INSTANCE_ID]
    if len(matches) != 1:
        raise ConfigError(f"SELECTED_INSTANCE_CARDINALITY_MISMATCH:{len(matches)}")
    row = matches[0]
    row_report = validate_selected_row(row, lock)
    config_path = harness_root / "mini-swe-agent" / lock["swebench_pro"]["mini_swe_scaffold"]
    if _sha256(config_path) != lock["swebench_pro"]["mini_swe_scaffold_sha256"]:
        raise ConfigError("MINI_SWE_SCAFFOLD_DIGEST_MISMATCH")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    agent = config.get("agent") if isinstance(config, dict) else None
    if not isinstance(agent, dict):
        raise ConfigError("MINI_SWE_SCAFFOLD_INVALID")
    messages = [
        {
            "role": "system",
            "content": Template(agent["system_template"], undefined=StrictUndefined).render(task=row["problem_statement"]),
        },
        {
            "role": "user",
            "content": Template(agent["instance_template"], undefined=StrictUndefined).render(task=row["problem_statement"]),
        },
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    serialized = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False
    )
    if not isinstance(token_ids, list):
        raise ConfigError("TOKENIZER_RETURN_SHAPE_UNSUPPORTED")
    tokenizer_files = {
        name: _sha256(model_root / name)
        for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    }
    prompt_report = validate_serialized_prompt(
        serialized,
        token_ids,
        lock,
        tokenizer_file_digests=tokenizer_files,
        tokenizer_class=type(tokenizer).__name__,
    )
    ephemeral_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    request = {
        "model": MODEL_ID,
        "prompt": token_ids,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(lock["runtime"]["max_new_tokens"]),
        "n": 1,
        "seed": int(lock["runtime"]["seed"]),
        "stream": False,
        "logprobs": 0,
        "return_token_ids": True,
    }
    _write_json(ephemeral_root / "completion_request.json", request)
    _write_jsonl(ephemeral_root / "official_raw_sample.jsonl", [row])
    _write_json(artifact_root / "selection_manifest.json", row_report)
    _write_json(artifact_root / "prompt_metadata.json", prompt_report)
    (ephemeral_root / "official_image.txt").write_text(
        f"docker.io/jefzda/sweap-images:{row['dockerhub_tag']}\n", encoding="utf-8"
    )
    return 0


def _control(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    lock = load_lock(args.lock)
    validate_lock(lock)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", args.run_id):
        raise ConfigError("Unsafe diagnostic run ID")
    payload = {
        "schema_version": 1,
        "enabled": args.mode == "ON",
        "mode": args.mode,
        "run_id": args.run_id,
        "instance_id": INSTANCE_ID,
        "prompt_token_count": int(lock["selection"]["serialized_prompt_token_count"]),
        "prompt_sha256": lock["selection"]["serialized_prompt_sha256"],
        "decode_samples": [0, 1, 8, 32],
    }
    _write_json(safe_absolute_path(args.output, "output"), payload)
    return 0


def _trace_equivalence(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    report = validate_trace_equivalence(load_json(args.off_response), load_json(args.on_response))
    if args.off_duration_ns <= 0 or args.on_duration_ns <= 0:
        raise ConfigError("TRACE_DURATION_INVALID")
    report["duration_ns"] = {"off": args.off_duration_ns, "on": args.on_duration_ns}
    report["instrumentation_overhead_ns"] = args.on_duration_ns - args.off_duration_ns
    report["instrumentation_overhead_ratio"] = args.on_duration_ns / args.off_duration_ns
    _write_json(safe_absolute_path(args.output, "output"), report)
    return 0


def _finalize_captures(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    lock = load_lock(args.lock)
    validate_lock(lock)
    raw_root = safe_absolute_path(args.raw_root, "raw_root")
    output_root = safe_absolute_path(args.output_root, "output_root")
    blockers = sorted(raw_root.glob("BLOCKED-*.json"))
    if blockers:
        report = {
            "schema_version": 1,
            "status": "BLOCKED",
            "failure_class": "NATIVE_DSA_EXPOSURE_BLOCKED",
            "blockers": [json.loads(path.read_text(encoding="utf-8")) for path in blockers],
            "fallback_attempted": False,
        }
        _write_json(output_root / "capture_manifest.json", report)
        return 3
    trace = load_json(args.trace_report)
    if trace.get("status") != "passed":
        raise ConfigError("TRACE_EQUIVALENCE_NOT_PASSED")
    raw_paths = sorted(raw_root.glob("*.json"))
    records, compressed, algorithm = compress_capture_records(raw_paths, output_root / "records")
    coverage = validate_capture_coverage(records, lock, output_token_count=int(trace["output_token_count"]))
    if coverage["run_id"] != args.run_id:
        raise ConfigError("CAPTURE_RUN_ID_MANIFEST_MISMATCH")
    manifest = build_artifact_manifest(compressed, compression=algorithm, coverage=coverage, run_id=args.run_id)
    _write_json(output_root / "validation_report.json", coverage)
    _write_json(output_root / "capture_manifest.json", manifest)
    checksums = "".join(f"{item['sha256']}  records/{item['name']}\n" for item in manifest["files"])
    (output_root / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    for path in raw_paths:
        path.unlink()
    return 0


def _extract_action(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    completion = extract_completion(load_json(args.response))
    actions = re.findall(r"```bash\s*\n(.*?)\n```", completion["raw_output"], re.DOTALL)
    if len(actions) != 1 or not actions[0].strip():
        raise ConfigError(f"AGENT_FORMAT_INVALID: expected one bash block, observed {len(actions)}")
    action_path = safe_absolute_path(args.action_output, "action_output")
    action_path.parent.mkdir(parents=True, exist_ok=True)
    action_path.write_text(actions[0].strip() + "\n", encoding="utf-8")
    metadata = {key: value for key, value in completion.items() if key not in {"raw_output", "normalized_output", "output_token_ids"}}
    metadata["schema_version"] = 1
    metadata["status"] = "passed"
    metadata["action_sha256"] = _sha256(action_path)
    _write_json(safe_absolute_path(args.metadata_output, "metadata_output"), metadata)
    return 0


def _make_prediction(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    patch_path = safe_absolute_path(args.patch, "patch")
    prediction = {
        INSTANCE_ID: {
            "model_name_or_path": MODEL_ID,
            "instance_id": INSTANCE_ID,
            "model_patch": patch_path.read_text(encoding="utf-8", errors="replace"),
        }
    }
    _write_json(safe_absolute_path(args.output, "output"), prediction)
    pred_root = safe_absolute_path(args.official_pred_root, "official_pred_root")
    pred_file = pred_root / INSTANCE_ID / f"{INSTANCE_ID}.pred"
    _write_json(pred_file, prediction[INSTANCE_ID])
    return 0


def _finalize_diagnostic(args: argparse.Namespace) -> int:
    allocation = require_slurm_allocation()
    if not re.fullmatch(r"[0-9a-f]{40}", args.project_commit):
        raise ConfigError("PROJECT_COMMIT_INVALID")
    revision = Path(args.model_revision).read_text(encoding="utf-8").strip()
    if revision != MODEL_REVISION:
        raise ConfigError("MODEL_REVISION_CHANGED_SINCE_PIN")
    inventory = load_json(args.inventory)
    if inventory.get("gpu_count") != 4 or len(inventory.get("gpu_uuids", [])) != 4:
        raise ConfigError("FINAL_ALLOCATION_INVENTORY_INVALID")
    validate_model_config(load_json(args.model_config))
    server_info = load_json(args.server_info)
    server = validate_server_info(server_info)
    trace_isolation = validate_diagnostic_server_isolation(server_info)
    runtime_log = validate_runtime_log(Path(args.server_log).read_text(encoding="utf-8", errors="replace"))
    trace = load_json(args.trace_report)
    captures = load_json(args.capture_manifest)
    if trace.get("status") != "passed" or captures.get("status") != "passed":
        raise ConfigError("DIAGNOSTIC_TRACE_OR_CAPTURE_NOT_PASSED")
    eval_results = load_json(args.eval_results)
    if set(eval_results) != {INSTANCE_ID} or not isinstance(eval_results[INSTANCE_ID], bool):
        raise ConfigError("OFFICIAL_SINGLE_ROW_EVALUATION_RESULT_INVALID")
    uuids = inventory["gpu_uuids"]
    load_hbm = summarize_hbm(parse_hbm_csv(Path(args.load_hbm).read_text(encoding="utf-8")), uuids)
    off_hbm = summarize_hbm(parse_hbm_csv(Path(args.off_hbm).read_text(encoding="utf-8")), uuids)
    on_hbm = summarize_hbm(parse_hbm_csv(Path(args.on_hbm).read_text(encoding="utf-8")), uuids)
    payload = {
        "schema_version": 1,
        "status": "PASS",
        "diagnostic": "glm52_nvfp4_native_dsa_single_swebench_pro_instance",
        "quality_score_eligible": False,
        "acceptance_threshold_evaluated": False,
        "instance_id": INSTANCE_ID,
        "official_evaluation_resolved": eval_results[INSTANCE_ID],
        "model": {"id": MODEL_ID, "revision": revision, "unmodified": True},
        "project_commit": args.project_commit,
        "sglang_commit": SGLANG_COMMIT,
        "sglang_image": SGLANG_IMAGE,
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "instance_count": 1},
        "official_harness_commit": HARNESS_COMMIT,
        "slurm": allocation,
        "runtime_contract": server,
        "trace_isolation_runtime": trace_isolation,
        "runtime_log_evidence": runtime_log,
        "trace_equivalence": trace,
        "capture_manifest": captures,
        "hbm": {"load": load_hbm, "trace_off": off_hbm, "trace_on": on_hbm},
        "minimum_positive_headroom_mib": min(
            load_hbm["minimum_headroom_mib"], off_hbm["minimum_headroom_mib"], on_hbm["minimum_headroom_mib"]
        ),
        "offload": False,
        "fallback_attempted": False,
    }
    _write_json(safe_absolute_path(args.output, "output"), payload)
    return 0


def _validate_harness(root: Path, lock: dict[str, Any]) -> None:
    if _git_head(root) != HARNESS_COMMIT or _git_head(root / "mini-swe-agent") != MINI_SWE_COMMIT:
        raise ConfigError("PINNED_HARNESS_OR_MINI_SWE_CHECKOUT_MISMATCH")
    files = {
        root / lock["swebench_pro"]["official_scorer"]: lock["swebench_pro"]["official_scorer_sha256"],
        root / lock["swebench_pro"]["official_gather_helper"]: lock["swebench_pro"]["official_gather_helper_sha256"],
    }
    if any(_sha256(path) != digest for path, digest in files.items()):
        raise ConfigError("OFFICIAL_HARNESS_FILE_DIGEST_MISMATCH")


def _git_head(root: Path) -> str:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    partial.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
