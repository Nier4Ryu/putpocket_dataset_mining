from __future__ import annotations

import argparse
import base64
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
from .glm52_vllm_diagnostic import (
    INSTANCE_ID,
    LOCK_PATH,
    MODEL_ID,
    MODEL_REVISION,
    compress_jsonl,
    build_runtime_jit_manifest,
    file_sha256,
    load_lock,
    validate_build_manifest,
    validate_capture_records,
    validate_inventory_csv,
    validate_lock,
    validate_model_config,
    validate_patched_tree,
    validate_source_tree,
    validate_source_identities,
    validate_source_provenance,
    validate_trace_equivalence,
)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return {
            "validate-lock": _validate_lock,
            "render-wrap": _render,
            "validate-source": _validate_source,
            "validate-patched": _validate_patched,
            "validate-build-bundle": _validate_bundle,
            "validate-inventory": _validate_inventory,
            "phase1": _phase1,
            "finalize-runtime-jit": _finalize_runtime_jit,
            "prepare": _prepare,
            "control": _control,
            "trace-equivalence": _trace,
            "finalize-captures": _captures,
            "extract-action": _extract_action,
            "make-prediction": _make_prediction,
        }[args.command](args)
    except (ConfigError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"schema_version": 1, "status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="putpocket-glm52-vllm-diagnostic")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate-lock")
    check.add_argument("--lock", default=str(LOCK_PATH))
    render = commands.add_parser("render-wrap")
    render.add_argument("--site", required=True)
    render.add_argument("--project-url", required=True)
    render.add_argument("--runtime-source-commit", required=True)
    render.add_argument("--build-source-commit", required=True)
    render.add_argument("--allow-runtime-source-split", action="store_true")
    render.add_argument("--login-safe-base64", action="store_true")
    render.add_argument("--cpu-partition", required=True)
    render.add_argument("--cpu-account", required=True)
    render.add_argument("--cpu-qos", required=True)
    render.add_argument("--cpu-cpus-per-task", required=True, type=int)
    render.add_argument("--cpu-memory", required=True)
    render.add_argument("--cpu-wall-time", required=True)
    render.add_argument("--cpu-local-scratch-root", required=True)
    render.add_argument("--container-executable", required=True)
    source = commands.add_parser("validate-source")
    source.add_argument("--lock", default=str(LOCK_PATH)); source.add_argument("--project-root", required=True); source.add_argument("--source-root", required=True)
    patched = commands.add_parser("validate-patched")
    patched.add_argument("--lock", default=str(LOCK_PATH)); patched.add_argument("--source-root", required=True)
    bundle = commands.add_parser("validate-build-bundle")
    bundle.add_argument("--lock", default=str(LOCK_PATH)); bundle.add_argument("--bundle-root", required=True)
    bundle.add_argument("--expected-build-source-commit", required=True)
    bundle.add_argument("--runtime-source-commit", required=True)
    bundle.add_argument("--observed-runtime-source-commit", required=True)
    bundle.add_argument("--wrapper-source-commit", required=True)
    bundle.add_argument("--allow-runtime-source-split", action="store_true")
    inventory = commands.add_parser("validate-inventory")
    inventory.add_argument("--csv", required=True); inventory.add_argument("--listing", required=True); inventory.add_argument("--output", required=True)
    phase1 = commands.add_parser("phase1")
    phase1.add_argument("--lock", default=str(LOCK_PATH)); phase1.add_argument("--metadata-root", required=True); phase1.add_argument("--artifact-root", required=True)
    jit = commands.add_parser("finalize-runtime-jit")
    jit.add_argument("--lock", default=str(LOCK_PATH)); jit.add_argument("--bundle-root", required=True); jit.add_argument("--cache-root", required=True); jit.add_argument("--audit-log", required=True); jit.add_argument("--started-utc", required=True); jit.add_argument("--completed-utc", required=True); jit.add_argument("--build-source-commit", required=True); jit.add_argument("--runtime-source-commit", required=True); jit.add_argument("--observed-runtime-source-commit", required=True); jit.add_argument("--wrapper-source-commit", required=True); jit.add_argument("--allow-runtime-source-split", action="store_true"); jit.add_argument("--runtime-image-id", required=True); jit.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--lock", default=str(LOCK_PATH)); prepare.add_argument("--model-root", required=True); prepare.add_argument("--harness-root", required=True); prepare.add_argument("--ephemeral-root", required=True); prepare.add_argument("--artifact-root", required=True)
    control = commands.add_parser("control")
    control.add_argument("--lock", default=str(LOCK_PATH)); control.add_argument("--mode", choices=["OFF", "ON"], required=True); control.add_argument("--run-id", required=True); control.add_argument("--output", required=True); control.add_argument("--build-source-commit", required=True); control.add_argument("--runtime-source-commit", required=True); control.add_argument("--observed-runtime-source-commit", required=True); control.add_argument("--wrapper-source-commit", required=True); control.add_argument("--allow-runtime-source-split", action="store_true"); control.add_argument("--runtime-image-id", required=True)
    trace = commands.add_parser("trace-equivalence")
    trace.add_argument("--off-response", required=True); trace.add_argument("--on-response", required=True); trace.add_argument("--off-duration-ns", type=int, required=True); trace.add_argument("--on-duration-ns", type=int, required=True); trace.add_argument("--output", required=True)
    captures = commands.add_parser("finalize-captures")
    captures.add_argument("--lock", default=str(LOCK_PATH)); captures.add_argument("--raw-root", required=True); captures.add_argument("--output-root", required=True); captures.add_argument("--trace-report", required=True)
    action = commands.add_parser("extract-action")
    action.add_argument("--response", required=True); action.add_argument("--action-output", required=True); action.add_argument("--metadata-output", required=True)
    pred = commands.add_parser("make-prediction")
    pred.add_argument("--patch", required=True); pred.add_argument("--output", required=True); pred.add_argument("--official-pred-root", required=True)
    return parser


def _validate_lock(args: argparse.Namespace) -> int:
    print(json.dumps(validate_lock(load_lock(args.lock)), indent=2, sort_keys=True)); return 0


def _render(args: argparse.Namespace) -> int:
    from .glm52_vllm_diagnostic_slurm import load_site, render_two_stage_submission
    site = load_site(
        args.site, cpu_partition=args.cpu_partition, cpu_account=args.cpu_account,
        cpu_qos=args.cpu_qos, cpu_cpus_per_task=args.cpu_cpus_per_task,
        cpu_memory=args.cpu_memory, cpu_wall_time=args.cpu_wall_time,
        cpu_local_scratch_root=args.cpu_local_scratch_root,
        container_executable=args.container_executable,
    )
    command = render_two_stage_submission(
        site=site,
        project_url=args.project_url,
        runtime_source_commit=args.runtime_source_commit,
        build_source_commit=args.build_source_commit,
        allow_runtime_source_split=args.allow_runtime_source_split,
        lock_path=LOCK_PATH,
    )
    if args.login_safe_base64:
        encoded = base64.b64encode((command + "\n").encode("utf-8")).decode("ascii")
        print(f"printf '%s' '{encoded}' | base64 --decode | /bin/bash")
    else:
        print(command)
    return 0


def _validate_source(args: argparse.Namespace) -> int:
    require_slurm_allocation(); print(json.dumps(validate_source_tree(args.project_root, args.source_root, load_lock(args.lock)), sort_keys=True)); return 0


def _validate_patched(args: argparse.Namespace) -> int:
    require_slurm_allocation(); print(json.dumps(validate_patched_tree(args.source_root, load_lock(args.lock)), sort_keys=True)); return 0


def _validate_bundle(args: argparse.Namespace) -> int:
    require_slurm_allocation(); root = safe_absolute_path(args.bundle_root, "bundle_root")
    manifest = _load_json(root / "build_manifest.json")
    report = validate_source_provenance(
        manifest,
        root,
        load_lock(args.lock),
        expected_build_source_commit=args.expected_build_source_commit,
        runtime_source_commit=args.runtime_source_commit,
        observed_runtime_source_commit=args.observed_runtime_source_commit,
        wrapper_source_commit=args.wrapper_source_commit,
        allow_runtime_source_split=args.allow_runtime_source_split,
    )
    print(json.dumps(report, sort_keys=True)); return 0


def _validate_inventory(args: argparse.Namespace) -> int:
    require_slurm_allocation(); report = validate_inventory_csv(Path(args.csv).read_text(encoding="utf-8"), Path(args.listing).read_text(encoding="utf-8")); _write_json(safe_absolute_path(args.output, "output"), report); return 0


def _phase1(args: argparse.Namespace) -> int:
    current_step = "initialization"

    def start(step: str) -> None:
        nonlocal current_step
        current_step = step
        print(f"PHASE1_STEP_START={step}", file=sys.stderr, flush=True)

    def passed() -> None:
        print(f"PHASE1_STEP_PASS={current_step}", file=sys.stderr, flush=True)

    try:
        start("slurm_allocation")
        require_slurm_allocation()
        passed()
        start("locked_contract")
        lock = load_lock(args.lock); validate_lock(lock)
        metadata = safe_absolute_path(args.metadata_root, "metadata_root")
        artifacts = safe_absolute_path(args.artifact_root, "artifact_root")
        metadata.mkdir(parents=True, exist_ok=True); artifacts.mkdir(parents=True, exist_ok=True)
        passed()
        start("runtime_imports")
        import transformers
        from huggingface_hub import HfApi, snapshot_download
        from packaging.version import Version
        passed()
        start("transformers_version")
        if Version(transformers.__version__) < Version("5.3"):
            raise ConfigError(f"TRANSFORMERS_5_3_REQUIRED:{transformers.__version__}")
        passed()
        start("model_revision_resolution")
        info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION)
        if info.sha != MODEL_REVISION:
            raise ConfigError("MODEL_REVISION_RESOLUTION_MISMATCH")
        passed()
        start("weightless_metadata_download")
        snapshot_download(
            repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=metadata,
            allow_patterns=["config.json", "hf_quant_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
        )
        passed()
        start("model_config_validation")
        config = _load_json(metadata / "config.json")
        config_report = validate_model_config(config, lock)
        passed()
        start("vllm_capability_validation")
        capabilities = _vllm_capability_probe()
        passed()
        start("artifact_publication")
        _write_json(artifacts / "model_config.json", config)
        _write_json(artifacts / "model_config_validation.json", config_report)
        _write_json(artifacts / "capability_probe.json", capabilities)
        (artifacts / "model_revision.txt").write_text(MODEL_REVISION + "\n", encoding="utf-8")
        passed()
        return 0
    except Exception as exc:
        detail = json.dumps(str(exc), ensure_ascii=True)
        print(
            f"PHASE1_STEP_FAILED={current_step} exception_type={type(exc).__name__} detail={detail}",
            file=sys.stderr,
            flush=True,
        )
        raise


def _finalize_runtime_jit(args: argparse.Namespace) -> int:
    require_slurm_allocation(); bundle = safe_absolute_path(args.bundle_root, "bundle_root")
    build = _load_json(bundle / "build_manifest.json"); build["_bundle_root"] = str(bundle)
    report = build_runtime_jit_manifest(args.cache_root, args.audit_log, started_utc=args.started_utc, completed_utc=args.completed_utc, build_source_commit=args.build_source_commit, runtime_source_commit=args.runtime_source_commit, observed_runtime_source_commit=args.observed_runtime_source_commit, wrapper_source_commit=args.wrapper_source_commit, allow_runtime_source_split=args.allow_runtime_source_split, runtime_image_id=args.runtime_image_id, build_manifest=build, lock=load_lock(args.lock))
    _write_json(safe_absolute_path(args.output, "output"), report)
    checksums = Path(args.output).with_name("runtime_jit_SHA256SUMS")
    checksums.write_text("".join(f"{item['sha256']}  cache/{item['path']}\n" for item in report["files"]) + f"{report['audit_log_sha256']}  compiler_audit.jsonl\n", encoding="utf-8")
    return 0


def _vllm_capability_probe() -> dict[str, Any]:
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config, ModelOptNvFp4W4A16LinearMethod
    from vllm.model_executor.layers.sparse_attn_indexer import sparse_attn_indexer
    from vllm.model_executor.models.deepseek_v2 import GlmMoeDsaForCausalLM
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
    import vllm.model_executor.layers.vllm_dsa_diagnostic_dump as hook
    values = [ModelOptNvFp4Config, ModelOptNvFp4W4A16LinearMethod, sparse_attn_indexer, GlmMoeDsaForCausalLM, FlashMLASparseBackend, hook.maybe_capture_native_dsa]
    if not all(value is not None for value in values):
        raise ConfigError("VLLM_REQUIRED_SYMBOL_MISSING")
    return {
        "schema_version": 1, "status": "passed", "vllm_commit": load_lock()["vllm"]["commit"],
        "symbols": [f"{value.__module__}.{value.__name__}" for value in values],
        "native_path": "fp8_fp4_mqa_logits/top_k_per_row_prefill and fp8_fp4_paged_mqa_logits/cooperative_topk_sm90",
    }


def _prepare(args: argparse.Namespace) -> int:
    require_slurm_allocation()
    lock = load_lock(args.lock); validate_lock(lock)
    model_root = safe_absolute_path(args.model_root, "model_root")
    harness = safe_absolute_path(args.harness_root, "harness_root")
    ephemeral = safe_absolute_path(args.ephemeral_root, "ephemeral_root")
    artifacts = safe_absolute_path(args.artifact_root, "artifact_root")
    _validate_harness(harness, lock)
    from datasets import load_dataset
    from jinja2 import StrictUndefined, Template
    from transformers import AutoTokenizer
    import yaml
    rows = load_dataset(lock["selection"]["dataset"], revision=lock["selection"]["dataset_revision"], split="test")
    matches = [dict(row) for row in rows if row.get("instance_id") == INSTANCE_ID]
    if len(matches) != 1: raise ConfigError(f"SELECTED_INSTANCE_CARDINALITY_MISMATCH:{len(matches)}")
    row = matches[0]; selection = lock["selection"]
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != selection["row_sha256"]: raise ConfigError("SELECTED_INSTANCE_ROW_DIGEST_MISMATCH")
    if row.get("repo") != selection["repo"] or row.get("dockerhub_tag") != selection["dockerhub_tag"]: raise ConfigError("SELECTED_INSTANCE_CONTENT_MISMATCH")
    scaffold = harness / lock["official_evaluation"]["mini_swe_scaffold"]
    if file_sha256(scaffold) != lock["official_evaluation"]["mini_swe_scaffold_sha256"]: raise ConfigError("MINI_SWE_SCAFFOLD_DIGEST_MISMATCH")
    config = yaml.safe_load(scaffold.read_text(encoding="utf-8")); agent = config["agent"]
    messages = [
        {"role": "system", "content": Template(agent["system_template"], undefined=StrictUndefined).render(task=row["problem_statement"])},
        {"role": "user", "content": Template(agent["instance_template"], undefined=StrictUndefined).render(task=row["problem_statement"])},
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    serialized = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    if not isinstance(ids, list) or len(ids) != 2071 or hashlib.sha256(serialized.encode()).hexdigest() != selection["serialized_prompt_sha256"] or len(serialized.encode()) != 8629:
        raise ConfigError("SERIALIZED_PROMPT_PIN_MISMATCH")
    ephemeral.mkdir(parents=True, exist_ok=True); artifacts.mkdir(parents=True, exist_ok=True)
    _write_json(ephemeral / "completion_request.json", {"model": MODEL_ID, "prompt": ids, "temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "n": 1, "seed": 0, "stream": False, "return_token_ids": True})
    (ephemeral / "official_raw_sample.jsonl").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    (ephemeral / "official_image.txt").write_text(f"docker.io/jefzda/sweap-images:{row['dockerhub_tag']}\n", encoding="utf-8")
    _write_json(artifacts / "prompt_metadata.json", {"schema_version": 1, "instance_id": INSTANCE_ID, "row_sha256": selection["row_sha256"], "prompt_token_count": len(ids), "prompt_utf8_bytes": len(serialized.encode()), "prompt_sha256": selection["serialized_prompt_sha256"], "raw_prompt_persisted": False})
    return 0


def _control(args: argparse.Namespace) -> int:
    require_slurm_allocation(); lock = load_lock(args.lock); validate_lock(lock)
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", args.run_id): raise ConfigError("TRACE_CONTROL_ID_INVALID")
    source_provenance = validate_source_identities(
        pinned_build_source_commit=lock["build"]["project_source_commit"],
        expected_build_source_commit=args.build_source_commit,
        runtime_source_commit=args.runtime_source_commit,
        observed_runtime_source_commit=args.observed_runtime_source_commit,
        wrapper_source_commit=args.wrapper_source_commit,
        allow_runtime_source_split=args.allow_runtime_source_split,
    )
    payload = {
        "schema_version": 1, "mode": args.mode, "run_id": args.run_id, "instance_id": INSTANCE_ID,
        "prompt_token_count": 2071, "full_indexer_layers": lock["model_layout"]["full_indexer_layers"],
        "shared_layer_mapping": lock["model_layout"]["shared_layer_mapping"],
        "shared_layer_mapping_sha256": lock["model_layout"]["shared_layer_mapping_sha256"],
        "source_provenance": source_provenance,
        "revisions": {
            "model": MODEL_REVISION,
            "vllm": lock["vllm"]["commit"],
            "project": args.runtime_source_commit,
            "build_source": args.build_source_commit,
            "runtime_source": args.runtime_source_commit,
            "wrapper_source": args.wrapper_source_commit,
            "patch_sha256": lock["vllm"]["patch_sha256"],
            "build_image": lock["build"]["base_image"],
            "bundle_key": lock["build"]["bundle_key"],
            "runtime_image_id": args.runtime_image_id,
        },
    }
    _write_json(safe_absolute_path(args.output, "output"), payload); return 0


def _trace(args: argparse.Namespace) -> int:
    require_slurm_allocation(); report = validate_trace_equivalence(_load_json(args.off_response), _load_json(args.on_response), off_ns=args.off_duration_ns, on_ns=args.on_duration_ns); _write_json(safe_absolute_path(args.output, "output"), report); return 0


def _captures(args: argparse.Namespace) -> int:
    require_slurm_allocation(); raw = safe_absolute_path(args.raw_root, "raw_root"); output = safe_absolute_path(args.output_root, "output_root")
    blocked = sorted(raw.glob("BLOCKED*.json"))
    if blocked:
        _write_json(output / "capture_manifest.json", {"schema_version": 1, "status": "BLOCKED", "failure_class": "NATIVE_DSA_EXPOSURE_BLOCKED", "blockers": [_load_json(path) for path in blocked]}); return 3
    records: list[dict[str, Any]] = []; paths = sorted(raw.glob("captures.rank-*.jsonl"))
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip(): records.append(json.loads(line))
    trace = _load_json(args.trace_report)
    coverage = validate_capture_records(records, output_token_count=int(trace["output_token_count"]), lock=load_lock(args.lock))
    files = compress_jsonl(paths, output / "compressed")
    _write_json(output / "coverage.json", coverage)
    _write_json(output / "capture_manifest.json", {"schema_version": 1, "status": "passed", "compression": "python-gzip-level9-mtime0", "files": files, "coverage": coverage, "raw_prompt_included": False})
    (output / "SHA256SUMS").write_text("".join(f"{item['sha256']}  compressed/{item['path']}\n" for item in files), encoding="utf-8")
    return 0


def _extract_action(args: argparse.Namespace) -> int:
    require_slurm_allocation(); choice = _load_json(args.response)["choices"][0]; text = choice.get("text", "")
    actions = re.findall(r"```bash\s*\n(.*?)\n```", text, re.S)
    if len(actions) != 1 or not actions[0].strip(): raise ConfigError(f"AGENT_FORMAT_INVALID:{len(actions)}")
    target = safe_absolute_path(args.action_output, "action_output"); target.parent.mkdir(parents=True, exist_ok=True); target.write_text(actions[0].strip() + "\n", encoding="utf-8")
    _write_json(safe_absolute_path(args.metadata_output, "metadata_output"), {"schema_version": 1, "status": "passed", "action_sha256": file_sha256(target)}); return 0


def _make_prediction(args: argparse.Namespace) -> int:
    require_slurm_allocation(); patch = safe_absolute_path(args.patch, "patch")
    value = {"model_name_or_path": MODEL_ID, "instance_id": INSTANCE_ID, "model_patch": patch.read_text(encoding="utf-8", errors="replace")}
    _write_json(safe_absolute_path(args.output, "output"), {INSTANCE_ID: value})
    pred = safe_absolute_path(args.official_pred_root, "official_pred_root") / INSTANCE_ID / f"{INSTANCE_ID}.pred"; _write_json(pred, value); return 0


def _validate_harness(root: Path, lock: dict[str, Any]) -> None:
    official = lock["official_evaluation"]
    if _git_head(root) != official["harness_commit"] or _git_head(root / "mini-swe-agent") != official["mini_swe_agent_commit"] or _git_head(root / "SWE-agent") != official["swe_agent_commit"]:
        raise ConfigError("PINNED_HARNESS_SUBMODULE_MISMATCH")
    for path, digest in ((official["official_scorer"], official["official_scorer_sha256"]), (official["official_gather_helper"], official["official_gather_helper_sha256"])):
        if file_sha256(root / path) != digest: raise ConfigError(f"OFFICIAL_HARNESS_FILE_DIGEST_MISMATCH:{path}")


def _git_head(root: Path) -> str:
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ConfigError("JSON_OBJECT_REQUIRED")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); partial = path.with_name(path.name + ".partial"); partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(partial, path)


if __name__ == "__main__":
    raise SystemExit(main())
