from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .glm52_sglang_gate import (
    MODEL_ID,
    MODEL_REF,
    SITE_PROFILE,
    SOURCE_LOCK,
    classify_startup_failure,
    load_json,
    parse_hbm_csv,
    parse_inventory_csv,
    summarize_hbm,
    validate_capability_report,
    validate_checkpoint_layout,
    validate_checkpoint_marker,
    validate_inventory_rows,
    validate_model_config,
    validate_runtime_log,
    validate_sentinel_response,
    validate_server_info,
    validate_source_lock,
    write_json,
)
from .glm52_sglang_gate_slurm import load_gate_site, render_compact_gate_submission


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "render-wrap":
            rendered = render_compact_gate_submission(
                site=load_gate_site(args.site), project_url=args.project_url, project_commit=args.project_commit
            )
            print(rendered)
            return 0
        if args.command == "validate-lock":
            print(json.dumps(validate_source_lock(load_json(args.lock)), indent=2, sort_keys=True))
            return 0
        if args.command == "validate-inventory":
            return _validate_inventory(args)
        if args.command == "phase1":
            return _phase1(args)
        if args.command == "download-model":
            return _download_model(args)
        if args.command == "validate-checkpoint":
            _require_exact_allocation()
            print(json.dumps(validate_checkpoint_marker(args.model_root, args.revision), indent=2, sort_keys=True))
            return 0
        if args.command == "validate-runtime":
            return _validate_runtime(args)
        if args.command == "classify-startup":
            print(classify_startup_failure(Path(args.server_log).read_text(encoding="utf-8", errors="replace")))
            return 0
    except (ConfigError, OSError, ValueError, json.JSONDecodeError, ImportError) as exc:
        print(json.dumps({"schema_version": 1, "status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    parser.error("a command is required")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="putpocket-glm52-sglang-gate")
    commands = parser.add_subparsers(dest="command")
    render = commands.add_parser("render-wrap", help="Render the dedicated compact four-H200 feasibility submission")
    render.add_argument("--site", default=str(SITE_PROFILE))
    render.add_argument("--project-url", required=True)
    render.add_argument("--project-commit", required=True)
    lock = commands.add_parser("validate-lock")
    lock.add_argument("--lock", default=str(SOURCE_LOCK))
    inventory = commands.add_parser("validate-inventory")
    inventory.add_argument("--csv", required=True)
    inventory.add_argument("--listing", required=True)
    inventory.add_argument("--output", required=True)
    phase1 = commands.add_parser("phase1")
    phase1.add_argument("--lock", default=str(SOURCE_LOCK))
    phase1.add_argument("--artifact-root", required=True)
    phase1.add_argument("--metadata-root", required=True)
    download = commands.add_parser("download-model")
    download.add_argument("--revision-file", required=True)
    download.add_argument("--model-root", required=True)
    checkpoint = commands.add_parser("validate-checkpoint")
    checkpoint.add_argument("--model-root", required=True)
    checkpoint.add_argument("--revision", required=True)
    runtime = commands.add_parser("validate-runtime")
    runtime.add_argument("--inventory", required=True)
    runtime.add_argument("--model-config", required=True)
    runtime.add_argument("--server-info", required=True)
    runtime.add_argument("--server-log", required=True)
    runtime.add_argument("--response", required=True)
    runtime.add_argument("--hbm-samples", required=True)
    runtime.add_argument("--model-revision", required=True)
    runtime.add_argument("--project-commit", required=True)
    runtime.add_argument("--source-lock-report", required=True)
    runtime.add_argument("--capability-report", required=True)
    runtime.add_argument("--exact-command", required=True)
    runtime.add_argument("--output", required=True)
    classify = commands.add_parser("classify-startup")
    classify.add_argument("--server-log", required=True)
    return parser


def _require_exact_allocation() -> None:
    if not str(os.environ.get("SLURM_JOB_ID", "")).isdigit():
        raise ConfigError("SLURM_ALLOCATION_REQUIRED")
    if os.environ.get("SLURM_JOB_NUM_NODES") != "1" or not os.environ.get("SLURM_JOB_NODELIST"):
        raise ConfigError("SLURM_NODE_COUNT_MISMATCH")
    if os.environ.get("SLURM_GPUS_ON_NODE") != "4":
        raise ConfigError("SLURM_GPU_COUNT_MISMATCH")


def _validate_inventory(args: argparse.Namespace) -> int:
    _require_exact_allocation()
    summary = validate_inventory_rows(
        parse_inventory_csv(Path(args.csv).read_text(encoding="utf-8")),
        mig_listing=Path(args.listing).read_text(encoding="utf-8"),
    )
    write_json(args.output, summary.as_dict())
    print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
    return 0


def _phase1(args: argparse.Namespace) -> int:
    _require_exact_allocation()
    artifact_root = Path(args.artifact_root)
    metadata_root = Path(args.metadata_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    lock = load_json(args.lock)
    lock_report = validate_source_lock(lock)
    write_json(artifact_root / "source_lock_validation.json", lock_report)

    imports: dict[str, bool] = {}
    modules: dict[str, Any] = {}
    for name in ("torch", "transformers", "sglang", "modelopt", "flashinfer", "flash_mla", "sgl_kernel"):
        try:
            modules[name] = importlib.import_module(name)
            imports[name] = True
        except ImportError:
            imports[name] = False
    if not all(imports.values()):
        report = {"transformers_version": _package_version("transformers"), "imports": imports, "symbols": {}, "server_defaults": {}, "server_controls": {}}
        validate_capability_report(report)

    modelopt = importlib.import_module("sglang.srt.layers.quantization.modelopt_quant")
    server_args_module = importlib.import_module("sglang.srt.server_args")
    server_args = getattr(server_args_module, "ServerArgs")
    sglang_root = Path(modules["sglang"].__file__).resolve().parent
    source_text = _read_python_sources(sglang_root)
    symbols = {
        "ModelOptFp4Config": hasattr(modelopt, "ModelOptFp4Config"),
        "ModelOptFp4LinearMethod": hasattr(modelopt, "ModelOptFp4LinearMethod"),
        "ModelOptNvFp4FusedMoEMethod": hasattr(modelopt, "ModelOptNvFp4FusedMoEMethod"),
        "prepare_nvfp4_layer_for_marlin": "prepare_nvfp4_layer_for_marlin" in source_text,
        "prepare_moe_nvfp4_layer_for_marlin": "prepare_moe_nvfp4_layer_for_marlin" in source_text,
        "marlin_w4a16": "w4a16" in source_text.lower() and "marlin" in source_text.lower(),
        "hopper_marlin_selection": "get_fp4_gemm_runner_backend" in source_text and "is_marlin" in source_text,
        "glm_moe_dsa_runtime": "GlmMoeDsaForCausalLM" in source_text and "glm_moe_dsa" in source_text,
        "flashmla_sparse": "flashmla_sparse" in source_text,
        "fa3": "dsa_decode_backend" in source_text and "fa3" in source_text,
        "sgl-kernel": "sgl-kernel" in source_text and "fast_topk_v2" in source_text,
    }
    defaults = _dataclass_defaults(server_args)
    controls = {
        "quantization": ["modelopt_fp4"] if "modelopt_fp4" in source_text and "--quantization" in source_text else [],
        "fp4_gemm_backend": ["marlin"] if "--fp4-gemm-backend" in source_text and "marlin" in source_text else [],
        "moe_runner_backend": ["marlin"] if "--moe-runner-backend" in source_text and "marlin" in source_text else [],
        "dsa_prefill_backend": ["flashmla_sparse"] if "--dsa-prefill-backend" in source_text and "flashmla_sparse" in source_text else [],
        "dsa_decode_backend": ["fa3"] if "--dsa-decode-backend" in source_text and "fa3" in source_text else [],
        "dsa_topk_backend": ["sgl-kernel"] if "--dsa-topk-backend" in source_text and "sgl-kernel" in source_text else [],
    }
    report = {
        "schema_version": 1,
        "transformers_version": _package_version("transformers"),
        "sglang_version": _package_version("sglang"),
        "torch_version": _package_version("torch"),
        "modelopt_version": _package_version("nvidia-modelopt"),
        "flashinfer_version": _package_version("flashinfer-python"),
        "torch_cuda_version": getattr(modules["torch"].version, "cuda", None),
        "torch_nccl_version": _nccl_version(modules["torch"]),
        "imports": imports,
        "symbols": symbols,
        "server_defaults": {key: defaults.get(key) for key in ("cpu_offload_gb", "disaggregation_mode", "speculative_algorithm", "weight_cache_mode")},
        "server_controls": controls,
    }
    validated_capabilities = validate_capability_report(report)
    write_json(artifact_root / "capability_probe.raw.json", report)
    write_json(artifact_root / "capability_probe.json", validated_capabilities)

    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().model_info(repo_id=MODEL_ID, revision=MODEL_REF).sha
    if not isinstance(revision, str) or len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ConfigError("MODEL_REVISION_UNRESOLVED: Hugging Face did not return a full lowercase commit")
    config_path = Path(
        hf_hub_download(repo_id=MODEL_ID, filename="config.json", revision=revision, local_dir=metadata_root)
    )
    forbidden = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.gguf")
    if any(path.is_file() for pattern in forbidden for path in metadata_root.rglob(pattern)):
        raise ConfigError("WEIGHTLESS_PROBE_VIOLATION: checkpoint data appeared during config probe")
    model_config = load_json(config_path)
    config_report = validate_model_config(model_config)
    write_json(artifact_root / "model_config_validation.json", config_report)
    (artifact_root / "model_revision.txt").write_text(revision + "\n", encoding="utf-8")
    (artifact_root / "model_config.json").write_text(json.dumps(model_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_json(
        artifact_root / "phase1_manifest.json",
        {
            "schema_version": 1,
            "status": "passed",
            "weightless": True,
            "model_id": MODEL_ID,
            "requested_revision": MODEL_REF,
            "resolved_revision": revision,
            "source_lock": lock_report,
            "capability_probe": validated_capabilities,
            "model_config": config_report,
        },
    )
    print(revision)
    return 0


def _download_model(args: argparse.Namespace) -> int:
    _require_exact_allocation()
    revision = Path(args.revision_file).read_text(encoding="utf-8").strip()
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ConfigError("MODEL_REVISION_UNRESOLVED")
    model_root = Path(args.model_root)
    model_root.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    resolved = snapshot_download(repo_id=MODEL_ID, revision=revision, local_dir=model_root)
    if Path(resolved).resolve() != model_root.resolve():
        raise ConfigError("CHECKPOINT_PATH_MISMATCH")
    layout = validate_checkpoint_layout(model_root)
    write_json(model_root / ".putpocket_checkpoint_ready.json", {"schema_version": 1, "model_id": MODEL_ID, "revision": revision, "status": "ready", "layout": layout})
    return 0


def _validate_runtime(args: argparse.Namespace) -> int:
    _require_exact_allocation()
    inventory = load_json(args.inventory)
    if inventory.get("status") != "passed" or inventory.get("gpu_count") != 4:
        raise ConfigError("ALLOCATION_INVENTORY_NOT_PASSED")
    uuids = inventory.get("gpu_uuids")
    if not isinstance(uuids, list):
        raise ConfigError("ALLOCATION_INVENTORY_INVALID")
    revision = Path(args.model_revision).read_text(encoding="utf-8").strip()
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision):
        raise ConfigError("MODEL_REVISION_UNRESOLVED")
    project_commit = args.project_commit
    if len(project_commit) != 40 or any(ch not in "0123456789abcdef" for ch in project_commit):
        raise ConfigError("PROJECT_COMMIT_INVALID")
    source_lock = load_json(args.source_lock_report)
    if source_lock.get("status") != "passed":
        raise ConfigError("SOURCE_LOCK_NOT_VALIDATED")
    capabilities = load_json(args.capability_report)
    if capabilities.get("status") != "passed":
        raise ConfigError("CAPABILITY_PROBE_NOT_PASSED")
    exact_command = Path(args.exact_command).read_text(encoding="utf-8").strip()
    if not exact_command or any(term in exact_command.lower() for term in ("token=", "password=", "secret=", "api_key=")):
        raise ConfigError("EXACT_COMMAND_INVALID_OR_SECRET_BEARING")
    config_report = validate_model_config(load_json(args.model_config))
    info_report = validate_server_info(load_json(args.server_info))
    log_text = Path(args.server_log).read_text(encoding="utf-8", errors="replace")
    log_report = validate_runtime_log(log_text)
    response = load_json(args.response)
    sentinel = validate_sentinel_response(response)
    hbm = summarize_hbm(parse_hbm_csv(Path(args.hbm_samples).read_text(encoding="utf-8")), uuids)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "gate": "glm52_sglang_minimal_feasibility",
        "model_id": MODEL_ID,
        "model_revision": revision,
        "project_commit": project_commit,
        "sglang_source_commit": source_lock["sglang_source_commit"],
        "runtime_image": source_lock["runtime_image"],
        "runtime_image_human_tag": source_lock["runtime_image_human_tag"],
        "runtime_versions": {
            "transformers": capabilities.get("transformers_version"),
            "torch_cuda": capabilities.get("torch_cuda_version"),
            "torch_nccl": capabilities.get("torch_nccl_version"),
        },
        "slurm": {
            "job_id": os.environ["SLURM_JOB_ID"],
            "nodelist": os.environ["SLURM_JOB_NODELIST"],
            "nodes": 1,
            "gpus_on_node": 4,
        },
        "gpu_uuids": uuids,
        "node_count": 1,
        "gpu_count": 4,
        "tensor_parallel": 4,
        "all_resident": True,
        "offload": False,
        "runtime": info_report,
        "runtime_log": log_report,
        "model_config": config_report,
        "sentinel": sentinel,
        "hbm": hbm,
        "exact_command": exact_command,
        "artifact_root": str(Path(args.output).parent),
        "claim": "FEASIBILITY_GATE_PASS",
        "next_phase_authorized": "separate_one_instance_swebench_pro_smoke_may_be_scheduled",
    }
    write_json(args.output, manifest)
    output_root = Path(args.output).parent
    (output_root / "sentinel.raw.txt").write_text(sentinel["raw_output"], encoding="utf-8")
    (output_root / "sentinel.raw.sha256").write_text(sentinel["raw_sha256"] + "\n", encoding="utf-8")
    (output_root / "sentinel.normalized.txt").write_text(sentinel["normalized_output"] + "\n", encoding="utf-8")
    (output_root / "sentinel.sha256").write_text(sentinel["normalized_sha256"] + "\n", encoding="utf-8")
    (output_root / "sentinel.normalized.sha256").write_text(sentinel["normalized_sha256"] + "\n", encoding="utf-8")
    write_json(output_root / "hbm_summary.json", hbm)
    write_json(output_root / "runtime_contract.json", {"model_config": config_report, "server_info": info_report, "runtime_log": log_report})
    (output_root / "runtime_contract.log").write_text(
        " ".join(
            (
                "architecture=GlmMoeDsaForCausalLM",
                "model_type=glm_moe_dsa",
                "layers=78",
                "indexer_layout=21-full/57-shared",
                "index_topk=2048",
                "tensor_parallel=4",
                "quantization=modelopt_fp4",
                "fp4_gemm_backend=marlin-W4A16",
                "dsa_prefill_backend=flashmla_sparse",
                "dsa_decode_backend=fa3",
                "dsa_topk_backend=sgl-kernel",
                "offload=false",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _read_python_sources(root: Path) -> str:
    chunks: list[str] = []
    for path in root.rglob("*.py"):
        try:
            chunks.append(path.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(chunks)


def _dataclass_defaults(cls: type[Any]) -> dict[str, Any]:
    if not dataclasses.is_dataclass(cls):
        raise ConfigError("BACKEND_CAPABILITY_MISSING: ServerArgs is not a dataclass")
    output: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING:
            output[field.name] = field.default
    return output


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "0"


def _nccl_version(torch_module: Any) -> Any:
    try:
        return torch_module.cuda.nccl.version()
    except (AttributeError, RuntimeError):
        return "unavailable"


if __name__ == "__main__":
    raise SystemExit(main())
