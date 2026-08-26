from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jsonschema

from .constants import REPO_ROOT
from .errors import ConfigError
from .glm52_attention_indexer import (
    analyze_capture,
    analyze_query_sum_capture,
    canonical_json_bytes,
    file_sha256,
    validate_report_digest,
)
from .glm52_indexer_propagation import (
    load_matrix_episode_manifest,
    score_matrix_capture,
)


PACKAGE_LOCK = REPO_ROOT / "configs/runpod/glm52_attention_indexer_package.lock.json"
SCHEDULE = REPO_ROOT / "configs/runpod/glm52_attention_indexer_schedule.json"
DOCTOR_SCHEMA = REPO_ROOT / "configs/runpod/schemas/glm52_runpod_doctor.schema.json"
REPORT_SCHEMA = REPO_ROOT / "configs/runpod/schemas/glm52_attention_indexer_report.schema.json"
SCHEDULE_SCHEMA = REPO_ROOT / "configs/runpod/schemas/glm52_attention_indexer_schedule.schema.json"
MODEL_REVISION = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
VLLM_COMMIT = "4a3447d200e5aa428d68d1a00aa00f1a19a1a729"
INSTANCE_ID = "instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
SCENARIO_ID = "glm52-sys-policy-equal-replacement-v1"
INDEXER_LAYERS = [
    0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66,
    70, 74,
]
_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ConfigError(reason)


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON_OBJECT_REQUIRED:{path}")
    return value


def load_package_lock(path: str | Path = PACKAGE_LOCK) -> dict[str, Any]:
    value = load_json(path)
    validate_package_lock(value)
    return value


def validate_package_lock(lock: Mapping[str, Any]) -> None:
    _require(lock.get("schema_version") == 1, "RUNPOD_PACKAGE_SCHEMA_INVALID")
    _require(lock.get("package_id") == "glm52-main-indexer-score-runpod-v1", "RUNPOD_PACKAGE_ID_INVALID")
    vllm = lock.get("vllm")
    runtime = lock.get("runtime")
    capture = lock.get("capture")
    matrix_capture = lock.get("matrix_capture")
    query_sum_capture = lock.get("query_sum_capture")
    provenance = lock.get("benchmark_provenance")
    layout = lock.get("model_layout")
    completion_audit = lock.get("completion_audit")
    rank_normalized = lock.get("rank_normalized_multihop")
    _require(all(isinstance(item, Mapping) for item in (vllm, runtime, capture, matrix_capture, query_sum_capture, provenance, layout)), "RUNPOD_PACKAGE_SECTION_INVALID")
    _require(isinstance(completion_audit, Mapping), "RUNPOD_COMPLETION_AUDIT_INVALID")
    _require(isinstance(rank_normalized, Mapping), "RUNPOD_RANK_MULTIHOP_INVALID")
    _require(
        rank_normalized.get("recurrence_id")
        == "putpocket_rank_normalized_indexer_multihop_v1"
        and rank_normalized.get("legacy_signed_recurrence_id")
        == "putpocket_raw_indexer_strict_causal_multihop_v1"
        and rank_normalized.get("legacy_outputs_preserved_immutable") is True
        and rank_normalized.get("offline_only_no_inference_decisions") is True
        and rank_normalized.get("primary_variant") == "rank_dcg_k64"
        and rank_normalized.get("rank_transition_k_sweep") == [16, 64, 256]
        and rank_normalized.get("evaluation_top_k") == [16, 64, 256]
        and rank_normalized.get("report_schema")
        == "configs/runpod/schemas/glm52_rank_normalized_multihop_report.schema.json"
        and rank_normalized.get("token_row_schema")
        == "configs/runpod/schemas/glm52_rank_normalized_multihop_token_row.schema.json",
        "RUNPOD_RANK_MULTIHOP_INVALID",
    )
    _require(
        completion_audit.get("cache_origin_plot_script_sha256")
        == "e1f5cdd967a48514750d9c3074a8ca4a33cd217060e9a7b5cbadf9c0df03363f"
        and completion_audit.get("cache_origin_transfer_prototype_sha256")
        == [
            "42b26b61940fb6ed5aec2d384890387416567f9e04c1dc2df12ce214270a5ad7",
            "26387857faf6225b76bafac58d55f7b5e0ef6cfe33efcc35682318b93dbdce70",
        ]
        and completion_audit.get("plot_dependency_versions")
        == {"matplotlib": "3.11.1", "numpy": "2.4.6"}
        and completion_audit.get("large_data_committed") is False,
        "RUNPOD_COMPLETION_AUDIT_INVALID",
    )
    _require(vllm["commit"] == VLLM_COMMIT, "RUNPOD_VLLM_COMMIT_INVALID")
    patch_chain = vllm.get("patch_chain")
    _require(isinstance(patch_chain, list) and len(patch_chain) == 3, "RUNPOD_PATCH_CHAIN_INVALID")
    _require(
        [item.get("role") for item in patch_chain]
        == [
            "required_legacy_packaging_base",
            "true_partial_overlay",
            "score_diagnostic_overlay_with_cuda129_deepgemm_host_include_fix",
        ],
        "RUNPOD_PATCH_ORDER_INVALID",
    )
    _require(
        patch_chain[0].get("apply_tool") == "patch"
        and patch_chain[0].get("apply_args") == ["-p1", "--forward", "--batch"],
        "RUNPOD_LEGACY_APPLY_CONTRACT_INVALID",
    )
    _require(
        patch_chain[1].get("apply_tool") == "git apply"
        and patch_chain[2].get("apply_tool") == "git apply"
        and patch_chain[1].get("apply_args") == ["--unidiff-zero"]
        and patch_chain[2].get("apply_args") == ["--unidiff-zero"],
        "RUNPOD_ZERO_CONTEXT_APPLY_ARGS_INVALID",
    )
    for item in patch_chain:
        _require(_SHA256.fullmatch(str(item.get("sha256", ""))) is not None, "RUNPOD_PATCH_DIGEST_INVALID")
    artifacts = lock.get("project_artifacts")
    _require(isinstance(artifacts, list) and artifacts, "RUNPOD_PROJECT_ARTIFACTS_INVALID")
    artifact_paths = [item.get("path") for item in artifacts if isinstance(item, Mapping)]
    _require(
        len(artifact_paths) == len(artifacts)
        and len(set(artifact_paths)) == len(artifact_paths)
        and all(isinstance(path, str) and path and not Path(path).is_absolute() for path in artifact_paths)
        and all(_SHA256.fullmatch(str(item.get("sha256", ""))) is not None for item in artifacts),
        "RUNPOD_PROJECT_ARTIFACT_ENTRY_INVALID",
    )
    _require(runtime.get("model_revision") == MODEL_REVISION, "RUNPOD_MODEL_REVISION_INVALID")
    _require(runtime.get("architecture") == "GlmMoeDsaForCausalLM", "RUNPOD_ARCHITECTURE_INVALID")
    _require(
        runtime.get("tensor_parallel_size") == 4
        and runtime.get("attention_backend") == "FLASHMLA_SPARSE"
        and runtime.get("indexer_backend") == "DEEPSEEK_V32_INDEXER"
        and runtime.get("block_size") == 64
        and runtime.get("kv_cache_dtype") == "bfloat16",
        "RUNPOD_RUNTIME_BOUNDARY_INVALID",
    )
    _require(
        layout.get("layers") == 78
        and layout.get("main_attention_heads") == 64
        and layout.get("qk_nope_head_dim") == 192
        and layout.get("qk_rope_head_dim") == 64
        and layout.get("v_head_dim") == 256
        and layout.get("kv_lora_rank") == 512
        and layout.get("indexer_heads") == 32
        and layout.get("indexer_head_dim") == 128
        and layout.get("index_topk") == 2048
        and layout.get("indexer_layers") == INDEXER_LAYERS,
        "RUNPOD_MODEL_LAYOUT_BOUNDARY_INVALID",
    )
    _require(capture.get("scenario_id") == SCENARIO_ID, "RUNPOD_SCENARIO_INVALID")
    _require(capture.get("instance_id") == INSTANCE_ID, "RUNPOD_INSTANCE_INVALID")
    _require(capture.get("probe_kind") == "ordinary_target_prefill_q1_boundary_no_q2", "RUNPOD_PROBE_KIND_INVALID")
    _require(
        matrix_capture.get("capture_mode") == "strict_causal_indexer_matrix"
        and matrix_capture.get("execution_scope")
        == "optional_default_off_capture_within_test_2_not_a_third_gpu_test"
        and matrix_capture.get("default_layers") == capture.get("layers")
        and matrix_capture.get("default_window_tokens") == 256
        and matrix_capture.get("hard_max_window_tokens") == 2176
        and matrix_capture.get("default_row_chunk_size") == 32
        and matrix_capture.get("hard_max_total_edges_per_rank") == 9465600
        and matrix_capture.get("default_max_propagation_level") == 3
        and matrix_capture.get("hard_max_propagation_level") == 16
        and matrix_capture.get("sampled_capture_unchanged") is True
        and matrix_capture.get("offline_only_no_inference_decisions") is True,
        "RUNPOD_MATRIX_CAPTURE_BOUNDARY_INVALID",
    )
    _require(
        query_sum_capture.get("capture_mode") == "query_range_attention_indexer_comparison"
        and query_sum_capture.get("probe_kind") == "benchmark_derived_two_query_q1_q2_score_probe"
        and query_sum_capture.get("layers") == capture.get("layers")
        and query_sum_capture.get("hard_max_query_tokens") == 2048
        and query_sum_capture.get("hard_max_candidate_tokens") == 2176
        and query_sum_capture.get("hard_max_main_logit_values_per_rank") == 150994944
        and query_sum_capture.get("candidate_history_scope")
        == "all frozen prompt positions [0,Q2_end); seed rows are all and only Q1/Q2 content tokens"
        and query_sum_capture.get("offline_analysis_version") == 2
        and query_sum_capture.get("normalized_distribution")
        == "independent_population_zscore_then_softmax_per_query_summed_vector"
        and query_sum_capture.get("topk_ranking") == "raw_descending_pre_softmax"
        and query_sum_capture.get("ndcg_relevance")
        == "main_raw_descending_rank_n_to_1"
        and query_sum_capture.get("native_softmax_policy")
        == "retain_as_saturation_diagnostic_only_not_primary_js_or_topk"
        and query_sum_capture.get("older_sampled_mode_is_final") is False
        and query_sum_capture.get("default_off") is True,
        "RUNPOD_QUERY_SUM_CAPTURE_BOUNDARY_INVALID",
    )
    _require(
        [
            matrix_capture.get("episode_schema"),
            matrix_capture.get("row_schema"),
            matrix_capture.get("report_schema"),
            matrix_capture.get("token_row_schema"),
        ]
        == [
            "configs/runpod/schemas/glm52_indexer_matrix_episode.schema.json",
            "configs/runpod/schemas/glm52_indexer_matrix_row.schema.json",
            "configs/runpod/schemas/glm52_indexer_multihop_report.schema.json",
            "configs/runpod/schemas/glm52_indexer_multihop_token_row.schema.json",
        ],
        "RUNPOD_MATRIX_SCHEMA_PATHS_INVALID",
    )
    _require(
        provenance.get("dataset") == "ScaleAI/SWE-bench_Pro"
        and provenance.get("dataset_revision") == "7ab5114912baf22bb098818e604c02fe7ad2c11f"
        and provenance.get("harness_commit")
        == "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
        and provenance.get("mini_swe_submodule_commit")
        == "d74716a3c8104a113f77cc9ab94cf407ecdcf1e9"
        and provenance.get("mini_swe_scaffold")
        == "mini-swe-agent/src/minisweagent/config/extra/swebench.yaml",
        "RUNPOD_DATASET_PROVENANCE_INVALID",
    )


def validate_schema(document: Mapping[str, Any], schema_path: str | Path) -> None:
    schema = load_json(schema_path)
    try:
        jsonschema.Draft202012Validator(schema).validate(document)
    except jsonschema.ValidationError as exc:
        raise ConfigError(f"SCHEMA_VALIDATION_FAILED:{Path(schema_path).name}:{exc.json_path}") from exc


def validate_schedule(path: str | Path = SCHEDULE) -> dict[str, Any]:
    value = load_json(path)
    validate_schema(value, SCHEDULE_SCHEMA)
    tests = value["tests"]
    _require([item["test_id"] for item in tests] == ["glm52-runpod-doctor-v1", "glm52-main-indexer-score-compare-v1"], "RUNPOD_TEST_ORDER_INVALID")
    _require(tests[1]["depends_on"] == [tests[0]["test_id"]], "RUNPOD_TEST_DEPENDENCY_INVALID")
    _require(tests[1]["dependency_evidence"] == "doctor.payload_sha256", "RUNPOD_TEST_EVIDENCE_DEPENDENCY_INVALID")
    postprocessors = value["postprocessors"]
    _require(
        len(postprocessors) == 1
        and postprocessors[0]["postprocessor_id"]
        == "glm52-raw-indexer-multihop-offline-v1"
        and postprocessors[0]["gpu_test_ordinal"] == 2
        and postprocessors[0]["default_enabled"] is False,
        "RUNPOD_POSTPROCESSOR_BOUNDARY_INVALID",
    )
    return value


def validate_project_artifacts(project_root: str | Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root)
    checked = []
    for item in lock["project_artifacts"]:
        relative, expected = item["path"], item["sha256"]
        path = root / relative
        _require(path.is_file(), f"RUNPOD_ARTIFACT_MISSING:{relative}")
        observed = file_sha256(path)
        _require(observed == expected, f"RUNPOD_ARTIFACT_DIGEST_MISMATCH:{relative}")
        checked.append({"path": relative, "sha256": observed})
    return {"status": "passed", "artifacts": checked}


def validate_vllm_tree(source_root: str | Path, lock: Mapping[str, Any], phase: str) -> dict[str, Any]:
    root = Path(source_root)
    phases = lock["vllm"]["source_hashes"]
    _require(phase in phases, f"RUNPOD_VLLM_PHASE_INVALID:{phase}")
    checked = []
    for relative, expected in phases[phase].items():
        path = root / relative
        _require(path.is_file(), f"RUNPOD_VLLM_SOURCE_MISSING:{relative}")
        observed = file_sha256(path)
        _require(observed == expected, f"RUNPOD_VLLM_SOURCE_DIGEST_MISMATCH:{phase}:{relative}")
        checked.append({"path": relative, "sha256": observed})
    return {"status": "passed", "phase": phase, "source_files": checked}


def _command(argv: Sequence[str], cwd: Path | None = None) -> str:
    result = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise ConfigError(f"COMMAND_FAILED:{argv[0]}:{result.stderr.strip()}")
    return result.stdout.strip()


def _validate_harness_scaffold(harness: Path, lock: Mapping[str, Any]) -> Path:
    """Bind the outer SWE-bench Pro checkout and its mini-swe-agent gitlink."""

    provenance = lock["benchmark_provenance"]
    _require(
        _command(["git", "rev-parse", "HEAD"], harness)
        == provenance["harness_commit"],
        "PROBE_HARNESS_COMMIT_MISMATCH",
    )
    mini_swe = harness / "mini-swe-agent"
    _require(
        _command(["git", "rev-parse", "HEAD"], mini_swe)
        == provenance["mini_swe_submodule_commit"],
        "PROBE_MINI_SWE_SUBMODULE_COMMIT_MISMATCH",
    )
    scaffold = harness / provenance["mini_swe_scaffold"]
    _require(
        file_sha256(scaffold) == provenance["mini_swe_scaffold_sha256"],
        "PROBE_SCAFFOLD_DIGEST_MISMATCH",
    )
    return scaffold


def _check(check_id: str, operation: Callable[[], Any]) -> dict[str, Any]:
    try:
        evidence = operation()
        return {"check_id": check_id, "status": "passed", "evidence": evidence}
    except BaseException as exc:
        return {
            "check_id": check_id,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _model_config_check(model_root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    marker = model_root / ".putpocket_model_revision"
    _require(model_root.is_absolute() and model_root.is_dir(), "DOCTOR_MODEL_ROOT_INVALID")
    _require(model_root.name == MODEL_REVISION, "DOCTOR_MODEL_ROOT_REVISION_NAME_MISMATCH")
    _require(marker.is_file() and marker.read_text(encoding="utf-8").strip() == MODEL_REVISION, "DOCTOR_MODEL_REVISION_MARKER_INVALID")
    config_path = model_root / "config.json"
    config = load_json(config_path)
    runtime, layout = lock["runtime"], lock["model_layout"]
    _require(config.get("architectures") == [runtime["architecture"]], "DOCTOR_MODEL_ARCHITECTURE_MISMATCH")
    _require(config.get("model_type") == runtime["model_type"], "DOCTOR_MODEL_TYPE_MISMATCH")
    _require(config.get("num_hidden_layers") == layout["layers"], "DOCTOR_MODEL_LAYERS_MISMATCH")
    _require(
        config.get("num_attention_heads") == layout["main_attention_heads"]
        and config.get("qk_nope_head_dim") == layout["qk_nope_head_dim"]
        and config.get("qk_rope_head_dim") == layout["qk_rope_head_dim"]
        and config.get("v_head_dim") == layout["v_head_dim"]
        and config.get("kv_lora_rank") == layout["kv_lora_rank"],
        "DOCTOR_MAIN_ATTENTION_LAYOUT_MISMATCH",
    )
    _require(config.get("index_topk") == layout["index_topk"], "DOCTOR_INDEX_TOPK_MISMATCH")
    _require(config.get("index_n_heads") == layout["indexer_heads"], "DOCTOR_INDEX_HEADS_MISMATCH")
    _require(config.get("index_head_dim") == layout["indexer_head_dim"], "DOCTOR_INDEX_HEAD_DIM_MISMATCH")
    quant = config.get("quantization_config")
    _require(isinstance(quant, Mapping) and str(quant.get("quant_algo", "")).upper() in {"NVFP4", "W4A16_NVFP4"}, "DOCTOR_MODEL_QUANTIZATION_MISMATCH")
    pattern = config.get("index_topk_pattern")
    types = config.get("indexer_types")
    if isinstance(pattern, str):
        observed = [index for index, value in enumerate(pattern) if value.upper() == "F"]
    elif isinstance(types, list):
        observed = [index for index, value in enumerate(types) if str(value).lower() == "full"]
    else:
        raise ConfigError("DOCTOR_INDEXER_PATTERN_MISSING")
    _require(observed == layout["indexer_layers"], "DOCTOR_INDEXER_LAYER_PATTERN_MISMATCH")
    tokenizer_hashes = {}
    for name, digest in runtime["tokenizer_sha256"].items():
        path = model_root / name
        _require(path.is_file() and file_sha256(path) == digest, f"DOCTOR_TOKENIZER_DIGEST_MISMATCH:{name}")
        tokenizer_hashes[name] = digest
    old_hf, old_transformers = os.environ.get("HF_HUB_OFFLINE"), os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)
    finally:
        if old_hf is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = old_hf
        if old_transformers is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = old_transformers
    return {
        "model_revision": MODEL_REVISION,
        "config_sha256_observed": file_sha256(config_path),
        "config_semantic_boundary": "exact_fields_checked_config_file_digest_observed_not_repository_pinned",
        "architecture": runtime["architecture"],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_sha256": tokenizer_hashes,
        "offline_local_files_only": True,
    }


def _python_import_check(lock: Mapping[str, Any]) -> dict[str, Any]:
    expected = lock["environment"]
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    _require(version == expected["python_major_minor"], "DOCTOR_PYTHON_VERSION_MISMATCH")
    versions = {}
    for distribution, expected_version in expected["python_distributions"].items():
        observed = importlib.metadata.version(distribution)
        _require(observed == expected_version, f"DOCTOR_DEPENDENCY_VERSION_MISMATCH:{distribution}:{observed}")
        versions[distribution] = observed
    import_versions = {}
    for module_name in expected["required_imports"]:
        module = importlib.import_module(module_name)
        import_versions[module_name] = str(getattr(module, "__version__", "not_exposed"))
    return {
        "python": sys.version.split()[0],
        "distributions": versions,
        "imports": expected["required_imports"],
        "import_versions_observed": import_versions,
    }


def _gpu_check(lock: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    runtime = lock["runtime"]
    hardware = lock["hardware"]
    _require(torch.cuda.is_available(), "DOCTOR_CUDA_UNAVAILABLE")
    _require(torch.version.cuda == lock["environment"]["torch_cuda"], "DOCTOR_TORCH_CUDA_VERSION_MISMATCH")
    count = torch.cuda.device_count()
    _require(count == runtime["tensor_parallel_size"], "DOCTOR_VISIBLE_GPU_COUNT_MISMATCH")
    _require(torch.cuda.is_bf16_supported(), "DOCTOR_BF16_UNSUPPORTED")
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        capability = torch.cuda.get_device_capability(index)
        _require(
            hardware["accelerator"].upper() in properties.name.upper(),
            "DOCTOR_GPU_MODEL_MISMATCH",
        )
        _require(tuple(capability) == tuple(hardware["compute_capability"]), "DOCTOR_GPU_CAPABILITY_MISMATCH")
        _require(total // (1024 * 1024) >= hardware["minimum_total_memory_mib"], "DOCTOR_GPU_TOTAL_MEMORY_TOO_SMALL")
        _require(free // (1024 * 1024) >= hardware["minimum_free_memory_mib"], "DOCTOR_GPU_FREE_MEMORY_TOO_SMALL")
        devices.append({
            "index": index,
            "name": properties.name,
            "capability": list(capability),
            "total_memory_mib": total // (1024 * 1024),
            "free_memory_mib": free // (1024 * 1024),
        })
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "bf16": True,
        "visible_gpu_count": count,
        "devices": devices,
    }


def _cuda_toolchain_check(lock: Mapping[str, Any]) -> dict[str, Any]:
    nvcc = shutil.which("nvcc")
    _require(nvcc is not None, "DOCTOR_NVCC_MISSING")
    output = _command([nvcc, "--version"])
    expected = lock["environment"]["torch_cuda"]
    _require(
        re.search(rf"release\s+{re.escape(expected)}(?:,|\s|$)", output) is not None,
        "DOCTOR_NVCC_VERSION_MISMATCH",
    )
    return {"path": nvcc, "expected_cuda": expected, "version_output": output}


def _runtime_contract_check(lock: Mapping[str, Any]) -> dict[str, Any]:
    runtime = lock["runtime"]
    layout = lock["model_layout"]
    return {
        "architecture": runtime["architecture"],
        "attention_backend": runtime["attention_backend"],
        "indexer_backend": runtime["indexer_backend"],
        "tensor_parallel_size": runtime["tensor_parallel_size"],
        "block_size": runtime["block_size"],
        "kv_cache_dtype": runtime["kv_cache_dtype"],
        "layers": layout["layers"],
        "indexer_layers": layout["indexer_layers"],
        "index_topk": layout["index_topk"],
    }


def _distributed_check(lock: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    _require(torch.distributed.is_available() and torch.distributed.is_nccl_available(), "DOCTOR_NCCL_UNAVAILABLE")
    peer = [
        {"left": left, "right": right, "accessible": bool(torch.cuda.can_device_access_peer(left, right))}
        for left in range(torch.cuda.device_count())
        for right in range(torch.cuda.device_count())
        if left != right
    ]
    _require(all(item["accessible"] for item in peer), "DOCTOR_GPU_PEER_ACCESS_INCOMPLETE")
    version = torch.cuda.nccl.version()
    return {"distributed_available": True, "nccl_available": True, "nccl_version": list(version) if isinstance(version, tuple) else version, "peer_access": peer}


def _vllm_symbol_check() -> dict[str, Any]:
    from vllm.model_executor.layers.glm52_attention_indexer_scores import (
        MATRIX_MODE,
        maybe_capture_indexer_native_logits,
        maybe_capture_main_attention_reference,
    )
    from vllm.model_executor.layers.sparse_attn_indexer import sparse_attn_indexer
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2Model,
        GlmMoeDsaForCausalLM,
    )
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, has_deep_gemm
    from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseBackend
    from vllm.v1.attention.backends.mla.indexer import DeepseekV32IndexerBackend
    from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.putpocket_true_partial_prefill import (
        build_target_plan,
        copy_reuse_rows_inplace,
        execution_evidence,
    )
    from vllm.v1.request import Request
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    supported, reason = is_flashmla_sparse_supported()
    _require(supported, f"DOCTOR_FLASHMLA_SPARSE_UNSUPPORTED:{reason}")
    _require(has_deep_gemm(), "DOCTOR_DEEPGEMM_UNAVAILABLE")
    required_source_symbols = {
        "request": (Request, "putpocket_true_partial_instruction"),
        "scheduler": (Scheduler, "putpocket_donor_registry"),
        "scheduler_output": (SchedulerOutput, "putpocket_true_partial_continuation"),
        "attention_metadata": (CommonAttentionMetadata, "putpocket_true_partial_positions"),
        "gpu_model_runner": (GPUModelRunner, "putpocket_true_partial_active_payload"),
        "glm_dsa_diagnostic_batch_attestation": (
            DeepseekV2Model,
            "maybe_set_score_diagnostic_batch",
        ),
    }
    for owner, (symbol, needle) in required_source_symbols.items():
        _require(needle in inspect.getsource(symbol), f"DOCTOR_PATCHED_SYMBOL_MISSING:{owner}:{needle}")
    _require(
        MATRIX_MODE == "strict_causal_indexer_matrix"
        and "MATRIX_MODE" in inspect.getsource(maybe_capture_indexer_native_logits),
        "DOCTOR_MATRIX_CAPTURE_SYMBOL_MISSING",
    )
    symbols = [
        sparse_attn_indexer,
        fp8_fp4_mqa_logits,
        DeepseekV2Model,
        GlmMoeDsaForCausalLM,
        FlashMLASparseBackend,
        DeepseekV32IndexerBackend,
        build_target_plan,
        copy_reuse_rows_inplace,
        execution_evidence,
        maybe_capture_main_attention_reference,
        maybe_capture_indexer_native_logits,
    ]
    return {
        "flashmla_sparse": True,
        "deepseek_v32_indexer": True,
        "symbols": [f"{item.__module__}.{item.__name__}" for item in symbols],
        "patched_boundaries": sorted(required_source_symbols),
        "optional_strict_causal_indexer_matrix_capture": True,
    }


def _default_off_check() -> dict[str, Any]:
    variables = {
        "PUTPOCKET_VLLM_TRUE_PARTIAL_PREFILL_ENABLE": os.getenv("PUTPOCKET_VLLM_TRUE_PARTIAL_PREFILL_ENABLE"),
        "PUTPOCKET_GLM52_FORCED_REUSE_CONTROL": os.getenv("PUTPOCKET_GLM52_FORCED_REUSE_CONTROL"),
        "PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_ENABLE": os.getenv("PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_ENABLE"),
    }
    _require(all(value in {None, ""} for value in variables.values()), "DOCTOR_EXPERIMENTAL_FEATURE_ALREADY_ENABLED")
    return {"default_off": True, "variables": variables, "mutually_exclusive_runtime_modes": True}


def run_doctor(
    *,
    project_root: str | Path,
    vllm_root: str | Path,
    model_root: str | Path,
    expected_project_commit: str,
    output: str | Path,
    lock_path: str | Path = PACKAGE_LOCK,
) -> dict[str, Any]:
    project = Path(project_root).resolve()
    vllm = Path(vllm_root).resolve()
    model = Path(model_root).resolve()
    _require(_SHA40.fullmatch(expected_project_commit) is not None, "DOCTOR_EXPECTED_PROJECT_COMMIT_INVALID")
    lock = load_package_lock(lock_path)
    checks = [
        _check("project_identity", lambda: {
            "commit": _command(["git", "rev-parse", "HEAD"], project),
            "branch": _command(["git", "branch", "--show-current"], project),
            "expected_commit": expected_project_commit,
            "commit_matches": _command(["git", "rev-parse", "HEAD"], project) == expected_project_commit,
            "branch_matches": _command(["git", "branch", "--show-current"], project) == lock["project_branch"],
        }),
        _check("project_artifact_hashes", lambda: validate_project_artifacts(project, lock)),
        _check("pinned_runtime_contract", lambda: _runtime_contract_check(lock)),
        _check("vllm_source_identity", lambda: {
            "commit": _command(["git", "rev-parse", "HEAD"], vllm),
            "expected_commit": VLLM_COMMIT,
            "matches": _command(["git", "rev-parse", "HEAD"], vllm) == VLLM_COMMIT,
            **validate_vllm_tree(vllm, lock, "post_score_diagnostic"),
        }),
        _check("python_dependency_imports", lambda: _python_import_check(lock)),
        _check("model_config_tokenizer_offline", lambda: _model_config_check(model, lock)),
        _check("cuda_toolchain", lambda: _cuda_toolchain_check(lock)),
        _check("torch_cuda_bf16_gpu_inventory", lambda: _gpu_check(lock)),
        _check("nccl_distributed_prerequisites", lambda: _distributed_check(lock)),
        _check("patched_backend_symbols", _vllm_symbol_check),
        _check("experimental_features_default_off", _default_off_check),
    ]
    # Identity checks return booleans so they remain structured, then fail closed here.
    for check in checks:
        evidence = check.get("evidence")
        if check["check_id"] == "project_identity" and isinstance(evidence, Mapping):
            if not evidence.get("commit_matches") or not evidence.get("branch_matches"):
                check.update(status="failed", error_type="ConfigError", error="DOCTOR_PROJECT_IDENTITY_MISMATCH")
                check.pop("evidence", None)
        if check["check_id"] == "vllm_source_identity" and isinstance(evidence, Mapping) and not evidence.get("matches"):
            check.update(status="failed", error_type="ConfigError", error="DOCTOR_VLLM_COMMIT_MISMATCH")
            check.pop("evidence", None)
    status = "passed" if all(item["status"] == "passed" for item in checks) else "failed"
    payload = {
        "schema_version": 1,
        "test_id": "glm52-runpod-doctor-v1",
        "status": status,
        "read_only": True,
        "project_commit": expected_project_commit,
        "vllm_commit": VLLM_COMMIT,
        "model_revision": MODEL_REVISION,
        "checks": checks,
    }
    report = {
        "payload": payload,
        "payload_sha256": hashlib.sha256(canonical_json_bytes(payload, newline=False)).hexdigest(),
    }
    validate_schema(report, DOCTOR_SCHEMA)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def load_successful_doctor(path: str | Path) -> tuple[dict[str, Any], str]:
    report = load_json(path)
    validate_schema(report, DOCTOR_SCHEMA)
    payload = report["payload"]
    expected = hashlib.sha256(canonical_json_bytes(payload, newline=False)).hexdigest()
    _require(report["payload_sha256"] == expected, "DOCTOR_PAYLOAD_DIGEST_MISMATCH")
    _require(payload["status"] == "passed", "DOCTOR_NOT_SUCCESSFUL")
    return report, expected


def prepare_probe(
    *,
    model_root: str | Path,
    harness_root: str | Path,
    output: str | Path,
    lock_path: str | Path = PACKAGE_LOCK,
) -> dict[str, Any]:
    lock = load_package_lock(lock_path)
    model = Path(model_root).resolve()
    harness = Path(harness_root).resolve()
    _model_config_check(model, lock)
    scaffold = _validate_harness_scaffold(harness, lock)
    from datasets import load_dataset
    from jinja2 import StrictUndefined, Template
    from transformers import AutoTokenizer
    import yaml

    rows = load_dataset(
        lock["benchmark_provenance"]["dataset"],
        revision=lock["benchmark_provenance"]["dataset_revision"],
        split=lock["benchmark_provenance"]["split"],
    )
    matches = [dict(row) for row in rows if row.get("instance_id") == INSTANCE_ID]
    _require(len(matches) == 1, "PROBE_INSTANCE_CARDINALITY_MISMATCH")
    row = matches[0]
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    _require(hashlib.sha256(canonical).hexdigest() == lock["benchmark_provenance"]["row_sha256"], "PROBE_DATASET_ROW_DIGEST_MISMATCH")
    agent = yaml.safe_load(scaffold.read_text(encoding="utf-8"))["agent"]
    messages = [
        {"role": "system", "content": Template(agent["system_template"], undefined=StrictUndefined).render(task=row["problem_statement"])},
        {"role": "user", "content": Template(agent["instance_template"], undefined=StrictUndefined).render(task=row["problem_statement"])},
    ]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    serialized = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    old_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    capture = lock["capture"]
    _require(
        isinstance(old_ids, list)
        and len(old_ids) == capture["expected_prompt_token_count"]
        and hashlib.sha256(serialized.encode()).hexdigest() == capture["baseline_serialized_prompt_sha256"],
        "PROBE_BASELINE_SERIALIZATION_MISMATCH",
    )
    edit = capture["equal_position_edit"]
    _require(old_ids[edit["position"]] == edit["old_token_id"], "PROBE_OLD_EDIT_TOKEN_MISMATCH")
    target_ids = list(old_ids)
    target_ids[edit["position"]] = edit["new_token_id"]
    target_digest = hashlib.sha256(json.dumps(target_ids, separators=(",", ":")).encode("ascii")).hexdigest()
    payload = {
        "schema_version": 1,
        "probe_id": capture["diagnostic_id"],
        "probe_kind": capture["probe_kind"],
        "instance_id": INSTANCE_ID,
        "scenario_id": SCENARIO_ID,
        "benchmark_native_components": ["problem_row", "repository_identifier", "container_mapping", "official_evaluator"],
        "project_authored_components": ["system_replacement", "ordinary_target_prefill_probe", "score_comparison"],
        "q2_in_probe": False,
        "q2_donor_reuse_claimed": False,
        "prompt_token_ids": target_ids,
        "prompt_token_count": len(target_ids),
        "prompt_token_ids_sha256": target_digest,
        "baseline_prompt_token_ids_sha256": hashlib.sha256(json.dumps(old_ids, separators=(",", ":")).encode("ascii")).hexdigest(),
        "edit": edit,
        "raw_prompt_persisted": False,
        "outcome_or_evaluator_data_used": False,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _content_token_range(
    serialized: str,
    content: str,
    offsets: Sequence[Sequence[int]],
) -> list[int]:
    start = serialized.rfind(content)
    _require(start >= 0 and serialized.find(content, start + 1) < 0, "FINAL_PROBE_CONTENT_NOT_UNIQUELY_LOCATED_FROM_END")
    end = start + len(content)
    positions = [index for index, pair in enumerate(offsets) if pair[1] > start and pair[0] < end]
    _require(positions and positions == list(range(positions[0], positions[-1] + 1)), "FINAL_PROBE_CONTENT_TOKEN_RANGE_NONCONTIGUOUS")
    return [positions[0], positions[-1] + 1]


def final_probe_candidate_window(
    q1_range: Sequence[int],
    q2_range: Sequence[int],
) -> list[int]:
    """Return the complete frozen candidate prefix for the two-query probe."""

    _require(
        len(q1_range) == 2
        and len(q2_range) == 2
        and 0 <= q1_range[0] < q1_range[1] <= q2_range[0] < q2_range[1],
        "FINAL_PROBE_QUERY_RANGES_INVALID",
    )
    return [0, q2_range[1]]


def prepare_final_two_query_probe(
    *,
    model_root: str | Path,
    harness_root: str | Path,
    output: str | Path,
    matrix_output: str | Path,
    lock_path: str | Path = PACKAGE_LOCK,
) -> dict[str, Any]:
    """Freeze an outcome-independent two-query probe when no executed A1 exists."""

    lock = load_package_lock(lock_path)
    model = Path(model_root).resolve()
    harness = Path(harness_root).resolve()
    _model_config_check(model, lock)
    scaffold = _validate_harness_scaffold(harness, lock)
    from datasets import load_dataset
    from jinja2 import StrictUndefined, Template
    from transformers import AutoTokenizer
    import yaml

    rows = load_dataset(
        lock["benchmark_provenance"]["dataset"],
        revision=lock["benchmark_provenance"]["dataset_revision"],
        split=lock["benchmark_provenance"]["split"],
    )
    matches = [dict(row) for row in rows if row.get("instance_id") == INSTANCE_ID]
    _require(len(matches) == 1, "PROBE_INSTANCE_CARDINALITY_MISMATCH")
    row = matches[0]
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    _require(hashlib.sha256(canonical).hexdigest() == lock["benchmark_provenance"]["row_sha256"], "PROBE_DATASET_ROW_DIGEST_MISMATCH")
    agent = yaml.safe_load(scaffold.read_text(encoding="utf-8"))["agent"]
    q1 = Template(agent["instance_template"], undefined=StrictUndefined).render(task=row["problem_statement"])
    q2 = lock["query_sum_capture"]["project_authored_q2_text"]
    bridge = lock["query_sum_capture"]["project_authored_assistant_bridge_text"]
    messages = [
        {"role": "system", "content": Template(agent["system_template"], undefined=StrictUndefined).render(task=row["problem_statement"])},
        {"role": "user", "content": q1},
        {"role": "assistant", "content": bridge},
        {"role": "user", "content": q2},
    ]
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    serialized = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    old_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    encoded = tokenizer(serialized, add_special_tokens=False, return_offsets_mapping=True)
    _require(isinstance(old_ids, list) and encoded["input_ids"] == old_ids, "FINAL_PROBE_SERIALIZATION_OFFSET_TOKEN_MISMATCH")
    q1_range = _content_token_range(serialized, q1, encoded["offset_mapping"])
    q2_range = _content_token_range(serialized, q2, encoded["offset_mapping"])
    _require(q1_range[1] <= q2_range[0], "FINAL_PROBE_QUERY_RANGES_OVERLAP")
    edit = lock["capture"]["equal_position_edit"]
    _require(old_ids[edit["position"]] == edit["old_token_id"], "PROBE_OLD_EDIT_TOKEN_MISMATCH")
    target_ids = list(old_ids)
    target_ids[edit["position"]] = edit["new_token_id"]
    target_digest = hashlib.sha256(json.dumps(target_ids, separators=(",", ":")).encode("ascii")).hexdigest()
    # Candidate history starts at absolute position zero so the score probe
    # includes the edited SYS row and the complete causal history of every
    # Q1/Q2 seed row. Only seed membership is restricted to Q1/Q2.
    window = final_probe_candidate_window(q1_range, q2_range)
    width = window[1] - window[0]
    query_count = (q1_range[1] - q1_range[0]) + (q2_range[1] - q2_range[0])
    declared = lock["query_sum_capture"]
    matrix = lock["matrix_capture"]
    matrix_edges = width * (width - 1) // 2 * len(declared["layers"])
    local_heads = lock["model_layout"]["main_attention_heads"] // lock["runtime"]["tensor_parallel_size"]
    main_values = sum(
        position - window[0]
        for start, end in (q1_range, q2_range)
        for position in range(start, end)
    ) * local_heads * len(declared["layers"])
    _require(
        query_count <= declared["hard_max_query_tokens"]
        and width <= declared["hard_max_candidate_tokens"]
        and main_values <= declared["hard_max_main_logit_values_per_rank"]
        and width <= matrix["hard_max_window_tokens"]
        and matrix_edges <= matrix["hard_max_total_edges_per_rank"],
        "FINAL_PROBE_COST_CAP_EXCEEDED",
    )
    target_decoded = tokenizer.decode(target_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    payload = {
        "schema_version": 1,
        "artifact_kind": "frozen_benchmark_derived_two_query_probe",
        "status": "frozen_before_model_capture_or_evaluator_outcome",
        "episode_id": "glm52-swebench-pro-two-query-equal-sys-probe-v1",
        "scenario_id": SCENARIO_ID,
        "instance_id": INSTANCE_ID,
        "benchmark_provenance": lock["benchmark_provenance"],
        "prompt": {
            "token_ids": target_ids,
            "token_count": len(target_ids),
            "token_ids_sha256": target_digest,
            "baseline_token_ids_sha256": hashlib.sha256(json.dumps(old_ids, separators=(",", ":")).encode("ascii")).hexdigest(),
            "serialized_old_prompt": serialized,
            "serialized_old_prompt_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "target_decoded_prompt": target_decoded,
            "target_decoded_prompt_sha256": hashlib.sha256(target_decoded.encode()).hexdigest(),
            "serializer_id": "nvidia/GLM-5.2-NVFP4:chat_template.jinja:add_generation_prompt=true",
            "tokenizer_revision": MODEL_REVISION,
        },
        "segments": {
            "q1_ranges": [q1_range],
            "q2_ranges": [q2_range],
            "membership_semantics": "content_tokens_overlapping_the_exact_user_message_content_character_span",
            "system_assistant_tool_tokens_are_seed_members": False,
        },
        "propagation_window": window,
        "equal_position_system_edit": edit,
        "cost_estimate": {
            "window_tokens": width,
            "q1_q2_query_tokens": query_count,
            "matrix_edge_values_per_rank": matrix_edges,
            "main_reference_logit_values_per_rank": main_values,
            "complexity": "O(layers * window_tokens^2)",
        },
        "leakage_policy": {
            "outcome_independent": True,
            "gold_patch_used": False,
            "evaluator_outcome_used": False,
            "q2_preregistered_sha256": hashlib.sha256(q2.encode()).hexdigest(),
            "assistant_bridge_preregistered_sha256": hashlib.sha256(bridge.encode()).hexdigest(),
        },
        "claim_boundary": {
            "real_a1_generated_or_executed": False,
            "q2_is_tool_observation": False,
            "stateful_cache_or_quality_claim_allowed": False,
            "probe_kind": "deterministic_two_query_benchmark_derived_score_diagnostic_only",
            "equal_position_edit_avoids_stale_rope_confound": True,
        },
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    matrix_payload = {
        "schema_version": 1,
        "artifact_kind": "frozen_indexer_matrix_episode",
        "status": "frozen_before_outcomes",
        "episode_id": payload["episode_id"],
        "scenario_id": SCENARIO_ID,
        "instance_id": INSTANCE_ID,
        "benchmark_provenance": {
            "dataset": lock["benchmark_provenance"]["dataset"],
            "dataset_revision": lock["benchmark_provenance"]["dataset_revision"],
            "split": lock["benchmark_provenance"]["split"],
            "instance_id": INSTANCE_ID,
            "native_components": ["problem", "repository identifier", "container mapping", "official evaluator"],
            "project_authored_components": ["equal-position SYS token replacement", "assistant bridge", "preregistered Q2", "score probe"],
            "outcome_data_used": False,
        },
        "source_frozen_episode_manifest": {"path": target.name, "sha256": file_sha256(target)},
        "prompt": {
            "kind": "frozen_first_post_edit_request_or_frozen_ordinary_prefill_probe",
            "token_ids": target_ids,
            "token_count": len(target_ids),
            "token_ids_sha256": target_digest,
            "serializer_id": payload["prompt"]["serializer_id"],
            "tokenizer_revision": MODEL_REVISION,
            "q2_included": True,
        },
        "segments": {
            "q1_ranges": [q1_range],
            "q2_ranges": [q2_range],
            "q2_semantics": "project_authored_preregistered_followup_query_in_non_stateful_probe",
        },
        "propagation_window": window,
        "capture": {
            "mode": "strict_causal_indexer_matrix",
            "layers": declared["layers"],
            "tensor_parallel_size": 4,
            "row_chunk_size": matrix["default_row_chunk_size"],
            "indexer_tp_max_abs_difference": declared["indexer_tp_max_abs_difference"],
            "hard_max_window_tokens": matrix["hard_max_window_tokens"],
            "hard_max_total_edges_per_rank": matrix["hard_max_total_edges_per_rank"],
            "cost_warning": "O(layers * window_tokens^2) capture cost; exact bounded estimate is frozen in the source probe",
        },
        "outcome_independent": True,
        "authorship_boundary": "SWE-bench Pro supplies problem/repository/container/evaluator; PutPocket authors both-query diagnostic probe, edit, and score transforms",
    }
    matrix_target = Path(matrix_output)
    _require(matrix_target.parent.resolve() == target.parent.resolve(), "FINAL_PROBE_OUTPUTS_MUST_SHARE_DIRECTORY")
    matrix_target.write_text(json.dumps(matrix_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    load_matrix_episode_manifest(matrix_target)
    return payload


def build_instrumentation_config(probe: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    capture = lock["capture"]
    _require(probe.get("prompt_token_ids_sha256") and probe.get("scenario_id") == SCENARIO_ID, "PROBE_ARTIFACT_INVALID")
    return {
        "schema_version": 1,
        "diagnostic_id": capture["diagnostic_id"],
        "instance_id": INSTANCE_ID,
        "scenario_id": SCENARIO_ID,
        "probe_kind": capture["probe_kind"],
        "expected_prompt_token_count": probe["prompt_token_count"],
        "expected_prompt_token_ids_sha256": probe["prompt_token_ids_sha256"],
        "expected_edit_position": capture["equal_position_edit"]["position"],
        "expected_target_token_id": capture["equal_position_edit"]["new_token_id"],
        "layers": capture["layers"],
        "query_positions": capture["query_positions"],
        "candidate_start_position": capture["candidate_start_position"],
        "max_candidate_tokens": capture["max_candidate_tokens"],
        "tensor_parallel_size": lock["runtime"]["tensor_parallel_size"],
        "global_main_attention_heads": lock["model_layout"]["main_attention_heads"],
        "indexer_heads": lock["model_layout"]["indexer_heads"],
        "indexer_head_dim": lock["model_layout"]["indexer_head_dim"],
        "main_qk_nope_head_dim": lock["model_layout"]["qk_nope_head_dim"],
        "main_qk_rope_head_dim": lock["model_layout"]["qk_rope_head_dim"],
    }


def build_query_sum_instrumentation_config(
    probe: Mapping[str, Any], lock: Mapping[str, Any]
) -> dict[str, Any]:
    declared = lock["query_sum_capture"]
    _require(
        probe.get("artifact_kind") == "frozen_benchmark_derived_two_query_probe"
        and probe.get("scenario_id") == SCENARIO_ID
        and probe["claim_boundary"]["real_a1_generated_or_executed"] is False,
        "QUERY_SUM_PROBE_ARTIFACT_INVALID",
    )
    return {
        "schema_version": 1,
        "capture_mode": "query_range_attention_indexer_comparison",
        "diagnostic_id": declared["diagnostic_id"],
        "instance_id": INSTANCE_ID,
        "scenario_id": SCENARIO_ID,
        "probe_kind": "benchmark_derived_two_query_q1_q2_score_probe",
        "expected_prompt_token_count": probe["prompt"]["token_count"],
        "expected_prompt_token_ids_sha256": probe["prompt"]["token_ids_sha256"],
        "expected_edit_position": lock["capture"]["equal_position_edit"]["position"],
        "expected_target_token_id": lock["capture"]["equal_position_edit"]["new_token_id"],
        "layers": declared["layers"],
        "q1_ranges": probe["segments"]["q1_ranges"],
        "q2_ranges": probe["segments"]["q2_ranges"],
        "candidate_window": probe["propagation_window"],
        "hard_max_query_tokens": declared["hard_max_query_tokens"],
        "hard_max_candidate_tokens": declared["hard_max_candidate_tokens"],
        "hard_max_main_logit_values_per_rank": declared["hard_max_main_logit_values_per_rank"],
        "tensor_parallel_size": lock["runtime"]["tensor_parallel_size"],
        "global_main_attention_heads": lock["model_layout"]["main_attention_heads"],
        "indexer_heads": lock["model_layout"]["indexer_heads"],
        "indexer_head_dim": lock["model_layout"]["indexer_head_dim"],
        "main_qk_nope_head_dim": lock["model_layout"]["qk_nope_head_dim"],
        "main_qk_rope_head_dim": lock["model_layout"]["qk_rope_head_dim"],
        "q2_in_probe": True,
    }


def build_matrix_instrumentation_config(
    episode: Mapping[str, Any], lock: Mapping[str, Any]
) -> dict[str, Any]:
    package = lock.get("matrix_capture")
    _require(isinstance(package, Mapping), "MATRIX_CAPTURE_PACKAGE_CONFIG_MISSING")
    declared = episode["capture"]
    window = episode["propagation_window"]
    width = window[1] - window[0]
    edges = width * (width - 1) // 2 * len(declared["layers"])
    _require(
        declared["mode"] == "strict_causal_indexer_matrix"
        and declared["layers"] == package["default_layers"]
        and declared["tensor_parallel_size"] == lock["runtime"]["tensor_parallel_size"]
        and declared["hard_max_window_tokens"] <= package["hard_max_window_tokens"]
        and declared["hard_max_total_edges_per_rank"]
        <= package["hard_max_total_edges_per_rank"]
        and width <= package["hard_max_window_tokens"]
        and edges <= package["hard_max_total_edges_per_rank"],
        "MATRIX_CAPTURE_PACKAGE_BOUNDARY_MISMATCH",
    )
    return {
        "schema_version": 1,
        "capture_mode": "strict_causal_indexer_matrix",
        "diagnostic_id": package["diagnostic_id"],
        "instance_id": episode["instance_id"],
        "scenario_id": episode["scenario_id"],
        "probe_kind": "frozen_episode_strict_causal_indexer_matrix",
        "expected_prompt_token_count": episode["prompt"]["token_count"],
        "expected_prompt_token_ids_sha256": episode["prompt"]["token_ids_sha256"],
        "layers": declared["layers"],
        "propagation_window": window,
        "row_chunk_size": declared["row_chunk_size"],
        "hard_max_window_tokens": declared["hard_max_window_tokens"],
        "hard_max_total_edges_per_rank": declared["hard_max_total_edges_per_rank"],
        "tensor_parallel_size": declared["tensor_parallel_size"],
        "indexer_heads": lock["model_layout"]["indexer_heads"],
        "indexer_head_dim": lock["model_layout"]["indexer_head_dim"],
        "q2_in_probe": True,
    }


def capture_matrix_episode(
    *,
    doctor_report: str | Path,
    episode_path: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    lock_path: str | Path = PACKAGE_LOCK,
    dry_run: bool = False,
) -> dict[str, Any]:
    _, doctor_digest = load_successful_doctor(doctor_report)
    lock = load_package_lock(lock_path)
    episode = load_matrix_episode_manifest(episode_path)
    config = build_matrix_instrumentation_config(episode, lock)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _require(
        not any(output.glob("matrix-rank-*-chunk-*.jsonl"))
        and not (output / "matrix-capture-run.json").exists(),
        "MATRIX_CAPTURE_OUTPUT_NOT_EMPTY",
    )
    config_path = output / "matrix-instrumentation-config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    width = config["propagation_window"][1] - config["propagation_window"][0]
    edge_count = width * (width - 1) // 2 * len(config["layers"])
    plan = {
        "schema_version": 1,
        "capture_mode": "strict_causal_indexer_matrix",
        "status": "dry_run" if dry_run else "armed",
        "doctor_payload_sha256": doctor_digest,
        "episode_sha256": file_sha256(episode_path),
        "model_root": str(Path(model_root).resolve()),
        "prompt_token_count": episode["prompt"]["token_count"],
        "propagation_window": config["propagation_window"],
        "layers": config["layers"],
        "edge_values_per_rank": edge_count,
        "cost_class": "O(layers * window_tokens^2)",
        "instrumentation_config": str(config_path),
        "instrumentation_config_sha256": file_sha256(config_path),
        "engine": lock["capture"]["engine_kwargs"],
    }
    if dry_run:
        return plan
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_ENABLE"] = "1"
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_CONFIG"] = str(config_path)
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_CONFIG_SHA256"] = file_sha256(config_path)
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_OUTPUT_ROOT"] = str(output)
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_RUN_ID"] = (
        f"glm52-matrix-{doctor_digest[:12]}-{episode['prompt']['token_ids_sha256'][:12]}"
    )
    _require(not os.getenv("PUTPOCKET_VLLM_TRUE_PARTIAL_PREFILL_ENABLE"), "CAPTURE_TRUE_PARTIAL_MODE_MUST_BE_OFF")
    _require(not os.getenv("PUTPOCKET_GLM52_FORCED_REUSE_CONTROL"), "CAPTURE_LEGACY_EMULATION_MODE_MUST_BE_OFF")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    engine = lock["capture"]["engine_kwargs"]
    _require(episode["prompt"]["token_count"] <= engine["max_model_len"], "MATRIX_CAPTURE_PROMPT_TOO_LONG")
    llm = LLM(
        model=str(Path(model_root).resolve()),
        tokenizer=str(Path(model_root).resolve()),
        tensor_parallel_size=engine["tensor_parallel_size"],
        dtype=engine["dtype"],
        quantization=engine["quantization"],
        block_size=engine["block_size"],
        kv_cache_dtype=engine["kv_cache_dtype"],
        max_model_len=engine["max_model_len"],
        max_num_seqs=engine["max_num_seqs"],
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=True,
        cpu_offload_gb=0,
        trust_remote_code=False,
        attention_config={"backend": "FLASHMLA_SPARSE", "sparse_mla_force_mqa": True},
        compilation_config=0,
        seed=0,
    )
    results = llm.generate(
        [TokensPrompt(prompt_token_ids=episode["prompt"]["token_ids"])],
        SamplingParams(temperature=0.0, max_tokens=1, seed=0),
        use_tqdm=False,
    )
    _require(len(results) == 1 and results[0].outputs, "MATRIX_CAPTURE_MODEL_OUTPUT_MISSING")
    plan.update(status="captured", generated_token_count=len(results[0].outputs[0].token_ids))
    (output / "matrix-capture-run.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan


def capture_probe(
    *,
    doctor_report: str | Path,
    probe_path: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    lock_path: str | Path = PACKAGE_LOCK,
    dry_run: bool = False,
) -> dict[str, Any]:
    _, doctor_digest = load_successful_doctor(doctor_report)
    lock = load_package_lock(lock_path)
    probe = load_json(probe_path)
    _require(probe.get("prompt_token_count") == lock["capture"]["expected_prompt_token_count"], "CAPTURE_PROBE_COUNT_INVALID")
    config = build_instrumentation_config(probe, lock)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "instrumentation-config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan = {
        "schema_version": 1,
        "status": "dry_run" if dry_run else "armed",
        "doctor_payload_sha256": doctor_digest,
        "model_root": str(Path(model_root).resolve()),
        "prompt_token_count": probe["prompt_token_count"],
        "instrumentation_config": str(config_path),
        "instrumentation_config_sha256": file_sha256(config_path),
        "engine": lock["capture"]["engine_kwargs"],
    }
    if dry_run:
        return plan
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_ENABLE"] = "1"
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_CONFIG"] = str(config_path)
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_CONFIG_SHA256"] = file_sha256(config_path)
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_OUTPUT_ROOT"] = str(output)
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_RUN_ID"] = capture_run_id(doctor_digest, probe["prompt_token_ids_sha256"])
    _require(not os.getenv("PUTPOCKET_VLLM_TRUE_PARTIAL_PREFILL_ENABLE"), "CAPTURE_TRUE_PARTIAL_MODE_MUST_BE_OFF")
    _require(not os.getenv("PUTPOCKET_GLM52_FORCED_REUSE_CONTROL"), "CAPTURE_LEGACY_EMULATION_MODE_MUST_BE_OFF")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    engine = lock["capture"]["engine_kwargs"]
    llm = LLM(
        model=str(Path(model_root).resolve()),
        tokenizer=str(Path(model_root).resolve()),
        tensor_parallel_size=engine["tensor_parallel_size"],
        dtype=engine["dtype"],
        quantization=engine["quantization"],
        block_size=engine["block_size"],
        kv_cache_dtype=engine["kv_cache_dtype"],
        max_model_len=engine["max_model_len"],
        max_num_seqs=engine["max_num_seqs"],
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=True,
        cpu_offload_gb=0,
        trust_remote_code=False,
        attention_config={"backend": "FLASHMLA_SPARSE", "sparse_mla_force_mqa": True},
        compilation_config=0,
        seed=0,
    )
    results = llm.generate(
        [TokensPrompt(prompt_token_ids=probe["prompt_token_ids"])],
        SamplingParams(temperature=0.0, max_tokens=1, seed=0),
        use_tqdm=False,
    )
    _require(len(results) == 1 and results[0].outputs, "CAPTURE_MODEL_OUTPUT_MISSING")
    plan.update(status="captured", generated_token_count=len(results[0].outputs[0].token_ids))
    (output / "capture-run.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def capture_query_sum_probe(
    *,
    doctor_report: str | Path,
    probe_path: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    lock_path: str | Path = PACKAGE_LOCK,
    dry_run: bool = False,
) -> dict[str, Any]:
    _, doctor_digest = load_successful_doctor(doctor_report)
    lock = load_package_lock(lock_path)
    probe = load_json(probe_path)
    config = build_query_sum_instrumentation_config(probe, lock)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _require(not any(output.glob("capture-rank-*.jsonl")), "QUERY_SUM_CAPTURE_OUTPUT_NOT_EMPTY")
    config_path = output / "query-sum-instrumentation-config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plan = {
        "schema_version": 1,
        "capture_mode": config["capture_mode"],
        "status": "dry_run" if dry_run else "armed",
        "doctor_payload_sha256": doctor_digest,
        "probe_sha256": file_sha256(probe_path),
        "model_root": str(Path(model_root).resolve()),
        "prompt_token_count": probe["prompt"]["token_count"],
        "q1_ranges": config["q1_ranges"],
        "q2_ranges": config["q2_ranges"],
        "candidate_window": config["candidate_window"],
        "layers": config["layers"],
        "instrumentation_config": str(config_path),
        "instrumentation_config_sha256": file_sha256(config_path),
        "engine": lock["capture"]["engine_kwargs"],
    }
    if dry_run:
        return plan
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_ENABLE"] = "1"
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_CONFIG"] = str(config_path)
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_CONFIG_SHA256"] = file_sha256(config_path)
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_OUTPUT_ROOT"] = str(output)
    os.environ["PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_RUN_ID"] = capture_run_id(doctor_digest, probe["prompt"]["token_ids_sha256"])
    _require(not os.getenv("PUTPOCKET_VLLM_TRUE_PARTIAL_PREFILL_ENABLE"), "CAPTURE_TRUE_PARTIAL_MODE_MUST_BE_OFF")
    _require(not os.getenv("PUTPOCKET_GLM52_FORCED_REUSE_CONTROL"), "CAPTURE_LEGACY_EMULATION_MODE_MUST_BE_OFF")
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    engine = lock["capture"]["engine_kwargs"]
    _require(probe["prompt"]["token_count"] <= engine["max_model_len"], "QUERY_SUM_CAPTURE_PROMPT_TOO_LONG")
    llm = LLM(
        model=str(Path(model_root).resolve()), tokenizer=str(Path(model_root).resolve()),
        tensor_parallel_size=engine["tensor_parallel_size"], dtype=engine["dtype"],
        quantization=engine["quantization"], block_size=engine["block_size"],
        kv_cache_dtype=engine["kv_cache_dtype"], max_model_len=engine["max_model_len"],
        max_num_seqs=engine["max_num_seqs"], enable_prefix_caching=False,
        enable_chunked_prefill=False, enforce_eager=True, cpu_offload_gb=0,
        trust_remote_code=False,
        attention_config={"backend": "FLASHMLA_SPARSE", "sparse_mla_force_mqa": True},
        compilation_config=0, seed=0,
    )
    results = llm.generate(
        [TokensPrompt(prompt_token_ids=probe["prompt"]["token_ids"])],
        SamplingParams(temperature=0.0, max_tokens=1, seed=0), use_tqdm=False,
    )
    _require(len(results) == 1 and results[0].outputs, "QUERY_SUM_CAPTURE_MODEL_OUTPUT_MISSING")
    plan.update(status="captured", generated_token_count=len(results[0].outputs[0].token_ids))
    (output / "query-sum-capture-run.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def capture_run_id(doctor_digest: str, probe_digest: str) -> str:
    _require(_SHA256.fullmatch(doctor_digest) is not None and _SHA256.fullmatch(probe_digest) is not None, "CAPTURE_RUN_DIGEST_INVALID")
    return f"glm52-score-{doctor_digest[:12]}-{probe_digest[:12]}"


def analyze_probe(
    *,
    doctor_report: str | Path,
    capture_root: str | Path,
    output_root: str | Path,
    lock_path: str | Path = PACKAGE_LOCK,
) -> dict[str, Any]:
    _, doctor_digest = load_successful_doctor(doctor_report)
    lock = load_package_lock(lock_path)
    report = analyze_capture(capture_root, output_root, lock, doctor_payload_sha256=doctor_digest)
    validate_schema(report, REPORT_SCHEMA)
    validate_report_digest(report)
    return report


def analyze_query_sum_probe(
    *,
    doctor_report: str | Path,
    probe_path: str | Path,
    capture_root: str | Path,
    output_root: str | Path,
    lock_path: str | Path = PACKAGE_LOCK,
) -> dict[str, Any]:
    _, doctor_digest = load_successful_doctor(doctor_report)
    lock = load_package_lock(lock_path)
    probe = load_json(probe_path)
    report = analyze_query_sum_capture(
        capture_root, output_root, lock, probe,
        doctor_payload_sha256=doctor_digest,
    )
    validate_schema(report, REPO_ROOT / lock["query_sum_capture"]["report_schema"])
    validate_report_digest(report)
    return report
