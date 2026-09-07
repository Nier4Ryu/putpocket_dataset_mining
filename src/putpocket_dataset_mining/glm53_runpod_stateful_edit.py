"""Fail-closed GLM-5.3 stateful-edit-v3 accuracy-ablation tooling.

The runtime represented by this module always computes the complete target
prefill.  A separate, default-OFF vLLM overlay may then replace selected
frozen-history cache rows with private donor snapshots.  Nothing here is a
selective-prefill, latency, skipped-FLOP, or production-serving mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TASK_ID = "T20260907-001__glm53-runpod-sm90"
RUNTIME_PROFILE_ENV = "PUTPOCKET_GLM53_RUNTIME_PROFILE"
MODEL_ID = "Intel/GLM-5.3-Flash-W4A16-AutoRound"
MODEL_REVISION = "5eee1846f0321058ed73745f9aa16f2aaf0fc0a0"
MODEL_ARCHITECTURE = "Glm5NextForConditionalGeneration"
VLLM_COMMIT = "9cd956c7e6cf54efa366b803cafa15ec6c2df827"
RUNTIME_PROFILES = {
    "sm90": {
        "compute_capability": "9.0",
        "attention_backend": "FLASHINFER_MLA_SPARSE_SM90",
        "kv_cache_dtype": "fp8_e4m3",
        "main_cache_row_bytes": 512,
    },
    "sm120": {
        "compute_capability": "12.0",
        "attention_backend": "FLASHINFER_MLA_SPARSE_SM120",
        "kv_cache_dtype": "fp8_ds_mla",
        "main_cache_row_bytes": 656,
    },
}
RUNTIME_PROFILE = os.getenv(RUNTIME_PROFILE_ENV, "sm90")
if RUNTIME_PROFILE not in RUNTIME_PROFILES:
    raise RuntimeError("GLM53_RUNTIME_PROFILE_INVALID")
ATTENTION_BACKEND = RUNTIME_PROFILES[RUNTIME_PROFILE]["attention_backend"]
KV_CACHE_DTYPE = RUNTIME_PROFILES[RUNTIME_PROFILE]["kv_cache_dtype"]
MODEL_LOCK = Path("configs/models/glm53_flash_w4a16_runpod_sm90_sm120.lock.json")
INSTANCE_ID = (
    "instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-"
    "vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
)
DATASET_ID = "ScaleAI/SWE-bench_Pro"
DATASET_REVISION = "7ab5114912baf22bb098818e604c02fe7ad2c11f"
HARNESS_COMMIT = "ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
EDIT_SENTENCE_OLD = (
    "Failure to follow these rules will cause your response to be rejected."
)
EDIT_SENTENCE_NEW = (
    "Failure to follow these rules will cause your response to be accepted."
)
RATIOS = tuple(range(0, 101, 10))
MLA_LAYERS = tuple(range(3, 45, 4))
INDEXER_LAYERS = MLA_LAYERS
KDA_LAYERS = tuple(layer for layer in range(45) if layer not in MLA_LAYERS)
INDEX_KPOOL = 4
UNSAFE_ACK = "I_UNDERSTAND_FULL_TARGET_PREFILL_THEN_DONOR_OVERWRITE"
TOKENIZER_SHA256 = "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
TOKENIZER_CONFIG_SHA256 = (
    "21848631f494177b449db5fab804cfb52e38077e0802dd4a8568978c010c3d3b"
)
CHAT_TEMPLATE_SHA256 = (
    "34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"
)


class StatefulEditContractError(RuntimeError):
    """A frozen episode, selector, control, or result violated the contract."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise StatefulEditContractError(reason)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_digest(tokens: Sequence[int]) -> str:
    return sha256_bytes(json.dumps(list(tokens), separators=(",", ":")).encode())


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_tokens(path: Path) -> list[int]:
    value = load_json(path)
    if isinstance(value, dict):
        value = value.get("prompt", value.get("prompt_token_ids", value.get("token_ids")))
    require(
        isinstance(value, list)
        and bool(value)
        and all(type(item) is int and item >= 0 for item in value),
        "TOKEN_FILE_INVALID",
    )
    return value


def validate_deployment_lock(lock: Mapping[str, Any]) -> dict[str, Any]:
    selection = lock.get("selection", {})
    runtime = lock.get("runtime", {})
    model = lock.get("model_config_contract", {})
    parallel = lock.get("capacity_plan", {}).get("parallelism", {})
    expected = {
        "repository": MODEL_ID,
        "revision": MODEL_REVISION,
        "architecture": MODEL_ARCHITECTURE,
    }
    for key, value in expected.items():
        require(selection.get(key) == value, f"DEPLOYMENT_SELECTION_{key.upper()}_MISMATCH")
    require(runtime.get("vllm_commit") == VLLM_COMMIT, "DEPLOYMENT_VLLM_COMMIT_MISMATCH")
    profile = runtime.get("profiles", {}).get(RUNTIME_PROFILE, {})
    require(profile.get("attention_backend") == ATTENTION_BACKEND, "DEPLOYMENT_ATTENTION_BACKEND_MISMATCH")
    require(profile.get("kv_cache_dtype") == KV_CACHE_DTYPE, "DEPLOYMENT_KV_DTYPE_MISMATCH")
    require(runtime.get("enable_prefix_caching") is False, "DEPLOYMENT_PREFIX_CACHE_MUST_BE_OFF")
    require(runtime.get("enable_mtp") is False, "DEPLOYMENT_MTP_MUST_BE_OFF")
    require(model.get("num_hidden_layers") == 45, "DEPLOYMENT_LAYER_COUNT_MISMATCH")
    require(model.get("index_kpool") == INDEX_KPOOL, "DEPLOYMENT_INDEX_KPOOL_MISMATCH")
    require(model.get("qk_rope_head_dim") == 0, "DEPLOYMENT_NOPE_CONTRACT_MISMATCH")
    require(model.get("kv_lora_rank") == 512, "DEPLOYMENT_KV_LORA_RANK_MISMATCH")
    require(parallel == {
        "tensor_parallel_size": 4,
        "data_parallel_size": 1,
        "expert_parallel": True,
        "expert_parallel_size": 4,
        "enable_ep_weight_filter": True,
        "pipeline_parallel_size": 1,
    }, "DEPLOYMENT_PARALLELISM_MISMATCH")
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "architecture": MODEL_ARCHITECTURE,
        "vllm_commit": VLLM_COMMIT,
        "mla_layers": list(MLA_LAYERS),
        "kda_layers": list(KDA_LAYERS),
        "indexer_layers": list(INDEXER_LAYERS),
        "index_kpool": INDEX_KPOOL,
    }


def validate_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    text = config.get("text_config", config)
    layer_types = text.get("layer_types")
    require(config.get("model_type") == "glm5_next", "MODEL_WRAPPER_TYPE_MISMATCH")
    require(text.get("model_type") == "glm5_next_text", "MODEL_TEXT_TYPE_MISMATCH")
    require(text.get("num_hidden_layers") == 45, "MODEL_LAYER_COUNT_MISMATCH")
    require(text.get("num_attention_heads") == 64, "MODEL_HEAD_COUNT_MISMATCH")
    require(text.get("index_topk") == 2048, "MODEL_INDEX_TOPK_MISMATCH")
    require(text.get("index_kpool") == INDEX_KPOOL, "MODEL_INDEX_KPOOL_MISMATCH")
    require(text.get("index_n_heads") == 32, "MODEL_INDEX_HEADS_MISMATCH")
    require(text.get("index_head_dim") == 128, "MODEL_INDEX_HEAD_DIM_MISMATCH")
    require(text.get("qk_rope_head_dim") == 0, "MODEL_NOPE_MISMATCH")
    require(text.get("mla_use_nope") is True, "MODEL_MLA_NOPE_MISMATCH")
    require(text.get("kv_lora_rank") == 512, "MODEL_KV_LORA_RANK_MISMATCH")
    require(isinstance(layer_types, list) and len(layer_types) == 45, "MODEL_LAYER_TYPES_INVALID")
    actual_mla = tuple(
        index for index, value in enumerate(layer_types) if value == "deepseek_sparse_attention"
    )
    actual_kda = tuple(
        index for index, value in enumerate(layer_types) if value == "linear_attention"
    )
    require(actual_mla == MLA_LAYERS, "MODEL_MLA_LAYER_MAPPING_MISMATCH")
    require(actual_kda == KDA_LAYERS, "MODEL_KDA_LAYER_MAPPING_MISMATCH")
    require(set(layer_types) == {"linear_attention", "deepseek_sparse_attention"}, "MODEL_LAYER_TYPE_UNSUPPORTED")
    return {
        "mla_layers": list(actual_mla),
        "kda_layers": list(actual_kda),
        "indexer_layers": list(actual_mla),
        "indexer_pool_size": INDEX_KPOOL,
        "runtime_profile": RUNTIME_PROFILE,
        "main_cache_row_bytes": RUNTIME_PROFILES[RUNTIME_PROFILE]["main_cache_row_bytes"],
        "indexer_cache_row_bytes": 132,
        "kda_cache_semantics": "final_recurrent_and_conv_state_not_token_addressable",
    }


def complete_indexer_pool_ends(
    selected_positions: Sequence[int],
    *,
    eligible_start: int,
    eligible_end: int,
    pool_size: int = INDEX_KPOOL,
) -> tuple[int, ...]:
    """Project selected token rows to exactly representable complete pools.

    A GLM-5.3 indexer cache entry is a learned compression of four consecutive
    token rows and exists only at the pool-end slot.  A pool may therefore be
    copied only if all its members were selected and all are inside frozen
    history.  Boundary or partially selected pools stay target-computed.
    """

    require(pool_size > 1, "INDEXER_POOL_SIZE_INVALID")
    selected = tuple(int(value) for value in selected_positions)
    require(len(selected) == len(set(selected)), "SELECTED_POSITION_DUPLICATE")
    require(tuple(sorted(selected)) == selected, "SELECTED_POSITIONS_NOT_SORTED")
    require(
        all(eligible_start <= value < eligible_end for value in selected),
        "SELECTED_POSITION_OUT_OF_RANGE",
    )
    selected_set = set(selected)
    ends: list[int] = []
    first_end = ((eligible_start + pool_size - 1) // pool_size) * pool_size - 1
    for end in range(first_end, eligible_end, pool_size):
        start = end - pool_size + 1
        if start < eligible_start or end >= eligible_end:
            continue
        if all(position in selected_set for position in range(start, end + 1)):
            ends.append(end)
    return tuple(ends)


def pool_covered_positions(pool_ends: Sequence[int], pool_size: int = INDEX_KPOOL) -> tuple[int, ...]:
    values = [position for end in pool_ends for position in range(end - pool_size + 1, end + 1)]
    return tuple(values)


def _validate_sha(value: Any, reason: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None, reason)
    return value


def validate_base_selector(
    selector: Mapping[str, Any],
    *,
    donor_initial_tokens: Sequence[int],
    edited_initial_tokens: Sequence[int],
    edit_position: int,
) -> tuple[list[int], dict[int, Any]]:
    require(selector.get("schema_version") == 1, "BASE_SELECTOR_SCHEMA_INVALID")
    require(selector.get("status") == "attested_before_benchmark_outcomes", "BASE_SELECTOR_NOT_PREATTESTED")
    model = selector.get("model", {})
    require(model.get("id") == MODEL_ID and model.get("revision") == MODEL_REVISION, "BASE_SELECTOR_MODEL_MISMATCH")
    require(model.get("vllm_commit") == VLLM_COMMIT, "BASE_SELECTOR_VLLM_MISMATCH")
    require(model.get("indexer_kind") == "glm53_native_raw_pre_topk", "BASE_SELECTOR_INDEXER_KIND_INVALID")
    require(model.get("tokenizer_sha256") == TOKENIZER_SHA256, "BASE_SELECTOR_TOKENIZER_MISMATCH")
    prompt = selector.get("prompt", {})
    require(prompt.get("donor_token_ids_sha256") == token_digest(donor_initial_tokens), "BASE_SELECTOR_DONOR_DIGEST_MISMATCH")
    require(prompt.get("edited_token_ids_sha256") == token_digest(edited_initial_tokens), "BASE_SELECTOR_EDITED_DIGEST_MISMATCH")
    require(prompt.get("edit_positions") == [edit_position], "BASE_SELECTOR_EDIT_POSITION_MISMATCH")
    require(prompt.get("token_count") == len(donor_initial_tokens), "BASE_SELECTOR_TOKEN_COUNT_MISMATCH")
    definition = selector.get("selection_definition", {})
    require(
        definition.get("algorithm")
        == "lexicographic_prefill_only_layer_stability_v2_glm53_kpool",
        "BASE_SELECTOR_ALGORITHM_INVALID",
    )
    require(
        definition.get("query_sample") == "prefill_last_query",
        "BASE_SELECTOR_QUERY_SAMPLE_INVALID",
    )
    require(
        definition.get("pool_score_expansion")
        == "assign_each_native_compressed_pool_score_to_its_four_constituent_token_positions",
        "BASE_SELECTOR_POOL_EXPANSION_INVALID",
    )
    source = selector.get("source_evidence", {})
    require(source.get("outcome_independent") is True, "BASE_SELECTOR_OUTCOME_LEAKAGE_UNATTESTED")
    require(source.get("benchmark_outcomes_read") is False, "BASE_SELECTOR_OUTCOME_LEAKAGE")
    require(source.get("native_raw_pre_topk_scores") is True, "BASE_SELECTOR_NOT_NATIVE_RAW_PRE_TOPK")
    require(source.get("layers") == list(INDEXER_LAYERS), "BASE_SELECTOR_LAYER_COVERAGE_INVALID")
    require(source.get("data_parallel_rank") == 0, "BASE_SELECTOR_DP_RANK_INVALID")
    ranking = selector.get("ranking")
    expected = set(range(edit_position + 1, len(donor_initial_tokens)))
    require(isinstance(ranking, list) and len(ranking) == len(expected) and set(ranking) == expected, "BASE_SELECTOR_RANKING_NOT_PERMUTATION")
    metrics = selector.get("token_metrics", {})
    require(isinstance(metrics, dict), "BASE_SELECTOR_METRICS_INVALID")
    return list(ranking), {int(key): value for key, value in metrics.items()}


def _finite(values: Sequence[Any]) -> bool:
    return all(
        isinstance(value, (int, float)) and math.isfinite(float(value))
        for value in values
    )


def _percentile(sorted_values: Sequence[float], percentile: float) -> float:
    require(bool(sorted_values), "PERCENTILE_VECTOR_EMPTY")
    coordinate = (len(sorted_values) - 1) * percentile
    lower = int(math.floor(coordinate))
    upper = int(math.ceil(coordinate))
    if lower == upper:
        return float(sorted_values[lower])
    weight = coordinate - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(
        sorted_values[upper]
    ) * weight


def _descending_ranks(values: Sequence[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    ranks = [0] * len(values)
    for rank, index in enumerate(order):
        ranks[index] = rank
    return ranks


def _load_capture_records(
    root: Path,
    *,
    prompt_side: str,
    prompt_digest: str,
    token_count: int,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    require(root.is_absolute() and root.is_dir(), "BASE_CAPTURE_ROOT_INVALID")
    require(not list(root.glob("BLOCKED*")), "BASE_CAPTURE_HAS_BLOCKED_MARKER")
    records: dict[int, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    for path in sorted(root.glob(f"base-selector.{prompt_side}.layer-*.json")):
        payload = load_json(path)
        recorded_digest = payload.pop("record_sha256", None)
        encoded = (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode()
        require(recorded_digest == sha256_bytes(encoded), "BASE_CAPTURE_RECORD_DIGEST_INVALID")
        layer = payload.get("layer")
        require(type(layer) is int and layer in INDEXER_LAYERS, "BASE_CAPTURE_LAYER_INVALID")
        require(layer not in records, "BASE_CAPTURE_DUPLICATE_LAYER")
        require(payload.get("schema_version") == 1, "BASE_CAPTURE_SCHEMA_INVALID")
        require(payload.get("capture_kind") == "glm53_native_raw_pre_topk_kpool", "BASE_CAPTURE_KIND_INVALID")
        require(payload.get("prompt_side") == prompt_side, "BASE_CAPTURE_SIDE_INVALID")
        require(
            payload.get("model")
            == {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "architecture": MODEL_ARCHITECTURE,
                "vllm_commit": VLLM_COMMIT,
            },
            "BASE_CAPTURE_MODEL_INVALID",
        )
        require(payload.get("prompt_token_ids_sha256") == prompt_digest, "BASE_CAPTURE_PROMPT_DIGEST_INVALID")
        require(payload.get("token_count") == token_count, "BASE_CAPTURE_TOKEN_COUNT_INVALID")
        require(payload.get("query_position") == token_count - 1, "BASE_CAPTURE_QUERY_POSITION_INVALID")
        require(payload.get("index_kpool") == INDEX_KPOOL, "BASE_CAPTURE_KPOOL_INVALID")
        require(payload.get("raw_score_kind") == "native_pre_topk_pre_normalization", "BASE_CAPTURE_SCORE_KIND_INVALID")
        require(payload.get("data_parallel_rank") == 0, "BASE_CAPTURE_DP_RANK_INVALID")
        require(payload.get("tp_rank") == 0, "BASE_CAPTURE_TP_RANK_INVALID")
        require(
            payload.get("native_scale")
            == "fp8_fp4_mqa_logits_output_before_top_k_per_row_prefill",
            "BASE_CAPTURE_NATIVE_SCALE_INVALID",
        )
        require(
            payload.get("head_aggregation")
            == "native_lightning_indexer_weights_fused_by_vllm",
            "BASE_CAPTURE_HEAD_AGGREGATION_INVALID",
        )
        pool_ids = payload.get("pool_ids")
        scores = payload.get("raw_pool_scores")
        expected_pool_count = token_count // INDEX_KPOOL
        require(pool_ids == list(range(expected_pool_count)), "BASE_CAPTURE_POOL_COVERAGE_INVALID")
        require(isinstance(scores, list) and len(scores) == expected_pool_count and _finite(scores), "BASE_CAPTURE_RAW_SCORES_INVALID")
        payload["record_sha256"] = recorded_digest
        records[layer] = payload
        files.append({"path": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)})
    require(set(records) == set(INDEXER_LAYERS), "BASE_CAPTURE_LAYER_COVERAGE_INCOMPLETE")
    return records, files


def freeze_base_selector_payload(
    *,
    donor_tokens: Sequence[int],
    edited_tokens: Sequence[int],
    donor_root: Path,
    edited_root: Path,
) -> dict[str, Any]:
    """Freeze the historical stability selector from GLM-5.3 pool logits."""

    require(len(donor_tokens) == len(edited_tokens) and len(donor_tokens) > INDEX_KPOOL, "BASE_SELECTOR_TOKEN_LENGTH_INVALID")
    differences = [
        index
        for index, pair in enumerate(zip(donor_tokens, edited_tokens, strict=True))
        if pair[0] != pair[1]
    ]
    require(len(differences) == 1, "BASE_SELECTOR_EDIT_NOT_SINGLE_TOKEN")
    edit_position = differences[0]
    donor_digest = token_digest(donor_tokens)
    edited_digest = token_digest(edited_tokens)
    donor, donor_files = _load_capture_records(
        donor_root,
        prompt_side="donor",
        prompt_digest=donor_digest,
        token_count=len(donor_tokens),
    )
    edited, edited_files = _load_capture_records(
        edited_root,
        prompt_side="edited",
        prompt_digest=edited_digest,
        token_count=len(edited_tokens),
    )

    eligible = tuple(range(edit_position + 1, len(donor_tokens)))
    scored_end = len(donor_tokens) // INDEX_KPOOL * INDEX_KPOOL
    scored_positions = tuple(position for position in eligible if position < scored_end)
    unscored_tail = tuple(position for position in eligible if position >= scored_end)
    aggregate: dict[int, dict[str, Any]] = {
        position: {
            "normalized_delta": [],
            "rank_shift": [],
            "cutoff_margin_retention": [],
            "membership_stable": 0,
            "membership_both": 0,
        }
        for position in scored_positions
    }
    for layer in INDEXER_LAYERS:
        before_pool = [float(value) for value in donor[layer]["raw_pool_scores"]]
        after_pool = [float(value) for value in edited[layer]["raw_pool_scores"]]
        before = [score for score in before_pool for _ in range(INDEX_KPOOL)]
        after = [score for score in after_pool for _ in range(INDEX_KPOOL)]
        require(len(before) == scored_end == len(after), "BASE_SELECTOR_EXPANDED_SCORE_LENGTH_INVALID")
        ordered = sorted(before)
        iqr = max(_percentile(ordered, 0.75) - _percentile(ordered, 0.25), 1e-12)
        before_ranks = _descending_ranks(before)
        after_ranks = _descending_ranks(after)
        select_pools = min(2048 // INDEX_KPOOL, len(before_pool))
        before_selected_pools = set(
            sorted(range(len(before_pool)), key=lambda value: (-before_pool[value], value))[:select_pools]
        )
        after_selected_pools = set(
            sorted(range(len(after_pool)), key=lambda value: (-after_pool[value], value))[:select_pools]
        )
        before_cutoff = min((before_pool[index] for index in before_selected_pools), default=min(before_pool))
        after_cutoff = min((after_pool[index] for index in after_selected_pools), default=min(after_pool))
        for position in scored_positions:
            entry = aggregate[position]
            pool = position // INDEX_KPOOL
            in_before = pool in before_selected_pools
            in_after = pool in after_selected_pools
            entry["normalized_delta"].append(abs(before[position] - after[position]) / iqr)
            entry["rank_shift"].append(float(abs(before_ranks[position] - after_ranks[position])))
            entry["cutoff_margin_retention"].append(
                min(
                    (before[position] - before_cutoff) / iqr,
                    (after[position] - after_cutoff) / iqr,
                )
            )
            entry["membership_stable"] += int(in_before == in_after)
            entry["membership_both"] += int(in_before and in_after)

    metrics: dict[int, dict[str, Any]] = {}
    for position in scored_positions:
        values = aggregate[position]
        deltas = sorted(values["normalized_delta"])
        shifts = sorted(values["rank_shift"])
        margins = sorted(values["cutoff_margin_retention"])
        metrics[position] = {
            "source": "glm53_native_compressed_pool_score_expanded_to_four_tokens",
            "membership_stable_count": values["membership_stable"],
            "membership_both_count": values["membership_both"],
            "max_iqr_normalized_delta": max(deltas),
            "p95_iqr_normalized_delta": _percentile(deltas, 0.95),
            "max_rank_shift": int(max(shifts)),
            "p95_rank_shift": _percentile(shifts, 0.95),
            "min_cutoff_margin_retention": min(margins),
        }
    for position in unscored_tail:
        metrics[position] = {
            "source": "native_kpool_incomplete_tail_has_no_pool_logit",
            "deterministic_tail_order": position,
        }

    ranking = sorted(
        eligible,
        key=lambda position: (
            position in unscored_tail,
            -int(metrics[position].get("membership_stable_count", 0)),
            -int(metrics[position].get("membership_both_count", 0)),
            float(metrics[position].get("max_iqr_normalized_delta", math.inf)),
            float(metrics[position].get("p95_iqr_normalized_delta", math.inf)),
            float(metrics[position].get("max_rank_shift", math.inf)),
            float(metrics[position].get("p95_rank_shift", math.inf)),
            -float(metrics[position].get("min_cutoff_margin_retention", -math.inf)),
            position,
        ),
    )
    return {
        "schema_version": 1,
        "status": "attested_before_benchmark_outcomes",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "vllm_commit": VLLM_COMMIT,
            "indexer_kind": "glm53_native_raw_pre_topk",
            "tokenizer_sha256": TOKENIZER_SHA256,
        },
        "prompt": {
            "donor_token_ids_sha256": donor_digest,
            "edited_token_ids_sha256": edited_digest,
            "edit_positions": [edit_position],
            "token_count": len(donor_tokens),
        },
        "selection_definition": {
            "algorithm": "lexicographic_prefill_only_layer_stability_v2_glm53_kpool",
            "query_sample": "prefill_last_query",
            "pool_score_expansion": "assign_each_native_compressed_pool_score_to_its_four_constituent_token_positions",
            "incomplete_tail_policy": "unscored_tail_positions_sorted_last_by_absolute_position",
            "fields_in_order": [
                "scored_before_unscored_tail",
                "membership_stable_count_desc",
                "membership_both_count_desc",
                "max_iqr_normalized_delta_asc",
                "p95_iqr_normalized_delta_asc",
                "max_rank_shift_asc",
                "p95_rank_shift_asc",
                "min_cutoff_margin_retention_desc",
                "absolute_position_asc_tie_break",
            ],
        },
        "source_evidence": {
            "outcome_independent": True,
            "benchmark_outcomes_read": False,
            "native_raw_pre_topk_scores": True,
            "layers": list(INDEXER_LAYERS),
            "data_parallel_rank": 0,
            "donor_files": donor_files,
            "edited_files": edited_files,
        },
        "ranking": ranking,
        "token_metrics": {str(position): metrics[position] for position in eligible},
    }


def freeze_base_selector(args: argparse.Namespace) -> None:
    payload = freeze_base_selector_payload(
        donor_tokens=load_tokens(args.donor_tokens.resolve()),
        edited_tokens=load_tokens(args.edited_tokens.resolve()),
        donor_root=args.donor_root.resolve(),
        edited_root=args.edited_root.resolve(),
    )
    output = args.output.resolve()
    require(not output.exists(), "BASE_SELECTOR_OUTPUT_ALREADY_EXISTS")
    atomic_json(output, payload)


def write_selector_capture_control(args: argparse.Namespace) -> None:
    tokens = load_tokens(args.tokens.resolve())
    require(len(tokens) > 2048, "SELECTOR_CAPTURE_REQUIRES_MORE_THAN_INDEX_TOPK_TOKENS")
    evidence = args.evidence_dir.resolve()
    output = args.output.resolve()
    require(evidence.is_absolute(), "SELECTOR_CAPTURE_EVIDENCE_DIR_INVALID")
    require(not output.exists(), "SELECTOR_CAPTURE_CONTROL_ALREADY_EXISTS")
    require(not evidence.exists() or not any(evidence.iterdir()), "SELECTOR_CAPTURE_EVIDENCE_NOT_EMPTY")
    atomic_json(
        output,
        {
            "schema_version": 1,
            "mode": "CAPTURE",
            "prompt_side": args.prompt_side,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "architecture": MODEL_ARCHITECTURE,
                "vllm_commit": VLLM_COMMIT,
                "attention_backend": ATTENTION_BACKEND,
                "kv_cache_dtype": KV_CACHE_DTYPE,
                "index_kpool": INDEX_KPOOL,
            },
            "token_count": len(tokens),
            "query_position": len(tokens) - 1,
            "prompt_token_ids_sha256": token_digest(tokens),
            "layers": list(INDEXER_LAYERS),
            "data_parallel_rank": 0,
            "raw_score_kind": "native_pre_topk_pre_normalization",
            "outcome_independent": True,
            "benchmark_outcomes_read": False,
            "evidence_dir": str(evidence),
        },
    )


def _chat_ids(tokenizer: Any, messages: list[dict[str, Any]], *, generation: bool) -> list[int]:
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation,
        return_dict=False,
    )
    require(isinstance(ids, list) and all(type(item) is int for item in ids), "CHAT_TEMPLATE_IDS_INVALID")
    return ids


def _extend_ranking(base_ranking: Sequence[int], base_count: int, history_count: int) -> list[int]:
    scored: list[tuple[float, int, int]] = []
    for ordinal, position in enumerate(base_ranking):
        scored.append(((ordinal + 0.5) / len(base_ranking), 0, position))
    for position in range(base_count, history_count):
        raw = hashlib.sha256(f"glm53-stateful-edit-v3:{position}".encode()).digest()[:8]
        scored.append((int.from_bytes(raw, "big") / 2**64, 1, position))
    return [position for _, _, position in sorted(scored)]


def build_episode_payloads(
    *,
    tokenizer: Any,
    capture: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    base_selector: Mapping[str, Any],
    source_capture_sha256: str,
    source_trajectory_sha256: str,
    base_selector_sha256: str,
) -> dict[str, Any]:
    request = capture.get("request")
    response = capture.get("response")
    require(capture.get("http_status") == 200 and isinstance(request, dict) and isinstance(response, dict), "EPISODE_CAPTURE_INVALID")
    request_messages = request.get("messages")
    require(isinstance(request_messages, list) and len(request_messages) == 2, "DONOR_TURN1_REQUEST_INVALID")
    require([item.get("role") for item in request_messages] == ["system", "user"], "DONOR_TURN1_ROLES_INVALID")
    messages = trajectory.get("messages")
    require(isinstance(messages, list) and len(messages) >= 4, "EPISODE_TRAJECTORY_TOO_SHORT")
    frozen = messages[:4]
    require([item.get("role") for item in frozen] == ["system", "user", "assistant", "user"], "EPISODE_MESSAGE_ROLES_INVALID")
    require(frozen[:2] == request_messages, "TRAJECTORY_REQUEST_DIVERGENCE")
    choices = response.get("choices", [])
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0].get("message"), dict), "EPISODE_RESPONSE_INVALID")
    require(choices[0]["message"].get("content") == frozen[2].get("content"), "FROZEN_A1_RESPONSE_DIVERGENCE")
    require(isinstance(frozen[3].get("content"), str) and bool(frozen[3]["content"]), "A1_WAS_NOT_EXECUTED")
    actions = re.findall(r"```bash\s*\n(.*?)\n```", str(frozen[2].get("content", "")), re.DOTALL)
    require(len(actions) == 1, "A1_NOT_EXACTLY_ONE_BASH_ACTION")

    old_system = frozen[0].get("content")
    require(isinstance(old_system, str), "OLD_SYSTEM_TEXT_INVALID")
    require(old_system.count(EDIT_SENTENCE_OLD) == 1 and EDIT_SENTENCE_NEW not in old_system, "OLD_SYSTEM_EDIT_SITE_INVALID")
    new_system = old_system.replace(EDIT_SENTENCE_OLD, EDIT_SENTENCE_NEW)
    donor_initial = _chat_ids(tokenizer, frozen[:2], generation=True)
    edited_initial_messages = [dict(frozen[0], content=new_system), frozen[1]]
    edited_initial = _chat_ids(tokenizer, edited_initial_messages, generation=True)
    require(len(donor_initial) == len(edited_initial), "INITIAL_PROMPT_TOKEN_COUNT_CHANGED")
    initial_differences = [
        index
        for index, pair in enumerate(zip(donor_initial, edited_initial, strict=True))
        if pair[0] != pair[1]
    ]
    require(len(initial_differences) == 1, "INITIAL_PROMPT_EDIT_NOT_SINGLE_TOKEN")
    edit_position = initial_differences[0]

    donor_history_messages = frozen[:3]
    edited_history_messages = [dict(item) for item in frozen[:3]]
    edited_history_messages[0] = {**edited_history_messages[0], "content": new_system}
    donor_history = _chat_ids(tokenizer, donor_history_messages, generation=False)
    edited_history = _chat_ids(tokenizer, edited_history_messages, generation=False)
    require(len(donor_history) == len(edited_history) and len(donor_history) > len(donor_initial), "STATEFUL_HISTORY_LENGTH_INVALID")
    history_differences = [
        index
        for index, pair in enumerate(zip(donor_history, edited_history, strict=True))
        if pair[0] != pair[1]
    ]
    require(history_differences == [edit_position], "STATEFUL_HISTORY_EDIT_MISMATCH")
    require(donor_history[: len(donor_initial)] == donor_initial, "DONOR_HISTORY_PREFIX_INVALID")
    require(edited_history[: len(edited_initial)] == edited_initial, "EDITED_HISTORY_PREFIX_INVALID")
    target_ids = _chat_ids(tokenizer, edited_history_messages + [frozen[3]], generation=True)
    require(target_ids[: len(edited_history)] == edited_history and len(target_ids) > len(edited_history), "Q2_NOT_STRICT_POST_HISTORY_EXTENSION")

    base_ranking, base_metrics = validate_base_selector(
        base_selector,
        donor_initial_tokens=donor_initial,
        edited_initial_tokens=edited_initial,
        edit_position=edit_position,
    )
    eligible_start = edit_position + 1
    history_count = len(donor_history)
    ranking = _extend_ranking(base_ranking, len(donor_initial), history_count)
    expected = set(range(eligible_start, history_count))
    require(len(ranking) == len(expected) and set(ranking) == expected, "STATEFUL_SELECTOR_NOT_PERMUTATION")
    by_ratio: dict[str, list[int]] = {}
    indexer_by_ratio: dict[str, list[int]] = {}
    for ratio in RATIOS:
        selected = sorted(ranking[: len(ranking) * ratio // 100])
        by_ratio[str(ratio)] = selected
        indexer_by_ratio[str(ratio)] = list(
            complete_indexer_pool_ends(
                selected,
                eligible_start=eligible_start,
                eligible_end=history_count,
            )
        )

    selector = {
        "schema_version": 3,
        "status": "attested_before_benchmark_outcomes",
        "unsafe_research_ablation": True,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "architecture": MODEL_ARCHITECTURE,
            "vllm_commit": VLLM_COMMIT,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        },
        "scenario": {
            "name": "glm53_single_instance_stateful_mid_trajectory_equal_token_edit_v3",
            "base_prompt_token_count": len(donor_initial),
            "history_token_count": history_count,
            "edit_positions": [edit_position],
            "old_token_ids": [donor_history[edit_position]],
            "new_token_ids": [edited_history[edit_position]],
            "eligible_history_range": [eligible_start, history_count],
            "same_length": True,
            "absolute_positions_unchanged": True,
            "q2_and_post_edit_extensions_recomputed": True,
        },
        "selection_definition": {
            "requested_ratio_denominator": len(ranking),
            "algorithm": "glm53_native_base_indexer_rank_interleaved_with_frozen_a1_hash_quantiles_v1",
            "base_selector_sha256": base_selector_sha256,
            "a1_ranking_is_outcome_independent": True,
            "higher_ranked_positions_are_donor_overwrite_rows": True,
            "indexer_projection": "copy_only_complete_selected_4_token_pools; partial_and_boundary_pools_remain_target_computed",
        },
        "source_evidence": {
            "donor_history_token_ids_sha256": token_digest(donor_history),
            "edited_history_token_ids_sha256": token_digest(edited_history),
            "base_selector_sha256": base_selector_sha256,
            "benchmark_outcomes_read": False,
        },
        "ranking": ranking,
        "positions_by_ratio": by_ratio,
        "indexer_pool_end_positions_by_ratio": indexer_by_ratio,
        "token_metrics": {
            **{
                str(position): base_metrics.get(position, {"source": "glm53_native_base_indexer"})
                for position in range(eligible_start, len(donor_initial))
            },
            **{
                str(position): {"source": "frozen_a1_deterministic_hash"}
                for position in range(len(donor_initial), history_count)
            },
        },
    }
    episode = {
        "schema_version": 3,
        "status": "frozen_before_ratio_outcomes",
        "artifact_kind": "frozen_per_instance_episode",
        "instance_id": INSTANCE_ID,
        "benchmark": {
            "dataset": DATASET_ID,
            "revision": DATASET_REVISION,
            "split": "test",
            "harness_commit": HARNESS_COMMIT,
            "native_components": ["problem_row", "repository_container_mapping", "official_evaluator"],
            "project_authored_components": ["A1_execute_Q2_freeze", "system_edit", "selector", "ratio_sweep"],
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "architecture": MODEL_ARCHITECTURE,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
        },
        "scenario": "SYS_old+Q1 -> frozen A1/action/Q2; edit SYS_old->SYS_new; SYS_new+Q1+A1+Q2 -> continuation",
        "system_edit": {
            "old_text": EDIT_SENTENCE_OLD,
            "new_text": EDIT_SENTENCE_NEW,
            "edit_positions": [edit_position],
            "old_token_ids": [donor_history[edit_position]],
            "new_token_ids": [edited_history[edit_position]],
            "equal_token_count": True,
        },
        "donor_turn1_messages": frozen[:2],
        "frozen_a1_message": frozen[2],
        "frozen_q2_message": frozen[3],
        "frozen_backend_response": response,
        "frozen_action_sha256": sha256_bytes(actions[0].encode()),
        "frozen_q2_sha256": sha256_bytes(frozen[3]["content"].encode()),
        "base_prompt_token_count": len(donor_initial),
        "history_token_count": history_count,
        "eligible_start": eligible_start,
        "eligible_end": history_count,
        "donor_history_token_ids_sha256": token_digest(donor_history),
        "edited_history_token_ids_sha256": token_digest(edited_history),
        "first_post_edit_prompt_token_count": len(target_ids),
        "q2_starts_at_or_after": history_count,
        "first_post_edit_prompt_prefix_attested": True,
        "source_capture_sha256": source_capture_sha256,
        "source_trajectory_sha256": source_trajectory_sha256,
        "outcome_leakage_policy": "freeze_episode_and_selector_before_reading_any_evaluator_outcome",
    }
    return {
        "episode": episode,
        "selector": selector,
        "donor_history_tokens": donor_history,
        "edited_history_tokens": edited_history,
    }


def build_episode(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    model_root = args.model_root.resolve()
    for relative, expected in (
        ("tokenizer.json", TOKENIZER_SHA256),
        ("tokenizer_config.json", TOKENIZER_CONFIG_SHA256),
        ("chat_template.jinja", CHAT_TEMPLATE_SHA256),
    ):
        path = model_root / relative
        require(path.is_file() and sha256_file(path) == expected, f"MODEL_ROOT_{relative.upper().replace('.', '_')}_MISMATCH")
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    capture_path = args.capture.resolve()
    trajectory_path = args.trajectory.resolve()
    selector_path = args.base_selector.resolve()
    payloads = build_episode_payloads(
        tokenizer=tokenizer,
        capture=load_json(capture_path),
        trajectory=load_json(trajectory_path),
        base_selector=load_json(selector_path),
        source_capture_sha256=sha256_file(capture_path),
        source_trajectory_sha256=sha256_file(trajectory_path),
        base_selector_sha256=sha256_file(selector_path),
    )
    output = args.output_root.resolve()
    require(not output.exists(), "EPISODE_OUTPUT_ROOT_ALREADY_EXISTS")
    atomic_json(output / "episode.json", payloads["episode"])
    atomic_json(output / "selector" / "selector.json", payloads["selector"])
    atomic_json(output / "donor-history-token-ids.json", {"prompt": payloads["donor_history_tokens"]})
    atomic_json(output / "edited-history-token-ids.json", {"prompt": payloads["edited_history_tokens"]})


def serialize_base_prompts(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    model_root = args.model_root.resolve()
    for relative, expected in (
        ("tokenizer.json", TOKENIZER_SHA256),
        ("tokenizer_config.json", TOKENIZER_CONFIG_SHA256),
        ("chat_template.jinja", CHAT_TEMPLATE_SHA256),
    ):
        path = model_root / relative
        require(path.is_file() and sha256_file(path) == expected, f"MODEL_ROOT_{relative.upper().replace('.', '_')}_MISMATCH")
    capture_path = args.capture.resolve()
    capture = load_json(capture_path)
    messages = capture.get("request", {}).get("messages")
    require(
        capture.get("http_status") == 200
        and isinstance(messages, list)
        and len(messages) == 2
        and [item.get("role") for item in messages] == ["system", "user"],
        "BASE_PROMPT_CAPTURE_INVALID",
    )
    old_system = messages[0].get("content")
    require(
        isinstance(old_system, str)
        and old_system.count(EDIT_SENTENCE_OLD) == 1
        and EDIT_SENTENCE_NEW not in old_system,
        "BASE_PROMPT_OLD_SYSTEM_INVALID",
    )
    edited_messages = [dict(item) for item in messages]
    edited_messages[0]["content"] = old_system.replace(
        EDIT_SENTENCE_OLD, EDIT_SENTENCE_NEW
    )
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    donor = _chat_ids(tokenizer, messages, generation=True)
    edited = _chat_ids(tokenizer, edited_messages, generation=True)
    differences = [
        index
        for index, pair in enumerate(zip(donor, edited, strict=True))
        if pair[0] != pair[1]
    ]
    require(len(donor) == len(edited) and len(differences) == 1, "BASE_PROMPT_EDIT_NOT_ONE_EQUAL_LENGTH_TOKEN")
    output = args.output_root.resolve()
    require(not output.exists(), "BASE_PROMPT_OUTPUT_ROOT_ALREADY_EXISTS")
    atomic_json(output / "donor-initial-token-ids.json", {"prompt": donor})
    atomic_json(output / "edited-initial-token-ids.json", {"prompt": edited})
    atomic_json(
        output / "base-prompt-manifest.json",
        {
            "schema_version": 1,
            "status": "frozen_before_benchmark_outcomes",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "capture_sha256": sha256_file(capture_path),
            "donor_token_ids_sha256": token_digest(donor),
            "edited_token_ids_sha256": token_digest(edited),
            "token_count": len(donor),
            "edit_positions": differences,
            "old_token_ids": [donor[differences[0]]],
            "new_token_ids": [edited[differences[0]]],
            "outcome_independent": True,
        },
    )


def configure_agent(args: argparse.Namespace) -> None:
    import yaml

    value = yaml.safe_load(args.input.resolve().read_text(encoding="utf-8"))
    system = value["agent"]["system_template"]
    require(system.count(EDIT_SENTENCE_OLD) == 1 and EDIT_SENTENCE_NEW not in system, "DONOR_SYSTEM_TEXT_INVALID")
    value["model"]["model_kwargs"]["api_base"] = args.api_base
    value["model"]["model_kwargs"]["temperature"] = 0.0
    value["agent"]["step_limit"] = int(args.step_limit)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(output)


def write_control(args: argparse.Namespace) -> None:
    donor = load_tokens(args.donor_history.resolve())
    edited = load_tokens(args.edited_history.resolve())
    episode = load_json(args.episode.resolve())
    selector_path = args.selector.resolve()
    selector = load_json(selector_path)
    require(episode.get("status") == "frozen_before_ratio_outcomes", "CONTROL_EPISODE_NOT_FROZEN")
    require(episode.get("instance_id") == INSTANCE_ID, "CONTROL_INSTANCE_MISMATCH")
    require(len(donor) == len(edited) == episode.get("history_token_count"), "CONTROL_HISTORY_LENGTH_INVALID")
    differences = [index for index, pair in enumerate(zip(donor, edited, strict=True)) if pair[0] != pair[1]]
    require(differences == episode.get("system_edit", {}).get("edit_positions"), "CONTROL_HISTORY_DIFF_INVALID")
    ratio = int(args.ratio)
    require(ratio in RATIOS and args.mode in {"OFF", "SNAPSHOT", "TRANSPLANT"}, "CONTROL_MODE_OR_RATIO_INVALID")
    require(selector.get("schema_version") == 3 and selector.get("status") == "attested_before_benchmark_outcomes", "CONTROL_SELECTOR_INVALID")
    scenario = selector.get("scenario", {})
    history_count = len(donor)
    eligible_start = int(episode["eligible_start"])
    eligible_end = int(episode["eligible_end"])
    require(scenario.get("history_token_count") == history_count, "CONTROL_SELECTOR_HISTORY_MISMATCH")
    require(scenario.get("eligible_history_range") == [eligible_start, eligible_end], "CONTROL_SELECTOR_RANGE_MISMATCH")
    ranking = selector.get("ranking", [])
    expected = set(range(eligible_start, eligible_end))
    require(len(ranking) == len(expected) and set(ranking) == expected, "CONTROL_SELECTOR_RANKING_INVALID")
    selected = selector.get("positions_by_ratio", {}).get(str(ratio))
    require(isinstance(selected, list) and selected == sorted(ranking[: len(ranking) * ratio // 100]), "CONTROL_SELECTED_PREFIX_INVALID")
    indexer_ends = list(
        complete_indexer_pool_ends(selected, eligible_start=eligible_start, eligible_end=eligible_end)
    )
    require(selector.get("indexer_pool_end_positions_by_ratio", {}).get(str(ratio)) == indexer_ends, "CONTROL_INDEXER_POOL_PROJECTION_INVALID")
    if args.mode == "SNAPSHOT":
        require(args.prompt_side == "donor" and ratio == 100, "SNAPSHOT_CONTROL_INVALID")
    if args.mode == "TRANSPLANT":
        require(args.prompt_side == "edited" and ratio > 0, "TRANSPLANT_CONTROL_INVALID")
    if args.mode == "OFF":
        require(ratio == 0, "OFF_CONTROL_RATIO_INVALID")
    evidence_dir = args.evidence_dir.resolve()
    require(evidence_dir.is_absolute(), "CONTROL_EVIDENCE_DIR_INVALID")
    atomic_json(args.output.resolve(), {
        "schema_version": 3,
        "mode": args.mode,
        "experiment_id": args.experiment_id,
        "phase_id": args.phase_id,
        "prompt_side": args.prompt_side,
        "unsafe_accuracy_ablation_ack": UNSAFE_ACK,
        "production_default_enabled": False,
        "compute_semantics": "full_target_prefill_then_selected_donor_cache_overwrite",
        "true_partial_prefill": False,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "architecture": MODEL_ARCHITECTURE,
            "vllm_commit": VLLM_COMMIT,
            "attention_backend": ATTENTION_BACKEND,
            "kv_cache_dtype": KV_CACHE_DTYPE,
            "index_kpool": INDEX_KPOOL,
        },
        "runtime": {
            "tensor_parallel_size": 4,
            "data_parallel_size": 1,
            "expert_parallel_size": 4,
            "data_parallel_rank_affinity": 0,
            "prefix_caching": False,
            "chunked_prefill": False,
        },
        "base_prompt_token_count": int(episode["base_prompt_token_count"]),
        "history_token_count": history_count,
        "eligible_start": eligible_start,
        "eligible_end": eligible_end,
        "edit_positions": differences,
        "mla_layers": list(MLA_LAYERS),
        "indexer_layers": list(INDEXER_LAYERS),
        "kda_layers": list(KDA_LAYERS),
        "kda_policy": "target_computed_no_donor_row_transplant",
        "indexer_tail_policy": "target_computed_never_transplanted",
        "donor_history_token_ids_sha256": token_digest(donor),
        "edited_history_token_ids_sha256": token_digest(edited),
        "selector_path": str(selector_path),
        "selector_sha256": sha256_file(selector_path),
        "requested_ratio_percent": ratio,
        "selected_main_positions": selected,
        "selected_main_positions_sha256": token_digest(selected),
        "selected_indexer_pool_end_positions": indexer_ends,
        "selected_indexer_pool_end_positions_sha256": token_digest(indexer_ends),
        "selected_indexer_covered_positions": list(pool_covered_positions(indexer_ends)),
        "evidence_dir": str(evidence_dir),
    })


def runtime_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("runtime.rank-*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return records


def finalize_ratio_payload(
    *,
    ratio: int,
    phase_id: str,
    episode: Mapping[str, Any],
    selector: Mapping[str, Any],
    proxy_records: Sequence[Mapping[str, Any]],
    runtime: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    require(ratio in RATIOS, "FINALIZE_RATIO_INVALID")
    history_count = int(episode["history_token_count"])
    eligible_start = int(episode["eligible_start"])
    ranking = selector["ranking"]
    selected = selector["positions_by_ratio"][str(ratio)]
    expected_main = len(ranking) * ratio // 100
    require(len(selected) == expected_main, "FINALIZE_MAIN_SELECTION_COUNT_INVALID")
    expected_indexer_ends = list(
        complete_indexer_pool_ends(selected, eligible_start=eligible_start, eligible_end=history_count)
    )
    records = [item for item in proxy_records if item.get("phase_id") == phase_id and item.get("path") == "/v1/chat/completions"]
    replay = [item for item in records if item.get("request_kind") == "replayed_frozen_old_turn1"]
    post_edit = [item for item in records if item.get("request_kind") == "forwarded_after_system_edit"]
    require(len(replay) == 1, "FROZEN_A1_REPLAY_COUNT_INVALID")
    require(bool(post_edit), "NO_POST_EDIT_MODEL_TURN")
    require(all(item.get("http_status") == 200 and item.get("system_edit_applied") is True for item in post_edit), "POST_EDIT_HTTP_OR_REWRITE_FAILURE")
    require(post_edit[0].get("frozen_q2_attested") is True, "FROZEN_Q2_REPLAY_DIVERGED")
    require(all(item.get("data_parallel_rank") == 0 for item in post_edit), "POST_EDIT_DP_AFFINITY_MISSING")
    prompt_tokens = [item.get("usage", {}).get("prompt_tokens") for item in post_edit]
    completion_tokens = [item.get("usage", {}).get("completion_tokens") for item in post_edit]
    require(all(type(value) is int and value > history_count for value in prompt_tokens), "POST_EDIT_PROMPT_NOT_EXTENSION")
    require(all(type(value) is int and value > 0 for value in completion_tokens), "POST_EDIT_COMPLETION_USAGE_INVALID")
    require(set(evaluation) == {INSTANCE_ID} and type(evaluation[INSTANCE_ID]) is bool, "OFFICIAL_RESULT_INVALID")

    transplant: dict[str, Any] = {
        "runtime_evidence_expected": ratio > 0,
        "main_selected_token_rows_per_layer": expected_main,
        "indexer_selected_complete_pool_rows_per_layer": len(expected_indexer_ends),
        "indexer_selected_covered_token_count_per_layer": INDEX_KPOOL * len(expected_indexer_ends),
        "kda_state_rows_transplanted": 0,
        "indexer_tail_rows_transplanted": 0,
    }
    if ratio == 0:
        require(not runtime, "RATIO_ZERO_RUNTIME_MUTATION_EVIDENCE_PRESENT")
    else:
        mutations = [item for item in runtime if item.get("action") == "destination_transplant"]
        attested = [item for item in runtime if item.get("action") == "prompt_attested"]
        require({int(item.get("rank", -1)) for item in mutations + attested} == {0}, "RUNTIME_DP_RANK_AFFINITY_INVALID")
        products = {("mla_kv", layer) for layer in MLA_LAYERS} | {("indexer_kpool", layer) for layer in INDEXER_LAYERS}
        for ordinal, prompt_count in enumerate(prompt_tokens):
            rows = [item for item in mutations if int(item.get("prefill_ordinal", -1)) == ordinal]
            require({(item.get("product"), int(item.get("layer", -1))) for item in rows} == products, "RUNTIME_LAYER_PRODUCT_COVERAGE_INVALID")
            for item in rows:
                expected_count = expected_main if item["product"] == "mla_kv" else len(expected_indexer_ends)
                require(item.get("actual_row_count") == expected_count, "RUNTIME_ROW_COUNT_INVALID")
                require(item.get("full_target_prefill_token_count") == prompt_count, "RUNTIME_FULL_PREFILL_EVIDENCE_INVALID")
                require(item.get("extension_rows_recomputed") == prompt_count - history_count, "POST_EDIT_EXTENSION_REUSE_DETECTED")
                require(item.get("true_partial_prefill") is False, "RUNTIME_TRUE_PARTIAL_MISLABEL")
                require(item.get("edited_token_excluded") is True, "RUNTIME_EDIT_TOKEN_REUSED")
        transplant["actual_transplant_records"] = len(mutations)

    return {
        "schema_version": 3,
        "status": "complete",
        "claim": "single_pinned_swebench_pro_instance_glm53_stateful_mid_trajectory_accuracy_ablation",
        "ratio_percent_requested": ratio,
        "ratio_main_rows_denominator": len(ranking),
        "ratio_main_rows_effective": expected_main / len(ranking),
        "history_token_count": history_count,
        "frozen_a1_replay_count": 1,
        "post_edit_agent_turn_count": len(post_edit),
        "official_evaluator_resolved": evaluation[INSTANCE_ID],
        "single_instance_accuracy": int(evaluation[INSTANCE_ID]),
        "total_post_edit_decode_tokens": sum(completion_tokens),
        "per_post_edit_turn_prompt_tokens": prompt_tokens,
        "transplant": transplant,
        "accuracy_ablation_only": True,
        "full_target_prefill": True,
        "true_partial_prefill": False,
        "latency_or_skipped_flop_claim": False,
        "historical_glm52_evidence_reused_as_glm53": False,
    }


def finalize_ratio(args: argparse.Namespace) -> None:
    proxy = [json.loads(line) for line in args.proxy_log.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    runtime = runtime_records(args.evidence_dir.resolve())
    payload = finalize_ratio_payload(
        ratio=int(args.ratio),
        phase_id=args.phase_id,
        episode=load_json(args.episode.resolve()),
        selector=load_json(args.selector.resolve()),
        proxy_records=proxy,
        runtime=runtime,
        evaluation=load_json(args.eval_results.resolve()),
    )
    payload["patches_sha256"] = sha256_file(args.patches.resolve())
    payload["eval_results_sha256"] = sha256_file(args.eval_results.resolve())
    atomic_json(args.output.resolve(), payload)


def finalize_sweep(args: argparse.Namespace) -> None:
    results = sorted(
        (load_json(Path(value).resolve()) for value in args.ratio_results),
        key=lambda item: item["ratio_percent_requested"],
    )
    expected = [int(item) for item in args.expected_ratios.split(",")]
    require([item["ratio_percent_requested"] for item in results] == expected, "SWEEP_RATIO_SET_INVALID")
    require(results and results[0]["ratio_percent_requested"] == 0, "RATIO_ZERO_MISSING")
    baseline = results[0]
    for result in results:
        result["post_edit_decode_delta_vs_ratio_0"] = (
            result["total_post_edit_decode_tokens"] - baseline["total_post_edit_decode_tokens"]
        )
    atomic_json(args.output.resolve(), {
        "schema_version": 3,
        "status": "complete",
        "primary_outcome": "official single-instance resolved accuracy after the frozen mid-trajectory system edit",
        "compute_semantics": "full_target_prefill_then_selected_donor_cache_overwrite_accuracy_ablation",
        "true_partial_prefill": False,
        "results": results,
    })


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-model-contract")
    validate.add_argument("--deployment-lock", type=Path, required=True)
    validate.add_argument("--model-config", type=Path, required=True)
    config = commands.add_parser("configure-agent")
    config.add_argument("--input", type=Path, required=True)
    config.add_argument("--output", type=Path, required=True)
    config.add_argument("--api-base", required=True)
    config.add_argument("--step-limit", type=int, default=0)
    episode = commands.add_parser("build-episode")
    episode.add_argument("--capture", type=Path, required=True)
    episode.add_argument("--trajectory", type=Path, required=True)
    episode.add_argument("--base-selector", type=Path, required=True)
    episode.add_argument("--model-root", type=Path, required=True)
    episode.add_argument("--output-root", type=Path, required=True)
    serialize = commands.add_parser("serialize-base-prompts")
    serialize.add_argument("--capture", type=Path, required=True)
    serialize.add_argument("--model-root", type=Path, required=True)
    serialize.add_argument("--output-root", type=Path, required=True)
    base_selector = commands.add_parser("freeze-base-selector")
    base_selector.add_argument("--donor-root", type=Path, required=True)
    base_selector.add_argument("--edited-root", type=Path, required=True)
    base_selector.add_argument("--donor-tokens", type=Path, required=True)
    base_selector.add_argument("--edited-tokens", type=Path, required=True)
    base_selector.add_argument("--output", type=Path, required=True)
    capture_control = commands.add_parser("selector-capture-control")
    capture_control.add_argument("--prompt-side", choices=("donor", "edited"), required=True)
    capture_control.add_argument("--tokens", type=Path, required=True)
    capture_control.add_argument("--evidence-dir", type=Path, required=True)
    capture_control.add_argument("--output", type=Path, required=True)
    control = commands.add_parser("control")
    control.add_argument("--mode", required=True)
    control.add_argument("--experiment-id", required=True)
    control.add_argument("--phase-id", required=True)
    control.add_argument("--prompt-side", required=True)
    control.add_argument("--ratio", type=int, required=True)
    control.add_argument("--episode", type=Path, required=True)
    control.add_argument("--selector", type=Path, required=True)
    control.add_argument("--donor-history", type=Path, required=True)
    control.add_argument("--edited-history", type=Path, required=True)
    control.add_argument("--evidence-dir", type=Path, required=True)
    control.add_argument("--output", type=Path, required=True)
    ratio = commands.add_parser("finalize-ratio")
    ratio.add_argument("--ratio", type=int, required=True)
    ratio.add_argument("--phase-id", required=True)
    ratio.add_argument("--episode", type=Path, required=True)
    ratio.add_argument("--selector", type=Path, required=True)
    ratio.add_argument("--proxy-log", type=Path, required=True)
    ratio.add_argument("--evidence-dir", type=Path, required=True)
    ratio.add_argument("--patches", type=Path, required=True)
    ratio.add_argument("--eval-results", type=Path, required=True)
    ratio.add_argument("--output", type=Path, required=True)
    sweep = commands.add_parser("finalize-sweep")
    sweep.add_argument("--ratio-results", nargs="+", required=True)
    sweep.add_argument("--expected-ratios", required=True)
    sweep.add_argument("--output", type=Path, required=True)
    return root


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "validate-model-contract":
        payload = {
            "deployment": validate_deployment_lock(load_json(args.deployment_lock.resolve())),
            "model_config": validate_model_config(load_json(args.model_config.resolve())),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "configure-agent":
        configure_agent(args)
    elif args.command == "build-episode":
        build_episode(args)
    elif args.command == "serialize-base-prompts":
        serialize_base_prompts(args)
    elif args.command == "freeze-base-selector":
        freeze_base_selector(args)
    elif args.command == "selector-capture-control":
        write_selector_capture_control(args)
    elif args.command == "control":
        write_control(args)
    elif args.command == "finalize-ratio":
        finalize_ratio(args)
    else:
        finalize_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
