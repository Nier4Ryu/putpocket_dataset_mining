# SPDX-License-Identifier: Apache-2.0
"""Default-off GLM-5.2 main-attention/indexer score capture.

This file is installed into ``vllm.model_executor.layers`` by the RunPod
package overlay.  It captures a bounded ordinary-prefill probe only.  Main MLA
scores are a dense, mathematically faithful reference recomputation from the
same post-RoPE Q and projected K state used by the layer.  Indexer scores are
the native FP8/FP4 kernel output before top-k.  The two origins are deliberately
kept distinct in every record.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


ENABLE_ENV = "PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_ENABLE"
CONFIG_ENV = "PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_CONFIG"
CONFIG_SHA256_ENV = "PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_CONFIG_SHA256"
OUTPUT_ROOT_ENV = "PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_OUTPUT_ROOT"
RUN_ID_ENV = "PUTPOCKET_GLM52_SCORE_DIAGNOSTIC_RUN_ID"
TRUE_PARTIAL_ENV = "PUTPOCKET_VLLM_TRUE_PARTIAL_PREFILL_ENABLE"
LEGACY_EMULATION_ENV = "PUTPOCKET_GLM52_FORCED_REUSE_CONTROL"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_config: dict[str, Any] | None = None
_batch: dict[str, Any] | None = None
_seen: set[tuple[str, int, int]] = set()


class ScoreDiagnosticError(RuntimeError):
    """The diagnostic contract was violated; partial evidence is unusable."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ScoreDiagnosticError(reason)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _token_digest(values: Sequence[int]) -> str:
    return _sha256(
        json.dumps(list(values), separators=(",", ":")).encode("ascii")
    )


def enabled() -> bool:
    return os.getenv(ENABLE_ENV) == "1"


def _load_config() -> dict[str, Any]:
    global _config
    if _config is not None:
        return _config
    _require(not os.getenv(TRUE_PARTIAL_ENV), "SCORE_DIAGNOSTIC_TRUE_PARTIAL_MODE_CONFLICT")
    _require(not os.getenv(LEGACY_EMULATION_ENV), "SCORE_DIAGNOSTIC_LEGACY_MODE_CONFLICT")
    raw_path = os.getenv(CONFIG_ENV)
    raw_digest = os.getenv(CONFIG_SHA256_ENV)
    _require(bool(raw_path), "SCORE_DIAGNOSTIC_CONFIG_NOT_SET")
    _require(
        isinstance(raw_digest, str) and _SHA256_RE.fullmatch(raw_digest) is not None,
        "SCORE_DIAGNOSTIC_CONFIG_DIGEST_NOT_SET",
    )
    path = Path(str(raw_path))
    _require(path.is_absolute() and path.is_file(), "SCORE_DIAGNOSTIC_CONFIG_INVALID")
    data = path.read_bytes()
    _require(_sha256(data) == raw_digest, "SCORE_DIAGNOSTIC_CONFIG_DIGEST_MISMATCH")
    value = json.loads(data)
    _require(isinstance(value, dict), "SCORE_DIAGNOSTIC_CONFIG_NOT_OBJECT")
    required = {
        "schema_version",
        "diagnostic_id",
        "instance_id",
        "scenario_id",
        "probe_kind",
        "expected_prompt_token_count",
        "expected_prompt_token_ids_sha256",
        "expected_edit_position",
        "expected_target_token_id",
        "layers",
        "query_positions",
        "candidate_start_position",
        "max_candidate_tokens",
        "tensor_parallel_size",
        "global_main_attention_heads",
        "indexer_heads",
        "indexer_head_dim",
        "main_qk_nope_head_dim",
        "main_qk_rope_head_dim",
    }
    _require(set(value) == required, "SCORE_DIAGNOSTIC_CONFIG_KEYS_INVALID")
    _require(value["schema_version"] == 1, "SCORE_DIAGNOSTIC_SCHEMA_INVALID")
    _require(
        value["probe_kind"] == "ordinary_target_prefill_q1_boundary_no_q2",
        "SCORE_DIAGNOSTIC_PROBE_KIND_INVALID",
    )
    layers = value["layers"]
    queries = value["query_positions"]
    _require(
        isinstance(layers, list)
        and layers
        and layers == sorted(set(layers))
        and all(isinstance(item, int) and item >= 0 for item in layers),
        "SCORE_DIAGNOSTIC_LAYERS_INVALID",
    )
    _require(
        isinstance(queries, list)
        and queries
        and queries == sorted(set(queries))
        and all(isinstance(item, int) and item >= 0 for item in queries),
        "SCORE_DIAGNOSTIC_QUERIES_INVALID",
    )
    _require(
        isinstance(value["expected_prompt_token_count"], int)
        and value["expected_prompt_token_count"] > max(queries),
        "SCORE_DIAGNOSTIC_PROMPT_COUNT_INVALID",
    )
    _require(
        isinstance(value["expected_prompt_token_ids_sha256"], str)
        and _SHA256_RE.fullmatch(value["expected_prompt_token_ids_sha256"]),
        "SCORE_DIAGNOSTIC_PROMPT_DIGEST_INVALID",
    )
    _require(
        value["tensor_parallel_size"] == 4
        and value["global_main_attention_heads"] == 64
        and value["indexer_heads"] == 64
        and value["indexer_head_dim"] == 128
        and value["main_qk_nope_head_dim"] == 128
        and value["main_qk_rope_head_dim"] == 64,
        "SCORE_DIAGNOSTIC_MODEL_LAYOUT_INVALID",
    )
    _config = value
    return value


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    for name in ("RANK", "LOCAL_RANK"):
        raw = os.getenv(name)
        if raw is not None:
            try:
                return int(raw)
            except ValueError as exc:
                raise ScoreDiagnosticError("SCORE_DIAGNOSTIC_RANK_INVALID") from exc
    return 0


def _layer_id(layer_name: str) -> int:
    match = _LAYER_RE.search(layer_name)
    _require(match is not None, "SCORE_DIAGNOSTIC_LAYER_NAME_INVALID")
    return int(match.group(1))


def _output_path() -> Path:
    raw_root = os.getenv(OUTPUT_ROOT_ENV)
    run_id = os.getenv(RUN_ID_ENV)
    _require(bool(raw_root) and bool(run_id), "SCORE_DIAGNOSTIC_OUTPUT_NOT_SET")
    _require(
        re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", str(run_id)) is not None,
        "SCORE_DIAGNOSTIC_RUN_ID_INVALID",
    )
    root = Path(str(raw_root))
    _require(root.is_absolute() and root.is_dir(), "SCORE_DIAGNOSTIC_OUTPUT_INVALID")
    return root / f"capture-rank-{_rank():02d}.jsonl"


def _write(record: Mapping[str, Any]) -> None:
    payload = dict(record)
    payload["record_sha256"] = _sha256(_canonical_bytes(payload))
    encoded = _canonical_bytes(payload)
    path = _output_path()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, encoded)
        _require(written == len(encoded), "SCORE_DIAGNOSTIC_SHORT_WRITE")
    finally:
        os.close(descriptor)


def maybe_set_score_diagnostic_batch(
    input_ids: torch.Tensor, positions: torch.Tensor
) -> None:
    """Attest the one full ordinary-prefill batch; ignore later decode steps."""

    global _batch
    if not enabled():
        return
    config = _load_config()
    ids = input_ids.detach().reshape(-1).to("cpu", dtype=torch.int64).tolist()
    pos = positions.detach().reshape(-1).to("cpu", dtype=torch.int64).tolist()
    expected_count = int(config["expected_prompt_token_count"])
    if not pos or pos[0] != 0:
        _batch = None
        return
    _require(len(ids) == len(pos) == expected_count, "SCORE_DIAGNOSTIC_PREFILL_COUNT_MISMATCH")
    _require(pos == list(range(expected_count)), "SCORE_DIAGNOSTIC_POSITIONS_NOT_CONTIGUOUS")
    _require(
        _token_digest(ids) == config["expected_prompt_token_ids_sha256"],
        "SCORE_DIAGNOSTIC_PROMPT_TOKEN_DIGEST_MISMATCH",
    )
    edit_position = int(config["expected_edit_position"])
    _require(
        ids[edit_position] == config["expected_target_token_id"],
        "SCORE_DIAGNOSTIC_TARGET_EDIT_TOKEN_MISMATCH",
    )
    _seen.clear()
    _batch = {"token_ids": ids, "positions": pos}


def _base_record(layer: int, query_position: int) -> dict[str, Any]:
    config = _load_config()
    _require(_batch is not None, "SCORE_DIAGNOSTIC_BATCH_NOT_ATTESTED")
    token_ids = _batch["token_ids"]
    return {
        "schema_version": 1,
        "diagnostic_id": config["diagnostic_id"],
        "instance_id": config["instance_id"],
        "scenario_id": config["scenario_id"],
        "probe_kind": config["probe_kind"],
        "q2_in_probe": False,
        "rank": _rank(),
        "tensor_parallel_size": config["tensor_parallel_size"],
        "layer": layer,
        "query_position": query_position,
        "query_token_id": token_ids[query_position],
        "dtype": "float32_capture_from_bfloat16_or_quantized_runtime_state",
    }


def maybe_capture_main_attention_reference(
    *,
    layer_name: str,
    positions: torch.Tensor,
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_b_proj: torch.nn.Module,
    scale: float,
    qk_nope_head_dim: int,
    v_head_dim: int,
) -> None:
    """Capture dense causal MLA QK logits for configured ordinary-prefill rows."""

    if not enabled() or _batch is None:
        return
    config = _load_config()
    layer = _layer_id(layer_name)
    if layer not in config["layers"]:
        return
    pos = positions.detach().reshape(-1).to("cpu", dtype=torch.int64).tolist()
    if pos != _batch["positions"]:
        return
    _require(q.ndim == 3 and kv_c_normed.ndim == 2 and k_pe.ndim == 3, "MAIN_REFERENCE_SHAPE_INVALID")
    _require(
        q.shape[0] == kv_c_normed.shape[0] == k_pe.shape[0] == len(pos),
        "MAIN_REFERENCE_TOKEN_SHAPE_MISMATCH",
    )
    _require(
        qk_nope_head_dim == config["main_qk_nope_head_dim"]
        and q.shape[-1]
        == config["main_qk_nope_head_dim"] + config["main_qk_rope_head_dim"],
        "MAIN_REFERENCE_HEAD_DIM_MISMATCH",
    )
    projected = kv_b_proj(kv_c_normed)[0].view(
        len(pos), q.shape[1], qk_nope_head_dim + v_head_dim
    )
    k_nope = projected[..., :qk_nope_head_dim]
    k_rope = k_pe.squeeze(1)
    candidate_start = int(config["candidate_start_position"])
    max_candidates = int(config["max_candidate_tokens"])
    for query_position in config["query_positions"]:
        key = ("main", layer, query_position)
        if key in _seen:
            raise ScoreDiagnosticError("MAIN_REFERENCE_DUPLICATE_CAPTURE")
        query_index = pos.index(query_position)
        first = max(candidate_start, query_position + 1 - max_candidates)
        candidate_positions = list(range(first, query_position + 1))
        candidate_indices = torch.tensor(
            candidate_positions, dtype=torch.long, device=q.device
        )
        query_nope = q[query_index, :, :qk_nope_head_dim].float()
        query_rope = q[query_index, :, qk_nope_head_dim:].float()
        candidate_nope = k_nope.index_select(0, candidate_indices).float()
        candidate_rope = k_rope.index_select(0, candidate_indices).float()
        logits = (
            torch.einsum("hd,chd->hc", query_nope, candidate_nope)
            + torch.einsum("hd,cd->hc", query_rope, candidate_rope)
        ) * float(scale)
        values = logits.detach().to("cpu", dtype=torch.float32).tolist()
        _require(
            all(len(head) == len(candidate_positions) for head in values),
            "MAIN_REFERENCE_VECTOR_SHAPE_INVALID",
        )
        record = _base_record(layer, query_position)
        record.update(
            {
                "record_kind": "main_attention_reference",
                "score_origin": "reference_recomputed_from_exact_post_rope_q_and_model_projected_k",
                "kernel_native": False,
                "reference_recomputed": True,
                "formula": "scale*((q_nope dot k_nope)+(q_rope dot k_rope)) per main head",
                "scale": float(scale),
                "mask": "causal_inclusive_candidate_position_le_query_position",
                "head_aggregation_at_capture": "none_tp_local_heads_preserved",
                "local_head_count": q.shape[1],
                "global_head_count": config["global_main_attention_heads"],
                "candidate_positions": candidate_positions,
                "candidate_token_ids": [
                    _batch["token_ids"][item] for item in candidate_positions
                ],
                "raw_logits_by_local_head": values,
            }
        )
        _write(record)
        _seen.add(key)


def maybe_capture_indexer_native_logits(
    *,
    layer_name: str,
    logits: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    token_start: int,
    token_end: int,
) -> None:
    """Capture native pre-top-k indexer logits with their exact causal span."""

    if not enabled() or _batch is None:
        return
    config = _load_config()
    layer = _layer_id(layer_name)
    if layer not in config["layers"]:
        return
    _require(logits.ndim == 2 and token_end > token_start, "INDEXER_NATIVE_SHAPE_INVALID")
    starts = cu_seqlen_ks.detach().reshape(-1).to("cpu", dtype=torch.int64).tolist()
    ends = cu_seqlen_ke.detach().reshape(-1).to("cpu", dtype=torch.int64).tolist()
    query_positions = _batch["positions"][token_start:token_end]
    _require(
        logits.shape[0] == len(query_positions),
        "INDEXER_NATIVE_QUERY_SHAPE_MISMATCH",
    )
    candidate_start = int(config["candidate_start_position"])
    max_candidates = int(config["max_candidate_tokens"])
    for query_position in config["query_positions"]:
        if query_position not in query_positions:
            continue
        key = ("indexer", layer, query_position)
        if key in _seen:
            raise ScoreDiagnosticError("INDEXER_NATIVE_DUPLICATE_CAPTURE")
        row = query_positions.index(query_position)
        _require(row < len(starts) and row < len(ends), "INDEXER_NATIVE_BOUNDS_MISSING")
        valid_start, valid_end = int(starts[row]), int(ends[row])
        _require(
            valid_start == 0 and valid_end == query_position + 1,
            "INDEXER_NATIVE_CAUSAL_ALIGNMENT_UNSUPPORTED",
        )
        first = max(candidate_start, valid_end - max_candidates)
        _require(first >= valid_start, "INDEXER_NATIVE_CANDIDATE_START_INVALID")
        candidate_positions = list(range(first, valid_end))
        vector = (
            logits[row, first:valid_end]
            .detach()
            .to("cpu", dtype=torch.float32)
            .tolist()
        )
        _require(
            len(vector) == len(candidate_positions),
            "INDEXER_NATIVE_VECTOR_SHAPE_INVALID",
        )
        record = _base_record(layer, query_position)
        record.update(
            {
                "record_kind": "indexer_native_pre_topk",
                "score_origin": "kernel_native_fp8_fp4_mqa_logits_before_top_k_per_row_prefill",
                "kernel_native": True,
                "reference_recomputed": False,
                "formula": "sum_h(weight_h*(128^-0.5)*(64^-0.5)*dot(fp8_q_h,quantized_k_with_scales))",
                "scale": {
                    "softmax_scale": 128**-0.5,
                    "indexer_head_scale": 64**-0.5,
                },
                "mask": "native_cu_seqlen_ks_ke_causal_inclusive",
                "head_aggregation_at_capture": "native_learned_weighted_sum_across_64_indexer_heads",
                "indexer_head_count": config["indexer_heads"],
                "indexer_head_dim": config["indexer_head_dim"],
                "candidate_positions": candidate_positions,
                "candidate_token_ids": [
                    _batch["token_ids"][item] for item in candidate_positions
                ],
                "raw_logits": vector,
                "native_valid_start": valid_start,
                "native_valid_end_exclusive": valid_end,
            }
        )
        _write(record)
        _seen.add(key)
