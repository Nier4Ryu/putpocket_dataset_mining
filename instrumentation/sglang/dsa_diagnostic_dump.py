"""Bounded native GLM DSA score capture for the pinned SGLang diagnostic.

This module is injected into an exact SGLang checkout.  It never performs a
second top-k operation.  The caller supplies the score tensor produced by the
native MQA kernel before SGLang's forced-token mask, then supplies the indices
returned by the unchanged fused ``sgl-kernel`` transform.  Logical token
coordinates are recovered from the transform metadata and selected scores are
gathered from the saved native row.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

import torch


SCHEMA_VERSION = 1
FULL_LAYERS = (0, 1, 2, *range(6, 75, 4))
DECODE_SAMPLES = (0, 1, 8, 32)
_SEEN: set[tuple[str, str, int, int, int]] = set()


def _rank() -> int:
    for name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value is not None and value.isdigit():
            return int(value)
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        return int(torch.cuda.current_device())
    except (AssertionError, RuntimeError):
        pass
    return -1


def _read_json_env(name: str) -> dict[str, Any]:
    path = os.environ.get(name, "")
    if not path:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _shared_layers(source_layer: int) -> list[int]:
    mapping: dict[int, list[int]] = {layer: [] for layer in FULL_LAYERS}
    source = 0
    for layer in range(78):
        if layer in FULL_LAYERS:
            source = layer
        else:
            mapping[source].append(layer)
    return mapping[source_layer]


def _phase_and_sample(forward_batch: Any, context_length: int, prompt_tokens: int) -> tuple[str, int] | None:
    mode = forward_batch.forward_mode
    if mode.is_extend_without_speculative():
        if context_length != prompt_tokens:
            return None
        return ("prefill_last_query", -1)
    if mode.is_decode_or_idle():
        decode_step = context_length - prompt_tokens - 1
        if decode_step in DECODE_SAMPLES:
            return ("decode", decode_step)
    return None


def prepare_native_dsa_capture(
    *,
    forward_batch: Any,
    layer_id: int,
    logits: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: Optional[torch.Tensor],
    metadata: Any,
    transform_kind: str,
    row_offset: int = 0,
    total_rows: Optional[int] = None,
    num_init_tokens: int = 0,
    num_local_tokens: int = 0,
) -> Optional[dict[str, Any]]:
    """Save one bounded native score row before the unchanged fused transform."""

    control = _read_json_env("PUTPOCKET_DSA_TRACE_CONTROL")
    if control.get("schema_version") != SCHEMA_VERSION or control.get("enabled") is not True:
        return None
    if control.get("mode") != "ON" or layer_id not in FULL_LAYERS:
        return None
    prompt_tokens = int(control.get("prompt_token_count", 0))
    if prompt_tokens <= 2048:
        _blocked("PREFILL_NATIVE_LOGITS_SKIPPED_AT_OR_BELOW_TOPK", layer_id, control)
    rows = int(logits.shape[0])
    if rows <= 0:
        return None
    if int(lengths.numel()) != rows or (row_starts is not None and int(row_starts.numel()) != rows):
        _blocked("NATIVE_QUERY_LENGTH_MAPPING_UNPROVEN", layer_id, control)
    if forward_batch.forward_mode.is_extend_without_speculative():
        expected_total = int(total_rows if total_rows is not None else rows)
        global_row = expected_total - 1
        if not (row_offset <= global_row < row_offset + rows):
            return None
        row = global_row - row_offset
    else:
        row = 0
    context_length = int(lengths[row].item())
    sample = _phase_and_sample(forward_batch, context_length, prompt_tokens)
    if sample is None:
        return None
    phase, decode_step = sample
    rank = _rank()
    if rank not in range(4):
        _blocked("NATIVE_TP_RANK_IDENTITY_UNPROVEN", layer_id, control)
    run_id = str(control.get("run_id", ""))
    identity = (run_id, phase, decode_step, layer_id, rank)
    if not run_id or identity in _SEEN:
        return None
    _SEEN.add(identity)
    start = int(row_starts[row].item()) if row_starts is not None else 0
    if start < 0 or context_length <= 0 or start + context_length > int(logits.shape[1]):
        _blocked("NATIVE_SCORE_VECTOR_BOUNDS_UNPROVEN", layer_id, control)
    raw = logits[row, start : start + context_length].detach().to(torch.float32).cpu().clone()
    attn = metadata.attn_metadata
    ticket: dict[str, Any] = {
        "control": control,
        "phase": phase,
        "decode_step": decode_step,
        "layer": layer_id,
        "rank": rank,
        "row": row,
        "row_start": start,
        "context_length": context_length,
        "raw": raw,
        "dtype": str(logits.dtype),
        "device": str(logits.device),
        "shape": list(logits.shape),
        "transform_kind": transform_kind,
        "num_init_tokens": int(num_init_tokens),
        "num_local_tokens": int(num_local_tokens),
        "page_size": int(attn.page_size),
    }
    offsets = attn.topk_indices_offset
    if transform_kind == "ragged" and offsets is None:
        transform_kind = ticket["transform_kind"] = "paged"
    if transform_kind == "paged":
        global_row = row_offset + row
        batch_indices = metadata.get_token_to_batch_idx()
        if batch_indices is None or int(batch_indices.numel()) <= global_row:
            _blocked("NATIVE_PAGED_QUERY_TO_SEQUENCE_MAPPING_UNPROVEN", layer_id, control)
        table_row = int(batch_indices[global_row].item())
        page_table = attn.real_page_table
        if table_row < 0 or table_row >= int(page_table.shape[0]):
            _blocked("NATIVE_PAGED_SEQUENCE_TABLE_BOUNDS_UNPROVEN", layer_id, control)
        ticket["page_table"] = page_table[table_row].detach().to(torch.int64).cpu().clone()
        ticket["page_table_row"] = table_row
    elif transform_kind == "ragged":
        if int(offsets.numel()) <= row_offset + row:
            _blocked("NATIVE_RAGGED_OFFSET_MAPPING_UNPROVEN", layer_id, control)
        ticket["native_offset"] = int(offsets[row_offset + row].item())
    else:
        _blocked("NATIVE_TRANSFORM_KIND_UNSUPPORTED", layer_id, control)
    return ticket


def finish_native_dsa_capture(ticket: Optional[dict[str, Any]], native_indices: torch.Tensor) -> None:
    """Recover native-selected logical IDs and atomically emit one record."""

    if ticket is None:
        return
    control = ticket["control"]
    layer_id = int(ticket["layer"])
    try:
        row = int(ticket["row"])
        native = native_indices[row].detach().to(torch.int64).cpu()
        if native.numel() != 2048 or bool((native < 0).any()):
            raise RuntimeError("native fused top-k did not return 2048 valid entries")
        context = int(ticket["context_length"])
        if ticket["transform_kind"] == "ragged":
            logical = native - int(ticket["native_offset"])
        else:
            page_size = int(ticket["page_size"])
            table = ticket["page_table"]
            pages = native // page_size
            offsets = native % page_size
            logical_values: list[int] = []
            valid_blocks = (context + page_size - 1) // page_size
            table_values = table[:valid_blocks].tolist()
            for page, offset in zip(pages.tolist(), offsets.tolist(), strict=True):
                matches = [index for index, value in enumerate(table_values) if int(value) == int(page)]
                if len(matches) != 1:
                    raise RuntimeError("physical page slot cannot be inverted uniquely")
                logical_values.append(matches[0] * page_size + int(offset))
            logical = torch.tensor(logical_values, dtype=torch.int64)
        if bool((logical < 0).any()) or bool((logical >= context).any()):
            raise RuntimeError("native selected coordinate is outside the causal source vector")
        if len(set(logical.tolist())) != 2048:
            raise RuntimeError("native selected coordinates are not unique")
        raw = ticket["raw"]
        selected_scores = raw[logical]
        provenance = _read_json_env("PUTPOCKET_DSA_TRACE_PROVENANCE")
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_type": "native_glm52_dsa_indexer_scores",
            "run_id": control["run_id"],
            "instance_id": control["instance_id"],
            "trace_mode": "ON",
            "phase": ticket["phase"],
            "layer": layer_id,
            "full_indexer_layer": layer_id,
            "shared_layers": _shared_layers(layer_id),
            "query_position": int(ticket["context_length"]) - 1,
            "decode_step": None if ticket["decode_step"] < 0 else int(ticket["decode_step"]),
            "context_length": int(ticket["context_length"]),
            "rank": int(ticket["rank"]),
            "backend_identities": provenance.get("backend_identities", {}),
            "dtype": ticket["dtype"],
            "device": ticket["device"],
            "native_logits_shape": ticket["shape"],
            "topk": 2048,
            "source_token_coordinate_semantics": "zero_based_logical_causal_position_within_exact_request",
            "native_transform_kind": ticket["transform_kind"],
            "native_selected_logical_token_ids": logical.tolist(),
            "native_selected_scores": selected_scores.tolist(),
            "native_pre_topk_raw_score_vector": raw.tolist(),
            "native_forced_token_mask": {
                "num_init_tokens": int(ticket["num_init_tokens"]),
                "num_local_tokens": int(ticket["num_local_tokens"]),
                "applied_by_sglang_after_raw_capture": True,
            },
            "revisions": provenance.get("revisions", {}),
        }
        _write_record(record)
    except Exception as exc:
        _blocked("NATIVE_SELECTED_COORDINATE_EXPOSURE_UNPROVEN", layer_id, control, str(exc))


def _write_record(record: dict[str, Any]) -> None:
    root = Path(os.environ["PUTPOCKET_DSA_TRACE_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    step = "prefill" if record["decode_step"] is None else f"decode-{record['decode_step']:02d}"
    name = f"{record['trace_mode'].lower()}-{step}-layer-{record['layer']:02d}-rank-{record['rank']:02d}.json"
    payload = json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    partial = root / (name + ".partial")
    partial.write_bytes(payload)
    partial.chmod(0o644)
    partial.replace(root / name)


def _blocked(
    failure_class: str,
    layer_id: int,
    control: dict[str, Any],
    detail: str = "",
) -> None:
    root = Path(os.environ.get("PUTPOCKET_DSA_TRACE_ROOT", "/tmp"))
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "BLOCKED",
        "failure_class": failure_class,
        "layer": layer_id,
        "run_id": control.get("run_id"),
        "instance_id": control.get("instance_id"),
        "trace_mode": control.get("mode"),
        "detail_sha256": hashlib.sha256(detail.encode("utf-8")).hexdigest(),
        "native_runtime_changed": False,
        "fallback_attempted": False,
        "maximum_native_evidence": "pre-transform native raw vector and fused transformed indices when available",
    }
    target = root / f"BLOCKED-layer-{layer_id:02d}-rank-{_rank():02d}.json"
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    target.chmod(0o644)
    raise RuntimeError(f"PUTPOCKET_DSA_BLOCKED:{failure_class}")
