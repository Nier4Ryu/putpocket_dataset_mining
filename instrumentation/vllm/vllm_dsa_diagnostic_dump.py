"""Bounded native vLLM DSA score capture for the pinned GLM diagnostic.

This module is copied into the pinned vLLM source before the wheel is built.
It observes the native logits and native top-k output in
``sparse_attn_indexer.py``; it never recomputes attention or changes selection.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_SEEN: set[tuple[str, int, str, int, int]] = set()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _blocked(root: Path, failure: str, details: dict[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "status": "BLOCKED",
        "failure_class": failure,
        "details": details,
    }
    _atomic_json(root / f"BLOCKED.rank-{_rank()}.json", payload)
    raise RuntimeError(f"{failure}:{json.dumps(details, sort_keys=True)}")


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get("RANK", "0"))


def _load_control() -> tuple[Path, dict[str, Any]] | None:
    value = os.environ.get("PUTPOCKET_VLLM_DSA_TRACE_CONTROL")
    if not value:
        return None
    path = Path(value)
    try:
        control = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"TRACE_CONTROL_UNREADABLE:{type(exc).__name__}") from exc
    if control.get("schema_version") != 1 or control.get("mode") not in {"OFF", "ON"}:
        raise RuntimeError("TRACE_CONTROL_INVALID")
    return path, control


def maybe_capture_native_dsa(
    *,
    phase: str,
    layer_name: str,
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    lengths: torch.Tensor,
    row_starts: torch.Tensor | None,
    topk: int,
    native_logits_backend: str,
    native_topk_backend: str,
    dcp_world_size: int,
) -> None:
    """Capture only configured points from the native fused indexer path."""

    loaded = _load_control()
    if loaded is None:
        return
    _, control = loaded
    if control["mode"] == "OFF":
        return
    root = Path(os.environ.get("PUTPOCKET_VLLM_DSA_TRACE_ROOT", ""))
    if not root.is_absolute():
        raise RuntimeError("TRACE_ROOT_MUST_BE_ABSOLUTE")
    match = _LAYER_RE.search(layer_name)
    if not match:
        _blocked(root, "LAYER_COORDINATE_UNAVAILABLE", {"layer_name": layer_name})
    layer = int(match.group(1))
    full_layers = [int(value) for value in control["full_indexer_layers"]]
    if layer not in full_layers:
        _blocked(root, "UNEXPECTED_INDEXER_LAYER", {"layer": layer})
    if dcp_world_size != 1:
        _blocked(root, "DCP_FORBIDDEN", {"dcp_world_size": dcp_world_size})
    if topk != 2048 or logits.dtype != torch.float32 or topk_indices.dtype != torch.int32:
        _blocked(
            root,
            "NATIVE_TENSOR_CONTRACT_MISMATCH",
            {"topk": topk, "logits_dtype": str(logits.dtype), "indices_dtype": str(topk_indices.dtype)},
        )
    if phase == "prefill" and native_topk_backend != "top_k_per_row_prefill":
        _blocked(root, "PREFILL_NATIVE_TOPK_AMBIGUOUS", {"backend": native_topk_backend})
    if phase == "decode" and native_topk_backend != "cooperative_topk_sm90":
        _blocked(root, "DECODE_NATIVE_TOPK_FALLBACK", {"backend": native_topk_backend})

    rank = _rank()
    prompt_tokens = int(control["prompt_token_count"])
    lengths_cpu = lengths.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
    starts_cpu = (
        [0] * len(lengths_cpu)
        if row_starts is None
        else row_starts.detach().to(device="cpu", dtype=torch.int64).reshape(-1).tolist()
    )
    if len(lengths_cpu) != logits.shape[0] or len(starts_cpu) != logits.shape[0]:
        _blocked(root, "NATIVE_ROW_METADATA_MISMATCH", {"rows": logits.shape[0]})

    for row, (length, start) in enumerate(zip(lengths_cpu, starts_cpu, strict=True)):
        if phase == "prefill":
            if length != prompt_tokens:
                continue
            sample = "prefill_last_query"
            decode_step = -1
        elif phase == "decode":
            decode_step = int(length) - prompt_tokens - 1
            if decode_step not in {0, 1, 8, 32}:
                continue
            sample = f"decode_{decode_step}"
        else:
            _blocked(root, "UNKNOWN_CAPTURE_PHASE", {"phase": phase})
        key = (str(control["run_id"]), rank, sample, layer, row)
        if key in _SEEN:
            _blocked(root, "DUPLICATE_CAPTURE", {"sample": sample, "layer": layer, "rank": rank})
        _SEEN.add(key)
        end = int(start) + int(length)
        if start < 0 or end > logits.shape[1] or length < topk:
            _blocked(
                root,
                "RAW_VECTOR_BOUNDS_MISMATCH",
                {"start": start, "end": end, "width": logits.shape[1], "length": length},
            )
        ids = topk_indices[row, :topk].detach().to(device="cpu", dtype=torch.int64)
        if bool(((ids < 0) | (ids >= length)).any().item()):
            _blocked(root, "NATIVE_SELECTED_ID_OUT_OF_BOUNDS", {"sample": sample, "layer": layer})
        raw = logits[row, start:end].detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(raw).all().item()):
            _blocked(root, "NONFINITE_NATIVE_RAW_SCORES", {"sample": sample, "layer": layer})
        selected = raw.index_select(0, ids)
        record = {
            "schema_version": 1,
            "run_id": control["run_id"],
            "instance_id": control["instance_id"],
            "trace_mode": "ON",
            "phase": phase,
            "sample_point": sample,
            "layer": layer,
            "full_indexer_layers": full_layers,
            "shared_layer_mapping": control["shared_layer_mapping"],
            "shared_layer_mapping_sha256": control["shared_layer_mapping_sha256"],
            "query_position": int(length) - 1,
            "decode_step": decode_step if phase == "decode" else None,
            "context_length": int(length),
            "rank": rank,
            "native_logits_backend": native_logits_backend,
            "native_topk_backend": native_topk_backend,
            "dtype": str(raw.dtype),
            "device": "cuda",
            "shape": [int(raw.numel())],
            "topk": topk,
            "source_token_coordinate_semantics": "zero_based_logical_causal_source_position",
            "revisions": control["revisions"],
            "raw_scores": raw.tolist(),
            "selected_ids": ids.tolist(),
            "selected_scores": selected.tolist(),
        }
        encoded = (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode()
        record["record_sha256"] = hashlib.sha256(encoded).hexdigest()
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"captures.rank-{rank}.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")
