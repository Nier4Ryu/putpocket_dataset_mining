# SPDX-License-Identifier: Apache-2.0
"""Fail-closed control plane for experimental donor-KV partial prefill.

This module is installed into ``vllm.v1`` by the maintained
vLLM overlay.  It is intentionally ignorant of SWE-bench messages and turns a
frozen, server-side manifest into exact cache-copy and recompute coordinates.
The feature is inert unless both a namespaced request hook and the server
enable environment variable are present.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


ENABLE_ENV = "PUTPOCKET_VLLM_TRUE_PARTIAL_PREFILL_ENABLE"
MANIFEST_ROOT_ENV = "PUTPOCKET_VLLM_TRUE_PARTIAL_MANIFEST_ROOT"
EVIDENCE_ROOT_ENV = "PUTPOCKET_VLLM_TRUE_PARTIAL_EVIDENCE_ROOT"
TOKENIZER_IDENTITY_ENV = "PUTPOCKET_VLLM_TRUE_PARTIAL_TOKENIZER_IDENTITY_SHA256"
SERIALIZER_IDENTITY_ENV = "PUTPOCKET_VLLM_TRUE_PARTIAL_SERIALIZER_IDENTITY_SHA256"
XARG_PREFIX = "putpocket_true_partial_"
XARG_OPERATION = f"{XARG_PREFIX}operation"
XARG_MANIFEST_PATH = f"{XARG_PREFIX}manifest_path"
XARG_MANIFEST_SHA256 = f"{XARG_PREFIX}manifest_sha256"
XARG_OWNER_TOKEN = f"{XARG_PREFIX}owner_token"
VALID_OPERATIONS = frozenset({"donor_register", "target_sparse"})
HEX64_RE = re.compile(r"[0-9a-f]{64}")
HANDLE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{15,127}")
RUNTIME_KEYS = frozenset(
    {
        "vllm_commit",
        "model",
        "model_revision",
        "architecture",
        "tokenizer",
        "tokenizer_revision",
        "tokenizer_identity_sha256",
        "serializer_identity_sha256",
        "kv_cache_dtype",
        "attention_backend",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    }
)
CACHE_KEYS = frozenset(
    {
        "block_size",
        "main_dtype",
        "main_row_width",
        "main_layer_names",
        "indexer_dtype",
        "indexer_value_bytes",
        "indexer_scale_bytes",
        "indexer_layer_names",
        "kv_lora_rank",
        "qk_rope_head_dim",
        "num_attention_heads",
    }
)


class TruePartialPrefillError(RuntimeError):
    """The experimental request failed before donor KV could be consumed."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise TruePartialPrefillError(reason)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256(encoded)


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    return canonical_json_sha256(list(token_ids))


def _hex64(value: Any, reason: str) -> str:
    _require(isinstance(value, str) and HEX64_RE.fullmatch(value) is not None, reason)
    return value


def _strict_keys(
    value: Any,
    *,
    required: set[str] | frozenset[str],
    optional: set[str] | frozenset[str] = frozenset(),
    reason: str,
) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{reason}_NOT_OBJECT")
    keys = set(value)
    _require(required <= keys, f"{reason}_MISSING_{sorted(required - keys)}")
    _require(keys <= required | optional, f"{reason}_UNKNOWN_{sorted(keys - required - optional)}")
    return value


def _positions(value: Any, *, upper: int, reason: str, ordered: bool = False) -> tuple[int, ...]:
    _require(isinstance(value, list), f"{reason}_NOT_ARRAY")
    result = tuple(value)
    _require(all(_is_int(item) and 0 <= item < upper for item in result), f"{reason}_OUT_OF_RANGE")
    _require(len(set(result)) == len(result), f"{reason}_DUPLICATE")
    if not ordered:
        _require(result == tuple(sorted(result)), f"{reason}_NOT_SORTED")
    return result


def _under_root(path: Path, root_env: str, reason: str) -> Path:
    _require(path.is_absolute(), f"{reason}_NOT_ABSOLUTE")
    raw_root = os.getenv(root_env)
    _require(bool(raw_root), f"{root_env}_NOT_CONFIGURED")
    root = Path(raw_root).resolve(strict=True)
    # Existing symlinks are resolved; ``strict=False`` also permits a new
    # evidence file whose parent is already under the configured root.
    resolved = path.resolve(strict=False)
    _require(resolved == root or root in resolved.parents, f"{reason}_OUTSIDE_ALLOWED_ROOT")
    return resolved


@dataclass(frozen=True)
class Alignment:
    target_position: int
    donor_position: int


@dataclass(frozen=True)
class EditOperation:
    kind: str
    donor_positions: tuple[int, ...]
    target_positions: tuple[int, ...]


@dataclass(frozen=True)
class RequestInstruction:
    operation: str
    handle: str
    owner_token_sha256: str
    manifest_path: Path
    manifest_sha256: str
    runtime: dict[str, Any]
    cache_contract: dict[str, Any]
    source_token_count: int
    source_token_ids_sha256: str
    target_request_token_count: int | None
    target_request_token_ids_sha256: str | None
    target_frozen_token_count: int | None
    target_frozen_token_ids_sha256: str | None
    continuation_start: int | None
    alignment: tuple[Alignment, ...]
    edit_operations: tuple[EditOperation, ...]
    eligible_recompute_positions: tuple[int, ...]
    selection_order: tuple[int, ...]
    mandatory_recompute_positions: tuple[int, ...]
    selected_recompute_positions: tuple[int, ...]
    recompute_positions: tuple[int, ...]
    evidence_path: Path

    @property
    def is_donor(self) -> bool:
        return self.operation == "donor_register"

    @property
    def is_target(self) -> bool:
        return self.operation == "target_sparse"


@dataclass(frozen=True)
class TargetPlan:
    handle: str
    donor_token_count: int
    frozen_history_token_count: int
    continuation_start: int
    alignment: tuple[Alignment, ...]
    reuse_alignment: tuple[Alignment, ...]
    recompute_positions: tuple[int, ...]
    mandatory_recompute_positions: tuple[int, ...]
    selected_recompute_positions: tuple[int, ...]
    selection_order: tuple[int, ...]
    shifted_reuse_alignment: tuple[Alignment, ...]
    recompute_token_ids_sha256: str

    def as_scheduler_payload(self) -> dict[str, Any]:
        return {
            "handle": self.handle,
            "donor_token_count": self.donor_token_count,
            "frozen_history_token_count": self.frozen_history_token_count,
            "continuation_start": self.continuation_start,
            "alignment": [
                [pair.target_position, pair.donor_position] for pair in self.alignment
            ],
            "reuse_alignment": [
                [pair.target_position, pair.donor_position]
                for pair in self.reuse_alignment
            ],
            "recompute_positions": list(self.recompute_positions),
            "mandatory_recompute_positions": list(self.mandatory_recompute_positions),
            "selected_recompute_positions": list(self.selected_recompute_positions),
            "selection_order": list(self.selection_order),
            "shifted_reuse_alignment": [
                [pair.target_position, pair.donor_position]
                for pair in self.shifted_reuse_alignment
            ],
            "recompute_token_ids_sha256": self.recompute_token_ids_sha256,
        }


def target_plan_from_scheduler_payload(payload: Mapping[str, Any]) -> TargetPlan:
    """Rehydrate the primitive scheduler payload on every worker rank."""

    alignment = tuple(Alignment(int(t), int(d)) for t, d in payload["alignment"])
    reuse = tuple(
        Alignment(int(t), int(d)) for t, d in payload["reuse_alignment"]
    )
    shifted = tuple(
        Alignment(int(t), int(d)) for t, d in payload["shifted_reuse_alignment"]
    )
    return TargetPlan(
        handle=str(payload["handle"]),
        donor_token_count=int(payload["donor_token_count"]),
        frozen_history_token_count=int(payload["frozen_history_token_count"]),
        continuation_start=int(payload["continuation_start"]),
        alignment=alignment,
        reuse_alignment=reuse,
        recompute_positions=tuple(int(v) for v in payload["recompute_positions"]),
        mandatory_recompute_positions=tuple(
            int(v) for v in payload["mandatory_recompute_positions"]
        ),
        selected_recompute_positions=tuple(
            int(v) for v in payload["selected_recompute_positions"]
        ),
        selection_order=tuple(int(v) for v in payload["selection_order"]),
        shifted_reuse_alignment=shifted,
        recompute_token_ids_sha256=str(payload["recompute_token_ids_sha256"]),
    )


def _validate_runtime(runtime: Any) -> dict[str, Any]:
    result = _strict_keys(runtime, required=RUNTIME_KEYS, reason="RUNTIME")
    for key in (
        "vllm_commit",
        "model",
        "model_revision",
        "architecture",
        "tokenizer",
        "tokenizer_revision",
        "kv_cache_dtype",
        "attention_backend",
    ):
        _require(isinstance(result[key], str) and bool(result[key]), f"RUNTIME_{key.upper()}_INVALID")
    _hex64(result["tokenizer_identity_sha256"], "RUNTIME_TOKENIZER_IDENTITY_INVALID")
    _hex64(result["serializer_identity_sha256"], "RUNTIME_SERIALIZER_IDENTITY_INVALID")
    for key in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    ):
        _require(_is_int(result[key]) and result[key] > 0, f"RUNTIME_{key.upper()}_INVALID")
    return dict(result)


def _validate_cache_contract(cache: Any) -> dict[str, Any]:
    result = _strict_keys(cache, required=CACHE_KEYS, reason="CACHE_CONTRACT")
    _require(_is_int(result["block_size"]) and result["block_size"] > 0, "CACHE_BLOCK_SIZE_INVALID")
    _require(result["main_dtype"] == "bfloat16", "CACHE_MAIN_DTYPE_UNSUPPORTED")
    _require(_is_int(result["main_row_width"]) and result["main_row_width"] > 0, "CACHE_MAIN_WIDTH_INVALID")
    _require(result["indexer_dtype"] == "uint8", "CACHE_INDEXER_DTYPE_UNSUPPORTED")
    _require(_is_int(result["indexer_value_bytes"]) and result["indexer_value_bytes"] > 0, "CACHE_INDEXER_VALUE_BYTES_INVALID")
    _require(_is_int(result["indexer_scale_bytes"]) and result["indexer_scale_bytes"] > 0, "CACHE_INDEXER_SCALE_BYTES_INVALID")
    for key in ("kv_lora_rank", "qk_rope_head_dim", "num_attention_heads"):
        _require(_is_int(result[key]) and result[key] > 0, f"CACHE_{key.upper()}_INVALID")
    for key in ("main_layer_names", "indexer_layer_names"):
        names = result[key]
        _require(isinstance(names, list) and names, f"CACHE_{key.upper()}_INVALID")
        _require(all(isinstance(name, str) and name for name in names), f"CACHE_{key.upper()}_INVALID")
        _require(len(names) == len(set(names)), f"CACHE_{key.upper()}_DUPLICATE")
    _require(
        result["main_row_width"] == result["kv_lora_rank"] + result["qk_rope_head_dim"],
        "CACHE_MAIN_HEAD_DIMENSION_MISMATCH",
    )
    return dict(result)


def _parse_alignment(value: Any, donor_count: int, target_count: int) -> tuple[Alignment, ...]:
    _require(isinstance(value, list), "ALIGNMENT_NOT_ARRAY")
    pairs: list[Alignment] = []
    for raw in value:
        item = _strict_keys(
            raw,
            required={"target_position", "donor_position"},
            reason="ALIGNMENT_ENTRY",
        )
        target = item["target_position"]
        donor = item["donor_position"]
        _require(_is_int(target) and 0 <= target < target_count, "ALIGNMENT_TARGET_OUT_OF_RANGE")
        _require(_is_int(donor) and 0 <= donor < donor_count, "ALIGNMENT_DONOR_OUT_OF_RANGE")
        pairs.append(Alignment(target, donor))
    _require(
        tuple(pair.target_position for pair in pairs)
        == tuple(sorted(pair.target_position for pair in pairs)),
        "ALIGNMENT_NOT_TARGET_SORTED",
    )
    _require(len({pair.target_position for pair in pairs}) == len(pairs), "ALIGNMENT_DUPLICATE_TARGET")
    _require(len({pair.donor_position for pair in pairs}) == len(pairs), "ALIGNMENT_DUPLICATE_DONOR")
    return tuple(pairs)


def _parse_edit_operations(value: Any, donor_count: int, target_count: int) -> tuple[EditOperation, ...]:
    _require(isinstance(value, list) and value, "EDIT_OPERATIONS_INVALID")
    operations: list[EditOperation] = []
    for raw in value:
        item = _strict_keys(
            raw,
            required={"kind", "donor_positions", "target_positions"},
            reason="EDIT_OPERATION",
        )
        kind = item["kind"]
        _require(kind in {"delete", "insert", "replace"}, "EDIT_OPERATION_KIND_INVALID")
        donors = _positions(item["donor_positions"], upper=donor_count, reason="EDIT_DONOR_POSITIONS")
        targets = _positions(item["target_positions"], upper=target_count, reason="EDIT_TARGET_POSITIONS")
        if kind == "delete":
            _require(bool(donors) and not targets, "DELETE_OPERATION_SHAPE_INVALID")
        elif kind == "insert":
            _require(not donors and bool(targets), "INSERT_OPERATION_SHAPE_INVALID")
        else:
            _require(bool(donors) and bool(targets), "REPLACE_OPERATION_SHAPE_INVALID")
        operations.append(EditOperation(kind, donors, targets))
    return tuple(operations)


def request_instruction_from_xargs(
    extra_args: Mapping[str, Any] | None,
    prompt_token_ids: Sequence[int] | None,
) -> RequestInstruction | None:
    """Resolve and validate the namespaced request hook.

    Ordinary requests return ``None`` without consulting experimental
    environment variables.  Any partial or disabled hook fails closed.
    """

    if not extra_args:
        return None
    present = {key for key in extra_args if key.startswith(XARG_PREFIX)}
    if not present:
        return None
    expected = {XARG_OPERATION, XARG_MANIFEST_PATH, XARG_MANIFEST_SHA256, XARG_OWNER_TOKEN}
    _require(present == expected, f"REQUEST_HOOK_KEYS_INVALID_{sorted(present)}")
    _require(os.getenv(ENABLE_ENV) == "1", "TRUE_PARTIAL_PREFILL_SERVER_DISABLED")
    _require(prompt_token_ids is not None, "TRUE_PARTIAL_PREFILL_REQUIRES_TOKEN_IDS")

    operation = extra_args[XARG_OPERATION]
    _require(operation in VALID_OPERATIONS, "REQUEST_OPERATION_INVALID")
    raw_path = extra_args[XARG_MANIFEST_PATH]
    expected_sha = extra_args[XARG_MANIFEST_SHA256]
    owner_token = extra_args[XARG_OWNER_TOKEN]
    _require(isinstance(raw_path, str), "REQUEST_MANIFEST_PATH_INVALID")
    _hex64(expected_sha, "REQUEST_MANIFEST_DIGEST_INVALID")
    _require(isinstance(owner_token, str) and len(owner_token) >= 32, "REQUEST_OWNER_TOKEN_INVALID")

    manifest_path = _under_root(Path(raw_path), MANIFEST_ROOT_ENV, "REQUEST_MANIFEST")
    actual_sha = file_sha256(manifest_path)
    _require(hmac.compare_digest(actual_sha, expected_sha), "REQUEST_MANIFEST_DIGEST_MISMATCH")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = _strict_keys(
        payload,
        required={
            "schema_version",
            "backend_kind",
            "operation",
            "handle",
            "owner_token_sha256",
            "runtime",
            "cache_contract",
            "source",
            "evidence_path",
            "production_default_enabled",
            "lifetime",
        },
        optional={"target", "alignment", "edit_operations", "selection"},
        reason="MANIFEST",
    )
    _require(root["schema_version"] == 1, "MANIFEST_SCHEMA_VERSION_INVALID")
    _require(root["backend_kind"] == "true_partial_prefill", "MANIFEST_BACKEND_KIND_INVALID")
    _require(root["operation"] == operation, "MANIFEST_OPERATION_MISMATCH")
    _require(root["production_default_enabled"] is False, "MANIFEST_DEFAULT_MUST_BE_OFF")
    _require(root["lifetime"] == "single_use_after_donor_finish", "MANIFEST_LIFETIME_UNSUPPORTED")
    handle = root["handle"]
    _require(isinstance(handle, str) and HANDLE_RE.fullmatch(handle) is not None, "MANIFEST_HANDLE_INVALID")
    owner_sha = _hex64(root["owner_token_sha256"], "MANIFEST_OWNER_DIGEST_INVALID")
    _require(
        hmac.compare_digest(_sha256(owner_token.encode("utf-8")), owner_sha),
        "REQUEST_OWNER_TOKEN_MISMATCH",
    )
    runtime = _validate_runtime(root["runtime"])
    cache_contract = _validate_cache_contract(root["cache_contract"])

    configured_tokenizer = _hex64(
        os.getenv(TOKENIZER_IDENTITY_ENV), "SERVER_TOKENIZER_IDENTITY_NOT_CONFIGURED"
    )
    configured_serializer = _hex64(
        os.getenv(SERIALIZER_IDENTITY_ENV), "SERVER_SERIALIZER_IDENTITY_NOT_CONFIGURED"
    )
    _require(
        hmac.compare_digest(runtime["tokenizer_identity_sha256"], configured_tokenizer),
        "SERVER_TOKENIZER_IDENTITY_MISMATCH",
    )
    _require(
        hmac.compare_digest(runtime["serializer_identity_sha256"], configured_serializer),
        "SERVER_SERIALIZER_IDENTITY_MISMATCH",
    )

    source = _strict_keys(
        root["source"],
        required={"token_count", "token_ids_sha256"},
        reason="SOURCE",
    )
    source_count = source["token_count"]
    _require(_is_int(source_count) and source_count > 0, "SOURCE_TOKEN_COUNT_INVALID")
    source_sha = _hex64(source["token_ids_sha256"], "SOURCE_TOKEN_DIGEST_INVALID")

    evidence_path = _under_root(Path(root["evidence_path"]), EVIDENCE_ROOT_ENV, "EVIDENCE_PATH")
    _require(evidence_path.is_file() or not evidence_path.exists(), "EVIDENCE_PATH_NOT_FILE")

    target_request_count: int | None = None
    target_request_sha: str | None = None
    target_frozen_count: int | None = None
    target_frozen_sha: str | None = None
    continuation_start: int | None = None
    alignment: tuple[Alignment, ...] = ()
    edits: tuple[EditOperation, ...] = ()
    eligible: tuple[int, ...] = ()
    ranking: tuple[int, ...] = ()
    mandatory: tuple[int, ...] = ()
    selected: tuple[int, ...] = ()
    recompute: tuple[int, ...] = ()

    prompt_ids = tuple(prompt_token_ids)
    _require(all(_is_int(token) and token >= 0 for token in prompt_ids), "REQUEST_TOKEN_IDS_INVALID")
    if operation == "donor_register":
        _require(not ({"target", "alignment", "edit_operations", "selection"} & set(root)), "DONOR_MANIFEST_HAS_TARGET_FIELDS")
        _require(len(prompt_ids) == source_count, "DONOR_PROMPT_LENGTH_MISMATCH")
        _require(token_ids_sha256(prompt_ids) == source_sha, "DONOR_PROMPT_DIGEST_MISMATCH")
    else:
        _require({"target", "alignment", "edit_operations", "selection"} <= set(root), "TARGET_MANIFEST_FIELDS_MISSING")
        target = _strict_keys(
            root["target"],
            required={
                "request_token_count",
                "request_token_ids_sha256",
                "frozen_history_token_count",
                "frozen_history_token_ids_sha256",
                "continuation_start",
            },
            reason="TARGET",
        )
        target_request_count = target["request_token_count"]
        target_frozen_count = target["frozen_history_token_count"]
        continuation_start = target["continuation_start"]
        _require(_is_int(target_request_count) and target_request_count > 0, "TARGET_REQUEST_LENGTH_INVALID")
        _require(
            _is_int(target_frozen_count)
            and 0 < target_frozen_count < target_request_count,
            "TARGET_REQUIRES_NORMAL_CONTINUATION_TOKEN",
        )
        _require(continuation_start == target_frozen_count, "TARGET_CONTINUATION_BOUNDARY_INVALID")
        target_request_sha = _hex64(target["request_token_ids_sha256"], "TARGET_REQUEST_DIGEST_INVALID")
        target_frozen_sha = _hex64(target["frozen_history_token_ids_sha256"], "TARGET_FROZEN_DIGEST_INVALID")
        _require(len(prompt_ids) == target_request_count, "TARGET_PROMPT_LENGTH_MISMATCH")
        _require(token_ids_sha256(prompt_ids) == target_request_sha, "TARGET_PROMPT_DIGEST_MISMATCH")
        _require(token_ids_sha256(prompt_ids[:target_frozen_count]) == target_frozen_sha, "TARGET_FROZEN_DIGEST_MISMATCH")

        alignment = _parse_alignment(root["alignment"], source_count, target_frozen_count)
        edits = _parse_edit_operations(root["edit_operations"], source_count, target_frozen_count)
        selection = _strict_keys(
            root["selection"],
            required={
                "semantic_quantity",
                "score_direction",
                "eligible_recompute_positions",
                "selection_order",
                "mandatory_recompute_positions",
                "selected_recompute_positions",
                "recompute_positions",
            },
            reason="SELECTION",
        )
        _require(selection["semantic_quantity"] == "importance_for_recompute", "SELECTION_QUANTITY_INVALID")
        _require(selection["score_direction"] == "higher_score_first", "SELECTION_DIRECTION_INVALID")
        eligible = _positions(selection["eligible_recompute_positions"], upper=target_frozen_count, reason="ELIGIBLE_RECOMPUTE")
        ranking = _positions(selection["selection_order"], upper=target_frozen_count, reason="SELECTION_ORDER", ordered=True)
        mandatory = _positions(selection["mandatory_recompute_positions"], upper=target_frozen_count, reason="MANDATORY_RECOMPUTE")
        selected = _positions(selection["selected_recompute_positions"], upper=target_frozen_count, reason="SELECTED_RECOMPUTE", ordered=True)
        recompute = _positions(selection["recompute_positions"], upper=target_frozen_count, reason="RECOMPUTE")
        _require(set(ranking) == set(eligible) and len(ranking) == len(eligible), "SELECTION_ORDER_NOT_ELIGIBLE_PERMUTATION")
        _require(selected == ranking[: len(selected)], "SELECTED_RECOMPUTE_NOT_RANKING_PREFIX")
        _require(set(mandatory).isdisjoint(selected), "MANDATORY_SELECTED_OVERLAP")
        _require(set(recompute) == set(mandatory) | set(selected), "RECOMPUTE_UNION_INVALID")

    return RequestInstruction(
        operation=operation,
        handle=handle,
        owner_token_sha256=owner_sha,
        manifest_path=manifest_path,
        manifest_sha256=actual_sha,
        runtime=runtime,
        cache_contract=cache_contract,
        source_token_count=source_count,
        source_token_ids_sha256=source_sha,
        target_request_token_count=target_request_count,
        target_request_token_ids_sha256=target_request_sha,
        target_frozen_token_count=target_frozen_count,
        target_frozen_token_ids_sha256=target_frozen_sha,
        continuation_start=continuation_start,
        alignment=alignment,
        edit_operations=edits,
        eligible_recompute_positions=eligible,
        selection_order=ranking,
        mandatory_recompute_positions=mandatory,
        selected_recompute_positions=selected,
        recompute_positions=recompute,
        evidence_path=evidence_path,
    )


def assert_runtime_compatible(declared: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    _require(set(declared) == RUNTIME_KEYS, "DECLARED_RUNTIME_FIELDS_INVALID")
    _require(set(actual) == RUNTIME_KEYS, "ACTUAL_RUNTIME_FIELDS_INVALID")
    for key in sorted(RUNTIME_KEYS):
        _require(declared[key] == actual[key], f"RUNTIME_MISMATCH_{key.upper()}")


def build_target_plan(
    instruction: RequestInstruction,
    donor_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
) -> TargetPlan:
    _require(instruction.is_target, "TARGET_PLAN_REQUIRES_TARGET_INSTRUCTION")
    frozen_count = instruction.target_frozen_token_count
    continuation_start = instruction.continuation_start
    _require(frozen_count is not None and continuation_start is not None, "TARGET_BOUNDARIES_MISSING")
    donor = tuple(donor_token_ids)
    target = tuple(target_token_ids[:frozen_count])
    _require(len(donor) == instruction.source_token_count, "LIVE_DONOR_LENGTH_MISMATCH")
    _require(token_ids_sha256(donor) == instruction.source_token_ids_sha256, "LIVE_DONOR_DIGEST_MISMATCH")
    _require(len(target) == frozen_count, "LIVE_TARGET_LENGTH_MISMATCH")
    _require(token_ids_sha256(target) == instruction.target_frozen_token_ids_sha256, "LIVE_TARGET_DIGEST_MISMATCH")

    aligned_targets = {pair.target_position for pair in instruction.alignment}
    aligned_donors = {pair.donor_position for pair in instruction.alignment}
    for pair in instruction.alignment:
        _require(
            target[pair.target_position] == donor[pair.donor_position],
            "ALIGNED_TOKEN_IDENTITY_MISMATCH",
        )

    removed_donors: set[int] = set()
    new_targets: set[int] = set()
    for edit in instruction.edit_operations:
        _require(removed_donors.isdisjoint(edit.donor_positions), "EDIT_DUPLICATE_DONOR_POSITION")
        _require(new_targets.isdisjoint(edit.target_positions), "EDIT_DUPLICATE_TARGET_POSITION")
        removed_donors.update(edit.donor_positions)
        new_targets.update(edit.target_positions)
    _require(aligned_donors.isdisjoint(removed_donors), "DELETED_OR_REPLACED_DONOR_REUSED")
    _require(aligned_targets.isdisjoint(new_targets), "INSERTED_OR_REPLACED_TARGET_ALIGNED")
    _require(aligned_donors | removed_donors == set(range(len(donor))), "DONOR_ALIGNMENT_NOT_TOTAL")
    _require(aligned_targets | new_targets == set(range(frozen_count)), "TARGET_ALIGNMENT_NOT_TOTAL")

    mandatory = set(instruction.mandatory_recompute_positions)
    selected = set(instruction.selected_recompute_positions)
    recompute = set(instruction.recompute_positions)
    _require(new_targets <= mandatory, "EDIT_TARGET_NOT_MANDATORY_RECOMPUTE")
    _require(selected <= aligned_targets, "SELECTED_RECOMPUTE_NOT_ALIGNED")
    _require(set(instruction.eligible_recompute_positions) <= aligned_targets, "ELIGIBLE_RECOMPUTE_NOT_ALIGNED")
    _require(recompute == mandatory | selected, "TARGET_RECOMPUTE_SET_INVALID")
    _require(set(range(frozen_count)) == aligned_targets | mandatory, "TARGET_POSITION_HAS_NO_REUSE_OR_RECOMPUTE_AUTHORITY")

    reuse = tuple(
        pair for pair in instruction.alignment if pair.target_position not in recompute
    )
    reuse_targets = {pair.target_position for pair in reuse}
    _require(reuse_targets.isdisjoint(recompute), "REUSE_RECOMPUTE_TARGET_OVERLAP")
    _require(reuse_targets | recompute == set(range(frozen_count)), "TARGET_WRITE_COVERAGE_INVALID")
    shifted = tuple(pair for pair in reuse if pair.target_position != pair.donor_position)
    recompute_ids = [target[position] for position in instruction.recompute_positions]
    return TargetPlan(
        handle=instruction.handle,
        donor_token_count=len(donor),
        frozen_history_token_count=frozen_count,
        continuation_start=continuation_start,
        alignment=instruction.alignment,
        reuse_alignment=reuse,
        recompute_positions=instruction.recompute_positions,
        mandatory_recompute_positions=instruction.mandatory_recompute_positions,
        selected_recompute_positions=instruction.selected_recompute_positions,
        selection_order=instruction.selection_order,
        shifted_reuse_alignment=shifted,
        recompute_token_ids_sha256=token_ids_sha256(recompute_ids),
    )


@dataclass
class DonorRecord:
    request_id: str
    instruction: RequestInstruction
    token_ids: tuple[int, ...]
    block_groups: tuple[tuple[Any, ...], ...]
    block_object_ids: tuple[tuple[int, ...], ...]
    block_ids: tuple[tuple[int, ...], ...]
    state: str = "live_pinned"
    consumer_request_id: str | None = None


class DonorRegistry:
    """Scheduler-owned, one-use registry of explicitly pinned donor blocks."""

    def __init__(self) -> None:
        self._records: dict[str, DonorRecord] = {}

    def register(
        self,
        request_id: str,
        instruction: RequestInstruction,
        token_ids: Sequence[int],
        block_groups: Sequence[Sequence[Any]],
    ) -> DonorRecord:
        _require(instruction.is_donor, "DONOR_REGISTRATION_REQUIRES_DONOR_INSTRUCTION")
        _require(instruction.handle not in self._records, "DONOR_HANDLE_ALREADY_REGISTERED")
        tokens = tuple(token_ids)
        _require(len(tokens) == instruction.source_token_count, "DONOR_REGISTER_LENGTH_MISMATCH")
        _require(token_ids_sha256(tokens) == instruction.source_token_ids_sha256, "DONOR_REGISTER_DIGEST_MISMATCH")
        groups = tuple(tuple(group) for group in block_groups)
        _require(bool(groups) and all(groups), "DONOR_BLOCK_GROUPS_EMPTY")
        for group in groups:
            for block in group:
                _require(getattr(block, "ref_cnt", 0) > 0, "DONOR_BLOCK_NOT_LIVE")
        record = DonorRecord(
            request_id=request_id,
            instruction=instruction,
            token_ids=tokens,
            block_groups=groups,
            block_object_ids=tuple(tuple(id(block) for block in group) for group in groups),
            block_ids=tuple(tuple(int(block.block_id) for block in group) for group in groups),
        )
        self._records[instruction.handle] = record
        return record

    def resolve(
        self,
        consumer_request_id: str,
        instruction: RequestInstruction,
        target_token_ids: Sequence[int],
    ) -> tuple[DonorRecord, TargetPlan]:
        _require(instruction.is_target, "DONOR_RESOLVE_REQUIRES_TARGET_INSTRUCTION")
        record = self._records.get(instruction.handle)
        _require(record is not None, "DONOR_HANDLE_NOT_LIVE")
        _require(record.state == "live_pinned", "DONOR_HANDLE_ALREADY_CONSUMED")
        _require(record.instruction.owner_token_sha256 == instruction.owner_token_sha256, "DONOR_OWNER_MISMATCH")
        _require(record.instruction.runtime == instruction.runtime, "DONOR_RUNTIME_IDENTITY_MISMATCH")
        _require(record.instruction.cache_contract == instruction.cache_contract, "DONOR_CACHE_CONTRACT_MISMATCH")
        _require(record.instruction.source_token_count == instruction.source_token_count, "DONOR_SOURCE_LENGTH_CONTRACT_MISMATCH")
        _require(record.instruction.source_token_ids_sha256 == instruction.source_token_ids_sha256, "DONOR_SOURCE_DIGEST_CONTRACT_MISMATCH")
        _require(len(record.block_groups) == len(record.block_ids), "DONOR_BLOCK_GROUP_COUNT_MUTATED")
        for objects, object_ids, ids in zip(
            record.block_groups, record.block_object_ids, record.block_ids, strict=True
        ):
            _require(len(objects) == len(object_ids) == len(ids), "DONOR_BLOCK_TABLE_MUTATED")
            for block, object_id, block_id in zip(objects, object_ids, ids, strict=True):
                _require(id(block) == object_id and int(block.block_id) == block_id, "DONOR_BLOCK_IDENTITY_MUTATED")
                _require(getattr(block, "ref_cnt", 0) > 0, "DONOR_BLOCK_EVICTED_OR_FREED")
        plan = build_target_plan(instruction, record.token_ids, target_token_ids)
        record.state = "consuming"
        record.consumer_request_id = consumer_request_id
        return record, plan

    def release(self, handle: str, consumer_request_id: str) -> DonorRecord:
        record = self._records.get(handle)
        _require(record is not None, "DONOR_RELEASE_HANDLE_UNKNOWN")
        _require(record.state == "consuming", "DONOR_RELEASE_STATE_INVALID")
        _require(record.consumer_request_id == consumer_request_id, "DONOR_RELEASE_CONSUMER_MISMATCH")
        record.state = "released"
        del self._records[handle]
        return record

    def cancel_consumption(self, handle: str, consumer_request_id: str) -> None:
        """Return a resolved donor to live state when target allocation stalls."""

        record = self._records.get(handle)
        _require(record is not None, "DONOR_CANCEL_HANDLE_UNKNOWN")
        _require(record.state == "consuming", "DONOR_CANCEL_STATE_INVALID")
        _require(record.consumer_request_id == consumer_request_id, "DONOR_CANCEL_CONSUMER_MISMATCH")
        record.state = "live_pinned"
        record.consumer_request_id = None

    def get(self, handle: str) -> DonorRecord | None:
        return self._records.get(handle)


def required_blocks(token_count: int, block_size: int) -> int:
    _require(token_count > 0 and block_size > 0, "BLOCK_COUNT_INPUT_INVALID")
    return (token_count + block_size - 1) // block_size


def verify_exact_slot_mappings(
    *,
    target_block_ids: Sequence[Sequence[int]],
    positions: Sequence[int],
    actual_slot_ids: Sequence[Sequence[int]],
    block_size: int,
) -> dict[str, Any]:
    """Attest that cache writes use the exact requested logical positions."""

    _require(block_size > 0, "SLOT_MAPPING_BLOCK_SIZE_INVALID")
    logical_positions = tuple(positions)
    _require(
        logical_positions
        and all(_is_int(position) and position >= 0 for position in logical_positions),
        "SLOT_MAPPING_POSITIONS_INVALID",
    )
    _require(
        len(target_block_ids) == len(actual_slot_ids) and bool(target_block_ids),
        "SLOT_MAPPING_GROUP_COUNT_MISMATCH",
    )
    expected_by_group: list[list[int]] = []
    for group_index, (blocks, actual) in enumerate(
        zip(target_block_ids, actual_slot_ids, strict=True)
    ):
        expected: list[int] = []
        for position in logical_positions:
            page, offset = divmod(position, block_size)
            _require(page < len(blocks), "SLOT_MAPPING_BLOCK_TABLE_TOO_SHORT")
            expected.append(int(blocks[page]) * block_size + offset)
        actual_values = [int(value) for value in actual]
        _require(
            actual_values == expected,
            f"SLOT_MAPPING_EXACT_POSITION_MISMATCH_GROUP_{group_index}",
        )
        expected_by_group.append(expected)
    return {
        "slot_mapping_positions": list(logical_positions),
        "slot_mapping_positions_sha256": canonical_json_sha256(
            list(logical_positions)
        ),
        "slot_mapping_by_group_sha256": canonical_json_sha256(expected_by_group),
        "slot_mapping_exact_positions_verified": True,
    }


CONTINUATION_IMMUTABLE_KEYS = (
    "request_id",
    "manifest_sha256",
    "evidence_path",
    "frozen_history_token_count",
    "target_request_token_count",
    "block_size",
    "sparse_recompute_count",
)


def build_continuation_payload(
    pending: Mapping[str, Any],
    current_target_block_ids: Sequence[Sequence[int]],
    num_scheduled_tokens: int,
) -> dict[str, Any]:
    """Bind immutable post-patch state to the refreshed Q2 block table."""

    required = set(CONTINUATION_IMMUTABLE_KEYS) | {"initial_target_block_ids"}
    _require(set(pending) == required, "CONTINUATION_PENDING_FIELDS_INVALID")
    _require(num_scheduled_tokens > 0, "CONTINUATION_SCHEDULED_TOKEN_COUNT_INVALID")
    initial = [list(group) for group in pending["initial_target_block_ids"]]
    current = [list(group) for group in current_target_block_ids]
    _require(
        len(initial) == len(current) and bool(current),
        "CONTINUATION_BLOCK_GROUP_COUNT_MUTATED",
    )
    for initial_group, current_group in zip(initial, current, strict=True):
        _require(
            current_group[: len(initial_group)] == initial_group,
            "CONTINUATION_INITIAL_BLOCK_PREFIX_MUTATED",
        )
    payload = {key: pending[key] for key in CONTINUATION_IMMUTABLE_KEYS}
    payload["target_block_ids"] = current
    payload["num_scheduled_tokens"] = num_scheduled_tokens
    return payload


def validate_continuation_payload(
    pending: Mapping[str, Any], emitted: Mapping[str, Any]
) -> None:
    rebuilt = build_continuation_payload(
        pending,
        emitted.get("target_block_ids", ()),
        emitted.get("num_scheduled_tokens", 0),
    )
    _require(dict(emitted) == rebuilt, "CONTINUATION_ATTESTATION_MUTATED")


def _copy_main_row(
    cache: torch.Tensor,
    src_block: int,
    src_offset: int,
    dst_block: int,
    dst_offset: int,
    row_width: int,
) -> None:
    _require(cache.ndim == 3 and cache.dtype == torch.bfloat16, "MAIN_CACHE_LAYOUT_UNSUPPORTED")
    _require(cache.shape[1] > max(src_offset, dst_offset) and cache.shape[2] == row_width, "MAIN_CACHE_SHAPE_MISMATCH")
    _require(max(src_block, dst_block) < cache.shape[0], "MAIN_CACHE_BLOCK_OUT_OF_RANGE")
    cache[dst_block, dst_offset].copy_(cache[src_block, src_offset])


def _copy_indexer_row(
    cache: torch.Tensor,
    src_block: int,
    src_offset: int,
    dst_block: int,
    dst_offset: int,
    block_size: int,
    value_bytes: int,
    scale_bytes: int,
) -> None:
    _require(cache.ndim == 3 and cache.dtype == torch.uint8, "INDEXER_CACHE_LAYOUT_UNSUPPORTED")
    _require(cache.shape[1] == block_size, "INDEXER_CACHE_BLOCK_SIZE_MISMATCH")
    _require(cache.shape[2] == value_bytes + scale_bytes, "INDEXER_CACHE_ROW_WIDTH_MISMATCH")
    _require(max(src_block, dst_block) < cache.shape[0], "INDEXER_CACHE_BLOCK_OUT_OF_RANGE")
    src = cache[src_block]
    dst = cache[dst_block]
    _require(src.is_contiguous() and dst.is_contiguous(), "INDEXER_CACHE_PAGE_NOT_CONTIGUOUS")
    src_flat = src.view(-1)
    dst_flat = dst.view(-1)
    src_value = src_offset * value_bytes
    dst_value = dst_offset * value_bytes
    dst_flat[dst_value : dst_value + value_bytes].copy_(
        src_flat[src_value : src_value + value_bytes]
    )
    scale_plane = block_size * value_bytes
    src_scale = scale_plane + src_offset * scale_bytes
    dst_scale = scale_plane + dst_offset * scale_bytes
    dst_flat[dst_scale : dst_scale + scale_bytes].copy_(
        src_flat[src_scale : src_scale + scale_bytes]
    )


def copy_reuse_rows_inplace(
    *,
    layer_caches: Iterable[tuple[str, int, torch.Tensor]],
    donor_block_ids: Sequence[Sequence[int]],
    target_block_ids: Sequence[Sequence[int]],
    plan: TargetPlan,
    cache_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy aligned reuse rows for every declared cache layer.

    ``layer_caches`` contains ``(layer_name, kv_cache_group_id, tensor)``.  The
    function understands the pinned FLASHMLA BF16 row layout and its packed
    FP8 indexer layout and rejects every other tensor.
    """

    _validate_cache_contract(dict(cache_contract))
    block_size = int(cache_contract["block_size"])
    _require(len(donor_block_ids) == len(target_block_ids), "CACHE_GROUP_COUNT_MISMATCH")
    source_needed = required_blocks(plan.donor_token_count, block_size)
    target_needed = required_blocks(plan.frozen_history_token_count, block_size)
    for source, target in zip(donor_block_ids, target_block_ids, strict=True):
        _require(len(source) >= source_needed, "DONOR_BLOCK_TABLE_TOO_SHORT")
        _require(len(target) >= target_needed, "TARGET_BLOCK_TABLE_TOO_SHORT")
        _require(set(source).isdisjoint(target), "DONOR_TARGET_BLOCK_ALIAS_UNSUPPORTED")

    expected_main = set(cache_contract["main_layer_names"])
    expected_indexer = set(cache_contract["indexer_layer_names"])
    seen_main: set[str] = set()
    seen_indexer: set[str] = set()
    writes_per_layer: dict[str, int] = {}
    seen_views: set[tuple[int, int, tuple[int, ...], tuple[int, ...]]] = set()
    for layer_name, group_id, cache in layer_caches:
        _require(0 <= group_id < len(donor_block_ids), "LAYER_CACHE_GROUP_OUT_OF_RANGE")
        view_key = (
            cache.untyped_storage().data_ptr(),
            cache.storage_offset(),
            tuple(cache.shape),
            tuple(cache.stride()),
        )
        if view_key in seen_views:
            continue
        seen_views.add(view_key)
        if layer_name in expected_main:
            seen_main.add(layer_name)
            copy = "main"
        elif layer_name in expected_indexer:
            seen_indexer.add(layer_name)
            copy = "indexer"
        else:
            raise TruePartialPrefillError(f"UNDECLARED_CACHE_LAYER_{layer_name}")
        source_blocks = donor_block_ids[group_id]
        target_blocks = target_block_ids[group_id]
        for pair in plan.reuse_alignment:
            src_page_index, src_offset = divmod(pair.donor_position, block_size)
            dst_page_index, dst_offset = divmod(pair.target_position, block_size)
            src_block = int(source_blocks[src_page_index])
            dst_block = int(target_blocks[dst_page_index])
            if copy == "main":
                _copy_main_row(
                    cache,
                    src_block,
                    src_offset,
                    dst_block,
                    dst_offset,
                    int(cache_contract["main_row_width"]),
                )
            else:
                _copy_indexer_row(
                    cache,
                    src_block,
                    src_offset,
                    dst_block,
                    dst_offset,
                    block_size,
                    int(cache_contract["indexer_value_bytes"]),
                    int(cache_contract["indexer_scale_bytes"]),
                )
        writes_per_layer[layer_name] = len(plan.reuse_alignment)

    _require(seen_main == expected_main, f"MAIN_LAYER_SET_MISMATCH_{sorted(expected_main - seen_main)}")
    _require(seen_indexer == expected_indexer, f"INDEXER_LAYER_SET_MISMATCH_{sorted(expected_indexer - seen_indexer)}")
    return {
        "copied_reuse_rows_per_layer": len(plan.reuse_alignment),
        "copied_main_layer_count": len(seen_main),
        "copied_indexer_layer_count": len(seen_indexer),
        "layer_write_counts_sha256": canonical_json_sha256(writes_per_layer),
        "shifted_reused_row_count": len(plan.shifted_reuse_alignment),
        "shifted_reused_rows_are_rope_correct": False,
        "shifted_reuse_policy": "preserve_donor_kv_bytes_without_rope_rerotation",
    }


def execution_evidence(
    *,
    request_id: str,
    manifest_sha256: str,
    plan: TargetPlan,
    model_input_positions: Sequence[int],
    model_input_token_ids: Sequence[int],
    copy_evidence: Mapping[str, Any],
    rank: int,
) -> dict[str, Any]:
    positions = tuple(model_input_positions)
    _require(positions == plan.recompute_positions, "MODEL_INPUT_RECOMPUTE_POSITIONS_MISMATCH")
    _require(len(positions) < plan.frozen_history_token_count, "FULL_PREFILL_PROHIBITION_TRIGGERED")
    _require(
        token_ids_sha256(model_input_token_ids) == plan.recompute_token_ids_sha256,
        "MODEL_INPUT_RECOMPUTE_TOKEN_DIGEST_MISMATCH",
    )
    return {
        "schema_version": 1,
        "event": "putpocket_true_partial_sparse_patch",
        "request_id": request_id,
        "handle": plan.handle,
        "rank": rank,
        "manifest_sha256": manifest_sha256,
        "backend_kind": "true_partial_prefill",
        "compute_saving_claim_requires_gpu_validation": True,
        "full_target_prefill_executed": False,
        "full_prefill_then_overwrite_prohibited": True,
        "frozen_history_token_count": plan.frozen_history_token_count,
        "model_input_frozen_history_row_count": len(positions),
        "model_input_positions": list(positions),
        "model_input_positions_sha256": canonical_json_sha256(list(positions)),
        "model_input_token_ids_sha256": token_ids_sha256(model_input_token_ids),
        "input_tokens_selected_at_exact_recompute_positions": True,
        "mandatory_recompute_positions": list(plan.mandatory_recompute_positions),
        "selected_recompute_positions": list(plan.selected_recompute_positions),
        "selection_order": list(plan.selection_order),
        "reuse_alignment": [
            [pair.target_position, pair.donor_position] for pair in plan.reuse_alignment
        ],
        "continuation_start": plan.continuation_start,
        **dict(copy_evidence),
    }


def continuation_execution_evidence(
    *,
    request_id: str,
    manifest_sha256: str,
    frozen_history_token_count: int,
    target_request_token_count: int,
    scheduler_num_computed_tokens: int,
    model_input_positions: Sequence[int],
    model_input_token_ids: Sequence[int],
    target_token_ids: Sequence[int],
    slot_mapping_evidence: Mapping[str, Any],
    rank: int,
) -> dict[str, Any]:
    """Fail closed if the first post-patch step retained a sparse-row boundary."""

    _require(
        target_request_token_count > frozen_history_token_count,
        "CONTINUATION_TOKEN_REQUIRED",
    )
    _require(
        scheduler_num_computed_tokens == frozen_history_token_count,
        "CONTINUATION_SCHEDULER_BOUNDARY_STALE",
    )
    positions = tuple(model_input_positions)
    expected_positions = tuple(
        range(frozen_history_token_count, frozen_history_token_count + len(positions))
    )
    _require(bool(positions), "CONTINUATION_MODEL_INPUT_EMPTY")
    _require(
        positions == expected_positions,
        "CONTINUATION_MODEL_POSITIONS_NOT_ABSOLUTE_Q2_PREFIX",
    )
    _require(
        positions[-1] < target_request_token_count,
        "CONTINUATION_MODEL_POSITION_OUT_OF_RANGE",
    )
    expected_token_ids = [target_token_ids[position] for position in positions]
    _require(
        list(model_input_token_ids) == expected_token_ids,
        "CONTINUATION_MODEL_TOKEN_SELECTION_MISMATCH",
    )
    return {
        "schema_version": 1,
        "event": "putpocket_true_partial_first_normal_continuation",
        "request_id": request_id,
        "rank": rank,
        "manifest_sha256": manifest_sha256,
        "backend_kind": "true_partial_prefill",
        "frozen_history_token_count": frozen_history_token_count,
        "scheduler_num_computed_tokens": scheduler_num_computed_tokens,
        "worker_num_computed_tokens": scheduler_num_computed_tokens,
        "normal_continuation_start": frozen_history_token_count,
        "model_input_positions": list(positions),
        "model_input_token_ids_sha256": token_ids_sha256(model_input_token_ids),
        "input_tokens_selected_at_exact_continuation_positions": True,
        "stale_recompute_count_boundary_observed": False,
        **dict(slot_mapping_evidence),
    }


def write_evidence_event(path: Path, event: Mapping[str, Any]) -> None:
    resolved = _under_root(path, EVIDENCE_ROOT_ENV, "EVIDENCE_PATH")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        dict(event), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ) + "\n"
    fd = os.open(resolved, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
