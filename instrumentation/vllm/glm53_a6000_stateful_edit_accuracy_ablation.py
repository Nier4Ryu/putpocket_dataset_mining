# SPDX-License-Identifier: Apache-2.0
"""Default-OFF GLM-5.3 stateful-edit accuracy-ablation hook for SM86 builds.

The current official vLLM source contains GLM5Next and its native compressed
pool indexer, but no sparse MLA/indexer execution backend supports SM86.  The
hook therefore preserves the exact control/schema boundary and cache-product
semantics while refusing every attempted activation on this image.  Absence of
the controls is a zero-overhead no-op.  It is never true partial prefill.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Sequence


CONTROL_ENV = "PUTPOCKET_GLM53_STATEFUL_EDIT_CONTROL"
SELECTOR_CAPTURE_ENV = "PUTPOCKET_GLM53_BASE_SELECTOR_CAPTURE"
MODEL_ID = "zai-org/GLM-5.3-Flash-BF16"
MODEL_REVISION = "a5b45eb41df6402735dedc900be14a42e8d5e538"
MODEL_ARCHITECTURE = "Glm5NextForConditionalGeneration"
VLLM_COMMIT = "9cd956c7e6cf54efa366b803cafa15ec6c2df827"
ATTENTION_BACKEND_POLICY = "official_sparse_mla_required_but_unavailable_on_sm86"
KV_CACHE_DTYPE = "bfloat16"
TARGET_COMPUTE_CAPABILITY = "8.6"
SM86_RUNTIME_SUPPORTED = False
MLA_LAYERS = tuple(range(3, 45, 4))
INDEXER_LAYERS = MLA_LAYERS
KDA_LAYERS = tuple(layer for layer in range(45) if layer not in MLA_LAYERS)
INDEX_KPOOL = 4
MAIN_CACHE_ROW_ELEMENTS = 512
MAIN_CACHE_ROW_BYTES = 1024
INDEXER_CACHE_ROW_BYTES = 132
UNSAFE_ACK = "I_UNDERSTAND_FULL_TARGET_PREFILL_THEN_DONOR_OVERWRITE"
_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


class StatefulEditInvariantError(RuntimeError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise StatefulEditInvariantError(reason)


def token_ids_sha256(values: Sequence[int]) -> str:
    encoded = json.dumps(list(values), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def complete_pool_ends(
    selected_positions: Sequence[int], eligible_start: int, eligible_end: int
) -> tuple[int, ...]:
    selected = tuple(int(value) for value in selected_positions)
    _require(selected == tuple(sorted(selected)), "SELECTED_POSITIONS_NOT_SORTED")
    _require(len(selected) == len(set(selected)), "SELECTED_POSITION_DUPLICATE")
    _require(
        all(eligible_start <= value < eligible_end for value in selected),
        "SELECTED_POSITION_OUT_OF_RANGE",
    )
    selected_set = set(selected)
    first_end = ((eligible_start + INDEX_KPOOL - 1) // INDEX_KPOOL) * INDEX_KPOOL - 1
    return tuple(
        end
        for end in range(first_end, eligible_end, INDEX_KPOOL)
        if end - INDEX_KPOOL + 1 >= eligible_start
        and all(
            position in selected_set
            for position in range(end - INDEX_KPOOL + 1, end + 1)
        )
    )


def _load_json_control(raw_path: str, expected_mode: str) -> dict[str, Any]:
    path = Path(raw_path)
    _require(path.is_absolute() and path.is_file(), "CONTROL_PATH_INVALID")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 4, "CONTROL_SCHEMA_INVALID")
    _require(payload.get("mode") == expected_mode, "CONTROL_MODE_INVALID")
    model = payload.get("model", {})
    expected_model = {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "architecture": MODEL_ARCHITECTURE,
        "vllm_commit": VLLM_COMMIT,
        "attention_backend_policy": ATTENTION_BACKEND_POLICY,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "index_kpool": INDEX_KPOOL,
    }
    _require(model == expected_model, "CONTROL_MODEL_RUNTIME_MISMATCH")
    _require(
        payload.get("target_compute_capability") == TARGET_COMPUTE_CAPABILITY,
        "CONTROL_COMPUTE_CAPABILITY_MISMATCH",
    )
    _require(payload.get("sm86_runtime_supported") is False, "SM86_SUPPORT_LIE")
    return payload


def validate_stateful_control(path: str | Path) -> dict[str, Any]:
    payload = _load_json_control(str(path), "STATEFUL_EDIT_ACCURACY_ABLATION")
    _require(
        payload.get("unsafe_accuracy_ablation_ack") == UNSAFE_ACK,
        "UNSAFE_ACK_MISSING",
    )
    _require(payload.get("production_default_enabled") is False, "DEFAULT_ON_FORBIDDEN")
    _require(
        payload.get("compute_semantics")
        == "full_target_prefill_then_selected_donor_cache_overwrite",
        "COMPUTE_SEMANTICS_INVALID",
    )
    _require(payload.get("true_partial_prefill") is False, "TRUE_PARTIAL_CLAIM_FORBIDDEN")
    _require(payload.get("speedup_claim") is False, "SPEEDUP_CLAIM_FORBIDDEN")
    history = payload.get("history_token_count")
    start = payload.get("eligible_start")
    edits = payload.get("edit_positions", [])
    selected = payload.get("selected_main_positions", [])
    _require(type(history) is int and history > 0, "HISTORY_COUNT_INVALID")
    _require(type(start) is int and 0 < start <= history, "ELIGIBLE_RANGE_INVALID")
    _require(
        isinstance(edits, list) and edits and max(edits) < start,
        "MANDATORY_EDIT_CLOSURE_INVALID",
    )
    _require(
        token_ids_sha256(selected) == payload.get("selected_main_positions_sha256"),
        "SELECTOR_POSITION_DIGEST_INVALID",
    )
    _require(not set(edits).intersection(selected), "EDIT_POSITION_REUSED")
    selector = Path(payload.get("selector_path", ""))
    selector_digest = payload.get("selector_sha256")
    _require(
        selector.is_absolute()
        and selector.is_file()
        and isinstance(selector_digest, str)
        and _DIGEST_RE.fullmatch(selector_digest) is not None,
        "SELECTOR_PROVENANCE_INVALID",
    )
    actual = hashlib.sha256(selector.read_bytes()).hexdigest()
    _require(actual == selector_digest, "SELECTOR_DIGEST_MISMATCH")
    projected = complete_pool_ends(selected, start, history)
    _require(
        list(projected) == payload.get("selected_indexer_pool_end_positions"),
        "COMPLETE_KPOOL_PROJECTION_INVALID",
    )
    return payload


def validate_selector_capture_control(path: str | Path) -> dict[str, Any]:
    payload = _load_json_control(str(path), "NATIVE_INDEXER_SELECTOR_CAPTURE")
    _require(
        payload.get("raw_score_kind") == "native_pre_topk_pre_normalization",
        "RAW_SCORE_KIND_INVALID",
    )
    _require(payload.get("layers") == list(INDEXER_LAYERS), "INDEXER_LAYERS_INVALID")
    _require(payload.get("outcome_independent") is True, "OUTCOME_ATTESTATION_MISSING")
    _require(payload.get("benchmark_outcomes_read") is False, "OUTCOME_LEAKAGE")
    return payload


def _activation_forbidden(control_name: str) -> None:
    raise StatefulEditInvariantError(
        "GLM53_SM86_RUNTIME_UNSUPPORTED:" + control_name
        + ":official sparse MLA/indexer backend unavailable"
    )


def accuracy_ablation_armed() -> bool:
    raw = os.getenv(CONTROL_ENV)
    if not raw:
        return False
    validate_stateful_control(raw)
    _activation_forbidden("stateful_edit")
    return False


def selector_capture_armed() -> bool:
    raw = os.getenv(SELECTOR_CAPTURE_ENV)
    if not raw:
        return False
    validate_selector_capture_control(raw)
    _activation_forbidden("selector_capture")
    return False


def maybe_attest_prompt(*, input_ids: Any, positions: Any) -> None:
    if os.getenv(CONTROL_ENV):
        accuracy_ablation_armed()
    if os.getenv(SELECTOR_CAPTURE_ENV):
        selector_capture_armed()


def maybe_capture_base_selector_scores(**_: Any) -> None:
    if os.getenv(SELECTOR_CAPTURE_ENV):
        selector_capture_armed()


def maybe_apply_main_cache(**_: Any) -> int:
    if os.getenv(CONTROL_ENV):
        accuracy_ablation_armed()
    return 0


def maybe_apply_indexer_cache(**_: Any) -> int:
    if os.getenv(CONTROL_ENV):
        accuracy_ablation_armed()
    return 0
