# SPDX-License-Identifier: Apache-2.0
"""Default-OFF GLM-5.3 stateful-edit-v3 cache overwrite hook.

This is an accuracy-emulation backend.  Native vLLM computes every target
prefill row first.  Only then may this hook overwrite explicitly selected MLA
rows and complete kpool indexer rows from private donor snapshots.  KDA
recurrent state and the kpool tail remain target-computed.  The module never
implements or claims selective prefill, skipped FLOPs, latency improvement, or
production-safe cache reuse.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


CONTROL_ENV = "PUTPOCKET_GLM53_STATEFUL_EDIT_CONTROL"
SELECTOR_CAPTURE_ENV = "PUTPOCKET_GLM53_BASE_SELECTOR_CAPTURE"
RUNTIME_PROFILE_ENV = "PUTPOCKET_GLM53_RUNTIME_PROFILE"
MODEL_ID = "Intel/GLM-5.3-Flash-W4A16-AutoRound"
MODEL_REVISION = "5eee1846f0321058ed73745f9aa16f2aaf0fc0a0"
MODEL_ARCHITECTURE = "Glm5NextForConditionalGeneration"
VLLM_COMMIT = "9cd956c7e6cf54efa366b803cafa15ec6c2df827"
RUNTIME_PROFILES = {
    "sm90": {
        "attention_backend": "FLASHINFER_MLA_SPARSE_SM90",
        "kv_cache_dtype": "fp8_e4m3",
        "main_row_bytes": 512,
    },
    "sm120": {
        "attention_backend": "FLASHINFER_MLA_SPARSE_SM120",
        "kv_cache_dtype": "fp8_ds_mla",
        "main_row_bytes": 656,
    },
}
RUNTIME_PROFILE = os.getenv(RUNTIME_PROFILE_ENV, "sm90")
if RUNTIME_PROFILE not in RUNTIME_PROFILES:
    raise RuntimeError("GLM53_RUNTIME_PROFILE_INVALID")
ATTENTION_BACKEND = RUNTIME_PROFILES[RUNTIME_PROFILE]["attention_backend"]
KV_CACHE_DTYPE = RUNTIME_PROFILES[RUNTIME_PROFILE]["kv_cache_dtype"]
MLA_LAYERS = tuple(range(3, 45, 4))
INDEXER_LAYERS = MLA_LAYERS
KDA_LAYERS = tuple(layer for layer in range(45) if layer not in MLA_LAYERS)
INDEX_KPOOL = 4
MAIN_ROW_BYTES = RUNTIME_PROFILES[RUNTIME_PROFILE]["main_row_bytes"]
INDEXER_ROW_BYTES = 132
RATIOS = tuple(range(0, 101, 10))
UNSAFE_ACK = "I_UNDERSTAND_FULL_TARGET_PREFILL_THEN_DONOR_OVERWRITE"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_ID_RE = re.compile(r"[A-Za-z0-9_.-]+")


class StatefulEditInvariantError(RuntimeError):
    """Runtime inputs are not exactly the frozen experimental contract."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise StatefulEditInvariantError(reason)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(values: Sequence[int]) -> str:
    return _sha256(json.dumps(list(values), separators=(",", ":")).encode())


def _complete_pool_ends(
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
        and end < eligible_end
        and all(position in selected_set for position in range(end - INDEX_KPOOL + 1, end + 1))
    )


@dataclass(frozen=True)
class _Control:
    mode: str
    experiment_id: str
    phase_id: str
    prompt_side: str
    donor_digest: str
    edited_digest: str
    history_token_count: int
    eligible_start: int
    eligible_end: int
    edit_positions: tuple[int, ...]
    requested_ratio: int
    selected_main_positions: tuple[int, ...]
    selected_indexer_pool_ends: tuple[int, ...]
    selector_sha256: str
    evidence_dir: Path
    data_parallel_rank: int

    @property
    def expected_prompt_digest(self) -> str:
        return self.donor_digest if self.prompt_side == "donor" else self.edited_digest


@dataclass
class _Snapshot:
    product: str
    layer: int
    positions: tuple[int, ...]
    source_slots: tuple[int, ...]
    rows: torch.Tensor
    physical_page_size: int
    ready: Any | None


@dataclass(frozen=True)
class _CaptureControl:
    mode: str
    prompt_side: str
    prompt_digest: str
    token_count: int
    query_position: int
    evidence_dir: Path


def _load_capture_control(path: Path) -> _CaptureControl:
    _require(path.is_absolute() and path.is_file(), "CAPTURE_CONTROL_PATH_INVALID")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "CAPTURE_CONTROL_SCHEMA_INVALID")
    _require(payload.get("mode") == "CAPTURE", "CAPTURE_CONTROL_MODE_INVALID")
    side = payload.get("prompt_side")
    _require(side in {"donor", "edited"}, "CAPTURE_CONTROL_SIDE_INVALID")
    model = payload.get("model", {})
    _require(model == {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "architecture": MODEL_ARCHITECTURE,
        "vllm_commit": VLLM_COMMIT,
        "attention_backend": ATTENTION_BACKEND,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "index_kpool": INDEX_KPOOL,
    }, "CAPTURE_CONTROL_MODEL_INVALID")
    token_count = payload.get("token_count")
    query = payload.get("query_position")
    digest = payload.get("prompt_token_ids_sha256")
    _require(type(token_count) is int and token_count > 2048, "CAPTURE_CONTROL_TOKEN_COUNT_INVALID")
    _require(query == token_count - 1, "CAPTURE_CONTROL_QUERY_INVALID")
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest), "CAPTURE_CONTROL_DIGEST_INVALID")
    _require(payload.get("layers") == list(INDEXER_LAYERS), "CAPTURE_CONTROL_LAYERS_INVALID")
    _require(payload.get("data_parallel_rank") == 0, "CAPTURE_CONTROL_DP_RANK_INVALID")
    _require(payload.get("raw_score_kind") == "native_pre_topk_pre_normalization", "CAPTURE_CONTROL_SCORE_KIND_INVALID")
    _require(payload.get("outcome_independent") is True, "CAPTURE_CONTROL_OUTCOME_ATTESTATION_MISSING")
    _require(payload.get("benchmark_outcomes_read") is False, "CAPTURE_CONTROL_OUTCOME_LEAKAGE")
    evidence = Path(payload.get("evidence_dir", ""))
    _require(evidence.is_absolute(), "CAPTURE_CONTROL_EVIDENCE_DIR_INVALID")
    evidence.mkdir(parents=True, exist_ok=True)
    return _CaptureControl("CAPTURE", side, digest, token_count, query, evidence)


def _current_capture_control() -> _CaptureControl | None:
    raw = os.getenv(SELECTOR_CAPTURE_ENV)
    if not raw:
        return None
    path = Path(raw)
    if not path.exists():
        return None
    control = _load_capture_control(path)
    return control if _dp_rank() == 0 else None


def _dp_rank() -> int:
    try:
        from vllm.distributed import get_dp_group

        return int(get_dp_group().rank_in_group)
    except Exception:
        raw = os.getenv("VLLM_DP_RANK", os.getenv("RANK", "0"))
        return int(raw) if raw.lstrip("-").isdigit() else 0


def _load_control(path: Path) -> _Control:
    _require(path.is_absolute() and path.is_file(), "CONTROL_PATH_INVALID")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 3, "CONTROL_SCHEMA_INVALID")
    mode = payload.get("mode")
    _require(mode in {"OFF", "SNAPSHOT", "TRANSPLANT"}, "CONTROL_MODE_INVALID")
    _require(payload.get("unsafe_accuracy_ablation_ack") == UNSAFE_ACK, "UNSAFE_ACK_MISSING")
    _require(payload.get("production_default_enabled") is False, "PRODUCTION_DEFAULT_MUST_BE_DISABLED")
    _require(payload.get("compute_semantics") == "full_target_prefill_then_selected_donor_cache_overwrite", "COMPUTE_SEMANTICS_INVALID")
    _require(payload.get("true_partial_prefill") is False, "TRUE_PARTIAL_PREFILL_MUST_BE_FALSE")
    model = payload.get("model", {})
    _require(model == {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "architecture": MODEL_ARCHITECTURE,
        "vllm_commit": VLLM_COMMIT,
        "attention_backend": ATTENTION_BACKEND,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "index_kpool": INDEX_KPOOL,
    }, "CONTROL_MODEL_RUNTIME_MISMATCH")
    runtime = payload.get("runtime", {})
    _require(runtime == {
        "tensor_parallel_size": 4,
        "data_parallel_size": 1,
        "expert_parallel_size": 4,
        "data_parallel_rank_affinity": 0,
        "prefix_caching": False,
        "chunked_prefill": False,
    }, "CONTROL_PARALLEL_OR_CACHE_RUNTIME_MISMATCH")
    _require(payload.get("mla_layers") == list(MLA_LAYERS), "CONTROL_MLA_LAYERS_INVALID")
    _require(payload.get("indexer_layers") == list(INDEXER_LAYERS), "CONTROL_INDEXER_LAYERS_INVALID")
    _require(payload.get("kda_layers") == list(KDA_LAYERS), "CONTROL_KDA_LAYERS_INVALID")
    _require(payload.get("kda_policy") == "target_computed_no_donor_row_transplant", "CONTROL_KDA_POLICY_INVALID")
    _require(payload.get("indexer_tail_policy") == "target_computed_never_transplanted", "CONTROL_INDEXER_TAIL_POLICY_INVALID")

    history = payload.get("history_token_count")
    eligible_start = payload.get("eligible_start")
    eligible_end = payload.get("eligible_end")
    edits = tuple(payload.get("edit_positions", []))
    _require(type(history) is int and history > 0, "CONTROL_HISTORY_COUNT_INVALID")
    _require(type(eligible_start) is int and eligible_start > 0 and eligible_end == history, "CONTROL_ELIGIBLE_RANGE_INVALID")
    _require(len(edits) == 1 and edits[0] + 1 == eligible_start, "CONTROL_EDIT_CLOSURE_INVALID")
    donor = payload.get("donor_history_token_ids_sha256")
    edited = payload.get("edited_history_token_ids_sha256")
    _require(isinstance(donor, str) and re.fullmatch(r"[0-9a-f]{64}", donor), "CONTROL_DONOR_DIGEST_INVALID")
    _require(isinstance(edited, str) and re.fullmatch(r"[0-9a-f]{64}", edited), "CONTROL_EDITED_DIGEST_INVALID")
    ratio = payload.get("requested_ratio_percent")
    _require(type(ratio) is int and ratio in RATIOS, "CONTROL_RATIO_INVALID")
    selected = tuple(payload.get("selected_main_positions", []))
    expected_count = (history - eligible_start) * ratio // 100
    _require(len(selected) == expected_count, "CONTROL_MAIN_SELECTED_COUNT_INVALID")
    _require(token_ids_sha256(selected) == payload.get("selected_main_positions_sha256"), "CONTROL_MAIN_SELECTED_DIGEST_INVALID")
    pool_ends = tuple(payload.get("selected_indexer_pool_end_positions", []))
    _require(pool_ends == _complete_pool_ends(selected, eligible_start, history), "CONTROL_INDEXER_POOL_PROJECTION_INVALID")
    _require(token_ids_sha256(pool_ends) == payload.get("selected_indexer_pool_end_positions_sha256"), "CONTROL_INDEXER_POOL_DIGEST_INVALID")
    covered = tuple(
        position
        for end in pool_ends
        for position in range(end - INDEX_KPOOL + 1, end + 1)
    )
    _require(tuple(payload.get("selected_indexer_covered_positions", [])) == covered, "CONTROL_INDEXER_COVERAGE_INVALID")
    _require(not set(edits).intersection(selected), "EDIT_POSITION_SELECTED_FOR_DONOR_OVERWRITE")

    experiment = payload.get("experiment_id")
    phase = payload.get("phase_id")
    side = payload.get("prompt_side")
    _require(isinstance(experiment, str) and _ID_RE.fullmatch(experiment), "EXPERIMENT_ID_INVALID")
    _require(isinstance(phase, str) and _ID_RE.fullmatch(phase), "PHASE_ID_INVALID")
    _require(side in {"donor", "edited"}, "PROMPT_SIDE_INVALID")
    if mode == "SNAPSHOT":
        _require(side == "donor" and ratio == 100, "SNAPSHOT_CONTROL_INVALID")
    elif mode == "TRANSPLANT":
        _require(side == "edited" and ratio > 0, "TRANSPLANT_CONTROL_INVALID")
    else:
        _require(ratio == 0, "OFF_CONTROL_INVALID")

    selector_path = Path(payload.get("selector_path", ""))
    selector_sha = payload.get("selector_sha256")
    _require(selector_path.is_absolute() and selector_path.is_file(), "SELECTOR_PATH_INVALID")
    _require(_file_sha256(selector_path) == selector_sha, "SELECTOR_DIGEST_MISMATCH")
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    _require(selector.get("schema_version") == 3 and selector.get("status") == "attested_before_benchmark_outcomes", "SELECTOR_SCHEMA_OR_STATUS_INVALID")
    _require(selector.get("source_evidence", {}).get("benchmark_outcomes_read") is False, "SELECTOR_OUTCOME_LEAKAGE")
    _require(selector.get("source_evidence", {}).get("donor_history_token_ids_sha256") == donor, "SELECTOR_DONOR_DIGEST_MISMATCH")
    _require(selector.get("source_evidence", {}).get("edited_history_token_ids_sha256") == edited, "SELECTOR_EDITED_DIGEST_MISMATCH")
    _require(tuple(selector.get("positions_by_ratio", {}).get(str(ratio), [])) == selected, "SELECTOR_MAIN_SELECTION_MISMATCH")
    _require(tuple(selector.get("indexer_pool_end_positions_by_ratio", {}).get(str(ratio), [])) == pool_ends, "SELECTOR_INDEXER_SELECTION_MISMATCH")

    evidence = Path(payload.get("evidence_dir", ""))
    _require(evidence.is_absolute(), "EVIDENCE_DIR_NOT_ABSOLUTE")
    evidence.mkdir(parents=True, exist_ok=True)
    return _Control(
        mode=mode,
        experiment_id=experiment,
        phase_id=phase,
        prompt_side=side,
        donor_digest=donor,
        edited_digest=edited,
        history_token_count=history,
        eligible_start=eligible_start,
        eligible_end=history,
        edit_positions=edits,
        requested_ratio=ratio,
        selected_main_positions=selected,
        selected_indexer_pool_ends=pool_ends,
        selector_sha256=selector_sha,
        evidence_dir=evidence,
        data_parallel_rank=int(runtime["data_parallel_rank_affinity"]),
    )


def _current_control() -> _Control | None:
    raw = os.getenv(CONTROL_ENV)
    if not raw:
        return None
    path = Path(raw)
    # A mounted but not-yet-created control path is the explicit default-OFF
    # state used while the server performs startup/profile work. Any present
    # malformed control still fails closed in _load_control. Ratio finalization
    # requires mutation evidence, so an absent file cannot masquerade as reuse.
    if not path.exists():
        return None
    control = _load_control(path)
    if _dp_rank() != control.data_parallel_rank:
        return None
    return control


def accuracy_ablation_armed() -> bool:
    control = _current_control()
    return control is not None and control.mode in {"SNAPSHOT", "TRANSPLANT"}


def selector_capture_armed() -> bool:
    return _current_capture_control() is not None


def _layer_from_name(name: str) -> int:
    match = _LAYER_RE.search(name)
    _require(match is not None, "LAYER_COORDINATE_UNAVAILABLE")
    return int(match.group(1))


def _ready_event(tensor: torch.Tensor) -> Any | None:
    if not tensor.is_cuda:
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(tensor.device))
    return event


def _wait(snapshot: _Snapshot, destination: torch.Tensor) -> None:
    if snapshot.ready is not None:
        torch.cuda.current_stream(destination.device).wait_event(snapshot.ready)


def _mapped_slots(
    slot_mapping: torch.Tensor,
    positions: torch.Tensor | Sequence[int],
    wanted: Sequence[int],
    physical_page_size: int,
) -> tuple[tuple[int, int, int], ...]:
    _require(slot_mapping.ndim == 1, "SLOT_MAPPING_RANK_INVALID")
    if isinstance(positions, torch.Tensor):
        _require(positions.ndim == 1, "ABSOLUTE_POSITION_RANK_INVALID")
        pos = positions.detach().to(device="cpu", dtype=torch.int64).tolist()
    else:
        pos = [int(value) for value in positions]
    _require(slot_mapping.numel() == len(pos), "SLOT_POSITION_COUNT_MISMATCH")
    slots = slot_mapping.detach().to(device="cpu", dtype=torch.int64).tolist()
    _require(len(pos) == len(set(pos)), "ABSOLUTE_POSITIONS_NOT_UNIQUE")
    mapping = {int(position): int(slot) for position, slot in zip(pos, slots, strict=True)}
    valid_slots = [slot for slot in slots if slot >= 0]
    _require(len(valid_slots) == len(set(valid_slots)), "VALID_SLOT_MAPPING_NOT_UNIQUE")
    result: list[tuple[int, int, int]] = []
    for position in wanted:
        _require(position in mapping and mapping[position] >= 0, "SELECTED_POSITION_HAS_NO_CACHE_SLOT")
        page, offset = divmod(mapping[position], physical_page_size)
        result.append((position, page, offset))
    return tuple(result)


def _snapshot_rows(
    product: str,
    layer: int,
    kv_cache: torch.Tensor,
    positions: torch.Tensor | Sequence[int],
    slot_mapping: torch.Tensor,
    wanted: tuple[int, ...],
) -> _Snapshot:
    expected_elements = MAIN_ROW_BYTES if product == "mla_kv" else INDEXER_ROW_BYTES
    _require(kv_cache.ndim == 3, f"{product.upper()}_CACHE_RANK_INVALID")
    _require(kv_cache.shape[2] == expected_elements, f"{product.upper()}_CACHE_ROW_ELEMENTS_INVALID")
    if product == "mla_kv":
        _require(kv_cache.element_size() == 1, "MLA_KV_CACHE_ELEMENT_SIZE_INVALID")
    else:
        _require(kv_cache.dtype == torch.uint8, "INDEXER_KPOOL_CACHE_DTYPE_INVALID")
    page_size = int(kv_cache.shape[1])
    _require(page_size > 0, f"{product.upper()}_CACHE_PAGE_SIZE_INVALID")
    mapping = _mapped_slots(slot_mapping, positions, wanted, page_size)
    _require(all(page < kv_cache.shape[0] for _, page, _ in mapping), f"{product.upper()}_SOURCE_PAGE_OUT_OF_RANGE")
    rows = torch.stack([kv_cache[page, offset].clone() for _, page, offset in mapping])
    slots = tuple(page * page_size + offset for _, page, offset in mapping)
    return _Snapshot(product, layer, wanted, slots, rows, page_size, _ready_event(kv_cache))


def _copy_rows(
    snapshot: _Snapshot,
    kv_cache: torch.Tensor,
    positions: torch.Tensor | Sequence[int],
    slot_mapping: torch.Tensor,
    wanted: tuple[int, ...],
) -> tuple[int, ...]:
    expected_elements = MAIN_ROW_BYTES if snapshot.product == "mla_kv" else INDEXER_ROW_BYTES
    _require(kv_cache.ndim == 3 and kv_cache.shape[2] == expected_elements, "DESTINATION_CACHE_LAYOUT_INVALID")
    _require(kv_cache.dtype == snapshot.rows.dtype, "SOURCE_DESTINATION_CACHE_DTYPE_MISMATCH")
    if snapshot.product == "mla_kv":
        _require(kv_cache.element_size() == 1, "DESTINATION_MLA_ELEMENT_SIZE_INVALID")
    else:
        _require(kv_cache.dtype == torch.uint8, "DESTINATION_INDEXER_DTYPE_INVALID")
    page_size = int(kv_cache.shape[1])
    _require(page_size == snapshot.physical_page_size, "SOURCE_DESTINATION_PAGE_SIZE_MISMATCH")
    _wait(snapshot, kv_cache)
    source = {position: index for index, position in enumerate(snapshot.positions)}
    mapping = _mapped_slots(slot_mapping, positions, wanted, page_size)
    destination: list[int] = []
    for position, page, offset in mapping:
        _require(position in source and page < kv_cache.shape[0], "DESTINATION_MAPPING_INVALID")
        kv_cache[page, offset].copy_(snapshot.rows[source[position]])
        destination.append(page * page_size + offset)
    return tuple(destination)


class _RuntimeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.snapshots: dict[tuple[str, int], _Snapshot] = {}
        self.snapshot_experiment: str | None = None
        self.snapshot_selector: str | None = None
        self.attested_phase: tuple[str, str, str, str] | None = None
        self.prefill_counts: dict[tuple[str, str, str], int] = {}
        self.current_prefill_ordinal: int | None = None
        self.current_prefill_token_count: int | None = None
        self.current_absolute_positions: tuple[int, ...] | None = None
        self.preflight_key: tuple[str, int] | None = None
        self.capture_attestation: tuple[str, str, int] | None = None
        self.capture_written_layers: set[tuple[str, int]] = set()

    def reset(self) -> None:
        with self.lock:
            self.snapshots.clear()
            self.snapshot_experiment = None
            self.snapshot_selector = None
            self.attested_phase = None
            self.prefill_counts.clear()
            self.current_prefill_ordinal = None
            self.current_prefill_token_count = None
            self.current_absolute_positions = None
            self.preflight_key = None
            self.capture_attestation = None
            self.capture_written_layers.clear()

    def attest_capture(
        self,
        control: _CaptureControl,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> None:
        if positions.ndim == 1 and positions.numel() == 1:
            position = int(positions.detach().to(device="cpu", dtype=torch.int64).item())
            if position >= control.token_count:
                return
        _require(input_ids is not None and input_ids.ndim == 1 and positions.ndim == 1, "CAPTURE_INPUT_IDS_REQUIRED")
        _require(input_ids.numel() == positions.numel() == control.token_count, "CAPTURE_SINGLE_COMPLETE_PREFILL_REQUIRED")
        absolute = positions.detach().to(device="cpu", dtype=torch.int64).tolist()
        _require(absolute == list(range(control.token_count)), "CAPTURE_ABSOLUTE_POSITIONS_INVALID")
        ids = input_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
        _require(token_ids_sha256(ids) == control.prompt_digest, "CAPTURE_PROMPT_DIGEST_MISMATCH")
        self.capture_attestation = (
            control.prompt_side,
            control.prompt_digest,
            control.token_count,
        )

    def capture_indexer_logits(
        self,
        control: _CaptureControl,
        *,
        layer: int,
        logits: torch.Tensor,
        query_positions: torch.Tensor,
        key_starts: torch.Tensor,
        key_ends: torch.Tensor,
    ) -> None:
        expected_attestation = (
            control.prompt_side,
            control.prompt_digest,
            control.token_count,
        )
        _require(self.capture_attestation == expected_attestation, "CAPTURE_PROMPT_NOT_ATTESTED")
        _require(layer in INDEXER_LAYERS, "CAPTURE_LAYER_INVALID")
        _require(logits.ndim == 2 and query_positions.ndim == 1, "CAPTURE_LOGIT_SHAPE_INVALID")
        _require(logits.shape[0] == query_positions.numel(), "CAPTURE_QUERY_ROW_COUNT_INVALID")
        queries = query_positions.detach().to(device="cpu", dtype=torch.int64).tolist()
        matches = [index for index, value in enumerate(queries) if value == control.query_position]
        if not matches:
            return
        _require(len(matches) == 1, "CAPTURE_QUERY_ROW_DUPLICATE")
        row = matches[0]
        starts = key_starts.detach().to(device="cpu", dtype=torch.int64).tolist()
        ends = key_ends.detach().to(device="cpu", dtype=torch.int64).tolist()
        _require(len(starts) == len(ends) == logits.shape[0], "CAPTURE_KEY_BOUNDARY_COUNT_INVALID")
        start = int(starts[row])
        end = int(ends[row])
        expected_pools = control.token_count // INDEX_KPOOL
        _require(start == 0 and end == expected_pools, "CAPTURE_POOL_BOUNDARY_INVALID")
        values = logits[row, start:end].detach().to(device="cpu", dtype=torch.float64).tolist()
        _require(len(values) == expected_pools and all(isinstance(value, float) and value == value and abs(value) != float("inf") for value in values), "CAPTURE_RAW_SCORE_NONFINITE")
        key = (control.prompt_side, layer)
        path = control.evidence_dir / f"base-selector.{control.prompt_side}.layer-{layer:03d}.json"
        with self.lock:
            _require(key not in self.capture_written_layers and not path.exists(), "CAPTURE_LAYER_ALREADY_WRITTEN")
            payload = {
                "schema_version": 1,
                "capture_kind": "glm53_native_raw_pre_topk_kpool",
                "prompt_side": control.prompt_side,
                "model": {
                    "id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "architecture": MODEL_ARCHITECTURE,
                    "vllm_commit": VLLM_COMMIT,
                },
                "layer": layer,
                "query_position": control.query_position,
                "token_count": control.token_count,
                "prompt_token_ids_sha256": control.prompt_digest,
                "index_kpool": INDEX_KPOOL,
                "pool_ids": list(range(expected_pools)),
                "raw_pool_scores": values,
                "raw_score_kind": "native_pre_topk_pre_normalization",
                "native_scale": "fp8_fp4_mqa_logits_output_before_top_k_per_row_prefill",
                "data_parallel_rank": 0,
                "tp_rank": 0,
                "head_aggregation": "native_lightning_indexer_weights_fused_by_vllm",
                "incomplete_tail_token_count": control.token_count % INDEX_KPOOL,
            }
            encoded = (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode()
            payload["record_sha256"] = _sha256(encoded)
            temporary = path.with_name(path.name + f".{os.getpid()}.partial")
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
            self.capture_written_layers.add(key)

    def record(self, control: _Control, payload: Mapping[str, Any]) -> None:
        record = {
            "schema_version": 1,
            "rank": _dp_rank(),
            "experiment_id": control.experiment_id,
            "phase_id": control.phase_id,
            "mode": control.mode,
            **payload,
        }
        path = control.evidence_dir / f"runtime.rank-{_dp_rank()}.jsonl"
        line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, line.encode())
        finally:
            os.close(descriptor)

    def attest_prompt(self, control: _Control, input_ids: torch.Tensor | None, positions: torch.Tensor) -> None:
        if positions.ndim == 1 and positions.numel() == 1:
            position = int(positions.detach().to(device="cpu", dtype=torch.int64).item())
            if position >= control.history_token_count:
                return
        _require(input_ids is not None and input_ids.ndim == 1 and positions.ndim == 1, "FROZEN_HISTORY_INPUT_IDS_REQUIRED")
        _require(input_ids.numel() == positions.numel() and input_ids.numel() >= control.history_token_count, "FROZEN_HISTORY_SINGLE_PREFILL_REQUIRED")
        absolute = positions.detach().to(device="cpu", dtype=torch.int64).tolist()
        _require(absolute == list(range(input_ids.numel())), "ABSOLUTE_POSITION_ALIGNMENT_INVALID")
        ids = input_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
        if control.mode == "SNAPSHOT":
            _require(len(ids) == control.history_token_count, "DONOR_MUST_BE_EXACT_FROZEN_HISTORY")
        else:
            _require(len(ids) > control.history_token_count, "TARGET_MUST_INCLUDE_Q2_CONTINUATION_TOKENS")
        history_ids = ids[: control.history_token_count]
        digest = token_ids_sha256(history_ids)
        _require(digest == control.expected_prompt_digest, "PROMPT_DIGEST_MISMATCH")
        self.attested_phase = (control.experiment_id, control.phase_id, control.prompt_side, digest)
        key = (control.experiment_id, control.phase_id, control.prompt_side)
        ordinal = self.prefill_counts.get(key, 0)
        self.prefill_counts[key] = ordinal + 1
        self.current_prefill_ordinal = ordinal
        self.current_prefill_token_count = len(ids)
        self.current_absolute_positions = tuple(absolute)
        self.record(control, {
            "action": "prompt_attested",
            "prefill_ordinal": ordinal,
            "frozen_history_token_count": control.history_token_count,
            "full_target_prefill_token_count": len(ids),
            "extension_token_count": len(ids) - control.history_token_count,
            "frozen_history_token_ids_sha256": digest,
            "edit_positions": list(control.edit_positions),
            "absolute_positions_unchanged": True,
            "kda_policy": "target_computed_no_donor_row_transplant",
            "indexer_tail_policy": "target_computed_never_transplanted",
            "true_partial_prefill": False,
        })

    def require_attested(self, control: _Control) -> None:
        expected = (control.experiment_id, control.phase_id, control.prompt_side, control.expected_prompt_digest)
        _require(self.attested_phase == expected, "PROMPT_PHASE_NOT_ATTESTED")

    def preflight(self, control: _Control) -> None:
        key = (control.selector_sha256, control.requested_ratio)
        if self.preflight_key == key:
            return
        _require(self.snapshot_experiment == control.experiment_id, "SOURCE_EXPERIMENT_MISMATCH")
        _require(self.snapshot_selector == control.selector_sha256, "SOURCE_SELECTOR_MISMATCH")
        expected = {("mla_kv", layer) for layer in MLA_LAYERS} | {
            ("indexer_kpool", layer) for layer in INDEXER_LAYERS
        }
        _require(set(self.snapshots) == expected, "SOURCE_SNAPSHOT_LAYER_PRODUCT_SET_INCOMPLETE")
        all_main = tuple(range(control.eligible_start, control.eligible_end))
        all_pools = _complete_pool_ends(all_main, control.eligible_start, control.eligible_end)
        for (product, _), snapshot in self.snapshots.items():
            _require(snapshot.positions == (all_main if product == "mla_kv" else all_pools), "SOURCE_SNAPSHOT_POSITION_SET_INCOMPLETE")
        self.preflight_key = key
        self.record(control, {
            "action": "transplant_preflight",
            "status": "passed",
            "source_layer_products": len(expected),
            "selected_main_positions": len(control.selected_main_positions),
            "selected_indexer_pool_rows": len(control.selected_indexer_pool_ends),
            "no_full_prefill_fallback": True,
            "full_target_prefill_required": True,
        })

    def apply(
        self,
        control: _Control,
        product: str,
        layer: int,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        positions: torch.Tensor | None,
    ) -> int:
        self.require_attested(control)
        _require(self.current_prefill_ordinal is not None and self.current_prefill_token_count is not None, "PREFILL_EVIDENCE_UNAVAILABLE")
        active_positions: torch.Tensor | Sequence[int]
        if positions is None:
            _require(self.current_absolute_positions is not None, "ABSOLUTE_POSITIONS_UNAVAILABLE")
            active_positions = self.current_absolute_positions
        else:
            active_positions = positions
        all_positions = tuple(range(control.eligible_start, control.eligible_end))
        snapshot_positions = (
            all_positions
            if product == "mla_kv"
            else _complete_pool_ends(all_positions, control.eligible_start, control.eligible_end)
        )
        key = (product, layer)
        with self.lock:
            if control.mode == "SNAPSHOT":
                if self.snapshot_experiment is None:
                    self.snapshot_experiment = control.experiment_id
                    self.snapshot_selector = control.selector_sha256
                _require(self.snapshot_experiment == control.experiment_id and self.snapshot_selector == control.selector_sha256, "SNAPSHOT_IDENTITY_CHANGED")
                _require(key not in self.snapshots, "DUPLICATE_SOURCE_LAYER_PRODUCT")
                snapshot = _snapshot_rows(product, layer, kv_cache, active_positions, slot_mapping, snapshot_positions)
                self.snapshots[key] = snapshot
                self.record(control, {
                    "action": "source_snapshot",
                    "product": product,
                    "layer": layer,
                    "row_count": len(snapshot_positions),
                    "physical_page_size": snapshot.physical_page_size,
                    "source_slots_sha256": token_ids_sha256(snapshot.source_slots),
                    "private_clone": True,
                })
                return 0
            _require(control.mode == "TRANSPLANT", "APPLY_MODE_INVALID")
            self.preflight(control)
            wanted = control.selected_main_positions if product == "mla_kv" else control.selected_indexer_pool_ends
            destination = _copy_rows(self.snapshots[key], kv_cache, active_positions, slot_mapping, wanted)
            self.record(control, {
                "action": "destination_transplant",
                "prefill_ordinal": self.current_prefill_ordinal,
                "full_target_prefill_token_count": self.current_prefill_token_count,
                "extension_rows_recomputed": self.current_prefill_token_count - control.history_token_count,
                "product": product,
                "layer": layer,
                "requested_ratio_percent": control.requested_ratio,
                "actual_row_count": len(wanted),
                "positions_sha256": token_ids_sha256(wanted),
                "destination_slots_sha256": token_ids_sha256(destination),
                "physical_page_size": int(kv_cache.shape[1]),
                "edited_token_excluded": not set(control.edit_positions).intersection(wanted),
                "destination_allocator_and_block_table_preserved": True,
                "native_cache_write_completed_before_overwrite": True,
                "full_target_prefill": True,
                "true_partial_prefill": False,
                "latency_or_skipped_flop_claim": False,
            })
            return len(wanted)


_STATE = _RuntimeState()


def maybe_attest_prompt(*, input_ids: torch.Tensor | None, positions: torch.Tensor) -> None:
    _require(
        not (os.getenv(CONTROL_ENV) and os.getenv(SELECTOR_CAPTURE_ENV)),
        "STATEFUL_AND_SELECTOR_CONTROLS_MUTUALLY_EXCLUSIVE",
    )
    capture = _current_capture_control()
    if capture is not None:
        _STATE.attest_capture(capture, input_ids, positions)
        return
    control = _current_control()
    if control is None or control.mode == "OFF":
        return
    _STATE.attest_prompt(control, input_ids, positions)


def maybe_capture_base_selector_scores(
    *,
    layer_name: str,
    logits: torch.Tensor,
    query_positions: torch.Tensor,
    key_starts: torch.Tensor,
    key_ends: torch.Tensor,
    num_prefills: int,
    num_decodes: int,
) -> None:
    control = _current_capture_control()
    if control is None:
        return
    _require(num_prefills == 1 and num_decodes == 0, "CAPTURE_REQUIRES_ONE_PREFILL_NO_DECODE")
    layer = _layer_from_name(layer_name)
    _STATE.capture_indexer_logits(
        control,
        layer=layer,
        logits=logits,
        query_positions=query_positions,
        key_starts=key_starts,
        key_ends=key_ends,
    )


def _prefill_or_decode_noop(
    control: _Control,
    *,
    num_prefills: int,
    num_decodes: int,
    num_prefill_tokens: int,
) -> bool:
    if num_prefills == 0 and num_decodes > 0:
        return False
    _require(num_prefills == 1 and num_decodes == 0, "FROZEN_HISTORY_SINGLE_PREFILL_METADATA_REQUIRED")
    _require(num_prefill_tokens >= control.history_token_count, "FROZEN_HISTORY_PREFILL_TOKEN_COUNT_INVALID")
    return True


def maybe_apply_main_cache(
    *,
    layer_name: str,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    num_prefills: int,
    num_decodes: int,
    num_prefill_tokens: int,
) -> int:
    control = _current_control()
    if control is None or control.mode == "OFF":
        return 0
    if not _prefill_or_decode_noop(
        control,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_prefill_tokens=num_prefill_tokens,
    ):
        return 0
    layer = _layer_from_name(layer_name)
    _require(layer in MLA_LAYERS, "MAIN_LAYER_OUT_OF_RANGE")
    return _STATE.apply(control, "mla_kv", layer, kv_cache, slot_mapping, positions)


def maybe_apply_indexer_cache(
    *,
    layer_name: str,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    num_prefills: int,
    num_decodes: int,
    num_prefill_tokens: int,
) -> int:
    control = _current_control()
    if control is None or control.mode == "OFF":
        return 0
    if not _prefill_or_decode_noop(
        control,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        num_prefill_tokens=num_prefill_tokens,
    ):
        return 0
    layer = _layer_from_name(layer_name)
    _require(layer in INDEXER_LAYERS, "INDEXER_LAYER_OUT_OF_RANGE")
    return _STATE.apply(control, "indexer_kpool", layer, kv_cache, slot_mapping, positions)
