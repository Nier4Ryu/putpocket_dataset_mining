# SPDX-License-Identifier: Apache-2.0
"""Default-OFF GLM-5.2 forced edit-reuse transplant experiment.

This module is deliberately unsafe for quality and deliberately narrow for
runtime safety.  It supports only the pinned 2,071-token, same-length SR-CC-1
system edit and only a single full-prefill request.  Donor cache rows are
cloned into private storage; destination requests retain their own allocator
slots, block tables, and sequence metadata.

The hook is inert unless ``PUTPOCKET_GLM52_FORCED_REUSE_CONTROL`` points to an
explicit control file.  A transplant control additionally requires the exact
unsafe acknowledgement string and a selector whose digest was frozen before
benchmark outcomes were read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


CONTROL_ENV = "PUTPOCKET_GLM52_FORCED_REUSE_CONTROL"
PROMPT_TOKENS = 2071
EDIT_POSITION = 114
OLD_TOKEN_ID = 17526
NEW_TOKEN_ID = 11660
DOWNSTREAM_START = 115
DOWNSTREAM_END = 2071
BLOCK_SIZE = 64
MAIN_LAYERS = tuple(range(78))
INDEXER_LAYERS = (0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74)
RATIOS = tuple(range(0, 101, 10))
UNSAFE_ACK = "I_UNDERSTAND_ZERO_SAFE_PAGES"
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_VALID_MODES = {"OFF", "SNAPSHOT", "TRANSPLANT"}


class ForcedReuseInvariantError(RuntimeError):
    """The experiment failed closed before producing valid reuse evidence."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ForcedReuseInvariantError(reason)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(token_ids: list[int]) -> str:
    encoded = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return _sha256(encoded)


def ratio_count(ratio: int) -> int:
    _require(ratio in RATIOS, "REQUESTED_RATIO_UNSUPPORTED")
    return (DOWNSTREAM_END - DOWNSTREAM_START) * ratio // 100


@dataclass(frozen=True)
class _Selector:
    sha256: str
    ranking: tuple[int, ...]
    positions_by_ratio: dict[int, tuple[int, ...]]
    source_baseline_digest: str
    source_edited_digest: str

    def selected(self, ratio: int) -> tuple[int, ...]:
        _require(ratio in self.positions_by_ratio, "SELECTOR_RATIO_MISSING")
        return self.positions_by_ratio[ratio]


@dataclass(frozen=True)
class _Control:
    mode: str
    experiment_id: str
    phase_id: str
    prompt_side: str
    donor_prompt_digest: str
    edited_prompt_digest: str
    selector: _Selector
    requested_ratio: int
    evidence_dir: Path

    @property
    def expected_prompt_digest(self) -> str:
        return self.donor_prompt_digest if self.prompt_side == "donor" else self.edited_prompt_digest


@dataclass
class _LayerSnapshot:
    product: str
    layer: int
    positions: tuple[int, ...]
    source_slots: tuple[int, ...]
    parts: tuple[torch.Tensor, ...]
    ready: torch.cuda.Event | None


def _load_selector(path: Path, expected_sha256: str) -> _Selector:
    _require(path.is_absolute(), "SELECTOR_PATH_NOT_ABSOLUTE")
    actual = _file_sha256(path)
    _require(actual == expected_sha256, "SELECTOR_DIGEST_MISMATCH")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "SELECTOR_SCHEMA_INVALID")
    _require(payload.get("status") == "attested_before_benchmark_outcomes", "SELECTOR_NOT_PREATTESTED")
    scenario = payload.get("scenario", {})
    _require(scenario.get("prompt_token_count") == PROMPT_TOKENS, "SELECTOR_PROMPT_LENGTH_INVALID")
    _require(scenario.get("edit_position") == EDIT_POSITION, "SELECTOR_EDIT_POSITION_INVALID")
    _require(scenario.get("old_token_id") == OLD_TOKEN_ID, "SELECTOR_OLD_TOKEN_INVALID")
    _require(scenario.get("new_token_id") == NEW_TOKEN_ID, "SELECTOR_NEW_TOKEN_INVALID")
    _require(scenario.get("downstream_range") == [DOWNSTREAM_START, DOWNSTREAM_END], "SELECTOR_RANGE_INVALID")
    _require(scenario.get("same_length") is True and scenario.get("rope_positions_unchanged") is True, "SELECTOR_POSITION_SEMANTICS_INVALID")
    ranking = tuple(payload.get("ranking", []))
    expected_positions = set(range(DOWNSTREAM_START, DOWNSTREAM_END))
    _require(len(ranking) == len(expected_positions) and set(ranking) == expected_positions, "SELECTOR_RANKING_NOT_PERMUTATION")
    raw_by_ratio = payload.get("positions_by_ratio", {})
    positions_by_ratio: dict[int, tuple[int, ...]] = {}
    for ratio in RATIOS:
        values = tuple(raw_by_ratio.get(str(ratio), []))
        _require(len(values) == ratio_count(ratio), f"SELECTOR_RATIO_{ratio}_COUNT_INVALID")
        _require(values == tuple(sorted(ranking[: ratio_count(ratio)])), f"SELECTOR_RATIO_{ratio}_PREFIX_INVALID")
        positions_by_ratio[ratio] = values
    _require(not positions_by_ratio[0], "SELECTOR_ZERO_ENDPOINT_INVALID")
    _require(set(positions_by_ratio[100]) == expected_positions, "SELECTOR_HUNDRED_ENDPOINT_INVALID")
    sources = payload.get("source_evidence", {})
    baseline = sources.get("baseline_prompt_token_ids_sha256")
    edited = sources.get("edited_prompt_token_ids_sha256")
    _require(isinstance(baseline, str) and re.fullmatch(r"[0-9a-f]{64}", baseline) is not None, "SELECTOR_BASELINE_DIGEST_INVALID")
    _require(isinstance(edited, str) and re.fullmatch(r"[0-9a-f]{64}", edited) is not None, "SELECTOR_EDITED_DIGEST_INVALID")
    return _Selector(actual, ranking, positions_by_ratio, baseline, edited)


def _load_control(path: Path) -> _Control:
    _require(path.is_absolute(), "CONTROL_PATH_NOT_ABSOLUTE")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "CONTROL_SCHEMA_INVALID")
    mode = payload.get("mode")
    _require(mode in _VALID_MODES, "CONTROL_MODE_INVALID")
    _require(payload.get("unsafe_forced_reuse_ack") == UNSAFE_ACK, "UNSAFE_ACK_MISSING")
    _require(payload.get("production_default_enabled") is False, "PRODUCTION_DEFAULT_MUST_BE_DISABLED")
    _require(payload.get("prompt_token_count") == PROMPT_TOKENS, "CONTROL_PROMPT_LENGTH_INVALID")
    _require(payload.get("edit_position") == EDIT_POSITION, "CONTROL_EDIT_POSITION_INVALID")
    _require(payload.get("real_block_size") == BLOCK_SIZE, "CONTROL_BLOCK_SIZE_INVALID")
    _require(payload.get("main_layers") == list(MAIN_LAYERS), "CONTROL_MAIN_LAYER_SET_INVALID")
    _require(payload.get("indexer_layers") == list(INDEXER_LAYERS), "CONTROL_INDEXER_LAYER_SET_INVALID")
    experiment = payload.get("experiment_id")
    phase = payload.get("phase_id")
    side = payload.get("prompt_side")
    _require(isinstance(experiment, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", experiment), "EXPERIMENT_ID_INVALID")
    _require(isinstance(phase, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", phase), "PHASE_ID_INVALID")
    _require(side in {"donor", "edited"}, "PROMPT_SIDE_INVALID")
    donor_digest = payload.get("donor_prompt_token_ids_sha256")
    edited_digest = payload.get("edited_prompt_token_ids_sha256")
    _require(donor_digest == payload.get("selector_source_baseline_digest"), "CONTROL_DONOR_SELECTOR_DIGEST_MISMATCH")
    _require(edited_digest == payload.get("selector_source_edited_digest"), "CONTROL_EDIT_SELECTOR_DIGEST_MISMATCH")
    selector_path = Path(payload.get("selector_path", ""))
    selector = _load_selector(selector_path, payload.get("selector_sha256", ""))
    _require(selector.source_baseline_digest == donor_digest, "SELECTOR_DONOR_DIGEST_MISMATCH")
    _require(selector.source_edited_digest == edited_digest, "SELECTOR_EDIT_DIGEST_MISMATCH")
    ratio = payload.get("requested_ratio_percent")
    _require(isinstance(ratio, int) and ratio in RATIOS, "CONTROL_RATIO_INVALID")
    if mode == "SNAPSHOT":
        _require(side == "donor" and ratio == 100, "SNAPSHOT_CONTROL_INVALID")
    if mode == "TRANSPLANT":
        _require(side == "edited" and ratio > 0, "TRANSPLANT_CONTROL_INVALID")
    evidence = Path(payload.get("evidence_dir", ""))
    _require(evidence.is_absolute(), "EVIDENCE_DIR_NOT_ABSOLUTE")
    evidence.mkdir(parents=True, exist_ok=True)
    return _Control(mode, experiment, phase, side, donor_digest, edited_digest, selector, ratio, evidence)


def _current_control() -> _Control | None:
    raw = os.getenv(CONTROL_ENV)
    if not raw:
        return None
    return _load_control(Path(raw))


def _layer_from_name(name: str) -> int:
    match = _LAYER_RE.search(name)
    _require(match is not None, "LAYER_COORDINATE_UNAVAILABLE")
    return int(match.group(1))


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    value = os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))
    return int(value) if value.lstrip("-").isdigit() else 0


def _ready_event(tensor: torch.Tensor) -> torch.cuda.Event | None:
    if not tensor.is_cuda:
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(tensor.device))
    return event


def _wait(snapshot: _LayerSnapshot, destination: torch.Tensor) -> None:
    if snapshot.ready is not None:
        torch.cuda.current_stream(destination.device).wait_event(snapshot.ready)


def _position_slots(slot_mapping: torch.Tensor, positions: tuple[int, ...], block_size: int) -> tuple[tuple[int, int, int], ...]:
    _require(slot_mapping.ndim == 1 and slot_mapping.numel() == PROMPT_TOKENS, "FULL_PROMPT_SLOT_MAPPING_REQUIRED")
    slots = slot_mapping.detach().to(device="cpu", dtype=torch.int64).tolist()
    _require(all(slot >= 0 for slot in slots), "PROMPT_SLOT_MAPPING_HAS_INVALID_SLOT")
    _require(len(set(slots)) == len(slots), "PROMPT_SLOT_MAPPING_NOT_UNIQUE")
    result = []
    for position in positions:
        slot = int(slots[position])
        page, offset = divmod(slot, block_size)
        result.append((position, page, offset))
    return tuple(result)


def _snapshot_main(kv_cache: torch.Tensor, layer: int, positions: tuple[int, ...], slot_mapping: torch.Tensor) -> _LayerSnapshot:
    _require(kv_cache.ndim == 3 and kv_cache.dtype == torch.bfloat16, "MAIN_CACHE_DTYPE_OR_RANK_INVALID")
    _require(kv_cache.shape[1] == BLOCK_SIZE and kv_cache.shape[2] == 576, "MAIN_CACHE_LAYOUT_INVALID")
    mapping = _position_slots(slot_mapping, positions, BLOCK_SIZE)
    _require(all(page < kv_cache.shape[0] for _, page, _ in mapping), "MAIN_SOURCE_PAGE_OUT_OF_RANGE")
    values = torch.stack([kv_cache[page, offset].clone() for _, page, offset in mapping])
    slots = tuple(page * BLOCK_SIZE + offset for _, page, offset in mapping)
    return _LayerSnapshot("mla_kv", layer, positions, slots, (values,), _ready_event(kv_cache))


def _snapshot_indexer(kv_cache: torch.Tensor, layer: int, positions: tuple[int, ...], slot_mapping: torch.Tensor) -> _LayerSnapshot:
    _require(kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8, "INDEXER_CACHE_DTYPE_OR_RANK_INVALID")
    _require(kv_cache.shape[1] == BLOCK_SIZE and kv_cache.shape[2] == 132, "INDEXER_CACHE_LAYOUT_INVALID")
    mapping = _position_slots(slot_mapping, positions, BLOCK_SIZE)
    _require(all(page < kv_cache.shape[0] for _, page, _ in mapping), "INDEXER_SOURCE_PAGE_OUT_OF_RANGE")
    values = []
    scales = []
    for _, page, offset in mapping:
        block = kv_cache[page]
        _require(block.is_contiguous(), "INDEXER_SOURCE_PAGE_NOT_CONTIGUOUS")
        flat = block.view(-1)
        values.append(flat[offset * 128 : (offset + 1) * 128].clone())
        scale_start = BLOCK_SIZE * 128 + offset * 4
        scales.append(flat[scale_start : scale_start + 4].clone())
    slots = tuple(page * BLOCK_SIZE + offset for _, page, offset in mapping)
    return _LayerSnapshot("indexer_k", layer, positions, slots, (torch.stack(values), torch.stack(scales)), _ready_event(kv_cache))


def _copy_main(snapshot: _LayerSnapshot, kv_cache: torch.Tensor, positions: tuple[int, ...], slot_mapping: torch.Tensor) -> tuple[int, ...]:
    _require(kv_cache.ndim == 3 and kv_cache.dtype == torch.bfloat16 and kv_cache.shape[1:] == (BLOCK_SIZE, 576), "MAIN_DESTINATION_LAYOUT_INVALID")
    _wait(snapshot, kv_cache)
    source_index = {position: index for index, position in enumerate(snapshot.positions)}
    mapping = _position_slots(slot_mapping, positions, BLOCK_SIZE)
    destination_slots = []
    for position, page, offset in mapping:
        _require(position in source_index and page < kv_cache.shape[0], "MAIN_DESTINATION_MAPPING_INVALID")
        kv_cache[page, offset].copy_(snapshot.parts[0][source_index[position]])
        destination_slots.append(page * BLOCK_SIZE + offset)
    return tuple(destination_slots)


def _copy_indexer(snapshot: _LayerSnapshot, kv_cache: torch.Tensor, positions: tuple[int, ...], slot_mapping: torch.Tensor) -> tuple[int, ...]:
    _require(kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8 and kv_cache.shape[1:] == (BLOCK_SIZE, 132), "INDEXER_DESTINATION_LAYOUT_INVALID")
    _wait(snapshot, kv_cache)
    source_index = {position: index for index, position in enumerate(snapshot.positions)}
    mapping = _position_slots(slot_mapping, positions, BLOCK_SIZE)
    destination_slots = []
    for position, page, offset in mapping:
        _require(position in source_index and page < kv_cache.shape[0], "INDEXER_DESTINATION_MAPPING_INVALID")
        block = kv_cache[page]
        _require(block.is_contiguous(), "INDEXER_DESTINATION_PAGE_NOT_CONTIGUOUS")
        flat = block.view(-1)
        index = source_index[position]
        flat[offset * 128 : (offset + 1) * 128].copy_(snapshot.parts[0][index])
        scale_start = BLOCK_SIZE * 128 + offset * 4
        flat[scale_start : scale_start + 4].copy_(snapshot.parts[1][index])
        destination_slots.append(page * BLOCK_SIZE + offset)
    return tuple(destination_slots)


def _page_accounting(
    selected_positions: tuple[int, ...], slot_mapping: torch.Tensor
) -> dict[str, int | float]:
    all_mapping = _position_slots(
        slot_mapping, tuple(range(DOWNSTREAM_START, DOWNSTREAM_END)), BLOCK_SIZE
    )
    selected_set = set(selected_positions)
    total_by_page: dict[int, int] = {}
    selected_by_page: dict[int, int] = {}
    for position, page, _ in all_mapping:
        total_by_page[page] = total_by_page.get(page, 0) + 1
        if position in selected_set:
            selected_by_page[page] = selected_by_page.get(page, 0) + 1
    touched = len(selected_by_page)
    complete = sum(
        selected_by_page.get(page, 0) == count
        for page, count in total_by_page.items()
    )
    mixed = sum(
        0 < selected_by_page.get(page, 0) < count
        for page, count in total_by_page.items()
    )
    return {
        "downstream_physical_page_count": len(total_by_page),
        "touched_physical_page_count": touched,
        "complete_reused_physical_page_count": complete,
        "mixed_physical_page_count": mixed,
        "untouched_physical_page_count": len(total_by_page) - touched,
        "effective_complete_page_ratio": complete / len(total_by_page),
    }


class _RuntimeState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.snapshots: dict[tuple[str, int], _LayerSnapshot] = {}
        self.snapshot_experiment: str | None = None
        self.snapshot_selector: str | None = None
        self.attested_phase: tuple[str, str, str, str] | None = None
        self.preflight_key: tuple[str, int] | None = None

    def reset(self) -> None:
        with self.lock:
            self.snapshots.clear()
            self.snapshot_experiment = None
            self.snapshot_selector = None
            self.attested_phase = None
            self.preflight_key = None

    def record(self, control: _Control, payload: dict[str, Any]) -> None:
        record = {"schema_version": 1, "rank": _rank(), "experiment_id": control.experiment_id, "phase_id": control.phase_id, "mode": control.mode, **payload}
        line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
        path = control.evidence_dir / f"runtime.rank-{_rank()}.jsonl"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(descriptor, line.encode("utf-8"))
        finally:
            os.close(descriptor)

    def attest_prompt(self, control: _Control, input_ids: torch.Tensor | None, positions: torch.Tensor) -> None:
        # The model hook is also reached for every decode token.  Decode must
        # leave the already-attested phase untouched; every other non-exact
        # shape is rejected so chunked/mixed prefills cannot be mistaken for
        # the pinned request.
        if positions.ndim == 1 and positions.numel() == 1:
            decode_position = int(positions.detach().to(device="cpu", dtype=torch.int64).item())
            if decode_position >= PROMPT_TOKENS:
                return
        _require(input_ids is not None and input_ids.ndim == 1, "FULL_PROMPT_INPUT_IDS_REQUIRED")
        _require(input_ids.numel() == PROMPT_TOKENS and positions.ndim == 1 and positions.numel() == PROMPT_TOKENS, "FULL_PROMPT_SINGLE_PREFILL_REQUIRED")
        positions_cpu = positions.detach().to(device="cpu", dtype=torch.int64).tolist()
        _require(positions_cpu == list(range(PROMPT_TOKENS)), "ABSOLUTE_POSITION_ALIGNMENT_INVALID")
        ids = input_ids.detach().to(device="cpu", dtype=torch.int64).tolist()
        digest = token_ids_sha256(ids)
        _require(digest == control.expected_prompt_digest, "PROMPT_DIGEST_MISMATCH")
        expected_edit = OLD_TOKEN_ID if control.prompt_side == "donor" else NEW_TOKEN_ID
        _require(ids[EDIT_POSITION] == expected_edit, "EDIT_TOKEN_ID_MISMATCH")
        self.attested_phase = (control.experiment_id, control.phase_id, control.prompt_side, digest)
        self.record(control, {"action": "prompt_attested", "prompt_token_count": len(ids), "prompt_token_ids_sha256": digest, "edit_position": EDIT_POSITION, "edit_token_id": expected_edit, "rope_positions_unchanged": True})

    def require_attested(self, control: _Control) -> None:
        expected = (control.experiment_id, control.phase_id, control.prompt_side, control.expected_prompt_digest)
        _require(self.attested_phase == expected, "PROMPT_PHASE_NOT_ATTESTED")

    def preflight(self, control: _Control) -> None:
        key = (control.selector.sha256, control.requested_ratio)
        if self.preflight_key == key:
            return
        _require(self.snapshot_experiment == control.experiment_id, "SOURCE_EXPERIMENT_MISMATCH")
        _require(self.snapshot_selector == control.selector.sha256, "SOURCE_SELECTOR_MISMATCH")
        expected = {("mla_kv", layer) for layer in MAIN_LAYERS} | {("indexer_k", layer) for layer in INDEXER_LAYERS}
        _require(set(self.snapshots) == expected, "SOURCE_SNAPSHOT_LAYER_PRODUCT_SET_INCOMPLETE")
        full_positions = tuple(sorted(control.selector.ranking))
        _require(all(snapshot.positions == full_positions for snapshot in self.snapshots.values()), "SOURCE_SNAPSHOT_POSITION_SET_INCOMPLETE")
        self.preflight_key = key
        self.record(control, {"action": "transplant_preflight", "status": "passed", "source_layer_products": len(expected), "selected_positions": len(control.selector.selected(control.requested_ratio))})

    def apply(self, control: _Control, product: str, layer: int, kv_cache: torch.Tensor, slot_mapping: torch.Tensor) -> int:
        self.require_attested(control)
        full_positions = tuple(sorted(control.selector.ranking))
        key = (product, layer)
        with self.lock:
            if control.mode == "SNAPSHOT":
                if self.snapshot_experiment is None:
                    self.snapshot_experiment = control.experiment_id
                    self.snapshot_selector = control.selector.sha256
                    self.snapshots.clear()
                _require(self.snapshot_experiment == control.experiment_id and self.snapshot_selector == control.selector.sha256, "SNAPSHOT_IDENTITY_CHANGED")
                _require(key not in self.snapshots, "DUPLICATE_SOURCE_LAYER_PRODUCT")
                snapshot = _snapshot_main(kv_cache, layer, full_positions, slot_mapping) if product == "mla_kv" else _snapshot_indexer(kv_cache, layer, full_positions, slot_mapping)
                self.snapshots[key] = snapshot
                self.record(control, {"action": "source_snapshot", "product": product, "layer": layer, "row_count": len(full_positions), "source_slots_sha256": _sha256(json.dumps(snapshot.source_slots, separators=(",", ":")).encode())})
                return 0
            _require(control.mode == "TRANSPLANT", "APPLY_MODE_INVALID")
            self.preflight(control)
            selected = control.selector.selected(control.requested_ratio)
            snapshot = self.snapshots[key]
            destination_slots = _copy_main(snapshot, kv_cache, selected, slot_mapping) if product == "mla_kv" else _copy_indexer(snapshot, kv_cache, selected, slot_mapping)
            page_accounting = _page_accounting(selected, slot_mapping)
            self.record(control, {"action": "destination_transplant", "product": product, "layer": layer, "requested_ratio_percent": control.requested_ratio, "requested_ratio_denominator": DOWNSTREAM_END - DOWNSTREAM_START, "actual_row_count": len(selected), "recomputed_downstream_row_count": (DOWNSTREAM_END - DOWNSTREAM_START) - len(selected), "effective_downstream_ratio": len(selected) / (DOWNSTREAM_END - DOWNSTREAM_START), "positions_sha256": _sha256(json.dumps(selected, separators=(",", ":")).encode()), "destination_slots_sha256": _sha256(json.dumps(destination_slots, separators=(",", ":")).encode()), "edited_token_excluded": EDIT_POSITION not in selected, "same_length_rope_positions_unchanged": True, "destination_allocator_and_block_table_preserved": True, "consumed_by_native_attention": True, **page_accounting})
            return len(selected)


_STATE = _RuntimeState()


def maybe_attest_prompt(*, input_ids: torch.Tensor | None, positions: torch.Tensor) -> None:
    control = _current_control()
    if control is None or control.mode == "OFF":
        return
    _STATE.attest_prompt(control, input_ids, positions)


def _full_prefill_or_decode_noop(control: _Control, slot_mapping: torch.Tensor, num_prefills: int, num_decodes: int, num_prefill_tokens: int) -> bool:
    if num_prefills == 0 and num_decodes > 0:
        return False
    _require(num_prefills == 1 and num_decodes == 0 and num_prefill_tokens == PROMPT_TOKENS, "FULL_PROMPT_PREFILL_METADATA_REQUIRED")
    _require(slot_mapping.numel() == PROMPT_TOKENS, "FULL_PROMPT_SLOT_COUNT_REQUIRED")
    return True


def maybe_apply_main_cache(*, layer_name: str, kv_cache: torch.Tensor, slot_mapping: torch.Tensor, num_prefills: int, num_decodes: int, num_prefill_tokens: int) -> int:
    control = _current_control()
    if control is None or control.mode == "OFF":
        return 0
    if not _full_prefill_or_decode_noop(control, slot_mapping, num_prefills, num_decodes, num_prefill_tokens):
        return 0
    layer = _layer_from_name(layer_name)
    _require(layer in MAIN_LAYERS, "MAIN_LAYER_OUT_OF_RANGE")
    return _STATE.apply(control, "mla_kv", layer, kv_cache, slot_mapping)


def maybe_apply_indexer_cache(*, layer_name: str, kv_cache: torch.Tensor, slot_mapping: torch.Tensor, num_prefills: int, num_decodes: int, num_prefill_tokens: int) -> int:
    control = _current_control()
    if control is None or control.mode == "OFF":
        return 0
    if not _full_prefill_or_decode_noop(control, slot_mapping, num_prefills, num_decodes, num_prefill_tokens):
        return 0
    layer = _layer_from_name(layer_name)
    _require(layer in INDEXER_LAYERS, "INDEXER_LAYER_SET_INVALID")
    return _STATE.apply(control, "indexer_k", layer, kv_cache, slot_mapping)
