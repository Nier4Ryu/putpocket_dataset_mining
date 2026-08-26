from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from jsonschema import Draft202012Validator

from instrumentation.vllm.putpocket_true_partial_prefill import (
    DonorRegistry,
    TruePartialPrefillError,
    build_target_plan,
    build_continuation_payload,
    canonical_json_sha256,
    continuation_execution_evidence,
    copy_reuse_rows_inplace,
    execution_evidence,
    request_instruction_from_xargs,
    token_ids_sha256,
    validate_continuation_payload,
    verify_exact_slot_mappings,
)


OWNER_TOKEN = "owner-token-with-at-least-thirty-two-bytes"
OWNER_SHA = hashlib.sha256(OWNER_TOKEN.encode()).hexdigest()
ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = ROOT / "configs/cluster/glm52_true_partial_prefill.lock.json"
SCHEMA_PATH = (
    ROOT
    / "configs/cluster/schemas/vllm_true_partial_prefill_server_manifest.schema.json"
)
PATCH_PATH = (
    ROOT
    / "patches/vllm/4a3447d200e5aa428d68d1a00aa00f1a19a1a729/putpocket_true_partial_prefill.patch"
)
INSTRUMENTATION_PATH = (
    ROOT / "instrumentation/vllm/putpocket_true_partial_prefill.py"
)


def _runtime() -> dict[str, object]:
    return {
        "vllm_commit": "4a3447d200e5aa428d68d1a00aa00f1a19a1a729",
        "model": "test/model",
        "model_revision": "model-revision",
        "architecture": "GlmMoeDsaForCausalLM",
        "tokenizer": "test/tokenizer",
        "tokenizer_revision": "tokenizer-revision",
        "tokenizer_identity_sha256": "1" * 64,
        "serializer_identity_sha256": "2" * 64,
        "kv_cache_dtype": "bfloat16",
        "attention_backend": "FLASHMLA_SPARSE",
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "prefill_context_parallel_size": 1,
    }


def _cache_contract() -> dict[str, object]:
    return {
        "block_size": 4,
        "main_dtype": "bfloat16",
        "main_row_width": 3,
        "main_layer_names": ["main.0"],
        "indexer_dtype": "uint8",
        "indexer_value_bytes": 2,
        "indexer_scale_bytes": 1,
        "indexer_layer_names": ["indexer.0"],
        "kv_lora_rank": 2,
        "qk_rope_head_dim": 1,
        "num_attention_heads": 8,
    }


def _base_manifest(
    tmp_path: Path, operation: str, source: list[int]
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "backend_kind": "true_partial_prefill",
        "operation": operation,
        "handle": "donor-handle-00000001",
        "owner_token_sha256": OWNER_SHA,
        "runtime": _runtime(),
        "cache_contract": _cache_contract(),
        "source": {
            "token_count": len(source),
            "token_ids_sha256": token_ids_sha256(source),
        },
        "evidence_path": str(tmp_path / "evidence.jsonl"),
        "production_default_enabled": False,
        "lifetime": "single_use_after_donor_finish",
    }


def _target_manifest(
    tmp_path: Path,
    source: list[int],
    frozen: list[int],
    request: list[int],
    alignment: list[tuple[int, int]],
    edits: list[dict[str, object]],
    mandatory: list[int],
    eligible: list[int],
    ranking: list[int],
    selected: list[int],
) -> dict[str, object]:
    payload = _base_manifest(tmp_path, "target_sparse", source)
    recompute = sorted(mandatory + selected)
    payload.update(
        {
            "target": {
                "request_token_count": len(request),
                "request_token_ids_sha256": token_ids_sha256(request),
                "frozen_history_token_count": len(frozen),
                "frozen_history_token_ids_sha256": token_ids_sha256(frozen),
                "continuation_start": len(frozen),
            },
            "alignment": [
                {"target_position": target, "donor_position": donor}
                for target, donor in alignment
            ],
            "edit_operations": edits,
            "selection": {
                "semantic_quantity": "importance_for_recompute",
                "score_direction": "higher_score_first",
                "eligible_recompute_positions": eligible,
                "selection_order": ranking,
                "mandatory_recompute_positions": mandatory,
                "selected_recompute_positions": selected,
                "recompute_positions": recompute,
            },
        }
    )
    return payload


def _write_instruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
    prompt: list[int],
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / f"{payload['operation']}.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv("PUTPOCKET_VLLM_TRUE_PARTIAL_PREFILL_ENABLE", "1")
    monkeypatch.setenv("PUTPOCKET_VLLM_TRUE_PARTIAL_MANIFEST_ROOT", str(tmp_path))
    monkeypatch.setenv("PUTPOCKET_VLLM_TRUE_PARTIAL_EVIDENCE_ROOT", str(tmp_path))
    monkeypatch.setenv(
        "PUTPOCKET_VLLM_TRUE_PARTIAL_TOKENIZER_IDENTITY_SHA256", "1" * 64
    )
    monkeypatch.setenv(
        "PUTPOCKET_VLLM_TRUE_PARTIAL_SERIALIZER_IDENTITY_SHA256", "2" * 64
    )
    return request_instruction_from_xargs(
        {
            "putpocket_true_partial_operation": payload["operation"],
            "putpocket_true_partial_manifest_path": str(path),
            "putpocket_true_partial_manifest_sha256": digest,
            "putpocket_true_partial_owner_token": OWNER_TOKEN,
        },
        prompt,
    )


def test_hook_is_inert_without_namespaced_xargs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUTPOCKET_VLLM_TRUE_PARTIAL_PREFILL_ENABLE", raising=False)
    assert request_instruction_from_xargs(None, [1, 2]) is None
    assert request_instruction_from_xargs({"unrelated": "value"}, [1, 2]) is None


def test_partial_hook_fails_closed_when_disabled(tmp_path: Path) -> None:
    with pytest.raises(TruePartialPrefillError, match="SERVER_DISABLED"):
        request_instruction_from_xargs(
            {
                "putpocket_true_partial_operation": "target_sparse",
                "putpocket_true_partial_manifest_path": str(tmp_path / "missing.json"),
                "putpocket_true_partial_manifest_sha256": "0" * 64,
                "putpocket_true_partial_owner_token": OWNER_TOKEN,
            },
            [1],
        )


def test_target_requires_a_normal_post_patch_continuation_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [1, 2, 3]
    frozen = [1, 9, 3]
    payload = _target_manifest(
        tmp_path,
        source,
        frozen,
        frozen,
        [(0, 0), (2, 2)],
        [{"kind": "replace", "donor_positions": [1], "target_positions": [1]}],
        [1],
        [2],
        [2],
        [],
    )
    with pytest.raises(TruePartialPrefillError, match="NORMAL_CONTINUATION_TOKEN"):
        _write_instruction(tmp_path, monkeypatch, payload, frozen)


@pytest.mark.parametrize(
    ("source", "frozen", "alignment", "edits", "mandatory", "eligible", "ranking", "selected"),
    [
        (
            [1, 2, 3, 4],
            [1, 9, 3, 4],
            [(0, 0), (2, 2), (3, 3)],
            [{"kind": "replace", "donor_positions": [1], "target_positions": [1]}],
            [1],
            [2, 3],
            [3, 2],
            [3],
        ),
        (
            [1, 2, 3],
            [1, 9, 2, 3],
            [(0, 0), (2, 1), (3, 2)],
            [{"kind": "insert", "donor_positions": [], "target_positions": [1]}],
            [1],
            [2, 3],
            [3, 2],
            [3],
        ),
        (
            [1, 2, 3, 4],
            [1, 3, 4],
            [(0, 0), (1, 2), (2, 3)],
            [{"kind": "delete", "donor_positions": [1], "target_positions": []}],
            [],
            [1, 2],
            [2, 1],
            [2],
        ),
    ],
)
def test_exact_replacement_insertion_and_deletion_plans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: list[int],
    frozen: list[int],
    alignment: list[tuple[int, int]],
    edits: list[dict[str, object]],
    mandatory: list[int],
    eligible: list[int],
    ranking: list[int],
    selected: list[int],
) -> None:
    request = frozen + [88, 89]
    payload = _target_manifest(
        tmp_path,
        source,
        frozen,
        request,
        alignment,
        edits,
        mandatory,
        eligible,
        ranking,
        selected,
    )
    instruction = _write_instruction(tmp_path, monkeypatch, payload, request)
    assert instruction is not None
    plan = build_target_plan(instruction, source, request)
    assert set(plan.recompute_positions) == set(mandatory) | set(selected)
    assert {pair.target_position for pair in plan.reuse_alignment}.isdisjoint(
        plan.recompute_positions
    )
    assert {pair.target_position for pair in plan.reuse_alignment} | set(
        plan.recompute_positions
    ) == set(range(len(frozen)))
    assert len(plan.shifted_reuse_alignment) == sum(t != d for t, d in alignment if t not in plan.recompute_positions)


class _Block:
    def __init__(self, block_id: int):
        self.block_id = block_id
        self.ref_cnt = 1


def test_donor_registry_ownership_incompatibility_and_eviction_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [1, 2, 3, 4]
    donor_payload = _base_manifest(tmp_path, "donor_register", source)
    donor = _write_instruction(tmp_path, monkeypatch, donor_payload, source)
    assert donor is not None
    target_tokens = [1, 9, 3, 4, 77]
    target_payload = _target_manifest(
        tmp_path,
        source,
        target_tokens[:4],
        target_tokens,
        [(0, 0), (2, 2), (3, 3)],
        [{"kind": "replace", "donor_positions": [1], "target_positions": [1]}],
        [1],
        [2, 3],
        [3, 2],
        [3],
    )
    target = _write_instruction(tmp_path, monkeypatch, target_payload, target_tokens)
    assert target is not None

    registry = DonorRegistry()
    blocks = ((_Block(10),),)
    registry.register("donor-request", donor, source, blocks)
    incompatible = replace(target, cache_contract={**target.cache_contract, "main_row_width": 4})
    with pytest.raises(TruePartialPrefillError, match="CACHE_CONTRACT_MISMATCH"):
        registry.resolve("target-request", incompatible, target_tokens)
    blocks[0][0].ref_cnt = 0
    with pytest.raises(TruePartialPrefillError, match="EVICTED_OR_FREED"):
        registry.resolve("target-request", target, target_tokens)


def test_cpu_cache_copy_preserves_only_reuse_rows_and_packed_indexer_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [1, 2, 3, 4]
    target_tokens = [1, 9, 3, 4, 77]
    payload = _target_manifest(
        tmp_path,
        source,
        target_tokens[:4],
        target_tokens,
        [(0, 0), (2, 2), (3, 3)],
        [{"kind": "replace", "donor_positions": [1], "target_positions": [1]}],
        [1],
        [2, 3],
        [3, 2],
        [3],
    )
    instruction = _write_instruction(tmp_path, monkeypatch, payload, target_tokens)
    assert instruction is not None
    plan = build_target_plan(instruction, source, target_tokens)

    main = torch.zeros((4, 4, 3), dtype=torch.bfloat16)
    main[0, 0] = torch.tensor([10, 11, 12], dtype=torch.bfloat16)
    main[0, 2] = torch.tensor([30, 31, 32], dtype=torch.bfloat16)
    indexer = torch.zeros((4, 4, 3), dtype=torch.uint8)
    source_flat = indexer[0].view(-1)
    source_flat[0:2] = torch.tensor([10, 11], dtype=torch.uint8)
    source_flat[4:6] = torch.tensor([30, 31], dtype=torch.uint8)
    source_flat[8] = 12
    source_flat[10] = 32

    evidence = copy_reuse_rows_inplace(
        layer_caches=[("main.0", 0, main), ("indexer.0", 0, indexer)],
        donor_block_ids=[[0]],
        target_block_ids=[[2]],
        plan=plan,
        cache_contract=_cache_contract(),
    )
    assert main[2, 0].tolist() == [10, 11, 12]
    assert main[2, 1].count_nonzero() == 0  # mandatory edit was not copied
    assert main[2, 2].tolist() == [30, 31, 32]
    assert main[2, 3].count_nonzero() == 0  # selected row must be recomputed
    target_flat = indexer[2].view(-1)
    assert target_flat[0:2].tolist() == [10, 11]
    assert int(target_flat[8]) == 12
    assert target_flat[4:6].tolist() == [30, 31]
    assert int(target_flat[10]) == 32
    assert evidence["copied_reuse_rows_per_layer"] == 2


def test_execution_evidence_attests_exact_rows_and_prohibits_full_prefill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = [1, 2, 3, 4]
    target_tokens = [1, 9, 3, 4, 77]
    payload = _target_manifest(
        tmp_path,
        source,
        target_tokens[:4],
        target_tokens,
        [(0, 0), (2, 2), (3, 3)],
        [{"kind": "replace", "donor_positions": [1], "target_positions": [1]}],
        [1],
        [2, 3],
        [3, 2],
        [3],
    )
    instruction = _write_instruction(tmp_path, monkeypatch, payload, target_tokens)
    assert instruction is not None
    plan = build_target_plan(instruction, source, target_tokens)
    row_ids = [target_tokens[position] for position in plan.recompute_positions]
    evidence = execution_evidence(
        request_id="target-request",
        manifest_sha256=instruction.manifest_sha256,
        plan=plan,
        model_input_positions=plan.recompute_positions,
        model_input_token_ids=row_ids,
        copy_evidence={"copied_reuse_rows_per_layer": 2},
        rank=0,
    )
    assert evidence["full_target_prefill_executed"] is False
    assert evidence["model_input_positions"] == list(plan.recompute_positions)
    assert evidence["model_input_positions_sha256"] == canonical_json_sha256(
        list(plan.recompute_positions)
    )
    full_plan = replace(plan, recompute_positions=tuple(range(4)))
    with pytest.raises(TruePartialPrefillError, match="FULL_PREFILL_PROHIBITION"):
        execution_evidence(
            request_id="target-request",
            manifest_sha256=instruction.manifest_sha256,
            plan=full_plan,
            model_input_positions=full_plan.recompute_positions,
            model_input_token_ids=target_tokens[:4],
            copy_evidence={},
            rank=0,
        )


def test_sparse_input_and_slot_mapping_use_exact_recompute_positions() -> None:
    positions = [1, 3, 6]
    evidence = verify_exact_slot_mappings(
        target_block_ids=[[10, 20], [30, 40]],
        positions=positions,
        actual_slot_ids=[[41, 43, 82], [121, 123, 162]],
        block_size=4,
    )
    assert evidence["slot_mapping_positions"] == positions
    assert evidence["slot_mapping_exact_positions_verified"] is True
    with pytest.raises(TruePartialPrefillError, match="EXACT_POSITION_MISMATCH"):
        verify_exact_slot_mappings(
            target_block_ids=[[10, 20]],
            positions=positions,
            actual_slot_ids=[[40, 41, 42]],
            block_size=4,
        )


def test_first_normal_continuation_uses_frozen_boundary_and_q2_tokens() -> None:
    target = [10, 11, 12, 20, 21, 22]
    slots = verify_exact_slot_mappings(
        target_block_ids=[[5, 6]],
        positions=[3, 4, 5],
        actual_slot_ids=[[23, 24, 25]],
        block_size=4,
    )
    evidence = continuation_execution_evidence(
        request_id="target-request",
        manifest_sha256="a" * 64,
        frozen_history_token_count=3,
        target_request_token_count=6,
        scheduler_num_computed_tokens=3,
        model_input_positions=[3, 4, 5],
        model_input_token_ids=[20, 21, 22],
        target_token_ids=target,
        slot_mapping_evidence=slots,
        rank=0,
    )
    assert evidence["normal_continuation_start"] == 3
    assert evidence["stale_recompute_count_boundary_observed"] is False
    with pytest.raises(TruePartialPrefillError, match="BOUNDARY_STALE"):
        continuation_execution_evidence(
            request_id="target-request",
            manifest_sha256="a" * 64,
            frozen_history_token_count=3,
            target_request_token_count=6,
            scheduler_num_computed_tokens=2,
            model_input_positions=[3],
            model_input_token_ids=[20],
            target_token_ids=target,
            slot_mapping_evidence=slots,
            rank=0,
        )


def test_continuation_refreshes_blocks_across_frozen_block_boundary() -> None:
    pending = {
        "request_id": "target-request",
        "manifest_sha256": "a" * 64,
        "evidence_path": "/tmp/evidence.jsonl",
        "frozen_history_token_count": 4,
        "target_request_token_count": 6,
        "block_size": 4,
        "sparse_recompute_count": 2,
        "initial_target_block_ids": [[5]],
    }
    payload = build_continuation_payload(
        pending,
        current_target_block_ids=[[5, 6]],
        num_scheduled_tokens=2,
    )
    assert payload["target_block_ids"] == [[5, 6]]
    assert "initial_target_block_ids" not in payload
    validate_continuation_payload(pending, payload)
    slots = verify_exact_slot_mappings(
        target_block_ids=payload["target_block_ids"],
        positions=[4, 5],
        actual_slot_ids=[[24, 25]],
        block_size=4,
    )
    assert slots["slot_mapping_exact_positions_verified"] is True

    mutated = {**payload, "manifest_sha256": "b" * 64}
    with pytest.raises(TruePartialPrefillError, match="ATTESTATION_MUTATED"):
        validate_continuation_payload(pending, mutated)
    with pytest.raises(TruePartialPrefillError, match="BLOCK_PREFIX_MUTATED"):
        build_continuation_payload(
            pending,
            current_target_block_ids=[[7, 6]],
            num_scheduled_tokens=2,
        )


def test_server_schema_and_lock_preserve_claim_boundary_and_hashes(
    tmp_path: Path,
) -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["backend_kind"]["const"] == "true_partial_prefill"
    selection = schema["$defs"]["selection"]["properties"]
    assert selection["semantic_quantity"]["const"] == "importance_for_recompute"
    assert selection["score_direction"]["const"] == "higher_score_first"
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    source = [1, 2, 3]
    donor_manifest = _base_manifest(tmp_path, "donor_register", source)
    validator.validate(donor_manifest)
    target_manifest = _target_manifest(
        tmp_path,
        source,
        [1, 9, 2, 3],
        [1, 9, 2, 3, 77],
        [(0, 0), (2, 1), (3, 2)],
        [{"kind": "insert", "donor_positions": [], "target_positions": [1]}],
        [1],
        [2, 3],
        [3, 2],
        [3],
    )
    validator.validate(target_manifest)
    assert lock["claim_boundary"]["legacy_accuracy_emulation_is_true_partial"] is False
    assert lock["claim_boundary"]["full_target_prefill_then_overwrite_prohibited_in_true_path"] is True
    assert lock["claim_boundary"]["runtime_completion_claimed"] is False
    assert lock["base_is_post_legacy_patch"] is True
    assert lock["legacy_patch_required_as_packaging_base"] is True
    assert lock["legacy_and_true_runtime_modes_mutually_exclusive"] is True
    assert (
        lock["patch_order"][0]["role"]
        == "required_post_legacy_packaging_base_runtime_mode_isolated"
    )
    assert lock["patch_order"][1]["apply_args"] == ["--unidiff-zero"]
    assert lock["patch_order"][1]["sha256"] == hashlib.sha256(
        PATCH_PATH.read_bytes()
    ).hexdigest()
    assert lock["instrumentation"]["sha256"] == hashlib.sha256(
        INSTRUMENTATION_PATH.read_bytes()
    ).hexdigest()
    assert lock["server_manifest_schema"]["sha256"] == hashlib.sha256(
        SCHEMA_PATH.read_bytes()
    ).hexdigest()


def test_overlay_wires_distinct_scheduler_runner_and_sparse_backend_path() -> None:
    patch = PATCH_PATH.read_text(encoding="utf-8")
    required = (
        "putpocket_instruction_from_xargs",
        "self.putpocket_donor_registry.resolve",
        "delay_cache_blocks=True",
        "putpocket_copy_reuse_rows_inplace",
        "positions_np = np.asarray",
        "self.positions[:total_num_scheduled_tokens].copy_(",
        "model_input_token_ids=self.input_ids.cpu[",
        "putpocket_verify_slot_mappings",
        "putpocket_virtual_decode_lanes",
        "TRUE_PARTIAL_CANNOT_COEXIST_WITH_ACCURACY_EMULATION_CONTROL",
        "TRUE_PARTIAL_RECOMPUTE_COUNT_NOT_SPARSE",
        "arbitrary_position_single_query_decode_lanes",
        "TRUE_PARTIAL_POST_PATCH_SCHEDULER_BOUNDARY_STALE",
        "TRUE_PARTIAL_WORKER_RECEIVED_STALE_RECOMPUTE_BOUNDARY",
        "putpocket_true_partial_continuation=putpocket_continuation_payload",
    )
    for marker in required:
        assert marker in patch
    assert "putpocket_true_partial_positions" in patch
    assert "common_attn_metadata.putpocket_virtual_decode_lanes()" in patch
    assert "putpocket_true_partial=putpocket_true_partial_payload" in patch
    assert "PUTPOCKET_GLM52_FORCED_REUSE_CONTROL =" not in patch
