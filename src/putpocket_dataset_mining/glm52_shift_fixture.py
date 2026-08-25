"""Build the canonical GLM-5.2 offset-shift KV-reuse fixture.

The source episode contains a duplicated user request.  The canonical edit
deletes the second copy.  This makes a realistic conversation-compaction
fixture without changing system, identity, authorization, or tool-permission
text.  This module creates offline evidence only; it never enables cache
mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
SCENARIO_ID = "mbpp_730_duplicate_user_turn_delete_v1"
SOURCE_ATTEMPT_ID = "attempt_5a8d1db9b812"
SOURCE_MESSAGES_SHA256 = "21ac22032eea683fbec5425daa36567f1e13b8adec7fbdc63d0e8d8c330133f6"
TOKENIZER_REVISION = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
TOKENIZER_FILES = {
    "tokenizer.json": "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
    "tokenizer_config.json": "77af7d4769cd62c107b90495cac9b0ba81573c86486821bfba2980c04285ec7a",
    "chat_template.jinja": "172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679",
}
BLOCK_SIZE = 64
DELETED_MESSAGE_INDEX = 2
EXPECTED_OLD_TOKENS = 909
EXPECTED_NEW_TOKENS = 861
EXPECTED_OLD_SPAN = (349, 397)
EXPECTED_NEW_SPAN = (349, 349)
EXPECTED_POSITION_DELTA = -48
EXPECTED_SUFFIX_TOKENS = 512
EXPECTED_OLD_TOKEN_SHA256 = "2fbd9d50825827d748ff56844d87941f1beba7b344b3e6e730905044416f16ae"
EXPECTED_NEW_TOKEN_SHA256 = "3dcceb2142dca9bb6272a2d677b32a3bd565ea17b1bdd645515a2c3bcf301297"
MODEL_ID = "nvidia/GLM-5.2-NVFP4"


class FixtureError(RuntimeError):
    """The source episode or tokenizer does not match the pinned fixture."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise FixtureError(reason)


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(payload))


def _token_digest(values: Sequence[int]) -> str:
    return _sha256_bytes(json.dumps(list(values), separators=(",", ":")).encode("utf-8"))


def _load_tokenizer(tokenizer_root: Path) -> Any:
    _require(tokenizer_root.is_absolute() and tokenizer_root.is_dir(), "TOKENIZER_ROOT_INVALID")
    for name, expected in TOKENIZER_FILES.items():
        path = tokenizer_root / name
        _require(path.is_file(), f"TOKENIZER_FILE_MISSING:{name}")
        _require(_sha256_file(path) == expected, f"TOKENIZER_FILE_DIGEST_MISMATCH:{name}")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - environment preflight
        raise FixtureError("TRANSFORMERS_UNAVAILABLE") from error
    return AutoTokenizer.from_pretrained(tokenizer_root, local_files_only=True)


def _render(tokenizer: Any, messages: Sequence[dict[str, str]]) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=True,
        return_dict=True,
    )
    values = encoded["input_ids"] if hasattr(encoded, "keys") and "input_ids" in encoded else encoded
    if values and isinstance(values[0], list):
        values = values[0]
    result = [int(value) for value in values]
    _require(tokenizer.encode(text, add_special_tokens=False) == result, "CHAT_TEMPLATE_TEXT_TOKEN_MISMATCH")
    return str(text), result


def _edit_span(old: Sequence[int], new: Sequence[int]) -> tuple[int, int, int, int]:
    prefix = 0
    while prefix < min(len(old), len(new)) and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < len(old) - prefix
        and suffix < len(new) - prefix
        and old[len(old) - suffix - 1] == new[len(new) - suffix - 1]
    ):
        suffix += 1
    return prefix, len(old) - suffix, prefix, len(new) - suffix


def _page_count(token_count: int) -> int:
    return math.ceil(token_count / BLOCK_SIZE)


def _page_range(start: int, end: int, total: int) -> list[int]:
    if start == end:
        return [min(start // BLOCK_SIZE, max(_page_count(total) - 1, 0))]
    return list(range(start // BLOCK_SIZE, (end - 1) // BLOCK_SIZE + 1))


def _candidate(
    tokenizer: Any,
    candidate_id: str,
    baseline_messages: list[dict[str, str]],
    edited_messages: list[dict[str, str]],
    *,
    provenance: str,
    edit_type: str,
    criticality: str,
    rationale: str,
) -> tuple[dict[str, Any], str, str]:
    old_text, old_ids = _render(tokenizer, baseline_messages)
    new_text, new_ids = _render(tokenizer, edited_messages)
    old_start, old_end, new_start, new_end = _edit_span(old_ids, new_ids)
    suffix = len(old_ids) - old_end
    delta = len(new_ids) - len(old_ids)
    suffix_identity = old_ids[old_end:] == new_ids[new_end:]
    return ({
        "candidate_id": candidate_id,
        "provenance": provenance,
        "edit_type": edit_type,
        "semantic_or_control_criticality": criticality,
        "rationale": rationale,
        "old_token_count": len(old_ids),
        "new_token_count": len(new_ids),
        "old_token_span": [old_start, old_end],
        "new_token_span": [new_start, new_end],
        "old_span_token_ids": old_ids[old_start:old_end],
        "new_span_token_ids": new_ids[new_start:new_end],
        "downstream_offset_delta": delta,
        "unchanged_suffix_token_count": suffix,
        "unchanged_suffix_token_identity_exact": suffix_identity,
        "unchanged_content_token_alignable": suffix_identity,
        "old_total_physical_pages": _page_count(len(old_ids)),
        "new_total_physical_pages": _page_count(len(new_ids)),
        "old_edit_physical_pages": _page_range(old_start, old_end, len(old_ids)),
        "new_edit_or_landing_physical_pages": _page_range(new_start, new_end, len(new_ids)),
        "old_suffix_start_page": old_end // BLOCK_SIZE if suffix else None,
        "new_suffix_start_page": new_end // BLOCK_SIZE if suffix else None,
        "old_prompt_token_ids_sha256": _token_digest(old_ids),
        "new_prompt_token_ids_sha256": _token_digest(new_ids),
        "baseline_text_sha256": _sha256_bytes(old_text.encode("utf-8")),
        "edited_text_sha256": _sha256_bytes(new_text.encode("utf-8")),
    }, old_text, new_text)


def _candidate_definitions(messages: list[dict[str, str]]) -> list[tuple[str, list[dict[str, str]], list[dict[str, str]], str, str, str, str]]:
    def without(*indices: int) -> list[dict[str, str]]:
        rejected = set(indices)
        return [dict(message) for index, message in enumerate(messages) if index not in rejected]

    system = messages[0]["content"]
    policy = "\n# Google Docstring Required\n\nThe primary function must include a concise Google-style docstring with Args and Returns sections."
    _require(policy in system, "SOURCE_SYSTEM_POLICY_FRAGMENT_MISSING")
    without_policy = [dict(message) for message in messages]
    without_policy[0]["content"] = system.replace(policy, "")
    base = [dict(message) for message in messages]
    return [
        ("delete_duplicate_query1", base, without(2), "prepared/messages_history2.json messages[1:3]", "deletion", "non-control semantic redundancy", "Deletes an accidentally duplicated user task while preserving the first copy."),
        ("insert_duplicate_query1", without(2), base, "inverse of prepared/messages_history2.json messages[1:3]", "insertion", "non-control semantic redundancy", "Models insertion of an accidental duplicate; retained as the exact inverse sensitivity candidate."),
        ("delete_write_tool_round", base, without(3, 4), "History1 write_to_file assistant/tool-result pair", "deletion", "task-state critical", "Removes the code-writing state transition."),
        ("delete_execute_tool_round", base, without(5, 6), "History1 execute_command assistant/tool-result pair", "deletion", "verification-state critical", "Removes execution evidence used by later turns."),
        ("delete_format_recovery_round", base, without(7, 8), "History1 invalid assistant prose and format-error feedback", "deletion", "tool-protocol control critical", "Removes protocol enforcement and recovery context."),
        ("delete_completion_round", base, without(9, 10), "History1 attempt_completion assistant/tool-result pair", "deletion", "task-control state", "Removes the completion state but leaves little downstream context."),
        ("delete_query2", base, without(11), "prepared/query2.txt", "deletion", "semantic task request", "Deletes the active refactor request and has no meaningful downstream suffix."),
        ("delete_system_docstring_policy", base, without_policy, "prepared/system_prompt_2.md policy delta", "deletion", "system/control critical", "Edits a system instruction and is excluded on safety grounds."),
    ]


def _request(prompt: list[int]) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "prompt": prompt,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 512,
        "n": 1,
        "seed": 0,
        "stream": False,
        "return_token_ids": True,
    }


def build_fixture(source_root: Path, tokenizer_root: Path, output_root: Path) -> dict[str, Any]:
    """Create the immutable fixture and return its manifest summary."""
    source_root = source_root.resolve()
    tokenizer_root = tokenizer_root.resolve()
    output_root = output_root.resolve()
    _require(source_root.name == SOURCE_ATTEMPT_ID and source_root.is_dir(), "SOURCE_ATTEMPT_INVALID")
    messages_path = source_root / "prepared/messages_history2.json"
    _require(_sha256_file(messages_path) == SOURCE_MESSAGES_SHA256, "SOURCE_MESSAGES_DIGEST_MISMATCH")
    _require(not output_root.exists(), "OUTPUT_ROOT_ALREADY_EXISTS")
    tokenizer = _load_tokenizer(tokenizer_root)
    messages = json.loads(messages_path.read_text(encoding="utf-8"))
    _require(isinstance(messages, list) and len(messages) == 12, "SOURCE_MESSAGE_COUNT_INVALID")
    _require(messages[1] == messages[DELETED_MESSAGE_INDEX], "DUPLICATE_MESSAGE_ATTESTATION_FAILED")
    _require(all(message.get("role") in {"system", "user", "assistant"} and isinstance(message.get("content"), str) for message in messages), "SOURCE_MESSAGE_SCHEMA_INVALID")
    edited_messages = [dict(message) for index, message in enumerate(messages) if index != DELETED_MESSAGE_INDEX]
    old_text, old_ids = _render(tokenizer, messages)
    new_text, new_ids = _render(tokenizer, edited_messages)
    old_start, old_end, new_start, new_end = _edit_span(old_ids, new_ids)
    _require((len(old_ids), len(new_ids)) == (EXPECTED_OLD_TOKENS, EXPECTED_NEW_TOKENS), "CANONICAL_TOKEN_COUNTS_CHANGED")
    _require((old_start, old_end) == EXPECTED_OLD_SPAN and (new_start, new_end) == EXPECTED_NEW_SPAN, "CANONICAL_EDIT_SPAN_CHANGED")
    _require(_token_digest(old_ids) == EXPECTED_OLD_TOKEN_SHA256, "CANONICAL_OLD_TOKEN_DIGEST_CHANGED")
    _require(_token_digest(new_ids) == EXPECTED_NEW_TOKEN_SHA256, "CANONICAL_NEW_TOKEN_DIGEST_CHANGED")
    _require(old_ids[old_end:] == new_ids[new_end:] and len(old_ids) - old_end == EXPECTED_SUFFIX_TOKENS, "CANONICAL_SUFFIX_ALIGNMENT_CHANGED")

    output_root.mkdir(parents=True)
    candidate_records = []
    for candidate_id, baseline, edited, provenance, edit_type, criticality, rationale in _candidate_definitions(messages):
        record, baseline_text, edited_text = _candidate(
            tokenizer, candidate_id, baseline, edited,
            provenance=provenance, edit_type=edit_type,
            criticality=criticality, rationale=rationale,
        )
        text_root = output_root / "candidate_texts"
        baseline_path = text_root / f"{candidate_id}.baseline.txt"
        edited_path = text_root / f"{candidate_id}.edited.txt"
        text_root.mkdir(exist_ok=True)
        baseline_path.write_text(baseline_text, encoding="utf-8")
        edited_path.write_text(edited_text, encoding="utf-8")
        record["baseline_text_path"] = str(baseline_path.relative_to(output_root))
        record["edited_text_path"] = str(edited_path.relative_to(output_root))
        record["selected"] = candidate_id == "delete_duplicate_query1"
        candidate_records.append(record)

    old_to_new: list[int | None] = list(range(old_start)) + [None] * (old_end - old_start) + [position + EXPECTED_POSITION_DELTA for position in range(old_end, len(old_ids))]
    new_to_old = list(range(new_start)) + [position - EXPECTED_POSITION_DELTA for position in range(new_start, len(new_ids))]
    _require(len(old_to_new) == len(old_ids) and len(new_to_old) == len(new_ids), "ALIGNMENT_LENGTH_INVALID")
    _require(all(old_ids[old] == new_ids[new] for old, new in enumerate(old_to_new) if new is not None), "ALIGNMENT_TOKEN_IDENTITY_INVALID")

    page_rows = []
    for page in range(_page_count(len(new_ids))):
        new_positions = list(range(page * BLOCK_SIZE, min((page + 1) * BLOCK_SIZE, len(new_ids))))
        old_positions = [new_to_old[position] for position in new_positions]
        page_rows.append({
            "new_page": page,
            "new_token_span": [new_positions[0], new_positions[-1] + 1],
            "mapped_old_token_span": [min(old_positions), max(old_positions) + 1],
            "mapped_old_pages": sorted({position // BLOCK_SIZE for position in old_positions}),
            "position_deltas": sorted({new - old for new, old in zip(new_positions, old_positions, strict=True)}),
            "single_old_page_copy_possible": len({position // BLOCK_SIZE for position in old_positions}) == 1 and all((new % BLOCK_SIZE) == (old % BLOCK_SIZE) for new, old in zip(new_positions, old_positions, strict=True)),
        })

    # Page 5 contains the deletion landing point.  Pages 4 and 6 are closed as
    # immediate physical/dependency neighbours.  With no semantic stability
    # evidence and no RoPE correction, causal fail-closed closure additionally
    # recomputes every post-edit token.
    edit_page = new_start // BLOCK_SIZE
    neighbour_pages = list(range(max(0, edit_page - 1), min(_page_count(len(new_ids)), edit_page + 2)))
    local_closure = list(range(neighbour_pages[0] * BLOCK_SIZE, min((neighbour_pages[-1] + 1) * BLOCK_SIZE, len(new_ids))))
    exact_raw_prefix = list(range(0, neighbour_pages[0] * BLOCK_SIZE))
    full_fail_closed_recompute = list(range(neighbour_pages[0] * BLOCK_SIZE, len(new_ids)))
    geometry_only_shifted = list(range((neighbour_pages[-1] + 1) * BLOCK_SIZE, len(new_ids)))

    fixture_contract = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": SCENARIO_ID,
        "status": "canonical_offline_offset_shift_fixture",
        "production_default_enabled": False,
        "source": {
            "attempt_id": SOURCE_ATTEMPT_ID,
            "dataset_id": "google-research-datasets/mbpp",
            "split": "train",
            "task_id": "730",
            "row_index": 129,
            "messages_path": str(messages_path),
            "messages_sha256": SOURCE_MESSAGES_SHA256,
            "gold_patch_in_prompt": False,
        },
        "tokenizer": {
            "model_id": MODEL_ID,
            "revision": TOKENIZER_REVISION,
            "files": TOKENIZER_FILES,
            "special_and_chat_template_tokens_included": True,
            "add_generation_prompt": True,
        },
        "edit": {
            "type": "deletion",
            "message_index": DELETED_MESSAGE_INDEX,
            "old_token_span": [old_start, old_end],
            "new_token_span": [new_start, new_end],
            "deleted_token_count": old_end - old_start,
            "old_token_ids": old_ids[old_start:old_end],
            "new_token_ids": [],
            "old_prompt_token_count": len(old_ids),
            "new_prompt_token_count": len(new_ids),
            "downstream_position_delta": EXPECTED_POSITION_DELTA,
            "same_length": False,
            "rope_absolute_positions_unchanged": False,
            "unchanged_suffix_old_span": [old_end, len(old_ids)],
            "unchanged_suffix_new_span": [new_end, len(new_ids)],
            "unchanged_suffix_token_count": EXPECTED_SUFFIX_TOKENS,
            "unchanged_suffix_token_identity_exact": True,
        },
        "runtime_support": {
            "existing_glm_hook": "same-position raw copy only",
            "raw_shifted_reuse_safe": False,
            "rope_correction_implemented": False,
            "rope_corrected_reuse_available": False,
            "forced_shifted_raw_reuse_available": False,
            "reason": "Shifted MLA latent state and packed indexer state cannot be copied by the current same-position hook; no mathematically validated RoPE correction exists.",
        },
        "request_attestation": {
            "baseline_prompt_token_count": len(old_ids),
            "edited_prompt_token_count": len(new_ids),
            "baseline_prompt_token_ids_sha256": _token_digest(old_ids),
            "edited_prompt_token_ids_sha256": _token_digest(new_ids),
            "baseline_serialized_sha256": _sha256_bytes(old_text.encode("utf-8")),
            "edited_serialized_sha256": _sha256_bytes(new_text.encode("utf-8")),
        },
        "artifact_files": {
            "baseline_request": "baseline_request.json",
            "edited_request": "edited_request.json",
            "alignment": "old-to-new-token-alignment.json",
            "rope_position_delta": "rope-position-delta-map.json",
            "physical_page_map": "physical-page-map.json",
            "candidate_sets": "donor-recompute-reuse-candidate-sets.json",
        },
    }
    _write_json(output_root / "baseline_messages.json", messages)
    _write_json(output_root / "edited_messages.json", edited_messages)
    (output_root / "baseline_serialized_prompt.txt").write_text(old_text, encoding="utf-8")
    (output_root / "edited_serialized_prompt.txt").write_text(new_text, encoding="utf-8")
    _write_json(output_root / "baseline_request.json", _request(old_ids))
    _write_json(output_root / "edited_request.json", _request(new_ids))
    _write_json(output_root / "tokenizer-attestation.json", fixture_contract["tokenizer"] | {
        "tokenizer_root_used": str(tokenizer_root),
        "baseline_token_count": len(old_ids),
        "edited_token_count": len(new_ids),
        "baseline_prompt_token_ids_sha256": _token_digest(old_ids),
        "edited_prompt_token_ids_sha256": _token_digest(new_ids),
        "baseline_serialized_bytes": len(old_text.encode("utf-8")),
        "edited_serialized_bytes": len(new_text.encode("utf-8")),
        "baseline_serialized_sha256": _sha256_bytes(old_text.encode("utf-8")),
        "edited_serialized_sha256": _sha256_bytes(new_text.encode("utf-8")),
    })
    _write_json(output_root / "edit-script.json", fixture_contract["edit"] | {
        "operation": "delete_message",
        "source_message_content_sha256": _sha256_bytes(messages[DELETED_MESSAGE_INDEX]["content"].encode("utf-8")),
        "duplicate_of_message_index": 1,
        "semantic_class": "non-control redundant user task",
    })
    _write_json(output_root / "old-to-new-token-alignment.json", {
        "schema_version": SCHEMA_VERSION,
        "alignment_algorithm": "exact longest-common-prefix plus longest-common-suffix for one deletion",
        "segments": [
            {"kind": "unchanged_same_position", "old_span": [0, old_start], "new_span": [0, new_start], "position_delta": 0},
            {"kind": "deleted", "old_span": [old_start, old_end], "new_span": [new_start, new_end], "position_delta": None},
            {"kind": "unchanged_shifted", "old_span": [old_end, len(old_ids)], "new_span": [new_end, len(new_ids)], "position_delta": EXPECTED_POSITION_DELTA},
        ],
        "old_to_new": old_to_new,
        "new_to_old": new_to_old,
        "all_mapped_token_ids_equal": True,
        "unknown_mapping_count": 0,
    })
    _write_json(output_root / "rope-position-delta-map.json", {
        "schema_version": SCHEMA_VERSION,
        "definition": "new absolute token position minus mapped old absolute token position",
        "segments": [
            {"new_span": [0, new_start], "old_span": [0, old_start], "delta": 0, "raw_donor_geometry_compatible": True},
            {"new_span": [new_start, len(new_ids)], "old_span": [old_end, len(old_ids)], "delta": EXPECTED_POSITION_DELTA, "raw_donor_geometry_compatible": False},
        ],
        "old_kv_can_be_transplanted_unchanged_when_delta_nonzero": False,
        "mathematically_and_implementation_valid_rope_correction_available": False,
    })
    _write_json(output_root / "physical-page-map.json", {
        "schema_version": SCHEMA_VERSION,
        "block_size_tokens": BLOCK_SIZE,
        "old_page_count": _page_count(len(old_ids)),
        "new_page_count": _page_count(len(new_ids)),
        "edit_landing_new_page": edit_page,
        "edit_touched_old_pages": _page_range(old_start, old_end, len(old_ids)),
        "dependency_neighbour_closure_new_pages": neighbour_pages,
        "rows": page_rows,
    })
    _write_json(output_root / "donor-recompute-reuse-candidate-sets.json", {
        "schema_version": SCHEMA_VERSION,
        "block_size_tokens": BLOCK_SIZE,
        "closure_rule": "Recompute the deletion landing page and one immediate physical-page neighbour on each side; because all later hidden states causally depend on edited context and neither native stability evidence nor RoPE correction exists, fail-closed dependency closure extends recompute through the live tail.",
        "local_page_closure_new_positions": local_closure,
        "production_safe_raw_donor_new_positions": exact_raw_prefix,
        "production_safe_raw_donor_old_positions": exact_raw_prefix,
        "mandatory_fail_closed_recompute_new_positions": full_fail_closed_recompute,
        "deleted_donor_only_old_positions": list(range(old_start, old_end)),
        "geometry_only_rope_correctable_candidate_new_positions": geometry_only_shifted,
        "geometry_only_rope_correctable_candidate_old_positions": [position - EXPECTED_POSITION_DELTA for position in geometry_only_shifted],
        "rope_corrected_candidate_is_safe": False,
        "rope_corrected_candidate_requirements": ["mathematically validated MLA latent RoPE transform", "packed indexer-key transform or recomputation", "native score stability gate", "TP consensus", "all-layer dependency closure"],
        "forced_unsafe_raw_shifted_candidate_new_positions": geometry_only_shifted,
        "forced_unsafe_raw_shifted_candidate_old_positions": [position - EXPECTED_POSITION_DELTA for position in geometry_only_shifted],
        "forced_unsafe_raw_shifted_candidate_label": "UNSAFE_ABLATION_ONLY_NOT_IMPLEMENTED",
        "unknown_positions": [],
        "live_tail_new_positions": list(range(new_start, len(new_ids))),
    })
    _write_json(output_root / "candidate-inventory.json", {
        "schema_version": SCHEMA_VERSION,
        "source_attempt": str(source_root),
        "baseline_definition": "prepared/messages_history2.json rendered with the pinned GLM tokenizer and add_generation_prompt=true",
        "candidate_count": len(candidate_records),
        "selected_candidate_id": "delete_duplicate_query1",
        "candidates": candidate_records,
    })
    _write_json(output_root / "fixture-contract.json", fixture_contract)
    readme = f"""# GLM-5.2 offset-shift KV-reuse fixture

Scenario `{SCENARIO_ID}` deletes the redundant second Query1 user message from
MBPP task 730 attempt `{SOURCE_ATTEMPT_ID}`.  It is a non-control conversation
compaction edit: system, identity, authorization, and tool-permission text is
unchanged.  Gold reference code and hidden tests are not included in either
model request.

The pinned GLM tokenizer produces 909 baseline tokens and 861 edited tokens,
including chat-template/special tokens and the assistant generation prefix.
Old tokens `[349,397)` are deleted.  The 512-token suffix maps exactly by token
identity from old `[397,909)` to new `[349,861)`, so its absolute position delta
is -48.  The suffix therefore begins on old physical page 6 and new page 5 for
the real 64-token block size.

Raw shifted KV reuse is not safe.  The current runtime hook only copies raw
same-position MLA and indexer rows and implements no RoPE correction.  This
fixture consequently permits only exact pre-edit prefix rows as raw donor
candidates; the fail-closed set recomputes page-neighbour closure and every
shifted/live-tail row.  The separately listed geometry-only candidates are for
future RoPE-correction research or an explicitly unsafe ablation, never a safe
reuse claim.

Rebuild with:

```bash
python -m putpocket_dataset_mining.glm52_shift_fixture \\
  --source-root {source_root} \\
  --tokenizer-root {tokenizer_root} \\
  --output-root <new-empty-output-directory>
```
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")

    file_rows = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS"}:
            file_rows.append({"path": str(path.relative_to(output_root)), "bytes": path.stat().st_size, "sha256": _sha256_file(path)})
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "glm52_offset_shift_fixture_cross_environment",
        "authoritative_cluster_center_pass": False,
        "actual_kv_reuse_result": False,
        "scenario_id": SCENARIO_ID,
        "generator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "fixture_contract_sha256": _sha256_file(output_root / "fixture-contract.json"),
        "files": file_rows,
    }
    _write_json(output_root / "MANIFEST.json", manifest)
    checksum_paths = [path for path in sorted(output_root.rglob("*")) if path.is_file() and path.name != "SHA256SUMS"]
    checksums = "".join(f"{_sha256_file(path)}  {path.relative_to(output_root)}\n" for path in checksum_paths)
    (output_root / "SHA256SUMS").write_text(checksums, encoding="utf-8")
    return {
        "output_root": str(output_root),
        "scenario_id": SCENARIO_ID,
        "manifest_sha256": _sha256_file(output_root / "MANIFEST.json"),
        "sha256sums_sha256": _sha256_file(output_root / "SHA256SUMS"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(json.dumps(build_fixture(arguments.source_root, arguments.tokenizer_root, arguments.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
