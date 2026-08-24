from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from putpocket_dataset_mining.glm52_forced_reuse import (
    DOWNSTREAM_END,
    DOWNSTREAM_START,
    EDIT_POSITION,
    NEW_TOKEN_ID,
    OLD_TOKEN_ID,
    PROMPT_TOKENS,
    FULL_LAYERS,
    SAMPLE_POINTS,
    build_selector,
)
from putpocket_dataset_mining.glm52_forced_reuse_sweep import (
    INDEXER_LAYERS as SWEEP_INDEXER_LAYERS,
    MAIN_LAYERS as SWEEP_MAIN_LAYERS,
    _control,
    _verify_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = ROOT / "instrumentation/vllm/glm52_forced_edit_reuse.py"
PATCH_PATH = (
    ROOT
    / "patches/vllm/4a3447d200e5aa428d68d1a00aa00f1a19a1a729"
    / "glm52_forced_edit_reuse.patch"
)
LOCK_PATH = ROOT / "configs/cluster/glm52_forced_reuse_ablation.lock.json"
SPEC = importlib.util.spec_from_file_location("glm52_forced_edit_reuse_hook", HOOK_PATH)
assert SPEC is not None and SPEC.loader is not None
HOOK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HOOK
SPEC.loader.exec_module(HOOK)


def _digest(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def _selector_payload(baseline: str, edited: str) -> dict[str, object]:
    ranking = list(range(HOOK.DOWNSTREAM_START, HOOK.DOWNSTREAM_END))
    return {
        "schema_version": 1,
        "status": "attested_before_benchmark_outcomes",
        "scenario": {
            "prompt_token_count": HOOK.PROMPT_TOKENS,
            "edit_position": HOOK.EDIT_POSITION,
            "old_token_id": HOOK.OLD_TOKEN_ID,
            "new_token_id": HOOK.NEW_TOKEN_ID,
            "downstream_range": [HOOK.DOWNSTREAM_START, HOOK.DOWNSTREAM_END],
            "same_length": True,
            "rope_positions_unchanged": True,
        },
        "source_evidence": {
            "baseline_prompt_token_ids_sha256": baseline,
            "edited_prompt_token_ids_sha256": edited,
        },
        "ranking": ranking,
        "positions_by_ratio": {
            str(ratio): sorted(ranking[: len(ranking) * ratio // 100])
            for ratio in HOOK.RATIOS
        },
    }


class HookContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop(HOOK.CONTROL_ENV, None)
        HOOK._STATE.reset()

    def test_default_path_is_inert(self) -> None:
        self.assertIsNone(HOOK._current_control())
        cache = torch.zeros((33, 64, 576), dtype=torch.bfloat16)
        slots = torch.arange(HOOK.PROMPT_TOKENS)
        self.assertEqual(
            HOOK.maybe_apply_main_cache(
                layer_name="model.layers.0.self_attn.attn",
                kv_cache=cache,
                slot_mapping=slots,
                num_prefills=1,
                num_decodes=0,
                num_prefill_tokens=HOOK.PROMPT_TOKENS,
            ),
            0,
        )

    def test_lock_binds_patch_instrumentation_and_runtime_contract(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        self.assertFalse(lock["production_default_enabled"])
        self.assertEqual(lock["source"]["vllm_commit"], "4a3447d200e5aa428d68d1a00aa00f1a19a1a729")
        self.assertEqual(lock["source"]["patch_sha256"], hashlib.sha256(PATCH_PATH.read_bytes()).hexdigest())
        self.assertEqual(lock["source"]["instrumentation_sha256"], hashlib.sha256(HOOK_PATH.read_bytes()).hexdigest())
        self.assertEqual(lock["model"]["tensor_parallel"], 4)
        self.assertEqual(lock["model"]["attention_backend"], "FLASHMLA_SPARSE")
        self.assertEqual(lock["model"]["cpu_offload_gb"], 0)
        self.assertEqual(lock["transplanted_state"]["source_layer_product_count_per_tp_rank"], 99)

    def test_ratio_endpoints_and_counts(self) -> None:
        self.assertEqual(
            [HOOK.ratio_count(ratio) for ratio in HOOK.RATIOS],
            [0, 195, 391, 586, 782, 978, 1173, 1369, 1564, 1760, 1956],
        )
        self.assertNotIn(HOOK.EDIT_POSITION, range(HOOK.DOWNSTREAM_START, HOOK.DOWNSTREAM_END))

    def test_selector_requires_frozen_prefixes(self) -> None:
        baseline = "1" * 64
        edited = "2" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.json"
            path.write_text(json.dumps(_selector_payload(baseline, edited)) + "\n", encoding="utf-8")
            loaded = HOOK._load_selector(path.resolve(), HOOK._file_sha256(path))
            self.assertEqual(len(loaded.selected(0)), 0)
            self.assertEqual(len(loaded.selected(100)), 1956)
            self.assertEqual(set(loaded.selected(100)), set(range(115, 2071)))

    def test_prompt_attestation_allows_only_decode_after_exact_prefill(self) -> None:
        baseline_ids = list(range(HOOK.PROMPT_TOKENS))
        edited_ids = baseline_ids.copy()
        baseline_ids[HOOK.EDIT_POSITION] = HOOK.OLD_TOKEN_ID
        edited_ids[HOOK.EDIT_POSITION] = HOOK.NEW_TOKEN_ID
        selector = HOOK._Selector(
            "a" * 64,
            tuple(range(115, 2071)),
            {ratio: tuple(range(115, 115 + HOOK.ratio_count(ratio))) for ratio in HOOK.RATIOS},
            _digest(baseline_ids),
            _digest(edited_ids),
        )
        with tempfile.TemporaryDirectory() as directory:
            control = HOOK._Control(
                "SNAPSHOT",
                "test",
                "donor",
                "donor",
                _digest(baseline_ids),
                _digest(edited_ids),
                selector,
                100,
                Path(directory),
            )
            HOOK._STATE.attest_prompt(
                control,
                torch.tensor(baseline_ids),
                torch.arange(HOOK.PROMPT_TOKENS),
            )
            HOOK._STATE.attest_prompt(control, torch.tensor([9]), torch.tensor([2071]))
            with self.assertRaisesRegex(HOOK.ForcedReuseInvariantError, "FULL_PROMPT_SINGLE_PREFILL_REQUIRED"):
                HOOK._STATE.attest_prompt(control, torch.tensor([9, 10]), torch.tensor([0, 1]))

    def test_main_cache_snapshot_is_private_and_destination_slots_are_owned(self) -> None:
        source = torch.arange(33 * 64 * 576, dtype=torch.int64).reshape(33, 64, 576).to(torch.bfloat16)
        source_slots = torch.arange(HOOK.PROMPT_TOKENS)
        positions = (115, 116, 2070)
        snapshot = HOOK._snapshot_main(source, 0, positions, source_slots)
        expected = snapshot.parts[0].clone()
        source.view(-1, 576)[list(snapshot.source_slots)].zero_()
        self.assertTrue(torch.equal(snapshot.parts[0], expected))
        destination = torch.zeros_like(source)
        destination_slots = torch.arange(HOOK.PROMPT_TOKENS - 1, -1, -1)
        copied = HOOK._copy_main(snapshot, destination, positions, destination_slots)
        for index, slot in enumerate(copied):
            self.assertTrue(torch.equal(destination.view(-1, 576)[slot], expected[index]))
        self.assertFalse(any(left == right for left, right in zip(snapshot.source_slots, copied, strict=True)))

    def test_packed_indexer_values_and_scale_planes_are_both_copied(self) -> None:
        source = torch.arange(33 * 64 * 132, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(33, 64, 132)
        positions = (115, 116, 2070)
        source_slots = torch.arange(HOOK.PROMPT_TOKENS)
        snapshot = HOOK._snapshot_indexer(source, 0, positions, source_slots)
        destination = torch.zeros_like(source)
        destination_slots = torch.arange(HOOK.PROMPT_TOKENS - 1, -1, -1)
        copied = HOOK._copy_indexer(snapshot, destination, positions, destination_slots)
        for index, slot in enumerate(copied):
            page, offset = divmod(slot, HOOK.BLOCK_SIZE)
            flat = destination[page].view(-1)
            self.assertTrue(torch.equal(flat[offset * 128 : (offset + 1) * 128], snapshot.parts[0][index]))
            scale = HOOK.BLOCK_SIZE * 128 + offset * 4
            self.assertTrue(torch.equal(flat[scale : scale + 4], snapshot.parts[1][index]))

    def test_page_accounting_uses_real_destination_slots(self) -> None:
        slots = torch.arange(HOOK.PROMPT_TOKENS)
        result = HOOK._page_accounting(tuple(range(128, 192)), slots)
        self.assertEqual(result["downstream_physical_page_count"], 32)
        self.assertEqual(result["touched_physical_page_count"], 1)
        self.assertEqual(result["complete_reused_physical_page_count"], 1)
        self.assertEqual(result["mixed_physical_page_count"], 0)
        mixed = HOOK._page_accounting((128, 130), slots)
        self.assertEqual(mixed["mixed_physical_page_count"], 1)
        self.assertEqual(mixed["complete_reused_physical_page_count"], 0)

    def test_preflight_requires_all_78_main_and_21_indexer_products(self) -> None:
        selector = HOOK._Selector(
            "a" * 64,
            tuple(range(115, 2071)),
            {ratio: tuple(range(115, 115 + HOOK.ratio_count(ratio))) for ratio in HOOK.RATIOS},
            "1" * 64,
            "2" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            control = HOOK._Control("TRANSPLANT", "test", "edited", "edited", "1" * 64, "2" * 64, selector, 10, Path(directory))
            HOOK._STATE.snapshot_experiment = "test"
            HOOK._STATE.snapshot_selector = selector.sha256
            with self.assertRaisesRegex(HOOK.ForcedReuseInvariantError, "SOURCE_SNAPSHOT_LAYER_PRODUCT_SET_INCOMPLETE"):
                HOOK._STATE.preflight(control)
            dummy = torch.empty(0)
            for product, layers in (("mla_kv", HOOK.MAIN_LAYERS), ("indexer_k", HOOK.INDEXER_LAYERS)):
                for layer in layers:
                    HOOK._STATE.snapshots[(product, layer)] = HOOK._LayerSnapshot(product, layer, tuple(range(115, 2071)), (), (dummy,), None)
            HOOK._STATE.preflight(control)
            self.assertEqual(HOOK._STATE.preflight_key, (selector.sha256, 10))

    def test_patch_places_mutation_after_native_cache_writes(self) -> None:
        text = PATCH_PATH.read_text(encoding="utf-8")
        self.assertIn("after the native row write and before attention", text)
        self.assertIn("after the native packed-row write and before top-k", text)
        self.assertEqual(text.count("maybe_apply_main_cache("), 1)
        self.assertEqual(text.count("maybe_apply_indexer_cache("), 1)
        self.assertIn("os.getenv(FORCED_REUSE_CONTROL_ENV)", text)


class SelectorTests(unittest.TestCase):
    def _write_capture(self, root: Path, edited: bool, *, complete: bool = False) -> None:
        root.mkdir()
        for rank in range(2):
            path = root / f"captures.rank-{rank}.jsonl"
            with path.open("w", encoding="utf-8") as stream:
                for layer in FULL_LAYERS if complete else (0,):
                    for sample in SAMPLE_POINTS if complete else ("prefill_last_query",):
                        raw = [float(index) + (0.01 if edited and index % 11 == 0 else 0.0) for index in range(PROMPT_TOKENS)]
                        selected = list(range(23, PROMPT_TOKENS))
                        record = {
                            "rank": rank,
                            "layer": layer,
                            "sample_point": sample,
                            "raw_scores": raw,
                            "selected_ids": selected,
                        }
                        encoded = (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode()
                        record["record_sha256"] = hashlib.sha256(encoded).hexdigest()
                        stream.write(json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n")

    def test_selector_rejects_incomplete_capture_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = root / "before"
            after = root / "after"
            self._write_capture(before, False)
            self._write_capture(after, True)
            old_ids = list(range(PROMPT_TOKENS))
            new_ids = old_ids.copy()
            old_ids[EDIT_POSITION] = OLD_TOKEN_ID
            new_ids[EDIT_POSITION] = NEW_TOKEN_ID
            old_path = root / "old.json"
            new_path = root / "new.json"
            old_path.write_text(json.dumps(old_ids), encoding="utf-8")
            new_path.write_text(json.dumps(new_ids), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "CAPTURE_COVERAGE_INCOMPLETE"):
                build_selector(before, after, root / "selector.json", baseline_prompt_path=old_path, edited_prompt_path=new_path)

    def test_selector_freezes_all_ratio_prefixes_before_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before = root / "before"
            after = root / "after"
            self._write_capture(before, False, complete=True)
            self._write_capture(after, True, complete=True)
            old_ids = list(range(PROMPT_TOKENS))
            new_ids = old_ids.copy()
            old_ids[EDIT_POSITION] = OLD_TOKEN_ID
            new_ids[EDIT_POSITION] = NEW_TOKEN_ID
            old_path = root / "old.json"
            new_path = root / "new.json"
            old_path.write_text(json.dumps(old_ids), encoding="utf-8")
            new_path.write_text(json.dumps(new_ids), encoding="utf-8")
            output = root / "selector.json"
            payload = build_selector(
                before,
                after,
                output,
                baseline_prompt_path=old_path,
                edited_prompt_path=new_path,
            )
            self.assertEqual(payload["status"], "attested_before_benchmark_outcomes")
            self.assertFalse(payload["selection_definition"]["uses_global_raw_score_threshold"])
            self.assertEqual(payload["positions_by_ratio"]["0"], [])
            self.assertEqual(len(payload["positions_by_ratio"]["100"]), 1956)
            self.assertNotIn(EDIT_POSITION, payload["ranking"])


class SweepDriverTests(unittest.TestCase):
    def test_control_is_default_off_and_binds_every_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            payload = _control(
                mode="OFF",
                experiment_id="test",
                phase_id="ratio-000",
                prompt_side="edited",
                ratio=0,
                selector_path=root / "selector.json",
                selector_sha256="a" * 64,
                donor_digest="b" * 64,
                edited_digest="c" * 64,
                evidence_dir=root / "evidence",
            )
            self.assertEqual(payload["mode"], "OFF")
            self.assertFalse(payload["production_default_enabled"])
            self.assertEqual(payload["real_block_size"], 64)
            self.assertEqual(payload["main_layers"], list(range(78)))
            self.assertEqual(payload["indexer_layers"], list(FULL_LAYERS))

    def test_runtime_verifier_requires_actual_all_layer_product_copies(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            count = 195
            for rank in range(4):
                path = root / f"runtime.rank-{rank}.jsonl"
                records = []
                for product, layers in (("mla_kv", SWEEP_MAIN_LAYERS), ("indexer_k", SWEEP_INDEXER_LAYERS)):
                    for layer in layers:
                        records.append({
                            "rank": rank,
                            "action": "destination_transplant",
                            "product": product,
                            "layer": layer,
                            "actual_row_count": count,
                            "consumed_by_native_attention": True,
                            "edited_token_excluded": True,
                        })
                path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            report = _verify_runtime(root, "destination_transplant", 10, 4)
            self.assertEqual(report["rank_count"], 4)
            self.assertEqual(report["by_rank"]["0"]["layer_product_count"], 99)
            self.assertEqual(report["by_rank"]["0"]["actual_transplanted_rows"], count * 99)
            first = root / "runtime.rank-0.jsonl"
            lines = first.read_text(encoding="utf-8").splitlines()
            first.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "RUNTIME_LAYER_PRODUCT_COVERAGE_INVALID"):
                _verify_runtime(root, "destination_transplant", 10, 4)


if __name__ == "__main__":
    unittest.main()
