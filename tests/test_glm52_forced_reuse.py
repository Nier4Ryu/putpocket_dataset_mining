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
RUNNER_PATH = ROOT / "scripts/cluster/run_glm52_forced_reuse_ablation.sh"
SUBMIT_PATH = ROOT / "scripts/cluster/submit_glm52_forced_reuse_ablation.sh"
PACKAGE_PATH = ROOT / "scripts/cluster/package_glm52_forced_reuse_ablation.sh"
SPEC = importlib.util.spec_from_file_location("glm52_forced_edit_reuse_hook", HOOK_PATH)
assert SPEC is not None and SPEC.loader is not None
HOOK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = HOOK
SPEC.loader.exec_module(HOOK)


def _digest(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


HISTORY_TOKENS = HOOK.PROMPT_TOKENS + 17


def _history_ids(history_count: int = HISTORY_TOKENS) -> tuple[list[int], list[int]]:
    donor = list(range(history_count))
    donor[HOOK.EDIT_POSITION] = HOOK.OLD_TOKEN_ID
    edited = donor.copy()
    edited[HOOK.EDIT_POSITION] = HOOK.NEW_TOKEN_ID
    return donor, edited


def _selector_payload(baseline: str, edited: str, history_count: int = HISTORY_TOKENS) -> dict[str, object]:
    ranking = list(range(HOOK.DOWNSTREAM_START, history_count))
    return {
        "schema_version": 2,
        "status": "attested_before_benchmark_outcomes",
        "scenario": {
            "base_prompt_token_count": HOOK.PROMPT_TOKENS,
            "history_token_count": history_count,
            "edit_position": HOOK.EDIT_POSITION,
            "old_token_id": HOOK.OLD_TOKEN_ID,
            "new_token_id": HOOK.NEW_TOKEN_ID,
            "eligible_history_range": [HOOK.DOWNSTREAM_START, history_count],
            "same_length": True,
            "rope_positions_unchanged": True,
            "q2_and_post_edit_extensions_recomputed": True,
        },
        "source_evidence": {
            "donor_history_token_ids_sha256": baseline,
            "edited_history_token_ids_sha256": edited,
        },
        "ranking": ranking,
        "positions_by_ratio": {
            str(ratio): sorted(ranking[: len(ranking) * ratio // 100])
            for ratio in HOOK.RATIOS
        },
    }


def _selector(donor: list[int], edited: list[int]) -> object:
    history_count = len(donor)
    ranking = tuple(range(HOOK.DOWNSTREAM_START, history_count))
    denominator = len(ranking)
    return HOOK._Selector(
        "a" * 64,
        ranking,
        {ratio: ranking[: HOOK.ratio_count(ratio, denominator)] for ratio in HOOK.RATIOS},
        _digest(donor),
        _digest(edited),
        history_count,
        HOOK.DOWNSTREAM_START,
        history_count,
    )


def _hook_control(mode: str, phase: str, side: str, ratio: int, donor: list[int], edited: list[int], selector: object, evidence: Path) -> object:
    return HOOK._Control(
        mode,
        "test",
        phase,
        side,
        _digest(donor),
        _digest(edited),
        selector,
        ratio,
        evidence,
        len(donor),
        HOOK.DOWNSTREAM_START,
        len(donor),
    )


class HookContractTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop(HOOK.CONTROL_ENV, None)
        HOOK._STATE.reset()

    def test_default_path_is_inert(self) -> None:
        self.assertIsNone(HOOK._current_control())
        self.assertFalse(HOOK.forced_reuse_mutation_armed())
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
        denominator = HISTORY_TOKENS - HOOK.DOWNSTREAM_START
        self.assertEqual(
            [HOOK.ratio_count(ratio, denominator) for ratio in HOOK.RATIOS],
            [denominator * ratio // 100 for ratio in HOOK.RATIOS],
        )
        self.assertNotIn(HOOK.EDIT_POSITION, range(HOOK.DOWNSTREAM_START, HISTORY_TOKENS))

    def test_selector_requires_frozen_prefixes(self) -> None:
        baseline = "1" * 64
        edited = "2" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selector.json"
            path.write_text(json.dumps(_selector_payload(baseline, edited)) + "\n", encoding="utf-8")
            loaded = HOOK._load_selector(
                path.resolve(),
                HOOK._file_sha256(path),
                HISTORY_TOKENS,
                HOOK.DOWNSTREAM_START,
                HISTORY_TOKENS,
            )
            self.assertEqual(len(loaded.selected(0)), 0)
            self.assertEqual(len(loaded.selected(100)), HISTORY_TOKENS - 115)
            self.assertEqual(set(loaded.selected(100)), set(range(115, HISTORY_TOKENS)))

    def test_prompt_attestation_requires_exact_frozen_history_and_q2_extension(self) -> None:
        baseline_ids, edited_ids = _history_ids()
        selector = _selector(baseline_ids, edited_ids)
        with tempfile.TemporaryDirectory() as directory:
            control = _hook_control("SNAPSHOT", "donor", "donor", 100, baseline_ids, edited_ids, selector, Path(directory))
            HOOK._STATE.attest_prompt(
                control,
                torch.tensor(baseline_ids),
                torch.arange(len(baseline_ids)),
            )
            HOOK._STATE.attest_prompt(control, torch.tensor([9]), torch.tensor([len(baseline_ids)]))
            with self.assertRaisesRegex(HOOK.ForcedReuseInvariantError, "FROZEN_HISTORY_SINGLE_PREFILL_REQUIRED"):
                HOOK._STATE.attest_prompt(control, torch.tensor([9, 10]), torch.tensor([0, 1]))
            edited_control = _hook_control("TRANSPLANT", "ratio-010", "edited", 10, baseline_ids, edited_ids, selector, Path(directory))
            with self.assertRaisesRegex(HOOK.ForcedReuseInvariantError, "TARGET_MUST_INCLUDE_POST_EDIT_Q2"):
                HOOK._STATE.attest_prompt(edited_control, torch.tensor(edited_ids), torch.arange(len(edited_ids)))
            extension = edited_ids + [41, 42, 43]
            HOOK._STATE.attest_prompt(
                edited_control,
                torch.tensor(extension),
                torch.arange(len(extension)),
            )
            self.assertEqual(HOOK._STATE.current_prefill_ordinal, 0)
            self.assertEqual(HOOK._STATE.current_prefill_token_count, len(extension))
            broken = extension.copy()
            broken[200] += 1
            with self.assertRaisesRegex(HOOK.ForcedReuseInvariantError, "PROMPT_DIGEST_MISMATCH"):
                HOOK._STATE.attest_prompt(edited_control, torch.tensor(broken), torch.arange(len(broken)))

    def test_main_cache_snapshot_is_private_and_destination_slots_are_owned(self) -> None:
        source = torch.arange(33 * 64 * 576, dtype=torch.int64).reshape(33, 64, 576).to(torch.bfloat16)
        source_slots = torch.arange(HISTORY_TOKENS)
        positions = (115, 116, HISTORY_TOKENS - 1)
        snapshot = HOOK._snapshot_main(source, 0, positions, source_slots)
        expected = snapshot.parts[0].clone()
        source.view(-1, 576)[list(snapshot.source_slots)].zero_()
        self.assertTrue(torch.equal(snapshot.parts[0], expected))
        destination = torch.zeros_like(source)
        destination_slots = torch.arange(HISTORY_TOKENS - 1, -1, -1)
        copied = HOOK._copy_main(snapshot, destination, positions, destination_slots)
        for index, slot in enumerate(copied):
            self.assertTrue(torch.equal(destination.view(-1, 576)[slot], expected[index]))
        self.assertFalse(any(left == right for left, right in zip(snapshot.source_slots, copied, strict=True)))

    def test_packed_indexer_values_and_scale_planes_are_both_copied(self) -> None:
        source = torch.arange(33 * 64 * 132, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(33, 64, 132)
        positions = (115, 116, HISTORY_TOKENS - 1)
        source_slots = torch.arange(HISTORY_TOKENS)
        snapshot = HOOK._snapshot_indexer(source, 0, positions, source_slots)
        destination = torch.zeros_like(source)
        destination_slots = torch.arange(HISTORY_TOKENS - 1, -1, -1)
        copied = HOOK._copy_indexer(snapshot, destination, positions, destination_slots)
        for index, slot in enumerate(copied):
            page, offset = divmod(slot, HOOK.BLOCK_SIZE)
            flat = destination[page].view(-1)
            self.assertTrue(torch.equal(flat[offset * 128 : (offset + 1) * 128], snapshot.parts[0][index]))
            scale = HOOK.BLOCK_SIZE * 128 + offset * 4
            self.assertTrue(torch.equal(flat[scale : scale + 4], snapshot.parts[1][index]))

    def test_page_accounting_uses_real_destination_slots(self) -> None:
        slots = torch.arange(HISTORY_TOKENS)
        result = HOOK._page_accounting(tuple(range(128, 192)), slots, HOOK.DOWNSTREAM_START, HISTORY_TOKENS)
        self.assertEqual(result["downstream_physical_page_count"], 32)
        self.assertEqual(result["touched_physical_page_count"], 1)
        self.assertEqual(result["complete_reused_physical_page_count"], 1)
        self.assertEqual(result["mixed_physical_page_count"], 0)
        mixed = HOOK._page_accounting((128, 130), slots, HOOK.DOWNSTREAM_START, HISTORY_TOKENS)
        self.assertEqual(mixed["mixed_physical_page_count"], 1)
        self.assertEqual(mixed["complete_reused_physical_page_count"], 0)

    def test_preflight_requires_all_78_main_and_21_indexer_products(self) -> None:
        donor, edited = _history_ids()
        selector = _selector(donor, edited)
        with tempfile.TemporaryDirectory() as directory:
            control = _hook_control("TRANSPLANT", "edited", "edited", 10, donor, edited, selector, Path(directory))
            HOOK._STATE.snapshot_experiment = "test"
            HOOK._STATE.snapshot_selector = selector.sha256
            with self.assertRaisesRegex(HOOK.ForcedReuseInvariantError, "SOURCE_SNAPSHOT_LAYER_PRODUCT_SET_INCOMPLETE"):
                HOOK._STATE.preflight(control)
            dummy = torch.empty(0)
            for product, layers in (("mla_kv", HOOK.MAIN_LAYERS), ("indexer_k", HOOK.INDEXER_LAYERS)):
                for layer in layers:
                    HOOK._STATE.snapshots[(product, layer)] = HOOK._LayerSnapshot(product, layer, tuple(range(115, HISTORY_TOKENS)), (), (dummy,), None)
            HOOK._STATE.preflight(control)
            self.assertEqual(HOOK._STATE.preflight_key, (selector.sha256, 10))

    def test_patch_places_mutation_after_native_cache_writes(self) -> None:
        text = PATCH_PATH.read_text(encoding="utf-8")
        self.assertIn("after the native row write and before attention", text)
        self.assertLess(
            text.index("if forced_reuse_mutation_armed():"),
            text.index('raise RuntimeError("GLM52_FORCED_REUSE_METADATA_UNAVAILABLE")'),
        )
        self.assertIn("after the native packed-row write and before top-k", text)
        self.assertEqual(text.count("maybe_apply_main_cache("), 1)
        self.assertEqual(text.count("maybe_apply_indexer_cache("), 1)
        self.assertIn("os.getenv(FORCED_REUSE_CONTROL_ENV)", text)

    def test_cluster_launcher_uses_immutable_docker_bundle_and_dependent_smoke(self) -> None:
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        submit = SUBMIT_PATH.read_text(encoding="utf-8")
        self.assertIn('"$CONTAINER" load --input "$BUNDLE/runtime-image.tar"', runner)
        self.assertIn("--entrypoint vllm", runner)
        self.assertIn("--attention-backend FLASHMLA_SPARSE", runner)
        self.assertIn("--linear-backend marlin", runner)
        self.assertIn("--cpu-offload-gb 0 --swap-space 0", runner)
        self.assertIn("MODEL_STAGE_SPACE_BELOW_550G", runner)
        self.assertNotIn("MODEL_STAGE_SPACE_BELOW_220G", runner)
        self.assertNotIn("PUTPOCKET_RUNTIME_PYTHON", runner)
        self.assertIn("PUTPOCKET_SWEEP_PROFILE=smoke", submit)
        self.assertIn('afterok:$SMOKE_JOB_ID', submit)
        self.assertIn("PUTPOCKET_SWEEP_PROFILE=full", submit)
        self.assertNotIn("--array", submit)

    def test_gitless_vllm_archive_is_bound_by_commit_marker(self) -> None:
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        packager = PACKAGE_PATH.read_text(encoding="utf-8")
        exact = "4a3447d200e5aa428d68d1a00aa00f1a19a1a729"
        self.assertIn('printf \'%s\\n\' ' + exact + ' > "$VLLM_STAGE/vllm/VLLM_COMMIT"', packager)
        self.assertIn('[[ -f $VLLM_SOURCE/VLLM_COMMIT', runner)
        self.assertIn('tr -d \'\\r\\n\' < "$VLLM_SOURCE/VLLM_COMMIT"', runner)
        self.assertIn(f"== {exact}", runner)
        self.assertIn("VLLM_ARCHIVE_COMMIT_MISMATCH", runner)
        self.assertNotIn('git -C "$VLLM_SOURCE" rev-parse HEAD', runner)
        self.assertIn('sha256sum -c artifacts/vllm-source.sha256', runner)


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
