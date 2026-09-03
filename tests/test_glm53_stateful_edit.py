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

from putpocket_dataset_mining import glm53_stateful_edit as edit


ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = ROOT / "instrumentation/vllm/glm53_stateful_edit_accuracy_ablation.py"
PATCH_PATH = (
    ROOT
    / "patches/vllm/878631b6079d2cf9fb80830ef9cb41b43aded098"
    / "glm53_stateful_edit_v3_accuracy_ablation.patch"
)
LOCK_PATH = ROOT / "configs/experiments/glm53_stateful_edit_v3_accuracy_ablation.lock.json"
INVENTORY_PATH = (
    ROOT
    / "agent/tasks/T20260903-001__glm53-retarget-all/GLM52_ACTIVE_PATH_INVENTORY.json"
)
SPEC = importlib.util.spec_from_file_location("glm53_stateful_hook_test", HOOK_PATH)
assert SPEC is not None and SPEC.loader is not None
hook = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hook
SPEC.loader.exec_module(hook)


def digest(values: list[int]) -> str:
    return hashlib.sha256(
        json.dumps(values, separators=(",", ":")).encode()
    ).hexdigest()


class FakeTokenizer:
    """Prefix-consistent fake for episode construction, not a model stand-in."""

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool,
    ) -> list[int]:
        assert tokenize and not return_dict
        system = str(messages[0]["content"])
        changed = edit.EDIT_SENTENCE_NEW in system
        base = [101, 102, 11660 if changed else 17526, 103, 104, 105]
        if len(messages) >= 3 or add_generation_prompt:
            base += [401]
        if len(messages) >= 3:
            base += [201, 202, 203]
        if len(messages) >= 4:
            base += [301, 302]
        if len(messages) >= 4 and add_generation_prompt:
            base += [402]
        return base


def base_selector(donor: list[int], target: list[int]) -> dict[str, object]:
    edit_position = 2
    ranking = list(range(edit_position + 1, len(donor)))
    return {
        "schema_version": 1,
        "status": "attested_before_benchmark_outcomes",
        "model": {
            "id": edit.MODEL_ID,
            "revision": edit.MODEL_REVISION,
            "vllm_commit": edit.VLLM_COMMIT,
            "indexer_kind": "glm53_native_raw_pre_topk",
            "tokenizer_sha256": edit.TOKENIZER_SHA256,
        },
        "prompt": {
            "donor_token_ids_sha256": digest(donor),
            "edited_token_ids_sha256": digest(target),
            "edit_positions": [edit_position],
            "token_count": len(donor),
        },
        "selection_definition": {
            "algorithm": "lexicographic_prefill_only_layer_stability_v2_glm53_kpool",
            "query_sample": "prefill_last_query",
            "pool_score_expansion": "assign_each_native_compressed_pool_score_to_its_four_constituent_token_positions",
        },
        "source_evidence": {
            "outcome_independent": True,
            "benchmark_outcomes_read": False,
            "native_raw_pre_topk_scores": True,
            "layers": list(edit.INDEXER_LAYERS),
            "data_parallel_rank": 0,
        },
        "ranking": ranking,
        "token_metrics": {str(position): {} for position in ranking},
    }


def model_config() -> dict[str, object]:
    layer_types = [
        "deepseek_sparse_attention" if layer in edit.MLA_LAYERS else "linear_attention"
        for layer in range(45)
    ]
    return {
        "model_type": "glm5_next",
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": 45,
            "num_attention_heads": 64,
            "index_topk": 2048,
            "index_kpool": 4,
            "index_n_heads": 32,
            "index_head_dim": 128,
            "qk_rope_head_dim": 0,
            "mla_use_nope": True,
            "kv_lora_rank": 512,
            "layer_types": layer_types,
        },
    }


class StaticContractTests(unittest.TestCase):
    def test_exact_glm53_architecture_and_cache_contract(self) -> None:
        result = edit.validate_model_config(model_config())
        self.assertEqual(result["mla_layers"], list(range(3, 45, 4)))
        self.assertEqual(len(result["kda_layers"]), 34)
        self.assertEqual(result["main_cache_row_bytes"], 656)
        self.assertEqual(result["indexer_cache_row_bytes"], 132)

    def test_complete_pool_projection_never_copies_partial_boundary_pool(self) -> None:
        selected = list(range(115, 124))
        self.assertEqual(
            edit.complete_indexer_pool_ends(
                selected, eligible_start=115, eligible_end=124
            ),
            (119, 123),
        )
        self.assertEqual(
            edit.pool_covered_positions((119, 123)), tuple(range(116, 124))
        )
        self.assertNotIn(115, edit.pool_covered_positions((119, 123)))

    def test_episode_derives_glm53_edit_position_and_freezes_q2(self) -> None:
        tokenizer = FakeTokenizer()
        old_system = "prefix " + edit.EDIT_SENTENCE_OLD + " suffix"
        request = {
            "messages": [
                {"role": "system", "content": old_system},
                {"role": "user", "content": "Q1"},
            ]
        }
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "```bash\nprintf ok\n```",
                    }
                }
            ]
        }
        trajectory = {
            "messages": request["messages"]
            + [response["choices"][0]["message"], {"role": "user", "content": "ok"}]
        }
        donor = tokenizer.apply_chat_template(
            request["messages"], tokenize=True, add_generation_prompt=True, return_dict=False
        )
        target_messages = [dict(request["messages"][0]), request["messages"][1]]
        target_messages[0]["content"] = str(target_messages[0]["content"]).replace(
            edit.EDIT_SENTENCE_OLD, edit.EDIT_SENTENCE_NEW
        )
        target = tokenizer.apply_chat_template(
            target_messages, tokenize=True, add_generation_prompt=True, return_dict=False
        )
        payloads = edit.build_episode_payloads(
            tokenizer=tokenizer,
            capture={"http_status": 200, "request": request, "response": response},
            trajectory=trajectory,
            base_selector=base_selector(donor, target),
            source_capture_sha256="a" * 64,
            source_trajectory_sha256="b" * 64,
            base_selector_sha256="c" * 64,
        )
        episode = payloads["episode"]
        self.assertEqual(episode["system_edit"]["edit_positions"], [2])
        self.assertGreater(episode["first_post_edit_prompt_token_count"], episode["history_token_count"])
        self.assertEqual(episode["q2_starts_at_or_after"], episode["history_token_count"])
        self.assertFalse(payloads["selector"]["scenario"].get("true_partial_prefill", False))

    def test_patch_hooks_follow_native_writes_and_never_claim_partial(self) -> None:
        text = PATCH_PATH.read_text(encoding="utf-8")
        self.assertEqual(text.count("maybe_apply_main_cache("), 1)
        self.assertEqual(text.count("maybe_apply_indexer_cache("), 1)
        self.assertIn("native target row", text)
        self.assertLess(text.index("native target row"), text.index("maybe_apply_main_cache("))
        self.assertIn("native target pool write", text)
        self.assertLess(
            text.index("native target pool write"), text.index("maybe_apply_indexer_cache(")
        )
        self.assertNotIn("num_computed_tokens", text)
        self.assertNotIn("partial_prefill", text.lower())

    def test_cpu_harness_has_no_gpu_or_scheduler_operations(self) -> None:
        text = (ROOT / "scripts/glm53/run_stateful_edit_v3_harness.sh").read_text()
        for forbidden in ("nvidia-smi", "--gpus", "--device", "sbatch", "srun", "CUDA_VISIBLE_DEVICES"):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)
        self.assertIn("X-data-parallel-rank: 0", text)
        self.assertIn("ScaleAI/SWE-bench_Pro", json.dumps(json.loads((ROOT / "configs/experiments/glm53_stateful_edit_v3_single_instance.template.json").read_text())))

    def test_inventory_is_complete_unique_and_targets_exist(self) -> None:
        inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
        entries = inventory["entries"]
        self.assertEqual(len(entries), inventory["source"]["path_count"])
        self.assertEqual(len(entries), 97)
        self.assertEqual(len({entry["source_path"] for entry in entries}), len(entries))
        for entry in entries:
            self.assertRegex(entry["source_sha256"], r"^[0-9a-f]{64}$")
            if entry["disposition"] == "ported_glm53":
                for target in entry["target_paths"]:
                    self.assertTrue((ROOT / target).exists(), target)

    def test_single_instance_authoring_template_matches_schema(self) -> None:
        import jsonschema

        template = json.loads(
            (
                ROOT
                / "configs/experiments/glm53_stateful_edit_v3_single_instance.template.json"
            ).read_text()
        )
        schema = json.loads(
            (
                ROOT
                / "configs/experiments/schemas/glm53_stateful_edit_v3_episode.schema.json"
            ).read_text()
        )
        jsonschema.Draft202012Validator(schema).validate(template)
        self.assertIsNone(template["edit"]["serialized_edit_position"])
        self.assertTrue(template["episode"]["single_instance_only"])

    def test_overlay_installs_hook_at_the_imported_vllm_module_path(self) -> None:
        prepare = (ROOT / "scripts/glm53/prepare_stateful_edit_v3_overlay.sh").read_text()
        dockerfile = (ROOT / "docker/glm53_sm120/Dockerfile.stateful-edit-v3").read_text()
        module = "vllm/model_executor/layers/glm53_stateful_edit_accuracy_ablation.py"
        self.assertIn(module, prepare)
        self.assertIn("model_executor/layers/glm53_stateful_edit_accuracy_ablation.py", dockerfile)
        self.assertIn("vllm.model_executor.layers.glm53_stateful_edit_accuracy_ablation", PATCH_PATH.read_text())
        self.assertIn("putpocket.glm53.deployment_base_image_id", dockerfile)
        launch = (ROOT / "scripts/glm53/launch_stateful_edit_v3_server.sh").read_text()
        self.assertIn("STATEFUL_IMAGE_CPU_DOCTOR_PASSED", launch)
        self.assertIn("accuracy_ablation_armed() is False", launch)

    def test_experiment_lock_binds_all_packaging_inputs(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        self.assertFalse(lock["runtime_semantics"]["production_default_enabled"])
        self.assertTrue(lock["runtime_semantics"]["full_target_prefill"])
        self.assertFalse(lock["runtime_semantics"]["true_partial_prefill"])
        self.assertEqual(lock["model"]["mla_layers"], list(range(3, 45, 4)))
        self.assertEqual(len(lock["model"]["kda_layers"]), 34)
        self.assertEqual(lock["patch_chain"][-1]["apply_args"], ["--unidiff-zero"])
        for artifact in lock["artifacts"]:
            path = ROOT / artifact["path"]
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, artifact["size"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), artifact["sha256"])


class HookTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop(hook.CONTROL_ENV, None)
        hook._STATE.reset()

    def test_absent_control_is_inert_but_present_malformed_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.json"
            os.environ[hook.CONTROL_ENV] = str(path)
            self.assertIsNone(hook._current_control())
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(hook.StatefulEditInvariantError, "CONTROL_SCHEMA_INVALID"):
                hook._current_control()

    def test_main_and_indexer_rows_copy_by_exact_position_and_slot(self) -> None:
        main = torch.arange(2 * 8 * 656, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(2, 8, 656)
        positions = torch.tensor([115, 116, 117, 118, 119, 120])
        source_slots = torch.tensor([1, 3, 5, 7, 8, 10])
        snapshot = hook._snapshot_rows("mla_kv", 3, main, positions, source_slots, (116, 119))
        expected = snapshot.rows.clone()
        main.zero_()
        destination_slots = torch.tensor([14, 13, 12, 11, 10, 9])
        copied = hook._copy_rows(snapshot, main, positions, destination_slots, (116, 119))
        self.assertEqual(copied, (13, 10))
        self.assertTrue(torch.equal(main[1, 5], expected[0]))
        self.assertTrue(torch.equal(main[1, 2], expected[1]))

        indexer = torch.arange(2 * 8 * 132, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(2, 8, 132)
        pool_positions = torch.tensor([115, 116, 117, 118, 119, 120, 121, 122, 123])
        pool_slots = torch.tensor([-1, -1, -1, -1, 3, -1, -1, -1, 9])
        pool_snapshot = hook._snapshot_rows(
            "indexer_kpool", 3, indexer, pool_positions, pool_slots, (119, 123)
        )
        self.assertEqual(pool_snapshot.source_slots, (3, 9))

    def test_kda_and_tail_are_not_hook_products(self) -> None:
        source = HOOK_PATH.read_text(encoding="utf-8")
        self.assertNotIn('product == "kda"', source)
        self.assertNotIn('product == "indexer_tail"', source)
        self.assertIn("target_computed_no_donor_row_transplant", source)
        self.assertIn("target_computed_never_transplanted", source)

    def test_native_selector_capture_records_exact_last_query_pool_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = list(range(2049))
            control = hook._CaptureControl(
                "CAPTURE", "donor", digest(ids), len(ids), len(ids) - 1, root
            )
            hook._STATE.attest_capture(
                control, torch.tensor(ids), torch.arange(len(ids))
            )
            hook._STATE.attest_capture(
                control, torch.tensor([7]), torch.tensor([len(ids)])
            )
            logits = torch.arange(512, dtype=torch.float32).reshape(1, 512)
            hook._STATE.capture_indexer_logits(
                control,
                layer=3,
                logits=logits,
                query_positions=torch.tensor([2048]),
                key_starts=torch.tensor([0]),
                key_ends=torch.tensor([512]),
            )
            path = root / "base-selector.donor.layer-003.json"
            payload = json.loads(path.read_text())
            recorded = payload.pop("record_sha256")
            encoded = (
                json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
            ).encode()
            self.assertEqual(recorded, hashlib.sha256(encoded).hexdigest())
            self.assertEqual(payload["pool_ids"], list(range(512)))
            self.assertEqual(payload["raw_pool_scores"][-1], 511.0)


class SelectorFreezeTests(unittest.TestCase):
    @staticmethod
    def _write_records(
        root: Path,
        side: str,
        tokens: list[int],
        delta: float,
    ) -> None:
        root.mkdir()
        for layer in edit.INDEXER_LAYERS:
            payload = {
                "schema_version": 1,
                "capture_kind": "glm53_native_raw_pre_topk_kpool",
                "prompt_side": side,
                "model": {
                    "id": edit.MODEL_ID,
                    "revision": edit.MODEL_REVISION,
                    "architecture": edit.MODEL_ARCHITECTURE,
                    "vllm_commit": edit.VLLM_COMMIT,
                },
                "layer": layer,
                "query_position": len(tokens) - 1,
                "token_count": len(tokens),
                "prompt_token_ids_sha256": digest(tokens),
                "index_kpool": 4,
                "pool_ids": [0, 1],
                "raw_pool_scores": [1.0 + delta, 2.0 - delta],
                "raw_score_kind": "native_pre_topk_pre_normalization",
                "native_scale": "fp8_fp4_mqa_logits_output_before_top_k_per_row_prefill",
                "data_parallel_rank": 0,
                "tp_rank": 0,
                "head_aggregation": "native_lightning_indexer_weights_fused_by_vllm",
                "incomplete_tail_token_count": 1,
            }
            encoded = (
                json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
            ).encode()
            payload["record_sha256"] = hashlib.sha256(encoded).hexdigest()
            (root / f"base-selector.{side}.layer-{layer:03d}.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )

    def test_freeze_preserves_pool_scores_and_marks_unscored_tail(self) -> None:
        donor_tokens = list(range(9))
        edited_tokens = donor_tokens.copy()
        donor_tokens[2] = 17526
        edited_tokens[2] = 11660
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            donor = root / "donor"
            edited = root / "edited"
            self._write_records(donor, "donor", donor_tokens, 0.0)
            self._write_records(edited, "edited", edited_tokens, 0.1)
            selector = edit.freeze_base_selector_payload(
                donor_tokens=donor_tokens,
                edited_tokens=edited_tokens,
                donor_root=donor.resolve(),
                edited_root=edited.resolve(),
            )
            self.assertEqual(set(selector["ranking"]), set(range(3, 9)))
            self.assertEqual(selector["ranking"][-1], 8)
            self.assertEqual(
                selector["token_metrics"]["8"]["source"],
                "native_kpool_incomplete_tail_has_no_pool_logit",
            )
            validated, _ = edit.validate_base_selector(
                selector,
                donor_initial_tokens=donor_tokens,
                edited_initial_tokens=edited_tokens,
                edit_position=2,
            )
            self.assertEqual(validated, selector["ranking"])


if __name__ == "__main__":
    unittest.main()
