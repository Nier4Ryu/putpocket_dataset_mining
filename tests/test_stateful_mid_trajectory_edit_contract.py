from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "configs/cluster/schemas/stateful_mid_trajectory_edit_scenario.schema.json"
EXAMPLE_PATH = ROOT / "configs/cluster/stateful_mid_trajectory_edit_scenario.example.yaml"
CONTRACT_PATH = ROOT / "docs/STATEFUL_MID_TRAJECTORY_EDIT_SCENARIO_CONTRACT.md"
GLM_REFERENCE_PATH = ROOT / "docs/CLUSTER_GLM52_STATEFUL_EDIT_V3.md"


def _canonical_array_sha256(value: list[int]) -> str:
    # Integer arrays serialize identically under RFC 8785 JCS and compact JSON.
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class StatefulMidTrajectoryEditContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.example = yaml.safe_load(EXAMPLE_PATH.read_text(encoding="utf-8"))

    def test_schema_is_draft_2020_12_and_encodes_claim_boundary(self) -> None:
        self.assertEqual(self.schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], 1)
        self.assertLess(SCHEMA_PATH.stat().st_size, 64 * 1024)

        selector = self.schema["$defs"]["selector"]["properties"]
        self.assertEqual(selector["ranked_quantity"]["const"], "importance_for_recompute")
        self.assertEqual(selector["score_direction"]["const"], "higher_score_first")

        accuracy_then = self.schema["$defs"]["runtime"]["allOf"][0]["then"]["properties"]
        self.assertFalse(accuracy_then["true_selective_prefill_claim"]["const"])
        self.assertFalse(accuracy_then["compute_saving_claim"]["const"])

        budget_required = set(self.schema["$defs"]["budget"]["required"])
        self.assertIn("selected_recompute_positions_in_selection_order_sha256", budget_required)
        self.assertIn("selected_recompute_positions_sorted_sha256", budget_required)
        self.assertIn("retained_old_kv_positions_sorted_sha256", budget_required)

    def test_authoring_example_satisfies_cross_field_set_and_boundary_invariants(self) -> None:
        manifest = self.example
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["status"], "authoring_example_not_executable")
        self.assertNotIn("glm", EXAMPLE_PATH.read_text(encoding="utf-8").lower())

        episode = manifest["episode"]
        self.assertTrue(episode["a1"]["generated_once"])
        self.assertTrue(episode["exact_replay_across_methods_budgets"])
        self.assertTrue(episode["selector_frozen_before_a2_or_outcomes"])
        self.assertEqual(episode["q2"]["kind"], "frozen_pre_edit_tool_observation")
        self.assertTrue(episode["q2"]["included_in_first_post_edit_request"])
        self.assertFalse(episode["q2"]["has_old_trajectory_kv"])

        tokenization = manifest["tokenization"]
        history_end = tokenization["frozen_history"]["history_token_count"]
        edit_positions = set(tokenization["edit"]["positions"])
        mandatory = set(tokenization["mandatory_recompute_positions"])
        eligible_start, eligible_end = tokenization["eligible_downstream_history_range"]
        eligible = set(range(eligible_start, eligible_end)) - mandatory
        q2_start, q2_end = tokenization["q2_token_range_in_first_post_edit_request"]

        self.assertTrue(edit_positions)
        self.assertTrue(edit_positions <= mandatory)
        self.assertEqual(len(tokenization["edit"]["old_token_ids"]), len(edit_positions))
        self.assertEqual(len(tokenization["edit"]["new_token_ids"]), len(edit_positions))
        self.assertEqual(eligible_end, history_end)
        self.assertFalse(mandatory & eligible)
        self.assertEqual(q2_start, history_end)
        self.assertEqual(q2_end, tokenization["first_genuinely_later_token_position"])

        selector = manifest["selector"]
        order = selector["selection_order_positions"]
        self.assertEqual(selector["ranked_quantity"], "importance_for_recompute")
        self.assertEqual(selector["score_direction"], "higher_score_first")
        self.assertTrue(selector["source_provenance"]["outcome_independent"])
        self.assertEqual(len(order), len(set(order)))
        self.assertEqual(set(order), eligible)
        self.assertEqual(selector["selection_order_sha256"], _canonical_array_sha256(order))

        q2_and_later = set(range(q2_start, q2_end + 1))
        self.assertFalse((mandatory | eligible | set(order)) & q2_and_later)
        for budget in manifest["budgets"]:
            count = budget["recompute_count"]
            selected_ordered = budget["selected_recompute_positions_in_selection_order"]
            selected_sorted = budget["selected_recompute_positions_sorted"]
            retained_sorted = budget["retained_old_kv_positions_sorted"]
            self.assertEqual(selected_ordered, order[:count])
            self.assertEqual(selected_sorted, sorted(selected_ordered))
            self.assertEqual(retained_sorted, sorted(eligible - set(selected_ordered)))
            self.assertEqual(
                budget["selected_recompute_positions_in_selection_order_sha256"],
                _canonical_array_sha256(selected_ordered),
            )
            self.assertEqual(
                budget["selected_recompute_positions_sorted_sha256"],
                _canonical_array_sha256(selected_sorted),
            )
            self.assertEqual(
                budget["retained_old_kv_positions_sorted_sha256"],
                _canonical_array_sha256(retained_sorted),
            )
            self.assertAlmostEqual(budget["recompute_ratio"], count / len(eligible))
            self.assertAlmostEqual(budget["reuse_ratio"], 1.0 - budget["recompute_ratio"])
            self.assertFalse((set(selected_ordered) | set(retained_sorted)) & q2_and_later)

        runtime = manifest["runtime"]
        self.assertEqual(runtime["baseline_cache"], "existing_old_trajectory_kv")
        self.assertEqual(
            runtime["selected_recompute_disposition"],
            "recompute_under_sys_new_and_patch_existing_cache",
        )
        self.assertEqual(runtime["unselected_disposition"], "retain_or_restore_old_kv")
        self.assertTrue(runtime["q2_and_later_tokens_run_normally"])

    def test_docs_separate_generic_recompute_ranking_from_glm_legacy_reuse_ranking(self) -> None:
        contract = CONTRACT_PATH.read_text(encoding="utf-8")
        normalized_contract = " ".join(contract.split())
        glm_reference = GLM_REFERENCE_PATH.read_text(encoding="utf-8")

        self.assertIn("selected for **recomputation**, never for donor reuse", normalized_contract)
        self.assertIn("accuracy_emulation_compute_then_overwrite", contract)
        self.assertIn("Q2 is the frozen pre-edit tool observation", contract)
        self.assertIn("no old-trajectory KV", contract)
        self.assertIn("STATEFUL_MID_TRAJECTORY_EDIT_SCENARIO_CONTRACT.md", glm_reference)
        self.assertIn("stability/reuse suitability", glm_reference)
        self.assertIn("MUST NOT be described as the most", glm_reference)
        self.assertIn("no-speedup reference backend", glm_reference)


if __name__ == "__main__":
    unittest.main()
