from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.glm52_dsa_diagnostic import (
    DECODE_SAMPLES,
    FULL_LAYERS,
    HARNESS_COMMIT,
    INSTANCE_ID,
    LOCK_PATH,
    MODEL_REVISION,
    SGLANG_COMMIT,
    build_artifact_manifest,
    canonical_row_sha256,
    compress_capture_records,
    expected_sample_points,
    full_shared_mapping,
    load_lock,
    validate_capture_coverage,
    validate_capture_record,
    validate_diagnostic_server_isolation,
    validate_lock,
    validate_patch_inputs,
    validate_selected_row,
    validate_serialized_prompt,
    validate_trace_equivalence,
)
from putpocket_dataset_mining.glm52_dsa_diagnostic_cli import main
from putpocket_dataset_mining.glm52_dsa_diagnostic_slurm import render_compact_diagnostic_submission
from putpocket_dataset_mining.glm52_sglang_gate_slurm import load_gate_site


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "configs/cluster/sites/herdr_h200_sglang_gate.json"
PROJECT_URL = "https://github.com/Nier4Ryu/putpocket_dataset_mining.git"
PROJECT_COMMIT = "8" * 40
ALLOCATION = {
    "SLURM_JOB_ID": "123",
    "SLURM_JOB_NUM_NODES": "1",
    "SLURM_JOB_NODELIST": "n87",
    "SLURM_GPUS_ON_NODE": "4",
}


def response(ids: list[int], text: str = "THOUGHT: inspect\n\n```bash\ngit status --short\n```") -> dict:
    return {"choices": [{"index": 0, "text": text, "token_ids": ids, "finish_reason": "stop"}]}


def capture_record(layer: int = 0, rank: int = 0, phase: str = "prefill_last_query", step: int | None = None) -> dict:
    context = 2071 if step is None else 2072 + int(step)
    raw = [float(context - index) / 1000.0 for index in range(context)]
    ids = [*range(2047), context - 1]
    mapping = full_shared_mapping()
    return {
        "schema_version": 1,
        "record_type": "native_glm52_dsa_indexer_scores",
        "run_id": "test-run",
        "instance_id": INSTANCE_ID,
        "trace_mode": "ON",
        "phase": phase,
        "layer": layer,
        "full_indexer_layer": layer,
        "shared_layers": [candidate for candidate, source in mapping.items() if source == layer and candidate != layer],
        "query_position": context - 1,
        "decode_step": step,
        "context_length": context,
        "rank": rank,
        "backend_identities": {
            "quantization": "modelopt_fp4",
            "fp4_gemm": "marlin_w4a16",
            "dsa_prefill": "flashmla_sparse",
            "dsa_decode": "fa3",
            "dsa_topk": "sgl-kernel",
        },
        "dtype": "torch.float32",
        "device": f"cuda:{rank}",
        "native_logits_shape": [1, context],
        "topk": 2048,
        "source_token_coordinate_semantics": "zero_based_logical_causal_position_within_exact_request",
        "native_transform_kind": "ragged" if phase == "prefill_last_query" else "paged",
        "native_selected_logical_token_ids": ids,
        "native_selected_scores": [raw[index] for index in ids],
        "native_pre_topk_raw_score_vector": raw,
        "native_forced_token_mask": {
            "num_init_tokens": 1,
            "num_local_tokens": 1,
            "applied_by_sglang_after_raw_capture": True,
        },
        "revisions": {
            "model": MODEL_REVISION,
            "sglang": SGLANG_COMMIT,
            "image": "sha256:3be8803490a8b899a44f7ab2e22d8f6a1fb877cab52faeb400769a1555317db4",
            "project": "b" * 40,
        },
    }


class DiagnosticLockAndSelectionTests(unittest.TestCase):
    def test_exact_authoritative_pins_and_selection(self) -> None:
        lock = load_lock()
        report = validate_lock(lock)
        self.assertEqual(report["full_layers"], list(FULL_LAYERS))
        self.assertEqual(lock["selection"]["instance_id"], INSTANCE_ID)
        self.assertEqual(lock["selection"]["row_sha256"], "78ff3ac298f276dfafaa311c26b7ace35be7c52d9094b4a6f658de2e7b5e25d1")
        self.assertEqual(lock["selection"]["serialized_prompt_token_count"], 2071)
        self.assertEqual(lock["selection"]["serialized_prompt_sha256"], "25d6597314bbb6c4df5afa886b064bebbdb7b57d27414d1e584e6a0127eeeab5")
        self.assertEqual(lock["swebench_pro"]["harness_commit"], HARNESS_COMMIT)
        self.assertFalse(lock["swebench_pro"]["score_eligible"])
        self.assertFalse(lock["swebench_pro"]["full_selection_reachable"])

    def test_lock_rejects_every_runtime_relaxation(self) -> None:
        for key, value in (
            ("tensor_parallel", 2),
            ("dsa_topk_backend", "torch"),
            ("offload", True),
            ("disable_radix_cache", False),
            ("cuda_graph_decode", "full"),
        ):
            lock = load_lock()
            lock["runtime"][key] = value
            with self.subTest(key=key), self.assertRaises(ConfigError):
                validate_lock(lock)

    def test_full_shared_mapping_is_exact_21_57(self) -> None:
        mapping = full_shared_mapping()
        self.assertEqual([layer for layer, source in mapping.items() if layer == source], list(FULL_LAYERS))
        self.assertEqual(sum(layer != source for layer, source in mapping.items()), 57)
        self.assertEqual({3: mapping[3], 5: mapping[5], 7: mapping[7], 77: mapping[77]}, {3: 2, 5: 2, 7: 6, 77: 74})

    def test_selected_row_digest_and_content_fail_closed(self) -> None:
        row = {"instance_id": INSTANCE_ID, "repo": "ansible/ansible", "dockerhub_tag": "pinned", "problem_statement": "task"}
        lock = load_lock()
        lock["selection"].update(
            {
                "dockerhub_tag": "pinned",
                "row_sha256": canonical_row_sha256(row),
                "problem_statement_sha256": hashlib.sha256(b"task").hexdigest(),
            }
        )
        self.assertEqual(validate_selected_row(row, lock)["status"], "passed")
        changed = dict(row); changed["problem_statement"] = "changed"
        with self.assertRaisesRegex(ConfigError, "ROW_DIGEST|PROBLEM_DIGEST"):
            validate_selected_row(changed, lock)

    def test_serialized_prompt_count_digest_and_tokenizer_identity(self) -> None:
        serialized = "exact prompt"
        token_ids = [1, 2, 3]
        lock = load_lock()
        lock["selection"].update(
            {
                "serialized_prompt_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
                "serialized_prompt_utf8_bytes": len(serialized),
                "serialized_prompt_token_count": len(token_ids),
            }
        )
        files = {
            "tokenizer.json": lock["selection"]["tokenizer"]["tokenizer_json_sha256"],
            "tokenizer_config.json": lock["selection"]["tokenizer"]["tokenizer_config_sha256"],
            "chat_template.jinja": lock["selection"]["tokenizer"]["chat_template_sha256"],
        }
        self.assertEqual(validate_serialized_prompt(serialized, token_ids, lock, tokenizer_file_digests=files, tokenizer_class="TokenizersBackend")["status"], "passed")
        with self.assertRaisesRegex(ConfigError, "TOKEN_COUNT"):
            validate_serialized_prompt(serialized, token_ids + [4], lock, tokenizer_file_digests=files, tokenizer_class="TokenizersBackend")

    def test_patch_and_instrumentation_digests_and_context_are_tracked(self) -> None:
        lock = load_lock()
        for key in ("patch_path", "instrumentation_source"):
            path = ROOT / lock["sglang"][key]
            digest_key = "patch_sha256" if key == "patch_path" else "instrumentation_sha256"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), lock["sglang"][digest_key])
        self.assertEqual(
            lock["sglang"]["patch_target_post_sha256"],
            "7985bffeb4f8e7b712e75b452b062fbcf02fd2299386dcf1b1f0a0864d28e050",
        )
        patch_text = (ROOT / lock["sglang"]["patch_path"]).read_text(encoding="utf-8")
        self.assertIn("prepare_native_dsa_capture", patch_text)
        self.assertIn("finish_native_dsa_capture", patch_text)
        self.assertTrue(all("topk_transform" in point for point in lock["sglang"]["native_points"]))
        self.assertNotIn("torch.topk", patch_text)
        instrumentation = (ROOT / lock["sglang"]["instrumentation_source"]).read_text(encoding="utf-8")
        self.assertNotIn("torch.topk", instrumentation)

    def test_patch_context_digest_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ConfigError, "PATCH_CONTEXT_DIGEST_MISMATCH"):
                validate_patch_inputs(ROOT, temp, load_lock())


class TraceAndCaptureTests(unittest.TestCase):
    def test_trace_requires_exact_output_token_id_equivalence(self) -> None:
        report = validate_trace_equivalence(response([4, 5, 6]), response([4, 5, 6]))
        self.assertEqual(report["output_token_count"], 3)
        with self.assertRaisesRegex(ConfigError, "TOKEN_ID_MISMATCH"):
            validate_trace_equivalence(response([4, 5, 6]), response([4, 5, 7]))
        with self.assertRaisesRegex(ConfigError, "TOKEN_IDS_MISSING"):
            validate_trace_equivalence(response([]), response([]))

    def test_expected_decode_samples_include_only_existing_steps(self) -> None:
        self.assertEqual(expected_sample_points(1), (("prefill_last_query", None),))
        self.assertEqual(expected_sample_points(3), (("prefill_last_query", None), ("decode", 0), ("decode", 1)))
        self.assertEqual(expected_sample_points(34)[-1], ("decode", 32))

    def test_capture_record_validates_vector_bounds_alignment_mapping_and_finiteness(self) -> None:
        lock = load_lock()
        self.assertEqual(validate_capture_record(capture_record(), lock)["context_length"], 2071)
        bad = capture_record(); bad["native_pre_topk_raw_score_vector"][0] = math.nan
        with self.assertRaisesRegex(ConfigError, "RAW_SCORE"):
            validate_capture_record(bad, lock)
        bad = capture_record(); bad["native_selected_scores"][0] += 1.0
        with self.assertRaisesRegex(ConfigError, "ALIGNMENT"):
            validate_capture_record(bad, lock)
        bad = capture_record()
        bad["native_selected_logical_token_ids"][-2] = 2048
        bad["native_selected_scores"][-2] = bad["native_pre_topk_raw_score_vector"][2048]
        with self.assertRaisesRegex(ConfigError, "TOPK_CONSISTENCY"):
            validate_capture_record(bad, lock)
        bad = capture_record(); bad["native_selected_logical_token_ids"][-1] = 2048
        bad["native_selected_scores"][-1] = bad["native_pre_topk_raw_score_vector"][2048]
        with self.assertRaisesRegex(ConfigError, "FORCED_TOKEN"):
            validate_capture_record(bad, lock)
        bad = capture_record(); bad["shared_layers"] = [77]
        with self.assertRaisesRegex(ConfigError, "MAPPING"):
            validate_capture_record(bad, lock)
        bad = capture_record(); bad["revisions"]["project"] = "floating-main"
        with self.assertRaisesRegex(ConfigError, "REVISION"):
            validate_capture_record(bad, lock)
        bad = capture_record(); bad["prompt"] = "forbidden"
        with self.assertRaisesRegex(ConfigError, "RAW_PROMPT|SCHEMA"):
            validate_capture_record(bad, lock)

    def test_coverage_requires_all_21_layers_and_four_ranks(self) -> None:
        records = [capture_record(layer, rank) for layer in FULL_LAYERS for rank in range(4)]
        report = validate_capture_coverage(records, load_lock(), output_token_count=1)
        self.assertEqual(report["record_count"], 84)
        self.assertEqual(len(report["shared_layer_to_full_layer"]), 57)
        with self.assertRaisesRegex(ConfigError, "INCOMPLETE"):
            validate_capture_coverage(records[:-1], load_lock(), output_token_count=1)
        records[-1]["run_id"] = "different-run"
        with self.assertRaisesRegex(ConfigError, "RUN_ID_COVERAGE"):
            validate_capture_coverage(records, load_lock(), output_token_count=1)

    def test_capture_schema_required_fields_match_runtime_record(self) -> None:
        schema = json.loads((ROOT / "configs/cluster/schemas/glm52_dsa_capture.schema.json").read_text())
        self.assertEqual(set(schema["required"]), set(capture_record()))
        self.assertFalse(schema["additionalProperties"])

    def test_compression_is_deterministic_and_hashes_compressed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "record.json"
            raw.write_text(json.dumps(capture_record(), sort_keys=True), encoding="utf-8")
            records1, files1, algorithm1 = compress_capture_records([raw], root / "one", prefer_zstd=False)
            records2, files2, algorithm2 = compress_capture_records([raw], root / "two", prefer_zstd=False)
            self.assertEqual(files1[0].read_bytes(), files2[0].read_bytes())
            self.assertEqual(algorithm1, algorithm2)
            manifest = build_artifact_manifest(files1, compression=algorithm1, coverage={"status": "passed"}, run_id="x")
            self.assertEqual(manifest["files"][0]["sha256"], hashlib.sha256(files1[0].read_bytes()).hexdigest())
            self.assertFalse(manifest["raw_prompt_included"])
            self.assertEqual(records1, records2)

    def test_missing_native_raw_vector_is_classified_blocked_by_entrypoint(self) -> None:
        script = (ROOT / "scripts/cluster/run_glm52_dsa_diagnostic.sh").read_text(encoding="utf-8")
        self.assertIn("NATIVE_DSA_EXPOSURE_BLOCKED", script)
        self.assertIn('JOB_STATUS=BLOCKED', script)
        self.assertIn('compgen -G "$TRACE_RAW/BLOCKED-*.json"', script)
        instrumentation = (ROOT / "instrumentation/sglang/dsa_diagnostic_dump.py").read_text(encoding="utf-8")
        self.assertIn("NATIVE_SELECTED_COORDINATE_EXPOSURE_UNPROVEN", instrumentation)
        self.assertIn("native_runtime_changed\": False", instrumentation)

    def test_trace_isolation_requires_runtime_proof(self) -> None:
        info = {
            "server_args": {
                "disable_radix_cache": True,
                "cuda_graph_backend_prefill": "disabled",
                "cuda_graph_backend_decode": "disabled",
            }
        }
        self.assertEqual(validate_diagnostic_server_isolation(info)["status"], "passed")
        info["server_args"]["disable_radix_cache"] = False
        with self.assertRaisesRegex(ConfigError, "TRACE_ISOLATION_RUNTIME_AMBIGUOUS"):
            validate_diagnostic_server_isolation(info)


class RendererAndAllocationTests(unittest.TestCase):
    def test_compact_renderer_is_exact_four_h200_diagnostic_only(self) -> None:
        command = render_compact_diagnostic_submission(
            site=load_gate_site(SITE), project_url=PROJECT_URL, project_commit=PROJECT_COMMIT
        )
        for fragment in (
            "--partition=H200",
            "--account=gsai-account",
            "--qos=hpgpu",
            "--nodes=1",
            "--ntasks=1",
            "--gres=gpu:H200:4",
            "--cpus-per-task=32",
            "--mem=512G",
            "--time=06:00:00",
            "--export=NONE",
            "run_glm52_dsa_diagnostic.sh",
        ):
            self.assertIn(fragment, command)
        lowered = command.lower()
        for forbidden in ("swe_bench_pro_eval", "swebench_pro_full", "--selection full", "run_swebench"):
            self.assertNotIn(forbidden, lowered)
        self.assertLess(command.index("docker info"), command.index("fetch --depth=1"))
        completed = subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        wrap_args = [value for value in shlex.split(command) if value.startswith("--wrap=")]
        self.assertEqual(len(wrap_args), 1)
        completed = subprocess.run(["bash", "-n"], input=wrap_args[0].split("=", 1)[1], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_entrypoint_preserves_gate_order_exact_backends_and_official_single_row_eval(self) -> None:
        script = (ROOT / "scripts/cluster/run_glm52_dsa_diagnostic.sh").read_text(encoding="utf-8")
        self.assertLess(script.index("validate-inventory"), script.index("docker pull \"$IMAGE\""))
        self.assertLess(script.index("validate-inventory"), script.index("download-model"))
        self.assertIn("--tp 4", script)
        self.assertIn("--quantization modelopt_fp4", script)
        self.assertIn("--fp4-gemm-backend marlin", script)
        self.assertIn("--dsa-prefill-backend flashmla_sparse", script)
        self.assertIn("--dsa-decode-backend fa3", script)
        self.assertIn("--dsa-topk-backend sgl-kernel", script)
        self.assertIn("--disable-radix-cache", script)
        self.assertIn("--cuda-graph-backend-prefill disabled", script)
        self.assertIn("--cuda-graph-backend-decode disabled", script)
        self.assertIn("--env SLURM_JOB_NAME", script)
        self.assertIn("SGLANG_PATCHED_TARGET_CACHE_MISMATCH", script)
        self.assertIn("apply --unidiff-zero --check", script)
        self.assertIn("server_identity_before.txt", script)
        self.assertIn("LIVE_SERVER_PROCESS_CHANGED", script)
        self.assertIn('"$HARNESS_ROOT/swe_bench_pro_eval.py"', script)
        self.assertIn("--num_workers 1", script)
        self.assertNotIn("--selection full", script)
        completed = subprocess.run(["bash", "-n", str(ROOT / "scripts/cluster/run_glm52_dsa_diagnostic.sh")], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_allocation_only_commands_refuse_montblanc(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            self.assertEqual(main(["control", "--mode", "OFF", "--run-id", "x", "--output", str(Path(temp) / "control.json")]), 2)
            self.assertEqual(main(["trace-equivalence", "--off-response", str(Path(temp) / "off.json"), "--on-response", str(Path(temp) / "on.json"), "--off-duration-ns", "1", "--on-duration-ns", "1", "--output", str(Path(temp) / "out.json")]), 2)

    def test_tracked_runtime_script_is_executable_and_contains_no_secret_or_login_action(self) -> None:
        path = ROOT / "scripts/cluster/run_glm52_dsa_diagnostic.sh"
        self.assertTrue(os.access(path, os.X_OK))
        text = path.read_text(encoding="utf-8").lower()
        self.assertNotIn("ssh ", text)
        self.assertNotIn("sbatch ", text)
        self.assertNotIn("hf_token", text)
        self.assertNotIn("api_key", text)


if __name__ == "__main__":
    unittest.main()
