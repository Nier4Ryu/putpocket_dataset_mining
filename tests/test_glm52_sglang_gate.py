from __future__ import annotations

import json
import os
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.glm52_sglang_gate import (
    SITE_PROFILE,
    SOURCE_LOCK,
    classify_startup_failure,
    load_json,
    parse_hbm_csv,
    parse_inventory_csv,
    summarize_hbm,
    validate_capability_report,
    validate_checkpoint_layout,
    validate_checkpoint_marker,
    validate_inventory_rows,
    validate_model_config,
    validate_public_project,
    validate_runtime_log,
    validate_sentinel_response,
    validate_server_info,
    validate_source_lock,
)
from putpocket_dataset_mining.glm52_sglang_gate_cli import main
from putpocket_dataset_mining.glm52_sglang_gate_slurm import GateSite, load_gate_site, render_compact_gate_submission


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "scripts" / "cluster" / "run_glm52_sglang_feasibility_gate.sh"
PROJECT_URL = "https://github.com/Nier4Ryu/putpocket_dataset_mining.git"
COMMIT = "a" * 40


def inventory_rows() -> list[dict[str, object]]:
    return [
        {
            "uuid": f"GPU-00000000-0000-0000-0000-{index:012d}",
            "name": "NVIDIA H200",
            "memory_total_mib": 143771,
            "memory_free_mib": 143000 - index,
            "mig_mode": "Disabled",
            "compute_capability": "9.0",
        }
        for index in range(4)
    ]


def model_config() -> dict[str, object]:
    return {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
        "indexer_types": ["full"] * 21 + ["shared"] * 57,
        "index_topk": 2048,
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16},
    }


def capability_report() -> dict[str, object]:
    return {
        "transformers_version": "5.12.1",
        "torch_cuda_version": "13.1",
        "torch_nccl_version": [2, 28, 9],
        "imports": {name: True for name in ("torch", "transformers", "sglang", "modelopt", "flashinfer", "flash_mla", "sgl_kernel")},
        "symbols": {
            name: True
            for name in (
                "ModelOptFp4Config",
                "ModelOptFp4LinearMethod",
                "ModelOptNvFp4FusedMoEMethod",
                "prepare_nvfp4_layer_for_marlin",
                "prepare_moe_nvfp4_layer_for_marlin",
                "marlin_w4a16",
                "hopper_marlin_selection",
                "glm_moe_dsa_runtime",
                "flashmla_sparse",
                "fa3",
                "sgl-kernel",
            )
        },
        "server_defaults": {
            "cpu_offload_gb": 0,
            "disaggregation_mode": None,
            "speculative_algorithm": None,
            "weight_cache_mode": "off",
        },
        "server_controls": {
            "quantization": ["modelopt_fp4"],
            "fp4_gemm_backend": ["marlin"],
            "moe_runner_backend": ["marlin"],
            "dsa_prefill_backend": ["flashmla_sparse"],
            "dsa_decode_backend": ["fa3"],
            "dsa_topk_backend": ["sgl-kernel"],
        },
    }


def server_info() -> dict[str, object]:
    return {
        "server_args": {
            "tp_size": 4,
            "quantization": "modelopt_fp4",
            "fp4_gemm_backend": "marlin",
            "moe_runner_backend": "marlin",
            "dsa_prefill_backend": "flashmla_sparse",
            "dsa_decode_backend": "fa3",
            "dsa_topk_backend": "sgl-kernel",
            "context_length": 4096,
            "max_running_requests": 1,
            "cpu_offload_gb": 0,
            "weight_cache_mode": "off",
            "disaggregation_mode": None,
            "speculative_algorithm": None,
        }
    }


RUNTIME_LOG = "GlmMoeDsaForCausalLM glm_moe_dsa modelopt_fp4 Marlin flashmla_sparse fa3 sgl-kernel ready"


class SourceAndSiteContractTests(unittest.TestCase):
    def test_tracked_source_lock_is_immutable_and_secret_free(self) -> None:
        result = validate_source_lock(load_json(SOURCE_LOCK))
        self.assertEqual(result["status"], "passed")
        self.assertRegex(result["sglang_source_commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(result["runtime_image"], r"@sha256:[0-9a-f]{64}$")
        text = SOURCE_LOCK.read_text(encoding="utf-8").lower()
        self.assertNotIn("hf_token", text)
        self.assertNotIn("api_key", text)

    def test_lock_rejects_mutable_runtime_identity_and_old_transformers(self) -> None:
        lock = load_json(SOURCE_LOCK)
        lock["runtime_image"]["linux_amd64_digest"] = "latest"
        with self.assertRaisesRegex(ConfigError, "immutable"):
            validate_source_lock(lock)
        lock = load_json(SOURCE_LOCK)
        lock["dependency_contract"]["transformers_minimum"] = "5.2"
        with self.assertRaisesRegex(ConfigError, "at least 5.3"):
            validate_source_lock(lock)

    def test_site_profile_is_exact_authoritative_request(self) -> None:
        site = load_gate_site(SITE_PROFILE)
        self.assertEqual((site.partition, site.account, site.qos), ("H200", "gsai-account", "hpgpu"))
        self.assertEqual((site.nodes, site.ntasks, site.cpus_per_task), (1, 1, 32))
        self.assertEqual(site.gpu_directive, "--gres=gpu:H200:4")
        self.assertEqual((site.memory, site.wall_time), ("512G", "06:00:00"))

    def test_site_rejects_any_resource_relaxation(self) -> None:
        raw = load_json(SITE_PROFILE)
        for key, value in (("nodes", 2), ("gpu_directive", "--gres=gpu:H200:8"), ("memory", "256G")):
            changed = json.loads(json.dumps(raw))
            changed["slurm"][key] = value
            with self.subTest(key=key), self.assertRaises(ConfigError):
                GateSite.from_mapping(changed)


class AllocationGateTests(unittest.TestCase):
    def test_full_physical_four_h200_inventory_passes(self) -> None:
        result = validate_inventory_rows(inventory_rows(), mig_listing="GPU 0: NVIDIA H200 (UUID: GPU-one)")
        self.assertEqual(result.as_dict()["gpu_count"], 4)
        self.assertEqual(result.as_dict()["mig"], "disabled")

    def test_inventory_csv_has_explicit_schema(self) -> None:
        text = "0, GPU-one, NVIDIA H200, 143771, 143000, Disabled, 9.0\n"
        self.assertEqual(parse_inventory_csv(text)[0]["uuid"], "GPU-one")
        with self.assertRaisesRegex(ConfigError, "seven CSV"):
            parse_inventory_csv("too,few\n")

    def test_each_allocation_mismatch_fails_closed(self) -> None:
        cases = []
        cases.append((inventory_rows()[:3], "GPU_COUNT"))
        wrong_name = inventory_rows(); wrong_name[0]["name"] = "NVIDIA B200"; cases.append((wrong_name, "GPU_TYPE"))
        wrong_mem = inventory_rows(); wrong_mem[0]["memory_total_mib"] = 120000; cases.append((wrong_mem, "MEMORY_CLASS"))
        mig = inventory_rows(); mig[0]["mig_mode"] = "Enabled"; cases.append((mig, "MIG_ENABLED"))
        arch = inventory_rows(); arch[0]["compute_capability"] = "10.0"; cases.append((arch, "GPU_ARCH"))
        duplicate = inventory_rows(); duplicate[1]["uuid"] = duplicate[0]["uuid"]; cases.append((duplicate, "UUID"))
        for rows, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ConfigError, error):
                validate_inventory_rows(rows)
        with self.assertRaisesRegex(ConfigError, "MIG_ENABLED"):
            validate_inventory_rows(inventory_rows(), mig_listing="MIG 1g.18gb Device 0")

    def test_allocation_only_cli_refuses_montblanc(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            csv = Path(temp) / "inventory.csv"; csv.write_text("", encoding="utf-8")
            listing = Path(temp) / "listing.txt"; listing.write_text("", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(main(["validate-inventory", "--csv", str(csv), "--listing", str(listing), "--output", str(Path(temp) / "out.json")]), 2)
                self.assertEqual(main(["phase1", "--artifact-root", temp, "--metadata-root", temp]), 2)
                self.assertEqual(main(["download-model", "--revision-file", str(csv), "--model-root", temp]), 2)
                self.assertEqual(
                    main(
                        [
                            "validate-runtime",
                            "--inventory", str(csv),
                            "--model-config", str(csv),
                            "--server-info", str(csv),
                            "--server-log", str(csv),
                            "--response", str(csv),
                            "--hbm-samples", str(csv),
                            "--model-revision", str(csv),
                            "--project-commit", COMMIT,
                            "--source-lock-report", str(csv),
                            "--capability-report", str(csv),
                            "--exact-command", str(csv),
                            "--output", str(Path(temp) / "out.json"),
                        ]
                    ),
                    2,
                )


class WeightlessContractTests(unittest.TestCase):
    def test_exact_official_model_config_passes(self) -> None:
        result = validate_model_config(model_config())
        self.assertEqual(result["indexer_layout"], {"full": 21, "shared": 57})

    def test_every_model_shape_or_quantization_change_fails(self) -> None:
        changes = (
            ("architectures", ["DenseModel"]),
            ("model_type", "glm_moe"),
            ("num_hidden_layers", 77),
            ("index_topk", 1024),
        )
        for key, value in changes:
            config = model_config(); config[key] = value
            with self.subTest(key=key), self.assertRaises(ConfigError):
                validate_model_config(config)
        config = model_config(); config["indexer_types"] = ["full"] * 20 + ["shared"] * 58
        with self.assertRaisesRegex(ConfigError, "INDEXER_LAYOUT"):
            validate_model_config(config)
        for key, value in (("quant_method", "awq"), ("quant_algo", "FP8"), ("group_size", 32)):
            config = model_config(); config["quantization_config"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ConfigError, "QUANTIZATION"):
                validate_model_config(config)

    def test_capability_report_requires_imports_symbols_controls_and_safe_defaults(self) -> None:
        self.assertEqual(validate_capability_report(capability_report())["status"], "passed")
        report = capability_report(); report["imports"]["flash_mla"] = False
        with self.assertRaisesRegex(ConfigError, "IMPORT_MISSING"):
            validate_capability_report(report)
        report = capability_report(); report["symbols"]["sgl-kernel"] = False
        with self.assertRaisesRegex(ConfigError, "CAPABILITY_MISSING"):
            validate_capability_report(report)
        report = capability_report(); report["server_controls"]["fp4_gemm_backend"] = []
        with self.assertRaisesRegex(ConfigError, "CONTROL_MISSING"):
            validate_capability_report(report)
        report = capability_report(); report["server_defaults"]["cpu_offload_gb"] = 10
        with self.assertRaisesRegex(ConfigError, "UNSAFE_RUNTIME_DEFAULT"):
            validate_capability_report(report)

    def test_checkpoint_layout_checks_indexed_shards_without_hashing_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "config.json").write_text(json.dumps(model_config()), encoding="utf-8")
            (root / "model-00001-of-00002.safetensors").write_bytes(b"one")
            (root / "model-00002-of-00002.safetensors").write_bytes(b"two")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.0": "model-00001-of-00002.safetensors", "layer.1": "model-00002-of-00002.safetensors"}}),
                encoding="utf-8",
            )
            result = validate_checkpoint_layout(root)
            self.assertEqual(result["hash_policy"], "no_full_tensor_hash")
            self.assertEqual(result["shard_count"], 2)
            revision = "b" * 40
            marker = {"schema_version": 1, "status": "ready", "model_id": "nvidia/GLM-5.2-NVFP4", "revision": revision, "layout": result}
            (root / ".putpocket_checkpoint_ready.json").write_text(json.dumps(marker), encoding="utf-8")
            self.assertEqual(validate_checkpoint_marker(root, revision)["shard_count"], 2)
            marker["revision"] = "c" * 40
            (root / ".putpocket_checkpoint_ready.json").write_text(json.dumps(marker), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "MARKER_MISMATCH"):
                validate_checkpoint_marker(root, revision)
            (root / "model-00002-of-00002.safetensors").unlink()
            with self.assertRaisesRegex(ConfigError, "INCOMPLETE"):
                validate_checkpoint_layout(root)


class RuntimeAndSentinelTests(unittest.TestCase):
    def test_final_runtime_cli_emits_pass_manifest_and_diagnostic_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            uuids = [str(row["uuid"]) for row in inventory_rows()]
            files = {
                "inventory.json": {"status": "passed", "gpu_count": 4, "gpu_uuids": uuids},
                "config.json": model_config(),
                "server.json": server_info(),
                "response.json": {"choices": [{"message": {"content": "Sentinel completed successfully."}}]},
                "source.json": {"status": "passed", "sglang_source_commit": "d" * 40, "runtime_image": "lmsysorg/sglang@sha256:" + "e" * 64, "runtime_image_human_tag": "lmsysorg/sglang:latest"},
                "capability.json": {"status": "passed", "transformers_version": "5.12.1", "torch_cuda_version": "13.1", "torch_nccl_version": [2, 28, 9]},
            }
            for name, value in files.items():
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            (root / "server.log").write_text(RUNTIME_LOG, encoding="utf-8")
            (root / "revision.txt").write_text("f" * 40, encoding="utf-8")
            (root / "command.txt").write_text("python3 -m sglang.launch_server --tp 4\n", encoding="utf-8")
            hbm_lines = ["timestamp,uuid,memory_total_mib,memory_used_mib,memory_free_mib"]
            hbm_lines.extend(f"t,{uuid},143771,120000,23771" for uuid in uuids)
            (root / "hbm.csv").write_text("\n".join(hbm_lines) + "\n", encoding="utf-8")
            output = root / "gate_manifest.json"
            environment = {"SLURM_JOB_ID": "123", "SLURM_JOB_NUM_NODES": "1", "SLURM_JOB_NODELIST": "n87", "SLURM_GPUS_ON_NODE": "4"}
            with patch.dict(os.environ, environment, clear=True):
                result = main(
                    [
                        "validate-runtime", "--inventory", str(root / "inventory.json"), "--model-config", str(root / "config.json"),
                        "--server-info", str(root / "server.json"), "--server-log", str(root / "server.log"), "--response", str(root / "response.json"),
                        "--hbm-samples", str(root / "hbm.csv"), "--model-revision", str(root / "revision.txt"), "--project-commit", COMMIT,
                        "--source-lock-report", str(root / "source.json"), "--capability-report", str(root / "capability.json"),
                        "--exact-command", str(root / "command.txt"), "--output", str(output),
                    ]
                )
            self.assertEqual(result, 0)
            manifest = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "PASS")
            self.assertEqual(manifest["slurm"]["nodelist"], "n87")
            self.assertTrue((root / "runtime_contract.log").is_file())
            self.assertTrue((root / "sentinel.raw.sha256").is_file())

    def test_server_info_proves_exact_runtime_and_rejects_ambiguity(self) -> None:
        self.assertEqual(validate_server_info(server_info())["status"], "passed")
        info = server_info(); info["server_args"]["tp_size"] = 2
        with self.assertRaisesRegex(ConfigError, "AMBIGUOUS"):
            validate_server_info(info)
        info = server_info(); info["server_args"]["disaggregation_mode"] = "decode"
        with self.assertRaisesRegex(ConfigError, "UNSAFE_RUNTIME_MODE"):
            validate_server_info(info)
        info = server_info(); info["metrics"] = {"loss": float("nan")}
        with self.assertRaisesRegex(ConfigError, "NONFINITE"):
            validate_server_info(info)

    def test_runtime_log_requires_all_backends_and_forbids_fallback_offload_dense_nonfinite(self) -> None:
        self.assertEqual(validate_runtime_log(RUNTIME_LOG)["status"], "passed")
        for suffix, failure in (
            ("", "AMBIGUOUS"),
            (" falling back", "SILENT_FALLBACK"),
            (" CPU offload enabled", "OFFLOAD_DETECTED"),
            (" dense attention", "DENSE_ATTENTION"),
            (" metric=NaN", "NONFINITE"),
        ):
            text = "modelopt_fp4 marlin flashmla_sparse fa3 sgl-kernel" if not suffix else RUNTIME_LOG + suffix
            with self.subTest(failure=failure), self.assertRaisesRegex(ConfigError, failure):
                validate_runtime_log(text)

    def test_sentinel_saves_deterministic_normalized_hash_and_rejects_bad_outputs(self) -> None:
        result = validate_sentinel_response({"choices": [{"message": {"content": "  Sentinel   completed. "}}]})
        self.assertEqual(result["normalized_output"], "Sentinel completed.")
        self.assertRegex(result["normalized_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(result["raw_sha256"], r"^[0-9a-f]{64}$")
        for value, failure in (("   ", "EMPTY"), ("value NaN", "NONFINITE"), ("go " * 20, "REPETITION")):
            with self.subTest(failure=failure), self.assertRaisesRegex(ConfigError, failure):
                validate_sentinel_response({"choices": [{"message": {"content": value}}]})
        with self.assertRaisesRegex(ConfigError, "NONFINITE"):
            validate_sentinel_response({"usage": {"metric": float("inf")}, "choices": [{"message": {"content": "okay"}}]})

    def test_hbm_summary_requires_all_four_devices_and_positive_headroom(self) -> None:
        uuids = [row["uuid"] for row in inventory_rows()]
        samples = []
        for uuid in uuids:
            samples.extend((
                {"uuid": uuid, "memory_total_mib": 143771, "memory_used_mib": 1000, "memory_free_mib": 142771},
                {"uuid": uuid, "memory_total_mib": 143771, "memory_used_mib": 120000, "memory_free_mib": 23771},
            ))
        result = summarize_hbm(samples, uuids)
        self.assertEqual(result["minimum_headroom_mib"], 23771)
        with self.assertRaisesRegex(ConfigError, "INCOMPLETE"):
            summarize_hbm(samples[:-2], uuids)
        bad = [dict(sample) for sample in samples]
        for sample in bad:
            if sample["uuid"] == uuids[0]:
                sample["memory_used_mib"] = 143771; sample["memory_free_mib"] = 0
        with self.assertRaisesRegex(ConfigError, "HEADROOM"):
            summarize_hbm(bad, uuids)

    def test_hbm_csv_and_startup_failure_classification(self) -> None:
        parsed = parse_hbm_csv("timestamp,uuid,memory_total_mib,memory_used_mib,memory_free_mib\nt,GPU-a,143771,100,143671\n")
        self.assertEqual(parsed[0]["memory_used_mib"], "100")
        self.assertEqual(classify_startup_failure("ModelOpt Marlin repack CUDA out of memory"), "MARLIN_REPACK_OOM")
        self.assertEqual(classify_startup_failure("flashmla_sparse unavailable"), "REQUIRED_BACKEND_STARTUP_FAILED")
        self.assertEqual(classify_startup_failure("unknown crash"), "MODEL_LOAD_FAILED")


class RendererAndEntrypointTests(unittest.TestCase):
    def test_compact_renderer_is_exact_dedicated_and_bash_valid(self) -> None:
        command = render_compact_gate_submission(site=load_gate_site(), project_url=PROJECT_URL, project_commit=COMMIT)
        for token in (
            "--partition=H200", "--account=gsai-account", "--qos=hpgpu", "--nodes=1", "--ntasks=1",
            "--gres=gpu:H200:4", "--cpus-per-task=32", "--mem=512G", "--time=06:00:00", "--export=NONE",
        ):
            self.assertIn(token, command)
        self.assertIn("run_glm52_sglang_feasibility_gate.sh", command)
        self.assertNotRegex(command.lower(), r"swe-?bench|swepro|selection full")
        completed = subprocess.run(["bash", "-n"], input=command, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        words = shlex.split(command.split("&&", 1)[1].strip())
        wrapper = next(word.removeprefix("--wrap=") for word in words if word.startswith("--wrap="))
        completed = subprocess.run(["bash", "-n"], input=wrapper, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('RUN_ROOT=/local-data/jslee202403/putpocket-glm52-sglang-gate/artifacts/"${SLURM_JOB_ID:-unknown}"', wrapper)
        self.assertLess(wrapper.index("SLURM_GPUS_ON_NODE"), wrapper.index("fetch --depth=1"))

    def test_project_url_rejects_credentials_queries_and_non_github(self) -> None:
        validate_public_project(PROJECT_URL, COMMIT)
        for url in ("https://token@github.com/x/y.git", "https://github.com/x/y.git?token=x", "https://example.com/x/y.git"):
            with self.subTest(url=url), self.assertRaises(ConfigError):
                validate_public_project(url, COMMIT)

    def test_entrypoint_is_executable_syntax_valid_and_phase_ordered(self) -> None:
        self.assertTrue(os.access(ENTRYPOINT, os.X_OK))
        completed = subprocess.run(["bash", "-n", str(ENTRYPOINT)], text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        text = ENTRYPOINT.read_text(encoding="utf-8")
        self.assertNotRegex(text.lower(), r"swe-?bench|swepro|selection=full")
        self.assertLess(text.index("validate-inventory"), text.index("docker pull"))
        self.assertLess(text.index("phase1"), text.index("download-model"))
        self.assertLess(text.index("docker info"), text.index("docker pull"))
        self.assertIn("module load cuda/12.9", text)
        self.assertIn("--tp 4", text)
        self.assertIn("--quantization modelopt_fp4", text)
        self.assertIn("--fp4-gemm-backend marlin", text)
        self.assertIn("--dsa-prefill-backend flashmla_sparse", text)
        self.assertIn("--dsa-decode-backend fa3", text)
        self.assertIn("--dsa-topk-backend sgl-kernel", text)
        self.assertIn("--context-length 4096", text)
        self.assertIn("--max-running-requests 1", text)
        self.assertNotRegex(text, r"--[^\n ]*trace")
        self.assertNotIn("snapshot_download", text)


if __name__ == "__main__":
    unittest.main()
