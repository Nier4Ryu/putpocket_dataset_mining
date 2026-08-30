from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.glm53_deployment import (
    GLM53DeploymentError,
    host_doctor,
    load_model_lock,
    selected_model_paths,
    validate_model_lock,
    verify_model_directory,
    verify_package_files,
)


class GLM53DeploymentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = load_model_lock()

    def test_committed_lock_is_self_consistent_and_bounded(self) -> None:
        validate_model_lock(self.lock)
        paths = selected_model_paths(self.lock)
        self.assertEqual(len([p for p in paths if p.startswith("model-")]), 10)
        self.assertIn("model_mtp.safetensors", paths)
        self.assertFalse(self.lock["runtime"]["enable_mtp"])
        self.assertEqual(
            self.lock["capacity_plan"]["parallelism"],
            {
                "tensor_parallel_size": 1,
                "data_parallel_size": 3,
                "expert_parallel": True,
                "expert_parallel_size": 3,
                "enable_ep_weight_filter": True,
                "pipeline_parallel_size": 1,
            },
        )
        self.assertEqual(
            self.lock["model_config_contract"]["effective_sparse_topk"], 2176
        )
        self.assertEqual(verify_package_files(self.lock)["status"], "ok")

    def test_lock_rejects_missing_mtp_and_unsafe_or_duplicate_paths(self) -> None:
        for mutation in ("missing_mtp", "unsafe", "duplicate"):
            lock = copy.deepcopy(self.lock)
            if mutation == "missing_mtp":
                lock["files"] = [
                    item for item in lock["files"] if item["path"] != "model_mtp.safetensors"
                ]
                lock["selected_files_total_bytes"] = sum(
                    item["size"] for item in lock["files"]
                )
            elif mutation == "unsafe":
                lock["files"][0]["path"] = "../README.md"
            else:
                lock["files"][1]["path"] = lock["files"][0]["path"]
            with self.subTest(mutation=mutation):
                with self.assertRaises(GLM53DeploymentError):
                    validate_model_lock(lock)

    def test_lock_rejects_tp3_or_nondivisible_expert_plan(self) -> None:
        lock = copy.deepcopy(self.lock)
        lock["model_config_contract"]["num_attention_heads"] = 63
        with self.assertRaisesRegex(GLM53DeploymentError, "TP=3"):
            validate_model_lock(lock)
        lock = copy.deepcopy(self.lock)
        lock["model_config_contract"]["n_routed_experts"] = 287
        with self.assertRaisesRegex(GLM53DeploymentError, "EP=3"):
            validate_model_lock(lock)

    def test_sizes_only_model_verification_checks_config_and_exact_index(self) -> None:
        lock = self._small_lock()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for item in lock["files"]:
                (root / item["path"]).write_bytes(b"x" * item["size"])
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Glm5NextForConditionalGeneration"],
                        "model_type": "glm5_next",
                        "text_config": {
                            "model_type": "glm5_next_text",
                            "num_hidden_layers": 45,
                            "num_attention_heads": 64,
                            "n_routed_experts": 288,
                            "num_experts_per_tok": 8,
                            "index_topk": 2048,
                            "index_kpool": 4,
                            "qk_rope_head_dim": 0,
                            "kv_lora_rank": 512,
                        },
                        "quantization_config": {
                            "quant_method": "compressed-tensors",
                            "format": "mixed-precision",
                        },
                    }
                ),
                encoding="utf-8",
            )
            lock_by_path = {item["path"]: item for item in lock["files"]}
            lock_by_path["config.json"]["size"] = (root / "config.json").stat().st_size
            shard_paths = [
                item["path"]
                for item in lock["files"]
                if item["path"].endswith(".safetensors")
            ]
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {f"weight.{i}": p for i, p in enumerate(shard_paths)}}),
                encoding="utf-8",
            )
            lock_by_path["model.safetensors.index.json"]["size"] = (
                root / "model.safetensors.index.json"
            ).stat().st_size
            lock["selected_files_total_bytes"] = sum(item["size"] for item in lock["files"])
            report = verify_model_directory(root, lock, verify_hashes=False)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["weight_index"]["referenced_shards"], sorted(shard_paths))

    def test_model_verification_rejects_index_shard_set_drift(self) -> None:
        lock = self._small_lock()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for item in lock["files"]:
                (root / item["path"]).write_bytes(b"x" * item["size"])
            (root / "config.json").write_text("{}", encoding="utf-8")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"bad": "unknown.safetensors"}}),
                encoding="utf-8",
            )
            report = verify_model_directory(root, lock, verify_hashes=False)
        self.assertEqual(report["status"], "failed")
        self.assertIn("index_shard_set_mismatch", report["failures"])

    def test_host_doctor_fails_closed_on_live_gpu_process(self) -> None:
        gpu = {
            "index": 0,
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "memory_total_mib": 97887,
            "memory_total_bytes": 97887 * 1024 * 1024,
            "compute_capability": "12.0",
        }
        disk = type("Disk", (), {"total": 2 * 1024**4, "free": 512 * 1024**3})()
        with (
            patch(
                "putpocket_dataset_mining.glm53_deployment._nvidia_smi_rows",
                return_value=[gpu, {**gpu, "index": 1}, {**gpu, "index": 2}],
            ),
            patch(
                "putpocket_dataset_mining.glm53_deployment._nvidia_compute_processes",
                return_value=[{"pid": 123, "process_name": "other", "used_memory_mib": 1}],
            ),
            patch("putpocket_dataset_mining.glm53_deployment.shutil.disk_usage", return_value=disk),
            patch(
                "putpocket_dataset_mining.glm53_deployment._command_output",
                return_value="580.159.03",
            ),
        ):
            report = host_doctor(self.lock)
        self.assertEqual(report["status"], "failed")
        self.assertIn("gpu_compute_processes_present", report["failures"])

    def _small_lock(self) -> dict:
        lock = copy.deepcopy(self.lock)
        files = []
        for i in range(1, 11):
            files.append(
                {
                    "path": f"model-{i:05d}-of-00010.safetensors",
                    "size": i,
                    "sha256": "0" * 64,
                }
            )
        files.extend(
            [
                {"path": "model_mtp.safetensors", "size": 11, "sha256": "0" * 64},
                {"path": "config.json", "size": 1, "sha256": "0" * 64},
                {
                    "path": "model.safetensors.index.json",
                    "size": 1,
                    "sha256": "0" * 64,
                },
            ]
        )
        lock["files"] = files
        lock["selected_files_total_bytes"] = sum(item["size"] for item in files)
        return lock


if __name__ == "__main__":
    unittest.main()
