from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import jsonschema

from putpocket_dataset_mining.glm53_runpod_image import (
    MODEL_ID,
    MODEL_REVISION,
    PROFILES,
    RunPodContractError,
    static_doctor,
    validate_lock,
    validate_model_metadata,
)


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "9cd956c7e6cf54efa366b803cafa15ec6c2df827"
LOCK_PATH = ROOT / "configs/models/glm53_flash_w4a16_runpod_sm90_sm120.lock.json"
PACKAGE_PATHS = (
    ROOT / "docker/glm53_sm90_sm120",
    ROOT / "scripts/glm53_runpod",
    ROOT / "src/putpocket_dataset_mining/glm53_runpod_image.py",
    ROOT / "src/putpocket_dataset_mining/glm53_runpod_stateful_edit.py",
    ROOT / "src/putpocket_dataset_mining/glm53_runpod_stateful_proxy.py",
    ROOT / "instrumentation/vllm/glm53_runpod_stateful_edit_accuracy_ablation.py",
    LOCK_PATH,
    ROOT / "configs/execution/glm53_runpod_sm90_sm120_w4a16.example.yaml",
    ROOT / "configs/experiments/schemas/glm53_runpod_stateful_edit_control.schema.json",
    ROOT / f"patches/vllm/{COMMIT}/glm53_sm90_sm120_build.patch",
    ROOT / f"patches/vllm/{COMMIT}/glm53_nope_fp8_ds_mla_cache.patch",
    ROOT / f"patches/vllm/{COMMIT}/glm53_sm120_nope_topk_lens.patch",
    ROOT / f"patches/vllm/{COMMIT}/glm53_stateful_edit_v3_accuracy_ablation.patch",
    ROOT / "docs/GLM53_RUNPOD_SM90_SM120_IMAGE.md",
)


def load_script(name: str):
    path = ROOT / "scripts/glm53_runpod" / name
    spec = importlib.util.spec_from_file_location(f"glm53_runpod_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GLM53RunPodImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))

    def test_under_200gb_w4a16_model_and_dual_profiles(self) -> None:
        validate_lock(self.lock)
        selected = self.lock["selection"]
        self.assertEqual(selected["repository"], MODEL_ID)
        self.assertEqual(selected["revision"], MODEL_REVISION)
        self.assertLess(selected["payload_bytes"], 200_000_000_000)
        self.assertEqual(selected["quant_method"], "auto-round")
        self.assertEqual(selected["runtime_quantization"], "inc")
        self.assertEqual(self.lock["runtime"]["profiles"], PROFILES)
        self.assertEqual(self.lock["runtime"]["default_gpu_count"], 4)
        self.assertFalse(self.lock["runtime"]["weights_in_image"])
        self.assertFalse(self.lock["runtime"]["silent_download_allowed"])

    def test_hybrid_and_accuracy_ablation_boundaries(self) -> None:
        hybrid = self.lock["hybrid_cache_contract"]
        self.assertEqual(hybrid["layer_count"], 45)
        self.assertEqual(hybrid["mla_indexer_layers"], list(range(3, 45, 4)))
        self.assertEqual(hybrid["kda_layer_count"], 34)
        self.assertEqual(hybrid["index_kpool"], 4)
        stateful = self.lock["stateful_edit"]
        self.assertFalse(stateful["default_enabled"])
        self.assertTrue(stateful["full_target_prefill"])
        self.assertTrue(stateful["donor_overwrite_after_native_write"])
        self.assertFalse(stateful["true_partial_prefill"])
        self.assertFalse(stateful["speedup_claim"])

    def test_static_doctor_has_no_torch_or_device_probe_and_passes(self) -> None:
        source = (ROOT / "src/putpocket_dataset_mining/glm53_runpod_image.py").read_text()
        node = next(
            item
            for item in ast.walk(ast.parse(source))
            if isinstance(item, ast.FunctionDef) and item.name == "static_doctor"
        )
        rendered = ast.unparse(node)
        self.assertNotIn("import torch", rendered)
        self.assertNotIn("device_count", rendered)
        self.assertNotIn("get_device_capability", rendered)
        report = static_doctor(LOCK_PATH, ROOT)
        self.assertTrue(report["package_integrity_ok"])
        self.assertFalse(report["runtime_gpu_validated"])

    def test_model_metadata_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RunPodContractError, "MODEL_PATH_NOT_DIRECTORY"):
                validate_model_metadata(Path(directory) / "missing", self.lock)
            report = validate_model_metadata(directory, self.lock)
            self.assertFalse(report["ok"])

    def test_control_schema_binds_profile_pairs_and_topology(self) -> None:
        schema = json.loads((ROOT / "configs/experiments/schemas/glm53_runpod_stateful_edit_control.schema.json").read_text())
        base = {
            "schema_version": 3,
            "mode": "OFF",
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "architecture": "Glm5NextForConditionalGeneration",
                "vllm_commit": COMMIT,
                "attention_backend": "FLASHINFER_MLA_SPARSE_SM90",
                "kv_cache_dtype": "fp8_e4m3",
                "index_kpool": 4,
            },
            "runtime": {
                "tensor_parallel_size": 4,
                "data_parallel_size": 1,
                "expert_parallel_size": 4,
                "data_parallel_rank_affinity": 0,
                "prefix_caching": False,
                "chunked_prefill": False,
            },
            "production_default_enabled": False,
            "compute_semantics": "full_target_prefill_then_selected_donor_cache_overwrite",
            "true_partial_prefill": False,
            "speedup_claim": False,
        }
        jsonschema.Draft7Validator(schema).validate(base)
        base["model"]["kv_cache_dtype"] = "fp8_ds_mla"
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft7Validator(schema).validate(base)

    def test_proxy_import_is_packaged_with_generic_stateful_module(self) -> None:
        dockerfile = (ROOT / "docker/glm53_sm90_sm120/Dockerfile").read_text()
        build_script = (ROOT / "scripts/glm53_runpod/build_image.sh").read_text()
        proxy = (ROOT / "src/putpocket_dataset_mining/glm53_runpod_stateful_proxy.py").read_text()
        self.assertIn("from .glm53_runpod_stateful_edit import", proxy)
        self.assertIn("glm53_runpod_stateful_edit.py", dockerfile)
        self.assertIn("glm53_runpod_stateful_proxy.py", dockerfile)
        self.assertIn('putpocket.source.commit="${PUTPOCKET_SOURCE_COMMIT}"', dockerfile)
        self.assertIn('--build-arg "PUTPOCKET_SOURCE_COMMIT=${PUTPOCKET_SOURCE_COMMIT}"', build_script)

    def test_new_package_has_no_a6000_sm86_or_old_model_binding(self) -> None:
        forbidden = ("A6000", "SM86", "RedHatAI/GLM-5.3", "878631b6079d")
        for entry in PACKAGE_PATHS:
            files = sorted(entry.rglob("*")) if entry.is_dir() else [entry]
            for path in files:
                if not path.is_file() or path.suffix == ".pyc":
                    continue
                text = path.read_text(encoding="utf-8")
                for term in forbidden:
                    with self.subTest(path=path, term=term):
                        self.assertNotIn(term, text)

    def test_build_is_cpu_only_and_profiles_are_fail_closed(self) -> None:
        scripts = "\n".join(
            path.read_text()
            for path in (ROOT / "scripts/glm53_runpod").glob("*")
            if path.is_file() and path.suffix != ".pyc"
        )
        self.assertNotIn("nvidia-smi", scripts)
        self.assertNotIn("--gpus", scripts)
        self.assertIn("torch_cuda_arch_list=9.0a 12.0a", scripts)
        launch = (ROOT / "scripts/glm53_runpod/launch_server.sh").read_text()
        self.assertIn("FLASHINFER_MLA_SPARSE_SM90", launch)
        self.assertIn("FLASHINFER_MLA_SPARSE_SM120", launch)
        self.assertIn("--quantization inc", launch)
        self.assertIn("--no-enable-chunked-prefill", launch)
        self.assertLess(launch.index("runtime_doctor") if "runtime_doctor" in launch else launch.index("glm53_runpod_image runtime"), launch.index("exec vllm serve"))

    def test_build_patch_omits_irrelevant_extensions_but_keeps_required(self) -> None:
        patch = (ROOT / f"patches/vllm/{COMMIT}/glm53_sm90_sm120_build.patch").read_text()
        self.assertIn("include(cmake/external_projects/deepgemm.cmake)", patch)
        self.assertIn("include(cmake/external_projects/flashkda.cmake)", patch)
        self.assertIn("bundled FA2/FA3", patch)
        self.assertIn("DeepEP intentionally omitted", patch)
        self.assertIn("putpocket_audit_dual_arch_wheel.py", patch)
        self.assertIn("printf '%s\\n' /opt/venv/bin/python3", patch)
        self.assertIn("find . -maxdepth 1 -type f -name '*.whl' -print -quit", patch)

    def test_audit_parser_rejects_foreign_arch_and_accepts_dual_union(self) -> None:
        audit = load_script("audit_dual_arch_wheel.py")
        self.assertEqual(audit.architectures_from_cuobjdump("arch = sm_90a code=compute_120a"), {"90", "120"})
        self.assertEqual(audit.REQUIRED_NATIVE_ARCHITECTURES, ("90", "120"))
        self.assertEqual(audit.ALLOWED_COMPATIBILITY_ARCHITECTURES, ("80", "89"))

    def test_overlay_and_artifact_hashes_are_exact(self) -> None:
        for item in self.lock["overlay"]["patches"]:
            path = ROOT / item["path"]
            self.assertEqual(__import__("hashlib").sha256(path.read_bytes()).hexdigest(), item["sha256"])
        for item in self.lock["artifacts"]:
            path = ROOT / item["path"]
            self.assertEqual(__import__("hashlib").sha256(path.read_bytes()).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
