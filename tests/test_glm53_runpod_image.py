from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import jsonschema

from putpocket_dataset_mining.glm53_runpod_image import (
    MODEL_ID,
    MODEL_REVISION,
    PROFILES,
    RunPodContractError,
    locate_vllm_flash_attn_extensions,
    runtime_doctor,
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
    ROOT / f"patches/vllm/{COMMIT}/glm53_sm120_nope_query_padding.patch",
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
        self.assertEqual(
            self.lock["runtime"]["common_import_dependencies"][
                "vllm_flash_attn_extensions"
            ],
            ["_vllm_fa2_C", "_vllm_fa3_C"],
        )

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

    def test_sm120_nope_adapter_preserves_model_semantics(self) -> None:
        adapter = self.lock["runtime"]["sm120_nope_query_adapter"]
        self.assertEqual(adapter["model_query_width"], 512)
        self.assertEqual(adapter["kernel_query_width"], 576)
        self.assertEqual(adapter["zero_padding_width"], 64)
        self.assertEqual(adapter["cache_bytes_per_token"], 656)
        self.assertEqual(adapter["kernel_qk_rope_head_dim"], 64)
        self.assertEqual(adapter["kv_scale_format"], "arbitrary_fp32")
        self.assertEqual(adapter["active_topk_length_argument"], "seq_lens")
        self.assertEqual(adapter["model_index_topk"], 2048)
        self.assertEqual(adapter["kernel_topk_width"], 2048)
        self.assertEqual(
            adapter["kpool_tail_policy"],
            "keep_valid_tail_drop_lowest_ranked_history",
        )
        self.assertEqual(
            adapter["attention_semantics"],
            "unchanged_nope_zero_dot_product_tail",
        )
        adapter_patch = (
            ROOT
            / f"patches/vllm/{COMMIT}/glm53_sm120_nope_query_padding.patch"
        ).read_text()
        self.assertIn("torch.nn.functional.pad(q, (0, 64), value=0.0)", adapter_patch)
        self.assertIn(
            "qk_rope_head_dim=flashinfer_qk_rope_head_dim", adapter_patch
        )
        self.assertNotIn("self.qk_rope_head_dim = 64", adapter_patch)

        topk_patch = (
            ROOT
            / f"patches/vllm/{COMMIT}/glm53_sm120_nope_topk_lens.patch"
        ).read_text()
        self.assertIn("seq_lens=active_topk_lens", topk_patch)
        self.assertNotIn("sparse_mla_top_k_lens=active_topk_lens", topk_patch)
        self.assertIn("_fit_kpool_indices_to_flashinfer", topk_patch)
        self.assertIn("topk_tokens - valid_tail", topk_patch)
        self.assertIn(
            "NUM_TOPK_TOKENS=attn_metadata.topk_tokens", topk_patch
        )

        cached_overlay = self.lock["runtime_image_overlay"][
            "cached_base_sm120_topk_abi"
        ]
        cached_patch = (ROOT / cached_overlay["path"]).read_text()
        self.assertIn("_fit_kpool_indices_to_flashinfer", cached_patch)
        self.assertIn("topk_tokens - valid_tail", cached_patch)

        dockerfile = (ROOT / "docker/glm53_sm90_sm120/Dockerfile").read_text()
        self.assertIn(
            "3b2ff18d2db7196f53c143acdbce146e0fd904ab4d797f0356c99a52c5823b50",
            dockerfile,
        )
        self.assertIn("glm53_sm120_cached_base_topk_abi.patch", dockerfile)
        self.assertIn(
            "c82a9697ea68a5039332e02e011afb5f07fc8a3fcc113012d552d2fc102edfb7",
            dockerfile,
        )

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
        self.assertIn('putpocket.task_id="T20260907-002__glm53-runpod-fa-fix"', dockerfile)
        self.assertIn('putpocket.vllm.flash_attn_import_extensions="fa2,fa3"', dockerfile)
        self.assertIn(
            'putpocket.vllm.sm120_nope_query_adapter="zero-pad-512-to-576-glm-layout"',
            dockerfile,
        )

    def test_flash_attention_extensions_are_located_without_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extension_root = root / "vllm_flash_attn"
            extension_root.mkdir()
            (extension_root / "_vllm_fa2_C.test.so").write_bytes(b"\x7fELFfa2")
            missing = locate_vllm_flash_attn_extensions(root)
            self.assertTrue(missing["vllm_fa2_extension"]["ok"])
            self.assertFalse(missing["vllm_fa3_extension"]["ok"])
            (extension_root / "_vllm_fa3_C.test.so").write_bytes(b"\x7fELFfa3")
            complete = locate_vllm_flash_attn_extensions(root)
            self.assertTrue(all(result["ok"] for result in complete.values()))
            self.assertTrue(complete["vllm_fa2_extension"]["path"].endswith(".so"))
            self.assertTrue(complete["vllm_fa3_extension"]["path"].endswith(".so"))

    def test_runtime_doctor_fails_before_torch_when_flash_extensions_missing(self) -> None:
        class FakeDistribution:
            def __init__(self, root: Path):
                self.root = root

            def locate_file(self, path: str) -> Path:
                return self.root / path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "vllm/vllm_flash_attn").mkdir(parents=True)
            with mock.patch(
                "putpocket_dataset_mining.glm53_runpod_image.importlib.metadata.distribution",
                return_value=FakeDistribution(root),
            ):
                with self.assertRaisesRegex(
                    RunPodContractError, "VLLM_FLASH_ATTN_EXTENSION_MISSING"
                ) as raised:
                    runtime_doctor(LOCK_PATH, "sm90", root / "model")
            diagnostic = str(raised.exception)
            self.assertIn("vllm_fa2_extension", diagnostic)
            self.assertIn("vllm_fa3_extension", diagnostic)
            self.assertIn("_vllm_fa2_C*.so", diagnostic)
            self.assertIn("_vllm_fa3_C*.so", diagnostic)

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
        self.assertIn("include(cmake/external_projects/vllm_flash_attn.cmake)", patch)
        self.assertIn("common FA2/FA3 import extensions enabled", patch)
        self.assertNotIn("bundled FA2/FA3, FlashMLA", patch)
        self.assertIn("DeepEP intentionally omitted", patch)
        self.assertIn("putpocket_audit_dual_arch_wheel.py", patch)
        self.assertIn("printf '%s\\n' /opt/venv/bin/python3", patch)
        self.assertIn("find . -maxdepth 1 -type f -name '*.whl' -print -quit", patch)

    def test_audit_parser_rejects_foreign_arch_and_accepts_dual_union(self) -> None:
        audit = load_script("audit_dual_arch_wheel.py")
        self.assertEqual(audit.architectures_from_cuobjdump("arch = sm_90a code=compute_120a"), {"90", "120"})
        self.assertEqual(audit.REQUIRED_NATIVE_ARCHITECTURES, ("90", "120"))
        self.assertEqual(audit.ALLOWED_COMPATIBILITY_ARCHITECTURES, ("80", "89"))
        self.assertEqual(
            audit.REQUIRED_EXTENSION_PREFIXES,
            (
                "vllm/vllm_flash_attn/_vllm_fa2_C",
                "vllm/vllm_flash_attn/_vllm_fa3_C",
            ),
        )
        with self.assertRaisesRegex(audit.DualArchAuditError, "missing required"):
            audit.required_extension_members([])
        members = audit.required_extension_members(
            [
                "vllm/vllm_flash_attn/_vllm_fa2_C.test.so",
                "vllm/vllm_flash_attn/_vllm_fa3_C.test.so",
            ]
        )
        self.assertEqual(set(members), set(audit.REQUIRED_EXTENSION_PREFIXES))

    def test_overlay_and_artifact_hashes_are_exact(self) -> None:
        for item in self.lock["overlay"]["patches"]:
            path = ROOT / item["path"]
            self.assertEqual(__import__("hashlib").sha256(path.read_bytes()).hexdigest(), item["sha256"])
        for item in self.lock["artifacts"]:
            path = ROOT / item["path"]
            self.assertEqual(__import__("hashlib").sha256(path.read_bytes()).hexdigest(), item["sha256"])


if __name__ == "__main__":
    unittest.main()
