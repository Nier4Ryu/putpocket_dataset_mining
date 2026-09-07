from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from putpocket_dataset_mining.glm53_a6000_image import (
    A6000ContractError,
    static_doctor,
    validate_lock,
)


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "9cd956c7e6cf54efa366b803cafa15ec6c2df827"
FORBIDDEN_LEGACY_TERMS = (
    "RedHatAI",
    "NVFP4",
    "SM120",
    "DSV4",
    "878631",
    "fp8_ds_mla",
)
PACKAGE_PATHS = (
    ROOT / "docker/glm53_sm86",
    ROOT / "scripts/glm53_a6000",
    ROOT / "instrumentation/vllm/glm53_a6000_compatibility_gate.py",
    ROOT / "instrumentation/vllm/glm53_a6000_stateful_edit_accuracy_ablation.py",
    ROOT / "src/putpocket_dataset_mining/glm53_a6000_image.py",
    ROOT / "src/putpocket_dataset_mining/glm53_a6000_stateful_edit.py",
    ROOT / "configs/models/glm53_flash_bf16_a6000_image.lock.json",
    ROOT / "configs/execution/glm53_a6000_compatibility_gated.example.yaml",
    ROOT / "configs/experiments/schemas/glm53_a6000_stateful_edit_control.schema.json",
    # The vLLM commit directory is shared by independent deployment packages.
    # Keep this historical SM86 assertion scoped to the two A6000 overlays;
    # later SM90/SM120 overlays legitimately contain terms forbidden here.
    ROOT / f"patches/vllm/{COMMIT}/glm53_a6000_sm86_only_build.patch",
    ROOT / f"patches/vllm/{COMMIT}/glm53_a6000_compatibility_gate.patch",
    ROOT / f"vendor/vllm/{COMMIT}/support_snapshot.txt",
)


def load_hook():
    path = ROOT / "instrumentation/vllm/glm53_a6000_stateful_edit_accuracy_ablation.py"
    spec = importlib.util.spec_from_file_location("glm53_a6000_hook", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_script(name: str):
    path = ROOT / "scripts/glm53_a6000" / name
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GLM53A6000ImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock_path = ROOT / "configs/models/glm53_flash_bf16_a6000_image.lock.json"
        self.lock = json.loads(self.lock_path.read_text(encoding="utf-8"))

    def test_lock_is_official_bf16_current_source_and_blocked(self) -> None:
        validate_lock(self.lock)
        self.assertEqual(self.lock["model"]["id"], "zai-org/GLM-5.3-Flash-BF16")
        self.assertEqual(
            self.lock["model"]["revision"],
            "a5b45eb41df6402735dedc900be14a42e8d5e538",
        )
        self.assertEqual(self.lock["runtime"]["vllm_commit"], COMMIT)
        self.assertFalse(self.lock["runtime"]["runtime_supported"])
        self.assertEqual(self.lock["runtime"]["torch_cuda_arch_list"], "8.6")
        self.assertEqual(self.lock["runtime"]["cmake_cuda_architectures"], "86")
        build_scope = self.lock["runtime"]["task_built_cuda_scope"]
        self.assertEqual(build_scope["exclusive_architecture"], "8.6")
        self.assertFalse(build_scope["deepep_included"])
        self.assertFalse(build_scope["bundled_fa2_included"])
        self.assertFalse(build_scope["non_sm86_optional_extensions_included"])
        self.assertFalse(
            self.lock["runtime"]["vendor_binary_scope"]["single_architecture_claim"]
        )
        self.assertEqual(
            self.lock["runtime"]["comparison_release"]["disposition"],
            "evidence_only_not_runtime_source",
        )

    def test_hybrid_and_accuracy_ablation_invariants(self) -> None:
        hybrid = self.lock["hybrid_cache_contract"]
        self.assertEqual(hybrid["layer_count"], 45)
        self.assertEqual(
            hybrid["mla_indexer_layers"],
            [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43],
        )
        self.assertEqual(hybrid["kda_layer_count"], 34)
        self.assertEqual(hybrid["index_kpool"], 4)
        stateful = self.lock["stateful_edit"]
        self.assertFalse(stateful["default_enabled"])
        self.assertTrue(stateful["full_target_prefill"])
        self.assertTrue(stateful["donor_overwrite_after_native_write"])
        self.assertFalse(stateful["true_partial_prefill"])
        self.assertFalse(stateful["speedup_claim"])

    def test_new_package_contains_no_legacy_runtime_binding(self) -> None:
        for entry in PACKAGE_PATHS:
            files = sorted(entry.rglob("*")) if entry.is_dir() else [entry]
            for path in files:
                if not path.is_file() or path.suffix == ".pyc":
                    continue
                text = path.read_text(encoding="utf-8")
                for term in FORBIDDEN_LEGACY_TERMS:
                    with self.subTest(path=path, term=term):
                        self.assertNotIn(term, text)

    def test_hook_signatures_and_binding_parse(self) -> None:
        path = ROOT / "instrumentation/vllm/glm53_a6000_stateful_edit_accuracy_ablation.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        self.assertTrue(
            {
                "accuracy_ablation_armed",
                "selector_capture_armed",
                "maybe_attest_prompt",
                "maybe_apply_main_cache",
                "maybe_apply_indexer_cache",
                "maybe_capture_base_selector_scores",
                "validate_stateful_control",
                "validate_selector_capture_control",
            }.issubset(names)
        )
        hook = load_hook()
        self.assertEqual(hook.MODEL_ID, "zai-org/GLM-5.3-Flash-BF16")
        self.assertEqual(hook.VLLM_COMMIT, COMMIT)
        self.assertEqual(hook.KV_CACHE_DTYPE, "bfloat16")
        self.assertFalse(hook.SM86_RUNTIME_SUPPORTED)
        self.assertFalse(hook.accuracy_ablation_armed())
        self.assertFalse(hook.selector_capture_armed())

    def test_complete_pool_projection_and_selector_digest_validation(self) -> None:
        hook = load_hook()
        self.assertEqual(hook.complete_pool_ends([4, 5, 6, 7, 8], 4, 12), (7,))
        with tempfile.TemporaryDirectory() as temporary:
            selector = Path(temporary) / "selector.json"
            selector.write_text('{"frozen":true}\n', encoding="utf-8")
            selected = [4, 5, 6, 7]
            control = {
                "schema_version": 4,
                "mode": "STATEFUL_EDIT_ACCURACY_ABLATION",
                "model": {
                    "id": hook.MODEL_ID,
                    "revision": hook.MODEL_REVISION,
                    "architecture": hook.MODEL_ARCHITECTURE,
                    "vllm_commit": hook.VLLM_COMMIT,
                    "attention_backend_policy": hook.ATTENTION_BACKEND_POLICY,
                    "kv_cache_dtype": hook.KV_CACHE_DTYPE,
                    "index_kpool": 4,
                },
                "target_compute_capability": "8.6",
                "sm86_runtime_supported": False,
                "unsafe_accuracy_ablation_ack": hook.UNSAFE_ACK,
                "production_default_enabled": False,
                "compute_semantics": "full_target_prefill_then_selected_donor_cache_overwrite",
                "true_partial_prefill": False,
                "speedup_claim": False,
                "history_token_count": 12,
                "eligible_start": 4,
                "edit_positions": [3],
                "selected_main_positions": selected,
                "selected_main_positions_sha256": hook.token_ids_sha256(selected),
                "selected_indexer_pool_end_positions": [7],
                "selector_path": str(selector),
                "selector_sha256": hashlib.sha256(selector.read_bytes()).hexdigest(),
            }
            path = Path(temporary) / "control.json"
            path.write_text(json.dumps(control), encoding="utf-8")
            self.assertEqual(hook.validate_stateful_control(path), control)
            control["selector_sha256"] = "0" * 64
            path.write_text(json.dumps(control), encoding="utf-8")
            with self.assertRaisesRegex(Exception, "SELECTOR_DIGEST_MISMATCH"):
                hook.validate_stateful_control(path)

    def test_static_doctor_has_no_device_probe_and_passes_integrity(self) -> None:
        source = (ROOT / "src/putpocket_dataset_mining/glm53_a6000_image.py").read_text(
            encoding="utf-8"
        )
        static_node = next(
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.FunctionDef) and node.name == "static_doctor"
        )
        static_text = ast.unparse(static_node)
        self.assertNotIn("torch", static_text)
        self.assertNotIn("device_count", static_text)
        self.assertNotIn("get_device_capability", static_text)
        report = static_doctor(self.lock_path, ROOT)
        self.assertTrue(report["package_integrity_ok"])
        self.assertFalse(report["glm53_runtime_supported"])

    def test_build_and_launch_scripts_cannot_use_gpu_runtime(self) -> None:
        build = (ROOT / "scripts/glm53_a6000/build_image.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("torch_cuda_arch_list=8.6", build)
        self.assertIn("putpocket_sm86_only=1", build)
        self.assertNotIn("--gpus", build)
        self.assertNotIn("nvidia-smi", build)
        wrapper = (ROOT / "docker/glm53_sm86/Dockerfile").read_text(
            encoding="utf-8"
        )
        self.assertIn("uv pip uninstall --system", wrapper)
        self.assertIn("nvidia-cutlass-dsl-libs-cu12", wrapper)
        self.assertIn(
            "/opt/putpocket/src/putpocket_dataset_mining/glm53_a6000_stateful_edit.py",
            wrapper,
        )
        self.assertIn(
            "/opt/putpocket/instrumentation/vllm/glm53_a6000_stateful_edit_accuracy_ablation.py",
            wrapper,
        )
        launch = (ROOT / "scripts/glm53_a6000/launch_template.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("exit 3", launch)
        self.assertNotIn("docker run", launch)
        entry = (ROOT / "scripts/glm53_a6000/entrypoint.sh").read_text(
            encoding="utf-8"
        )
        self.assertLess(entry.index("GLM-5.3-Flash is blocked"), entry.index("--runtime"))
        self.assertNotIn("vllm serve", entry)

    def test_sm86_packaging_patch_uses_real_empty_wheel_checks(self) -> None:
        attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn(
            f"patches/vllm/{COMMIT}/*.patch -whitespace",
            attributes,
        )
        patch = (
            ROOT
            / f"patches/vllm/{COMMIT}/glm53_a6000_sm86_only_build.patch"
        ).read_text(encoding="utf-8")
        postimage_lines = "\n".join(
            line[1:]
            for line in patch.splitlines()
            if line.startswith("+") and not line.startswith("+++")
        )
        self.assertIn("CMAKE_CUDA_ARCHITECTURES=86", postimage_lines)
        self.assertIn("bundled FA2 omitted to prevent SM80 SASS/PTX", postimage_lines)
        self.assertIn("if not PUTPOCKET_SM86_ONLY", postimage_lines)
        self.assertIn(
            "find . -maxdepth 1 -type f -name '*.whl' -print -quit",
            postimage_lines,
        )
        self.assertNotIn("test ! -e '*.whl'", postimage_lines)
        self.assertIn("sha256sum omission.json > wheels.sha256", postimage_lines)
        self.assertNotIn(
            "sha256sum /tmp/ep_kernels_workspace/dist/omission.json",
            postimage_lines,
        )
        self.assertNotIn(
            "uv pip install --system ep_kernels/dist/*.whl", postimage_lines
        )

    def test_empty_wheel_check_fails_when_a_real_wheel_exists(self) -> None:
        check = "test -z \"$(find . -maxdepth 1 -type f -name '*.whl' -print -quit)\""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = subprocess.run(
                ["bash", "-ceu", check], cwd=root, check=False
            )
            self.assertEqual(empty.returncode, 0)
            (root / "deepep-real.whl").touch()
            populated = subprocess.run(
                ["bash", "-ceu", check], cwd=root, check=False
            )
            self.assertNotEqual(populated.returncode, 0)

    def test_sm86_cuda_arch_parser_and_build_log_audit(self) -> None:
        wheel_audit = load_script("audit_sm86_wheel.py")
        self.assertEqual(
            wheel_audit.architectures_from_cuobjdump(
                "ELF file: x.sm_86.cubin\nPTX file: x.compute_86.ptx"
            ),
            {"86"},
        )
        self.assertEqual(
            wheel_audit.architectures_from_cuobjdump("x.sm_90a.cubin"),
            {"90"},
        )
        log_audit = load_script("audit_sm86_build_log.py")
        markers = "\n".join(
            (
                "-- CUDA target architectures: 8.6",
                "PutPocket SM86-only build: excluding non-SM86 optional external projects",
                "PutPocket SM86-only build: bundled FA2 omitted to prevent SM80 SASS/PTX",
                "PutPocket SM86-only build: FlashInfer JIT cache omitted",
                '{"component":"DeepEP","included":false}',
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "build.log"
            log.write_text(
                markers
                + "\nnvcc source.cu -gencode=arch=compute_86,code=sm_86\n",
                encoding="utf-8",
            )
            self.assertTrue(
                log_audit.audit_build_log(log)[
                    "no_non_sm86_compiler_flags_observed"
                ]
            )
            cached_report = {
                "all_task_built_cuda_payloads_sm86_only": True,
                "observed_architectures": ["86"],
                "members": {
                    "vllm.whl:vllm/_C.abi3.so": {
                        "architectures": ["86"],
                        "device_code_present": True,
                    }
                },
            }
            log.write_text(
                "\n".join(
                    (
                        "PutPocket SM86-only build: FlashInfer JIT cache omitted",
                        '{"component":"DeepEP","included":false}',
                        "#40 CACHED",
                        json.dumps(cached_report),
                    )
                ),
                encoding="utf-8",
            )
            cached = log_audit.audit_build_log(log)
            self.assertEqual(
                cached["configuration_evidence"],
                "cached_layer_plus_embedded_wheel_audit",
            )
            cached_report["members"]["vllm.whl:vllm/_vllm_fa2_C.so"] = {
                "architectures": ["86"]
            }
            log.write_text(
                "\n".join(
                    (
                        "PutPocket SM86-only build: FlashInfer JIT cache omitted",
                        '{"component":"DeepEP","included":false}',
                        json.dumps(cached_report),
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Exception, "cached build log"):
                log_audit.audit_build_log(log)
            log.write_text(
                markers
                + "\nnvcc source.cu -gencode=arch=compute_90,code=sm_90\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Exception, "must be.*86"):
                log_audit.audit_build_log(log)

    def test_wheel_audit_accepts_host_only_shared_object(self) -> None:
        wheel_audit = load_script("audit_sm86_wheel.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared_object = root / "host_only.so"
            shared_object.write_bytes(b"host-only test fixture")
            cuobjdump = root / "cuobjdump"
            cuobjdump.write_text(
                "#!/usr/bin/env bash\n"
                "echo \"cuobjdump info: File '$2' does not contain device code\" >&2\n"
                "exit 1\n",
                encoding="utf-8",
            )
            cuobjdump.chmod(0o755)
            record = wheel_audit.inspect_shared_object(shared_object, str(cuobjdump))
            self.assertFalse(record["device_code_present"])
            self.assertEqual(record["architectures"], [])
            cuobjdump.write_text(
                "#!/usr/bin/env bash\necho 'unexpected parser failure' >&2\nexit 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Exception, "cuobjdump failed"):
                wheel_audit.inspect_shared_object(shared_object, str(cuobjdump))

    def test_patch_applies_and_postimages_compile_when_source_available(self) -> None:
        source = Path(
            os.environ.get(
                "PUTPOCKET_VLLM_9CD956_SOURCE",
                str(
                    Path.home()
                    / ".cache/putpocket-runtime"
                    / "T20260904-001__glm53-a6000-image/vllm-9cd956"
                ),
            )
        )
        if not source.is_dir():
            self.skipTest("exact task-local current-vLLM source unavailable")
        self.assertEqual(
            subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
            COMMIT,
        )
        patches = (
            ROOT / f"patches/vllm/{COMMIT}/glm53_a6000_sm86_only_build.patch",
            ROOT / f"patches/vllm/{COMMIT}/glm53_a6000_compatibility_gate.patch",
        )
        for patch in patches:
            subprocess.run(
                ["git", "-C", str(source), "apply", "--check", str(patch)],
                check=True,
            )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared"
            subprocess.run(
                [
                    "bash",
                    str(ROOT / "scripts/glm53_a6000/prepare_vllm_source.sh"),
                    str(source),
                    str(output),
                ],
                check=True,
            )
            self.assertTrue((output / ".git").is_dir())
            expected = self.lock["overlay"]["postimages"]
            for relative, digest in expected.items():
                postimage = output / relative
                self.assertEqual(hashlib.sha256(postimage.read_bytes()).hexdigest(), digest)
                if postimage.suffix == ".py":
                    ast.parse(postimage.read_text(encoding="utf-8"))

    def test_validate_lock_rejects_runtime_enablement(self) -> None:
        altered = json.loads(json.dumps(self.lock))
        altered["runtime"]["runtime_supported"] = True
        with self.assertRaises(A6000ContractError):
            validate_lock(altered)


if __name__ == "__main__":
    unittest.main()
