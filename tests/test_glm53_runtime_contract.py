from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class GLM53RuntimeContractTests(unittest.TestCase):
    def test_runtime_files_are_generic_and_python_compiles(self) -> None:
        files = [
            ROOT / "docker/glm53_sm120/Dockerfile.flashinfer-overlay",
            ROOT / "scripts/glm53/bootstrap_runtime.sh",
            ROOT / "scripts/glm53/download_model.sh",
            ROOT / "scripts/glm53/launch_server.sh",
            ROOT / "scripts/glm53/run_smoke.sh",
            ROOT / "scripts/glm53/stop_server.sh",
            ROOT / "scripts/glm53/smoke_client.py",
            ROOT / "docs/GLM53_MONTBLANC_DEPLOYMENT.md",
        ]
        for path in files:
            with self.subTest(path=path):
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("/home/dyryu", text)
                self.assertNotIn("HF_TOKEN=", text)
        ast.parse((ROOT / "scripts/glm53/smoke_client.py").read_text(encoding="utf-8"))

    def test_overlay_fetches_only_exact_official_pr_source(self) -> None:
        text = (
            ROOT / "docker/glm53_sm120/Dockerfile.flashinfer-overlay"
        ).read_text(encoding="utf-8")
        self.assertIn("https://github.com/flashinfer-ai/flashinfer.git", text)
        self.assertIn('test "$(git -C /opt/flashinfer-source rev-parse HEAD)"', text)
        self.assertIn("python3 -m pip uninstall -y flashinfer-python flashinfer-jit-cache", text)
        self.assertIn("FLASHINFER_BUILD_NO_PIP=1", text)
        self.assertIn("FLASHINFER_DISABLE_VERSION_CHECK=1", text)
        self.assertNotIn("chriswritescode", text)
        self.assertNotIn("cstechdev", text)

    def test_bootstrap_selects_the_pinned_upstream_dockerfile(self) -> None:
        text = (ROOT / "scripts/glm53/bootstrap_runtime.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('--file "${VLLM_SOURCE_DIR}/docker/Dockerfile"', text)
        self.assertIn('--target vllm-openai', text)
        self.assertIn(
            '--build-arg "torch_cuda_arch_list=$(lock_value runtime.torch_cuda_arch_list)"',
            text,
        )
        self.assertIn("vllm_dockerfile_post_patch_sha256", text)
        self.assertIn('git -C "${VLLM_SOURCE_DIR}" apply --check', text)
        self.assertIn('status --porcelain)" = " M docker/Dockerfile"', text)

    def test_packaging_patch_only_skips_unpublished_flashinfer_release(self) -> None:
        patch_text = (
            ROOT
            / "patches/vllm/878631b6079d2cf9fb80830ef9cb41b43aded098/"
            "glm53_skip_unpublished_flashinfer_release.patch"
        ).read_text(encoding="utf-8")
        self.assertIn("FlashInfer 0.6.18rc10 release assets", patch_text)
        self.assertEqual(patch_text.count("+RUN true"), 2)
        self.assertIn("do not restore the unavailable release", patch_text)
        self.assertIn("flashinfer_python-${FLASHINFER_VERSION}", patch_text)
        self.assertNotIn("vllm/", patch_text)

    def test_lock_makes_sm120_only_build_boundary_explicit(self) -> None:
        lock = json.loads(
            (ROOT / "configs/models/glm53_flash_nvfp4_montblanc.lock.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(lock["runtime"]["torch_cuda_arch_list"], "12.0")
        self.assertEqual(
            lock["runtime"]["flash_attention_arch_policy"],
            "upstream_per_kernel_forward_compatible_defaults",
        )

    def test_launch_uses_only_supported_three_gpu_layout_and_bounded_smoke(self) -> None:
        text = (ROOT / "scripts/glm53/launch_server.sh").read_text(encoding="utf-8")
        required = [
            "/dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidiactl",
            "/dev/nvidia-uvm /dev/nvidia-uvm-tools",
            "libcuda.so.1",
            "libnvidia-ptxjitcompiler.so.1",
            "libnvidia-nvvm.so.4",
            "libnvidia-gpucomp.so.${EXPECTED_DRIVER_VERSION}",
            "explicit_devices_and_driver_libs",
            "--tensor-parallel-size 1",
            "--pipeline-parallel-size 1",
            "--data-parallel-size 3",
            "--enable-expert-parallel",
            "--enable-ep-weight-filter",
            "--kv-cache-dtype fp8_ds_mla",
            "--block-size 512",
            "--max-model-len 4096",
            "--max-num-seqs 1",
            "--no-enable-prefix-caching",
            '"moe_backend":"marlin"',
            '"backend":"FLASHINFER_MLA_SPARSE_SM120"',
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
        ]
        for value in required:
            with self.subTest(value=value):
                self.assertIn(value, text)
        self.assertNotIn("--tensor-parallel-size 3", text)
        self.assertNotIn("--pipeline-parallel-size 3", text)
        self.assertNotIn("--gpus", text)
        self.assertNotIn("model_mtp", text)

    def test_safe_stop_never_escalates_to_sigkill(self) -> None:
        text = (ROOT / "scripts/glm53/stop_server.sh").read_text(encoding="utf-8")
        self.assertIn("docker stop --signal SIGTERM --timeout -1", text)
        self.assertIn("putpocket.task_id", text)
        self.assertIn("putpocket.run_id", text)
        self.assertNotIn("SIGKILL", text.replace("never escalates to SIGKILL", ""))
        self.assertNotIn("pkill", text)

    def test_smoke_uses_all_endpoint_layers_and_exact_prompt_replay(self) -> None:
        text = (ROOT / "scripts/glm53/smoke_client.py").read_text(encoding="utf-8")
        self.assertIn("/v1/models", text)
        self.assertIn("/v1/chat/completions", text)
        self.assertIn("OpenAICompatibleHTTPGenerationEngine", text)
        self.assertIn("first_text != second_text", text)
        self.assertIn("rendered_prompt=rendered", text)
        self.assertIn('"clear_thinking": True', text)

    def test_execution_example_is_machine_readable_and_consistent(self) -> None:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - project dependency
            self.fail(str(exc))
        config = yaml.safe_load(
            (
                ROOT / "configs/execution/server2_glm53_flash_nvfp4_ep3.example.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["parallelism"]["tensor_parallel_size"], 1)
        self.assertEqual(config["parallelism"]["data_parallel_size"], 3)
        self.assertEqual(config["parallelism"]["expert_parallel_size"], 3)
        self.assertFalse(config["runtime_boundaries"]["mtp"])
        self.assertFalse(config["runtime_boundaries"]["prefix_caching"])
        lock = json.loads(
            (ROOT / config["model_lock"]).read_text(encoding="utf-8")
        )
        self.assertEqual(lock["runtime"]["moe_backend"], "marlin")


if __name__ == "__main__":
    unittest.main()
