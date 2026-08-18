from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

from putpocket_dataset_mining.cluster_config import (
    ClusterProfile,
    ClusterSite,
    VLLM_026_SHA,
    load_cluster_profile,
    validate_environment_lock,
)
from putpocket_dataset_mining.cluster_manifest import ProbeResult, ProbeRunner, capture_run_manifest, execute_guarded
from putpocket_dataset_mining.cluster_safety import (
    E_SECRET_BEARING_COMMAND,
    E_SLURM_ALLOCATION_REQUIRED,
    E_UNSAFE_CLUSTER_PATH,
    require_slurm_allocation,
    safe_absolute_path,
    validate_secret_free_command,
)
from putpocket_dataset_mining.cluster_slurm import render_slurm_job
from putpocket_dataset_mining.errors import ConfigError


ROOT = Path(__file__).resolve().parents[1]


class ManifestProbe(ProbeRunner):
    def run(self, command):  # type: ignore[no-untyped-def]
        if "rev-parse" in command:
            stdout = "0123456789abcdef0123456789abcdef01234567\n"
        elif "topo" in command:
            stdout = "GPU0 GPU1 NV4\n"
        elif "--version" in command:
            stdout = "Cuda compilation tools, release 12.9, V12.9.86\n"
        else:
            stdout = "0, NVIDIA H200, GPU-0, 999.1, 0000:01:00.0, 9.0\n"
        return ProbeResult(tuple(command), 0, stdout, "")


def site_mapping() -> dict:
    return {
        "schema_version": 1,
        "site": {
            "partition": "gpu",
            "account": "research",
            "constraint": "h200",
            "wall_time": "04:00:00",
            "cpus_per_task": 32,
            "repository_root": str(ROOT),
            "python_executable": "/opt/python/bin/python",
            "uv_executable": "/opt/uv/bin/uv",
            "git_executable": "/usr/bin/git",
            "nvidia_smi_executable": "/usr/bin/nvidia-smi",
            "nvcc_executable": "/opt/cuda/bin/nvcc",
            "environment_root": "/cluster/envs/glm52",
            "vllm_source_root": "/cluster/src/vllm-026",
            "cache_root": "/cluster/cache/glm52",
            "checkpoint_root": "/cluster/checkpoints",
            "artifact_root": "/cluster/artifacts",
            "slurm_log_root": "/cluster/slurm-logs",
        },
        "models": {
            "glm52_nvfp4_tp1_pcp4_ep": {"path": "/cluster/checkpoints/nvfp4", "revision": "rev-a"},
            "glm52_nvfp4_tp2_pcp2_ep": {"path": "/cluster/checkpoints/nvfp4", "revision": "rev-a"},
            "glm52_fp8_tp8_reference": {"path": "/cluster/checkpoints/fp8", "revision": "rev-b"},
        },
    }


class ClusterPackageTests(unittest.TestCase):
    def test_exact_phase1_profiles(self) -> None:
        cases = {
            "glm52_nvfp4_tp1_pcp4_ep": ("nvidia/GLM-5.2-NVFP4", 4, 1, 4, True),
            "glm52_nvfp4_tp2_pcp2_ep": ("nvidia/GLM-5.2-NVFP4", 4, 2, 2, True),
            "glm52_fp8_tp8_reference": ("zai-org/GLM-5.2-FP8", 8, 8, 1, False),
        }
        for profile_id, expected in cases.items():
            with self.subTest(profile_id=profile_id):
                profile = load_cluster_profile(profile_id)
                self.assertEqual(
                    (
                        profile.model_id,
                        profile.gpus_per_node,
                        profile.tensor_parallel_size,
                        profile.prefill_context_parallel_size,
                        profile.expert_parallel,
                    ),
                    expected,
                )

    def test_parallelism_mismatch_is_rejected(self) -> None:
        profile = {
            "schema_version": 1,
            "profile_id": "bad",
            "phase": "1_foundation",
            "model": {"id": "m", "path": None, "revision": None, "quantization": "fp8"},
            "hardware": {
                "accelerator_name_pattern": "H200",
                "compute_capability": "9.0",
                "nodes": 1,
                "gpus_per_node": 4,
            },
            "parallelism": {"tensor_parallel_size": 2, "prefill_context_parallel_size": 1, "expert_parallel": False},
            "runtime": {
                "engine": "vllm",
                "environment_lock": "configs/env/cluster_h200_sm90_vllm026.lock.yaml",
                "engine_args": ["--tensor-parallel-size", "2"],
            },
            "readiness": {"accepted_quantization_markers": ["fp8"], "required_imports": {"torch": ["cuda"]}},
        }
        with self.assertRaisesRegex(ConfigError, "parallelism mismatch"):
            ClusterProfile.from_mapping(profile)

    def test_environment_lock_is_h200_sm90_and_vllm_026_commit(self) -> None:
        lock = validate_environment_lock()
        self.assertEqual(lock["hardware"]["accelerator_family"], "H200")
        self.assertEqual(lock["hardware"]["torch_cuda_arch_list"], "9.0")
        self.assertEqual(lock["vllm"]["commit"], VLLM_026_SHA)
        self.assertTrue(lock["vllm"]["source_clean_required"])
        self.assertNotIn("server2", str(lock).lower())
        self.assertNotIn("runpod", str(lock).lower())

    def test_committed_json_schemas_remain_small_source_files(self) -> None:
        for name in ("profile.schema.json", "site.schema.json"):
            path = ROOT / "configs/cluster/schemas" / name
            schema = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertLess(path.stat().st_size, 64 * 1024)

    def test_slurm_render_is_provider_neutral_and_never_submits(self) -> None:
        profile = load_cluster_profile("glm52_nvfp4_tp1_pcp4_ep")
        rendered = render_slurm_job(profile, ClusterSite.from_mapping(site_mapping()), "readiness")
        self.assertIn("#SBATCH --gpus-per-node=4", rendered)
        self.assertIn("#SBATCH --partition=gpu", rendered)
        self.assertIn("#SBATCH --account=research", rendered)
        self.assertIn("#SBATCH --constraint=h200", rendered)
        self.assertIn("#SBATCH --export=NONE", rendered)
        self.assertIn("allocation-check", rendered)
        self.assertIn("--stage all", rendered)
        self.assertIn("--model-revision \"$MODEL_REVISION\"", rendered)
        self.assertNotRegex(rendered, r"(^|\s)(sbatch|salloc)(\s|$)")
        self.assertNotIn("runpod", rendered.lower())
        self.assertNotIn("montblanc", rendered.lower())

    def test_environment_render_wraps_bootstrap_in_guarded_action(self) -> None:
        profile = load_cluster_profile("glm52_fp8_tp8_reference")
        rendered = render_slurm_job(profile, ClusterSite.from_mapping(site_mapping()), "environment")
        self.assertIn("#SBATCH --gpus-per-node=8", rendered)
        self.assertIn("--action environment-build", rendered)
        self.assertIn("--execute", rendered)
        self.assertIn("cluster_h200_sm90_vllm026.lock.yaml", rendered)

    def test_hostname_alone_does_not_satisfy_allocation_guard(self) -> None:
        with self.assertRaisesRegex(ConfigError, E_SLURM_ALLOCATION_REQUIRED):
            require_slurm_allocation({"HOSTNAME": "compute-42"})
        evidence = require_slurm_allocation(
            {
                "SLURM_JOB_ID": "1234",
                "SLURM_JOB_NODELIST": "gpu[01]",
                "SLURM_JOB_NUM_NODES": "1",
                "SLURM_JOB_NAME": "readiness",
            }
        )
        self.assertEqual(evidence["job_id"], "1234")

    def test_every_heavy_action_refuses_without_allocation_before_execution(self) -> None:
        profile = load_cluster_profile("glm52_nvfp4_tp1_pcp4_ep")
        for action in (
            "environment-build",
            "dependency-install",
            "checkpoint-stage",
            "gpu-smoke",
            "model-load",
            "benchmark",
            "one-shot-generation",
        ):
            with self.subTest(action=action), self.assertRaisesRegex(ConfigError, E_SLURM_ALLOCATION_REQUIRED):
                execute_guarded(
                    action=action,
                    profile=profile,
                    command=["/bin/true"],
                    artifact_root="/cluster/artifacts/test",
                    git_executable="/usr/bin/git",
                    nvidia_smi_executable="/usr/bin/nvidia-smi",
                    nvcc_executable="/opt/cuda/bin/nvcc",
                    model_revision=None,
                    env={},
                )

    def test_secret_and_path_sanitization(self) -> None:
        with self.assertRaisesRegex(ConfigError, E_SECRET_BEARING_COMMAND):
            validate_secret_free_command(["python", "run.py", "--token=top-secret"])
        with self.assertRaisesRegex(ConfigError, E_SECRET_BEARING_COMMAND):
            validate_secret_free_command(["HF_TOKEN=value", "python", "run.py"])
        with self.assertRaisesRegex(ConfigError, E_UNSAFE_CLUSTER_PATH):
            safe_absolute_path("/home/user/.ssh/output", "artifact_root")
        with self.assertRaisesRegex(ConfigError, E_UNSAFE_CLUSTER_PATH):
            safe_absolute_path("/cluster/runs/../secret", "artifact_root")
        self.assertEqual(validate_secret_free_command(["python", "run.py", "--model", "nvidia/GLM-5.2-NVFP4"])[0], "python")

    def test_run_manifest_captures_allowlisted_provenance_without_environment_dump(self) -> None:
        profile = load_cluster_profile("glm52_nvfp4_tp1_pcp4_ep")
        manifest = capture_run_manifest(
            profile=profile,
            command=["/cluster/env/bin/python", "workload.py", "--model", "/cluster/checkpoints/nvfp4"],
            artifact_root="/cluster/artifacts/run-1",
            git_executable="/usr/bin/git",
            nvidia_smi_executable="/usr/bin/nvidia-smi",
            nvcc_executable="/opt/cuda/bin/nvcc",
            model_revision="revision-a",
            env={
                "SLURM_JOB_ID": "1234",
                "SLURM_JOB_NODELIST": "gpu01",
                "SLURM_JOB_NUM_NODES": "1",
                "SLURM_JOB_NAME": "manifest-test",
                "SLURM_JOB_GPUS": "0,1,2,3",
                "HF_TOKEN": "must-not-appear",
            },
            runner=ManifestProbe(),
        )
        serialized = str(manifest)
        self.assertEqual(manifest["git_sha"], "0123456789abcdef0123456789abcdef01234567")
        self.assertEqual(manifest["slurm"]["job_id"], "1234")  # type: ignore[index]
        self.assertIn("NV4", manifest["gpu"]["topology"]["stdout"])  # type: ignore[index]
        self.assertEqual(manifest["gpu"]["allocated_selector"], "0,1,2,3")  # type: ignore[index]
        self.assertEqual(manifest["model"]["revision"], "revision-a")  # type: ignore[index]
        self.assertIn("no_full_tensor_hash", manifest["checkpoint_hash_policy"])
        self.assertNotIn("must-not-appear", serialized)
        self.assertNotIn("HF_TOKEN", serialized)

    def test_gitignore_protects_runtime_artifacts_but_not_source_fixtures(self) -> None:
        ignored = [
            "models/shard-00001.safetensors",
            "scratch/pytorch_model-00001-of-00004.bin",
            "scratch/model.safetensors.index.json",
            "checkpoints/model.ckpt",
            ".cache/huggingface/hub/blob",
            ".cache/vllm/compile-cache.bin",
            "scratch/hf_cache/models/blob",
            "slurm-readiness-123.out",
            "cluster_artifacts/run/manifest.json",
            "benchmark_outputs/result.json",
            "swebench_outputs/report.json",
        ]
        tracked_sources = [
            "tests/fixtures/tiny_checkpoint.safetensors",
            "configs/cluster/schemas/profile.schema.json",
            "configs/cluster/profiles/glm52_nvfp4_tp1_pcp4_ep.yaml",
        ]
        for relative in ignored:
            with self.subTest(relative=relative):
                result = subprocess.run(["git", "check-ignore", "--no-index", "-q", relative], cwd=ROOT)
                self.assertEqual(result.returncode, 0)
        for relative in tracked_sources:
            with self.subTest(relative=relative):
                result = subprocess.run(["git", "check-ignore", "--no-index", "-q", relative], cwd=ROOT)
                self.assertEqual(result.returncode, 1)


if __name__ == "__main__":
    unittest.main()
