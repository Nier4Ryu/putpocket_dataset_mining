from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from putpocket_dataset_mining.cluster_safety import E_SECRET_BEARING_COMMAND, E_SLURM_ALLOCATION_REQUIRED
from putpocket_dataset_mining.errors import ConfigError
from putpocket_dataset_mining.swebench_pro import (
    DATASET_REVISION,
    DOCKERHUB_NAMESPACE,
    FULL_TEST_ROWS,
    HARNESS_SHA,
    MINI_SWE_AGENT_SHA,
    SWE_AGENT_SHA,
    Selection,
    build_runtime_agent_config,
    classify_container_preflight,
    finalize_official_results,
    load_selection,
    official_image_uri,
    run_restartable_stage,
    select_rows,
    validate_agent_overlay,
    validate_source_lock,
)
from putpocket_dataset_mining.swebench_pro_slurm import BaselineSite, render_baseline_job


ROOT = Path(__file__).resolve().parents[1]
ALLOCATION = {
    "SLURM_JOB_ID": "1234",
    "SLURM_JOB_NODELIST": "h200-01",
    "SLURM_JOB_NUM_NODES": "1",
    "SLURM_JOB_NAME": "swepro-test",
}
PROJECT_SHA = "0123456789abcdef0123456789abcdef01234567"


def site_mapping() -> dict:
    return {
        "schema_version": 1,
        "site": {
            "partition": "gpu",
            "account": "research",
            "qos": "normal",
            "wall_time": "2-00:00:00",
            "memory": "512G",
            "cpus_per_task": 32,
            "h200_gpu_directive": "--gres=gpu:H200:4",
            "base_python": "/usr/bin/python3",
            "uv_executable": "/usr/bin/uv",
            "git_executable": "/usr/bin/git",
            "nvidia_smi_executable": "/usr/bin/nvidia-smi",
            "nvcc_executable": "/opt/cuda/bin/nvcc",
            "curl_executable": "/usr/bin/curl",
            "storage_root": "/cluster/project",
            "cache_root": "/cluster/project/cache",
            "artifact_root": "/cluster/project/artifacts",
            "slurm_log_root": "/cluster/project/slurm",
        },
        "model": {
            "path": "/cluster/project/models/glm52-nvfp4",
            "revision": "model-revision",
            "source": "huggingface",
        },
        "job": {"experiment_id": "glm52-swepro-baseline", "evaluation_workers": 4, "agent_workers": 1},
    }


class SwebenchProContractTests(unittest.TestCase):
    def test_pinned_official_sources_and_dataset_are_exact(self) -> None:
        lock = validate_source_lock()
        self.assertEqual(lock["harness"]["commit"], HARNESS_SHA)
        self.assertEqual(lock["submodules"]["SWE-agent"]["commit"], SWE_AGENT_SHA)
        self.assertEqual(lock["submodules"]["mini-swe-agent"]["commit"], MINI_SWE_AGENT_SHA)
        self.assertEqual(lock["dataset"]["revision"], DATASET_REVISION)
        self.assertEqual(lock["dataset"]["expected_rows"], FULL_TEST_ROWS)
        self.assertEqual(lock["dataset"]["required_image_field"], "dockerhub_tag")
        self.assertTrue(lock["contract"]["official_scorer_unchanged"])

    def test_dockerhub_tag_is_the_only_image_tag_source(self) -> None:
        row = {
            "instance_id": "instance_org__repo-deadbeef-v1",
            "repo": "org/repo",
            "dockerhub_tag": "org.repo-org__repo-deadbeef-v1",
        }
        self.assertEqual(
            official_image_uri(row),
            f"docker.io/{DOCKERHUB_NAMESPACE}:org.repo-org__repo-deadbeef-v1",
        )
        selected = select_rows([row], load_selection("smoke"))
        self.assertEqual(selected[0]["image_name"], official_image_uri(row))
        with self.assertRaisesRegex(ConfigError, "dockerhub_tag"):
            official_image_uri({"instance_id": "x", "repo": "org/repo"})

    def test_smoke_can_never_claim_score_or_threshold_pass(self) -> None:
        report = finalize_official_results(
            selection=load_selection("smoke"),
            expected_instance_ids=["instance_one"],
            eval_results={"instance_one": True},
        )
        self.assertEqual(report["status"], "complete")
        self.assertFalse(report["score_eligible"])
        self.assertIsNone(report["score_percent"])
        self.assertIsNone(report["acceptance_pass"])

    def test_full_score_requires_complete_public_coverage(self) -> None:
        ids = [f"instance_{index:04d}" for index in range(FULL_TEST_ROWS)]
        incomplete = finalize_official_results(
            selection=load_selection("full"),
            expected_instance_ids=ids,
            eval_results={key: True for key in ids[:-1]},
        )
        self.assertEqual(incomplete["status"], "incomplete")
        self.assertIsNone(incomplete["score_percent"])
        results = {key: index < 293 for index, key in enumerate(ids)}
        complete = finalize_official_results(
            selection=load_selection("full"), expected_instance_ids=ids, eval_results=results
        )
        self.assertEqual(complete["resolved_count"], 293)
        self.assertGreaterEqual(complete["score_percent"], 40.0)
        self.assertTrue(complete["acceptance_pass"])

    def test_agent_overlay_is_local_openai_compatible_and_not_cloud(self) -> None:
        overlay = validate_agent_overlay()
        rendered = build_runtime_agent_config({"agent": {"step_limit": 10}, "environment": {}}, overlay, "singularity")
        model = rendered["model"]
        self.assertEqual(model["model_name"], "openai/nvidia/GLM-5.2-NVFP4")
        self.assertEqual(model["model_kwargs"]["api_base"], "http://127.0.0.1:8000/v1")
        self.assertEqual(model["model_kwargs"]["api_key"], "local-vllm-no-auth")
        self.assertEqual(rendered["environment"]["environment_class"], "singularity")
        self.assertNotIn("api.openai.com", str(rendered))

    def test_container_preflight_fails_closed_for_non_docker_runtimes(self) -> None:
        ready = classify_container_preflight(
            docker_present=True,
            docker_usable=True,
            podman_present=False,
            apptainer_present=False,
            singularity_present=False,
        )
        self.assertTrue(ready["official_evaluation_supported"])
        blocked = classify_container_preflight(
            docker_present=False,
            docker_usable=False,
            podman_present=False,
            apptainer_present=True,
            singularity_present=False,
        )
        self.assertEqual(blocked["failure_class"], "OFFICIAL_EVALUATION_DOCKER_REQUIRED")
        self.assertFalse(blocked["official_evaluation_supported"])


class SwebenchProStageTests(unittest.TestCase):
    def test_heavy_stage_refuses_before_creating_artifacts_outside_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            root = Path(tmp_name) / "run"
            with self.assertRaisesRegex(ConfigError, E_SLURM_ALLOCATION_REQUIRED):
                run_restartable_stage(
                    stage="inference",
                    command=["/bin/true"],
                    artifact_root=root,
                    fingerprint="test-inference",
                    env={},
                )
            self.assertFalse(root.exists())

    def test_successful_stage_is_idempotently_resumed(self) -> None:
        calls = 0

        def runner(command):  # type: ignore[no-untyped-def]
            nonlocal calls
            calls += 1
            return 0

        with tempfile.TemporaryDirectory() as tmp_name:
            first = run_restartable_stage(
                stage="gather",
                command=["/bin/true"],
                artifact_root=tmp_name,
                fingerprint="same-inputs",
                env=ALLOCATION,
                runner=runner,
            )
            second = run_restartable_stage(
                stage="gather",
                command=["/bin/true"],
                artifact_root=tmp_name,
                fingerprint="same-inputs",
                env=ALLOCATION,
                runner=runner,
            )
        self.assertEqual(first.status, "passed")
        self.assertEqual(second.status, "skipped_complete")
        self.assertEqual(calls, 1)

    def test_partial_failure_is_marked_and_not_claimed_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            result = run_restartable_stage(
                stage="evaluate",
                command=["/bin/false"],
                artifact_root=tmp_name,
                fingerprint="failed-inputs",
                env=ALLOCATION,
                runner=lambda command: 17,
            )
            self.assertEqual(result.status, "failed")
            self.assertTrue(result.marker.name.endswith(".failed.json"))
            self.assertFalse((Path(tmp_name) / "markers" / "evaluate.complete.json").exists())

    def test_secret_bearing_stage_command_is_rejected_before_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            with self.assertRaisesRegex(ConfigError, E_SECRET_BEARING_COMMAND):
                run_restartable_stage(
                    stage="prepare",
                    command=["HF_TOKEN=should-not-log", "/bin/true"],
                    artifact_root=tmp_name,
                    fingerprint="secret-test",
                    env=ALLOCATION,
                )
            self.assertFalse((Path(tmp_name) / "markers").exists())


class SwebenchProSlurmTests(unittest.TestCase):
    def test_renderer_requests_exactly_one_node_and_four_h200s(self) -> None:
        rendered = render_baseline_job(
            site=BaselineSite.from_mapping(site_mapping()),
            project_url="https://github.com/Nier4Ryu/putpocket_dataset_mining.git",
            project_commit=PROJECT_SHA,
        )
        self.assertIn("#SBATCH --nodes=1", rendered)
        self.assertIn("#SBATCH --gres=gpu:H200:4", rendered)
        self.assertNotIn("#SBATCH --gpus-per-node=8", rendered)
        self.assertIn("PROFILE_PRIMARY=glm52_nvfp4_tp1_pcp4_ep", rendered)
        self.assertIn("PROFILE_FALLBACK=glm52_nvfp4_tp2_pcp2_ep", rendered)
        self.assertIn("dockerhub_username jefzda", rendered)
        self.assertIn("swe_bench_pro_eval.py", rendered)
        self.assertIn("dockerhub_tag", (ROOT / "configs/cluster/swebench_pro_sources.lock.yaml").read_text())
        self.assertNotRegex(rendered, r"(^|\s)(sbatch|salloc)(\s|$)")
        completed = subprocess.run(["bash", "-n"], input=rendered, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_runtime_preflight_precedes_all_clone_install_and_model_actions(self) -> None:
        rendered = render_baseline_job(
            site=BaselineSite.from_mapping(site_mapping()),
            project_url="https://github.com/Nier4Ryu/putpocket_dataset_mining.git",
            project_commit=PROJECT_SHA,
        )
        preflight = rendered.index("docker info")
        self.assertLess(preflight, rendered.index("fetch --depth=1"))
        self.assertLess(preflight, rendered.index("pip install"))
        self.assertLess(preflight, rendered.index("snapshot_download"))
        self.assertLess(preflight, rendered.index("vllm\" serve"))
        self.assertLess(preflight, rendered.index("swe_bench_pro_eval.py"))

    def test_preflight_only_job_is_fail_closed_and_shell_syntax_valid(self) -> None:
        rendered = render_baseline_job(
            site=BaselineSite.from_mapping(site_mapping()),
            project_url="https://github.com/Nier4Ryu/putpocket_dataset_mining.git",
            project_commit=PROJECT_SHA,
            preflight_only=True,
        )
        self.assertIn("exit 42", rendered)
        self.assertIn("JOB_STATUS=preflight_passed", rendered)
        self.assertNotIn("git clone", rendered)
        completed = subprocess.run(["bash", "-n"], input=rendered, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_non_exact_gpu_directive_is_rejected(self) -> None:
        mapping = site_mapping()
        mapping["site"]["h200_gpu_directive"] = "--gres=gpu:H200:8"
        with self.assertRaisesRegex(ConfigError, "exactly four H200"):
            BaselineSite.from_mapping(mapping)


if __name__ == "__main__":
    unittest.main()
