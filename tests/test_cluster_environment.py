from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from putpocket_dataset_mining.cluster_environment import build_bootstrap_plan, execute_bootstrap_plan
from putpocket_dataset_mining.cluster_safety import E_SLURM_ALLOCATION_REQUIRED
from putpocket_dataset_mining.errors import ConfigError


ROOT = Path(__file__).resolve().parents[1]


class ClusterEnvironmentTests(unittest.TestCase):
    def test_bootstrap_plan_is_commit_addressed_and_sm90_only(self) -> None:
        plan = build_bootstrap_plan(
            lock_path=ROOT / "configs/env/cluster_h200_sm90_vllm026.lock.yaml",
            repository_root=ROOT,
            environment_root="/cluster/envs/glm52",
            vllm_source_root="/cluster/src/vllm-026",
            cache_root="/cluster/cache/glm52",
            python_executable="/opt/python/bin/python",
            uv_executable="/opt/uv/bin/uv",
            git_executable="/usr/bin/git",
            build_jobs=32,
        )
        payload = plan.as_dict()
        flattened = " ".join(arg for command in payload["commands"] for arg in command)
        self.assertIn("568afb3a13806beb53bb2e6bd518269357b237c0", flattened)
        self.assertIn("torch-2.11.0%2Bcu129", flattened)
        self.assertEqual(payload["environment"]["TORCH_CUDA_ARCH_LIST"], "9.0")
        self.assertNotIn("server2", flattened.lower())
        self.assertNotIn("runpod", flattened.lower())

    def test_execute_refuses_without_allocation_before_creating_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_name:
            base = Path(tmp_name)
            plan = build_bootstrap_plan(
                lock_path=ROOT / "configs/env/cluster_h200_sm90_vllm026.lock.yaml",
                repository_root=ROOT,
                environment_root=base / "env",
                vllm_source_root=base / "src/vllm",
                cache_root=base / "cache",
                python_executable="/opt/python/bin/python",
                uv_executable="/opt/uv/bin/uv",
                git_executable="/usr/bin/git",
                build_jobs=4,
            )
            with self.assertRaisesRegex(ConfigError, E_SLURM_ALLOCATION_REQUIRED):
                execute_bootstrap_plan(plan, env={})
            self.assertFalse((base / "env").exists())
            self.assertFalse((base / "src").exists())
            self.assertFalse((base / "cache").exists())


if __name__ == "__main__":
    unittest.main()
