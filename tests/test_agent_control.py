from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.agent_control import (
    AgentConfig,
    allocate_task_id,
    classify_source_ownership,
    classify_worktrees,
    detect_context,
    render_task,
    require_production_allowed,
    slugify,
)


class AgentControlTests(unittest.TestCase):
    repo_root = Path(__file__).resolve().parents[1]

    def test_slugify_and_task_id_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(slugify("Agent Worktree Control Plane!"), "agent-worktree-control-plane")
            with patch("putpocket_dataset_mining.agent_control._git_ref_exists", return_value=False):
                self.assertEqual(allocate_task_id("Agent Worktree Control Plane!", root=root, date="20260812"), "T20260812-001__agent-worktree-control-plane")
            (root / "T20260812-001__agent-worktree-control-plane").mkdir()
            with patch("putpocket_dataset_mining.agent_control._git_ref_exists", return_value=False):
                self.assertEqual(allocate_task_id("Agent Worktree Control Plane!", root=root, date="20260812"), "T20260812-002__agent-worktree-control-plane")

    def test_task_template_contains_runtime_mode_and_forbidden_paths(self) -> None:
        text = render_task(
            "T20260812-001__topic",
            "topic",
            "agent/T20260812-001__topic",
            Path("/tmp/wt"),
            "abc123",
            "shared-python-overlay",
        )
        self.assertIn("runtime mode: shared-python-overlay", text)
        self.assertIn("Putpocket_env/", text)
        self.assertIn("final handoff link", text)

    def test_activation_script_context_policy(self) -> None:
        text = (self.repo_root / "scripts" / "env" / "env_activate.sh").read_text(encoding="utf-8")
        self.assertIn("PUTPOCKET_EXECUTION_CONTEXT", text)
        self.assertIn("task-worktree", text)
        self.assertIn("PUTPOCKET_PRODUCTION_ALLOWED=\"0\"", text)
        self.assertNotIn("pip install", text)
        self.assertNotIn("uv pip install", text)
        self.assertNotIn("export CUDA_VISIBLE_DEVICES", text)

    def test_context_detection_for_task_worktree(self) -> None:
        cfg = AgentConfig(
            canonical_root=Path("/tmp/canonical"),
            worktree_root=Path("/tmp/worktrees"),
            artifact_root=Path("/tmp/artifacts"),
            active_env=Path("/tmp/canonical/Putpocket_env"),
            external_vllm_root=Path("/tmp/canonical/externals/vllm"),
            external_lmcache_root=Path("/tmp/canonical/externals/lmcache"),
        )
        with patch("putpocket_dataset_mining.agent_control.git_top_level", return_value=Path("/tmp/worktrees/T20260812-001__x")), \
            patch("putpocket_dataset_mining.agent_control.git") as mocked_git, \
            patch("putpocket_dataset_mining.agent_control.classify_source_ownership", return_value="TASK_OVERLAY_ACTIVE"):
            mocked_git.return_value.stdout = "abc123\n"
            mocked_git.return_value.returncode = 0
            info = detect_context(Path("/tmp/worktrees/T20260812-001__x"), cfg)
        self.assertEqual(info.execution_context, "task-worktree")
        self.assertFalse(info.production_allowed)

    def test_production_guard_blocks_task_context(self) -> None:
        cfg = AgentConfig(
            canonical_root=Path("/tmp/canonical"),
            worktree_root=Path("/tmp/worktrees"),
            artifact_root=Path("/tmp/artifacts"),
            active_env=Path("/tmp/canonical/Putpocket_env"),
            external_vllm_root=Path("/tmp/canonical/externals/vllm"),
            external_lmcache_root=Path("/tmp/canonical/externals/lmcache"),
        )
        with patch("putpocket_dataset_mining.agent_control.detect_context") as mocked:
            mocked.return_value.production_allowed = False
            mocked.return_value.execution_context = "task-worktree"
            with self.assertRaises(SystemExit):
                require_production_allowed("remote job submission", config=cfg)

    def test_worktree_audit_classifies_agent_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            canonical = root / "repo"
            subprocess.run(["git", "init", str(canonical)], check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "-C", str(canonical), "config", "user.email", "test@example.com"], check=True)
            subprocess.run(["git", "-C", str(canonical), "config", "user.name", "Test"], check=True)
            (canonical / "README.md").write_text("x\n", encoding="utf-8")
            subprocess.run(["git", "-C", str(canonical), "add", "README.md"], check=True)
            subprocess.run(["git", "-C", str(canonical), "commit", "-m", "init"], check=True, stdout=subprocess.PIPE)
            subprocess.run(["git", "-C", str(canonical), "branch", "-M", "master"], check=True)
            subprocess.run(["git", "-C", str(canonical), "update-ref", "refs/remotes/origin/master", "HEAD"], check=True)
            worktree_root = root / "worktrees"
            task = worktree_root / "T20260812-001__x"
            subprocess.run(["git", "-C", str(canonical), "worktree", "add", "-b", "agent/T20260812-001__x", str(task), "HEAD"], check=True, stdout=subprocess.PIPE)
            cfg = AgentConfig(canonical, worktree_root, root / "artifacts", canonical / "Putpocket_env", canonical / "externals/vllm", canonical / "externals/lmcache")
            rows = classify_worktrees(cfg)
        self.assertTrue(any(row["disposition"] == "ACTIVE_CURRENT_TASK" for row in rows))

    def test_source_ownership_can_report_task_overlay(self) -> None:
        cfg = AgentConfig(
            canonical_root=self.repo_root,
            worktree_root=self.repo_root.parent,
            artifact_root=self.repo_root / "data",
            active_env=self.repo_root / "Putpocket_env",
            external_vllm_root=self.repo_root / "externals/vllm",
            external_lmcache_root=self.repo_root / "externals/lmcache",
        )
        with patch("putpocket_dataset_mining.agent_control.source_ownership_details", return_value={
            "editable_worktree_leakage": False,
            "task_overlay_active": True,
            "putpocket_origin_ok": False,
        }):
            self.assertEqual(classify_source_ownership(self.repo_root, cfg), "TASK_OVERLAY_ACTIVE")


if __name__ == "__main__":
    unittest.main()
