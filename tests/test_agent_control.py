from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining.agent_control import (
    AgentConfig,
    acquire_agent_locks,
    active_agent_locks,
    agent_lock_root,
    allocate_task_id,
    classify_source_ownership,
    classify_worktrees,
    detect_context,
    pending_agent_lock_requests,
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
        for suffix in ("glm52", "glm52_v025"):
            self.assertNotIn(f"Putpocket_env_{suffix}/", text)
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

    def test_agent_lock_acquire_release_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = AgentConfig(root / "repo", root / "worktrees", root / "artifacts", root / "repo/Putpocket_env", root / "repo/externals/vllm", root / "repo/externals/lmcache")
            lock_root = root / "locks"
            with patch.dict(os.environ, {"PUTPOCKET_AGENT_LOCK_ROOT": str(lock_root)}):
                self.assertEqual(agent_lock_root(cfg), lock_root)
                with acquire_agent_locks(cfg, ["build"], operation="test build"):
                    locks = active_agent_locks(cfg)
                    self.assertEqual(len(locks), 1)
                    self.assertEqual(locks[0]["resource"], "build")
                    self.assertEqual(locks[0]["operation"], "test build")
                    self.assertFalse(locks[0]["stale"])
                self.assertEqual(active_agent_locks(cfg), [])

    def test_agent_lock_conflict_records_pending_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = AgentConfig(root / "repo", root / "worktrees", root / "artifacts", root / "repo/Putpocket_env", root / "repo/externals/vllm", root / "repo/externals/lmcache")
            with patch.dict(os.environ, {"PUTPOCKET_AGENT_LOCK_ROOT": str(root / "locks")}):
                with acquire_agent_locks(cfg, ["build"], operation="first build"):
                    with self.assertRaises(SystemExit) as caught:
                        with acquire_agent_locks(cfg, ["build"], operation="second build", wait_seconds=0):
                            pass
                    self.assertIn("LOCK_HELD_PENDING_RECORDED", str(caught.exception))
                    pending = pending_agent_lock_requests(cfg)
                    self.assertEqual(len(pending), 1)
                    self.assertEqual(pending[0]["status"], "pending")
                    self.assertEqual(pending[0]["requested"]["operation"], "second build")
                    self.assertEqual(pending[0]["blocking"]["operation"], "first build")

    def test_lock_protocol_is_documented_for_agents(self) -> None:
        text = (self.repo_root / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("putpocket-agent locks status", text)
        self.assertIn("<git-common-dir>/putpocket-locks/", text)
        self.assertIn("putpocket-agent task start", text)

    def test_bootstrap_mutating_presets_acquire_build_lock(self) -> None:
        text = (self.repo_root / "src" / "putpocket_dataset_mining" / "bootstrap_sr.py").read_text(encoding="utf-8")
        self.assertIn('["canonical-runtime", "build"]', text)
        self.assertIn('operation="bootstrap runpod-dev build"', text)
        self.assertIn('operation="bootstrap server2 build"', text)
        self.assertIn("_run_server2_preset_locked(args, run, plan, resolved_arch)", text)

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
