from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from .config import dump_yaml, load_yaml
from .constants import DEFAULT_DOCKER_IMAGE, DEFAULT_MODEL_ID, REPO_ROOT, RUNS_ROOT, ensure_data_dirs
from .dataset import SourceTask, dataset_adapter_from_config, initial_workspace_files_for_task
from .docker_workspace import DockerImageManager, snapshot_workspace, workspace_from_execution_config
from .errors import DatasetMiningError, InfraError
from .execution_config import DockerBackend, ExecutionConfig
from .judge import CodexJudge, read_text_files
from .prompts import ChatTemplateRenderer, PromptPreparer
from .runtime import EpisodeTimeline, HeadlessClineRuntime, RolloutResult
from .serving import GenerationEngine, LocalVLLMEngine
from .storage import AttemptRecord, DatasetMaterializer, MiningIndex
from .verifier import HiddenVerifier, VerificationResult
from .ssh_transport import SshRsyncTransport


class SingleSampleRunner:
    def __init__(self, config: dict[str, Any], config_path: Path | None = None) -> None:
        self.config = config
        self.config_path = config_path
        self.execution_config = ExecutionConfig.from_env_and_mapping(config.get("execution", {}))
        self._remote_preflight_passed = False

    @classmethod
    def from_config_path(cls, path: str | Path) -> "SingleSampleRunner":
        config_path = Path(path)
        return cls(load_yaml(config_path), config_path=config_path)

    def run(
        self,
        sample_index: int = 0,
        split: str | None = None,
        run_id: str | None = None,
        attempt_id: str | None = None,
        write_index: bool = True,
        engine: GenerationEngine | None = None,
        dataset_version: str = "single_sample",
        gpu_devices: list[int] | None = None,
    ) -> dict[str, Any]:
        adapter = dataset_adapter_from_config(self.config)
        task = adapter.get_by_flat_index(sample_index, split=split)
        return self.run_task(
            task=task,
            run_id=run_id,
            attempt_id=attempt_id,
            write_index=write_index,
            engine=engine,
            dataset_version=dataset_version,
            gpu_devices=gpu_devices,
        )

    def run_task(
        self,
        task: SourceTask,
        run_id: str | None = None,
        attempt_id: str | None = None,
        write_index: bool = True,
        engine: GenerationEngine | None = None,
        dataset_version: str = "single_sample",
        gpu_devices: list[int] | None = None,
    ) -> dict[str, Any]:
        ensure_data_dirs()
        run_id = run_id or f"run_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        attempt_id = attempt_id or f"attempt_{uuid.uuid4().hex[:12]}"
        output_root_cfg = self.config.get("run", {}).get("output_root")
        output_root = Path(output_root_cfg) if output_root_cfg else RUNS_ROOT
        if not output_root.is_absolute():
            output_root = REPO_ROOT / output_root
        attempt_dir = output_root / run_id / "samples" / task.sample_id / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        timeline = EpisodeTimeline(attempt_dir)
        timeline.append("attempt.start", f"Starting sample {task.sample_id}.")

        summary: dict[str, Any] = {
            "run_id": run_id,
            "dataset_version": dataset_version,
            "sample_id": task.sample_id,
            "split": task.split,
            "row_index": task.row_index,
            "task_id": task.task_id,
            "attempt_id": attempt_id,
            "artifact_path": str(attempt_dir),
            "final_status": "failed_infra",
            "failure_class": None,
            "query1": None,
            "query2": None,
            "policy_delta": None,
        }

        try:
            self.execution_config.validate_for_evaluation_start()
            self._write_static_artifacts(attempt_dir, task)
            docker_image = self._docker_image()
            if self.execution_config.workspace_backend == DockerBackend.LOCAL_DOCKER or self.execution_config.verifier_backend == DockerBackend.LOCAL_DOCKER:
                DockerImageManager(docker_image, self._dockerfile()).ensure_image(
                    build_if_missing=bool(self.config.get("docker", {}).get("build_if_missing", True)),
                    timeout_sec=int(self.config.get("docker", {}).get("timeouts", {}).get("image_build_sec", 900)),
                )
            self._preflight_remote_verifier_if_needed(docker_image)

            workspace_dir = attempt_dir / "workspace"
            self._create_initial_workspace(workspace_dir, task)
            snapshot_workspace(workspace_dir, attempt_dir / "workspace_snapshots" / "initial")

            model_id = self.config.get("model", {}).get("generation_model_id", DEFAULT_MODEL_ID)
            renderer = ChatTemplateRenderer(model_id=model_id)
            preparer = PromptPreparer(
                attempt_dir=attempt_dir,
                model_id=model_id,
                profile=self.config.get("prompt", {}).get("cline_prompt_profile", "compact"),
                mining_seed=int(self.config.get("run", {}).get("mining_seed", 42)),
                renderer=renderer,
            )
            messages1, query1 = preparer.prepare_history1(task)
            summary["query1"] = query1

            engine = engine or LocalVLLMEngine(model_id=model_id, gpu_devices=gpu_devices)
            docker_cfg = self.config.get("docker", {})
            timeouts = docker_cfg.get("timeouts", {})
            with workspace_from_execution_config(
                host_workspace=workspace_dir,
                image=docker_image,
                cpus=docker_cfg.get("cpus", 8),
                memory=docker_cfg.get("memory", "8g"),
                startup_timeout_sec=int(timeouts.get("container_startup_sec", 120)),
                execution_config=self.execution_config,
            ) as workspace:
                runtime = HeadlessClineRuntime(
                    attempt_dir=attempt_dir,
                    renderer=renderer,
                    engine=engine,
                    workspace=workspace,
                    timeline=timeline,
                    max_parse_failures=int(self.config.get("history", {}).get("max_parse_failures_per_history", 3)),
                    per_generation_timeout_sec=int(timeouts.get("per_generation_sec", 300)),
                    per_tool_timeout_sec=int(timeouts.get("per_tool_command_sec", 120)),
                )
                history1 = runtime.run_history(
                    "history1",
                    messages1,
                    max_turns=int(self.config.get("history", {}).get("history1_max_turns", 30)),
                )
                self._write_rollout_artifact(attempt_dir, history1)
                snapshot_workspace(workspace_dir, attempt_dir / "workspace_snapshots" / "after_history1")

                if not history1.completed:
                    summary["failure_class"] = history1.failure_class
                    self._write_skipped_history2_artifacts(attempt_dir, "history1 rollout did not complete")
                    self._write_skipped_verification(attempt_dir, "history1", history1.failure_class or "history1.rollout.failed")
                    self._write_skipped_verification(attempt_dir, "history2", "history2.skipped")
                    summary["final_status"] = "rejected"
                    return self._finalize(attempt_dir, summary, task, write_index)

                verifier = self._verifier(attempt_dir, docker_image)
                verification1 = verifier.verify("history1", attempt_dir / "workspace_snapshots" / "after_history1", task)
                if not verification1.passed:
                    summary["failure_class"] = verification1.failure_class
                    self._write_skipped_history2_artifacts(attempt_dir, "history1 verification did not pass")
                    self._write_skipped_verification(attempt_dir, "history2", "history2.skipped")
                    CodexJudge(attempt_dir).write_skipped("history1 verification did not pass")
                    summary["final_status"] = "rejected"
                    return self._finalize(attempt_dir, summary, task, write_index)

                messages2, query2, policy_delta = preparer.prepare_history2(task, query1, history1.messages)
                summary["query2"] = query2
                summary["policy_delta"] = policy_delta
                history2 = runtime.run_history(
                    "history2",
                    messages2,
                    max_turns=int(self.config.get("history", {}).get("history2_max_turns", 30)),
                )
                self._write_rollout_artifact(attempt_dir, history2)
                snapshot_workspace(workspace_dir, attempt_dir / "workspace_snapshots" / "after_history2")

            if not history2.completed:
                summary["failure_class"] = history2.failure_class
                self._write_skipped_verification(attempt_dir, "history2", history2.failure_class or "history2.rollout.failed")
                CodexJudge(attempt_dir).write_skipped("history2 rollout did not complete")
                summary["final_status"] = "rejected"
                return self._finalize(attempt_dir, summary, task, write_index)

            verifier = self._verifier(attempt_dir, docker_image)
            verification2 = verifier.verify("history2", attempt_dir / "workspace_snapshots" / "after_history2", task)

            final_status, failure_class = self._judge_and_decide(attempt_dir, verification1, verification2)
            summary["final_status"] = final_status
            summary["failure_class"] = failure_class
            return self._finalize(attempt_dir, summary, task, write_index)

        except DatasetMiningError as exc:
            summary["final_status"] = "failed_infra"
            summary["failure_class"] = self._failure_class_for_exception(exc)
            summary["error"] = str(exc)
            timeline.append("attempt.failed", str(exc), {"failure_class": summary["failure_class"]})
            self._ensure_failure_placeholders(attempt_dir, summary["failure_class"])
            return self._finalize(attempt_dir, summary, task, write_index)
        except Exception as exc:  # noqa: BLE001 - record unexpected blocker in artifacts.
            summary["final_status"] = "failed_infra"
            summary["failure_class"] = "infra.unexpected_error"
            summary["error"] = f"{exc.__class__.__name__}: {exc}"
            timeline.append("attempt.failed", summary["error"], {"failure_class": summary["failure_class"]})
            self._ensure_failure_placeholders(attempt_dir, summary["failure_class"])
            return self._finalize(attempt_dir, summary, task, write_index)

    def _write_static_artifacts(self, attempt_dir: Path, task: SourceTask) -> None:
        task.write_json(attempt_dir / "source_task.json")
        if self.config_path is not None:
            shutil.copy2(self.config_path, attempt_dir / "scenario_config.yaml")
        else:
            dump_yaml(self.config, attempt_dir / "scenario_config.yaml")
        for path in [
            attempt_dir / "prepared",
            attempt_dir / "serving",
            attempt_dir / "trajectories",
            attempt_dir / "workspace_snapshots",
            attempt_dir / "verification",
            attempt_dir / "judge",
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _create_initial_workspace(self, workspace_dir: Path, task: SourceTask) -> None:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        initial_files = initial_workspace_files_for_task(task, self.config)
        for rel_path, content in initial_files.items():
            target = workspace_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
        clinerules = workspace_dir / ".clinerules"
        if clinerules.exists():
            clinerules.unlink()
        tests_dir = workspace_dir / "tests"
        if tests_dir.exists():
            shutil.rmtree(tests_dir)

    def _docker_image(self) -> str:
        return self.config.get("docker", {}).get("image", DEFAULT_DOCKER_IMAGE)

    def _dockerfile(self) -> Path:
        path = Path(self.config.get("docker", {}).get("dockerfile", "docker/default_python/Dockerfile"))
        return path if path.is_absolute() else REPO_ROOT / path

    def _verifier(self, attempt_dir: Path, docker_image: str) -> HiddenVerifier:
        docker_cfg = self.config.get("docker", {})
        verifier_cfg = self.config.get("verifier", {})
        timeout_sec = verifier_cfg.get("timeout_sec") or self.execution_config.verifier_timeout_sec
        return HiddenVerifier(
            attempt_dir=attempt_dir,
            docker_image=docker_image,
            cpus=docker_cfg.get("cpus", 8),
            memory=docker_cfg.get("memory", "8g"),
            test_command=verifier_cfg.get("test_command", "pytest -q tests/test_solution.py"),
            timeout_sec=int(timeout_sec),
            execution_config=self.execution_config,
        )

    def _preflight_remote_verifier_if_needed(self, docker_image: str) -> None:
        if self.execution_config.verifier_backend != DockerBackend.REMOTE_SSH_DOCKER or self._remote_preflight_passed:
            return
        result = SshRsyncTransport(self.execution_config.remote).lightweight_preflight(docker_image)
        if result.status != "REMOTE_DOCKER_PREFLIGHT_PASSED":
            raise InfraError(f"REMOTE_DOCKER_PREFLIGHT_FAILED: {result.detail or result.error_class or 'unknown remote preflight failure'}")
        self._remote_preflight_passed = True

    def _judge_and_decide(
        self,
        attempt_dir: Path,
        verification1: VerificationResult,
        verification2: VerificationResult,
    ) -> tuple[str, str | None]:
        judge = CodexJudge(attempt_dir)
        if not verification1.passed or not verification2.passed:
            reason = "unit tests failed; judge skipped"
            judge.write_skipped(reason)
            return "rejected", verification1.failure_class or verification2.failure_class
        if not bool(self.config.get("judge", {}).get("enabled", True)):
            judge.write_skipped("judge disabled by configuration")
            return "accepted", None

        prepared = attempt_dir / "prepared"
        result = judge.run(
            cline_rules_v1=(prepared / "cline_rules_v1.md").read_text(encoding="utf-8"),
            files_after_history1=read_text_files(attempt_dir / "workspace_snapshots" / "after_history1"),
            cline_rules_v2=(prepared / "cline_rules_v2.md").read_text(encoding="utf-8"),
            query2=(prepared / "query2.txt").read_text(encoding="utf-8"),
            files_after_history2=read_text_files(attempt_dir / "workspace_snapshots" / "after_history2"),
            history2_unit_test_summary=verification2.to_dict(),
        )
        if result.failure_class == "judge.cli_error":
            return "failed_infra", result.failure_class
        if result.decision == "pass":
            return "accepted", None
        if result.decision == "fail":
            return "rejected", "judge.failed"
        return "uncertain", result.failure_class or "judge.uncertain"

    def _write_rollout_artifact(self, attempt_dir: Path, result: RolloutResult) -> None:
        path = attempt_dir / "trajectories" / f"{result.history_name}_rollout_summary.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def _write_skipped_history2_artifacts(self, attempt_dir: Path, reason: str) -> None:
        prepared = attempt_dir / "prepared"
        for name in ["messages_history2.json", "tokenization_history2.json"]:
            (prepared / name).write_text(json.dumps({"skipped": True, "reason": reason}, indent=2), encoding="utf-8")
        (prepared / "rendered_prompt_history2.txt").write_text(f"SKIPPED: {reason}\n", encoding="utf-8")
        (attempt_dir / "trajectories" / "history2_trajectory.jsonl").write_text("", encoding="utf-8")
        initial = attempt_dir / "workspace_snapshots" / "after_history1"
        target = attempt_dir / "workspace_snapshots" / "after_history2"
        if initial.exists() and not target.exists():
            shutil.copytree(initial, target)

    def _write_skipped_verification(self, attempt_dir: Path, stage: str, failure_class: str) -> None:
        verification_dir = attempt_dir / "verification" / stage
        verification_dir.mkdir(parents=True, exist_ok=True)
        checklist = {
            "stage": stage,
            "checks": [{"name": "hidden_mbpp_pytest", "passed": False, "skipped": True}],
            "final_status": "skipped",
            "failure_class": failure_class,
        }
        (verification_dir / "checklist.json").write_text(json.dumps(checklist, indent=2, sort_keys=True), encoding="utf-8")

    def _ensure_failure_placeholders(self, attempt_dir: Path, failure_class: str | None) -> None:
        prepared = attempt_dir / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        for history in ["history1", "history2"]:
            messages_path = prepared / f"messages_{history}.json"
            rendered_path = prepared / f"rendered_prompt_{history}.txt"
            tokenization_path = prepared / f"tokenization_{history}.json"
            if not messages_path.exists():
                messages_path.write_text(json.dumps({"failed_before_prepare": True, "failure_class": failure_class}, indent=2), encoding="utf-8")
            if not rendered_path.exists():
                rendered_path.write_text(f"FAILED BEFORE RENDER: {failure_class}\n", encoding="utf-8")
            if not tokenization_path.exists():
                tokenization_path.write_text(json.dumps({"failed_before_tokenization": True, "failure_class": failure_class}, indent=2), encoding="utf-8")
            (attempt_dir / "trajectories" / f"{history}_trajectory.jsonl").parent.mkdir(parents=True, exist_ok=True)
            (attempt_dir / "trajectories" / f"{history}_trajectory.jsonl").touch(exist_ok=True)
            if not (attempt_dir / "verification" / history / "checklist.json").exists():
                self._write_skipped_verification(attempt_dir, history, failure_class or f"{history}.skipped")
        judge = CodexJudge(attempt_dir)
        if not (attempt_dir / "judge" / "judge_decision.json").exists():
            judge.write_skipped(failure_class or "failed before judge")

    def _failure_class_for_exception(self, exc: DatasetMiningError) -> str:
        if isinstance(exc, InfraError):
            text = str(exc).lower()
            if "docker" in text:
                return "infra.docker_start_failed"
            if "vllm" in text or "tokenizer" in text:
                return "infra.vllm_generation_failed"
        return "infra.filesystem_permission_error"

    def _finalize(self, attempt_dir: Path, summary: dict[str, Any], task: SourceTask, write_index: bool) -> dict[str, Any]:
        summary["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (attempt_dir / "episode_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if write_index:
            index = MiningIndex.default()
            index.record_attempt(
                AttemptRecord(
                    run_id=summary["run_id"],
                    dataset_version=summary["dataset_version"],
                    sample_id=task.sample_id,
                    split=task.split,
                    row_index=task.row_index,
                    task_id=task.task_id,
                    attempt_id=summary["attempt_id"],
                    final_status=summary["final_status"],
                    failure_class=summary.get("failure_class"),
                    artifact_path=str(attempt_dir),
                    summary=summary,
                )
            )
            if summary["final_status"] == "accepted":
                DatasetMaterializer(index).materialize_dataset(summary["dataset_version"])
        return summary
