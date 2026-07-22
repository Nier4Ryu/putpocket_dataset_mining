from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import shutil
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from putpocket_dataset_mining.config import dump_yaml, load_yaml
from putpocket_dataset_mining.constants import (
    ALLOWED_CUDA_DEVICES,
    BUILD_ENV_OVERRIDES,
    DEFAULT_DOCKERFILE,
    DEFAULT_DOCKER_IMAGE,
    GLM52_08B_MODEL_ID,
    MODEL_EVALUATION_RUNS_ROOT,
    REPO_ROOT,
    ensure_model_evaluation_dirs,
)
from putpocket_dataset_mining.dataset import SourceTask
from putpocket_dataset_mining.docker_workspace import DockerImageManager, DockerWorkspace, snapshot_workspace
from putpocket_dataset_mining.errors import ConfigError, DatasetMiningError, InfraError
from putpocket_dataset_mining.judge import CodexJudge, read_text_files
from putpocket_dataset_mining.jsonl import append_jsonl, write_jsonl
from putpocket_dataset_mining.model_evaluation.dataset_loader import AcceptedDatasetSample, load_accepted_samples
from putpocket_dataset_mining.prompts import (
    CLINE_RULES_V1,
    COMPACT_CLINE_TOOL_INSTRUCTIONS,
    FULL_CLINE_TOOL_INSTRUCTIONS,
    POLICY_DELTAS,
    ChatTemplateRenderer,
    Message,
)
from putpocket_dataset_mining.runtime import EpisodeTimeline, HeadlessClineRuntime, RolloutResult
from putpocket_dataset_mining.serving import GenerationEngine, LocalVLLMEngine
from putpocket_dataset_mining.verifier import HiddenVerifier, VerificationResult


EVALUATION_MODE = "full_two_turn_glm_rerun"


class EvalPromptPreparer:
    def __init__(
        self,
        attempt_dir: Path,
        model_id: str,
        prompt_profile: str,
        renderer: ChatTemplateRenderer,
    ) -> None:
        self.prepared_dir = attempt_dir / "prepared_glm"
        self.model_id = model_id
        self.prompt_profile = prompt_profile
        self.renderer = renderer
        self.prepared_dir.mkdir(parents=True, exist_ok=True)

    def prepare_history1(self, sample: AcceptedDatasetSample) -> list[Message]:
        query1 = str(sample.row["query1"])
        system_prompt = self._tool_instructions() + "\n\n" + CLINE_RULES_V1
        (self.prepared_dir / "system_prompt_1.md").write_text(system_prompt, encoding="utf-8")
        (self.prepared_dir / "cline_rules_v1.md").write_text(CLINE_RULES_V1, encoding="utf-8")
        (self.prepared_dir / "query1.txt").write_text(query1, encoding="utf-8")
        messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query1},
        ]
        self.save_messages("history1", messages)
        return messages

    def prepare_history2(
        self,
        sample: AcceptedDatasetSample,
        history1_messages: list[Message],
    ) -> list[Message]:
        query1 = str(sample.row["query1"])
        query2 = str(sample.row["query2"])
        policy_delta = str(sample.row["policy_delta"])
        if policy_delta not in POLICY_DELTAS:
            raise ConfigError(f"Unknown policy_delta for {sample.sample_id}: {policy_delta}")
        rules_v2 = CLINE_RULES_V1 + "\n\n" + POLICY_DELTAS[policy_delta]
        system_prompt = self._tool_instructions() + "\n\n" + rules_v2
        (self.prepared_dir / "system_prompt_2.md").write_text(system_prompt, encoding="utf-8")
        (self.prepared_dir / "cline_rules_v2.md").write_text(rules_v2, encoding="utf-8")
        (self.prepared_dir / "query2.txt").write_text(query2, encoding="utf-8")
        (self.prepared_dir / "query2_metadata.json").write_text(
            json.dumps(
                {
                    "policy_delta": policy_delta,
                    "generated_by_mining_loop": True,
                    "source": "accepted_dataset_row",
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        messages: list[Message] = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": query1})
        for message in history1_messages:
            if message.get("role") != "system":
                messages.append(message)
        messages.append({"role": "user", "content": query2})
        self.save_messages("history2", messages)
        return messages

    def save_messages(self, history_name: str, messages: list[Message]) -> None:
        (self.prepared_dir / f"messages_{history_name}.json").write_text(
            json.dumps(messages, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def write_skipped_history2(self, reason: str) -> None:
        (self.prepared_dir / "messages_history2.json").write_text(
            json.dumps({"skipped": True, "reason": reason}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (self.prepared_dir / "rendered_prompt_history2_skipped.txt").write_text(f"SKIPPED: {reason}\n", encoding="utf-8")
        (self.prepared_dir / "tokenization_history2_skipped.json").write_text(
            json.dumps({"skipped": True, "reason": reason}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _tool_instructions(self) -> str:
        return FULL_CLINE_TOOL_INSTRUCTIONS if self.prompt_profile == "full" else COMPACT_CLINE_TOOL_INSTRUCTIONS


class GLMSampleEvaluator:
    def __init__(
        self,
        run_config: dict[str, Any],
        engine: GenerationEngine | None = None,
        gpu_slot: list[int] | None = None,
    ) -> None:
        self.run_config = run_config
        self.single_config = dict(run_config["single_config"])
        self.run_root = Path(run_config["run_root"])
        self.engine = engine
        self.gpu_slot = gpu_slot or []

    def evaluate(self, sample: AcceptedDatasetSample) -> dict[str, Any]:
        attempt_id = f"attempt_{uuid.uuid4().hex[:12]}"
        attempt_dir = self.run_root / "per_sample" / sample.sample_id / attempt_id
        attempt_dir.mkdir(parents=True, exist_ok=True)
        timeline = EpisodeTimeline(attempt_dir)
        result = self._initial_result(sample, attempt_id, attempt_dir)
        started = time.time()
        current_stage = "prepare"
        timeline.append("eval_sample.start", f"Starting GLM evaluation for {sample.sample_id}.")

        try:
            self._write_static_artifacts(sample, attempt_dir)
            if sample.missing_artifacts:
                (attempt_dir / "source_artifact_reference.json").write_text(
                    json.dumps(self._source_reference(sample), indent=2, sort_keys=True),
                    encoding="utf-8",
                )

            docker_image = self._docker_image()
            workspace_dir = attempt_dir / "workspace"
            self._create_initial_workspace(workspace_dir)
            snapshot_workspace(workspace_dir, attempt_dir / "workspace_snapshots" / "initial")

            model_id = str(self.run_config["model_id"])
            renderer = ChatTemplateRenderer(model_id=model_id)
            prompt_preparer = EvalPromptPreparer(
                attempt_dir=attempt_dir,
                model_id=model_id,
                prompt_profile=str(self.run_config["prompt_profile"]),
                renderer=renderer,
            )
            messages1 = prompt_preparer.prepare_history1(sample)
            engine = self.engine or self._engine()
            docker_cfg = self.single_config.get("docker", {})
            timeouts = docker_cfg.get("timeouts", {})

            with DockerWorkspace(
                host_workspace=workspace_dir,
                image=docker_image,
                cpus=docker_cfg.get("cpus", 8),
                memory=docker_cfg.get("memory", "8g"),
                startup_timeout_sec=int(timeouts.get("container_startup_sec", 120)),
            ) as workspace:
                current_stage = "history1_rollout"
                runtime = HeadlessClineRuntime(
                    attempt_dir=attempt_dir,
                    renderer=renderer,
                    engine=engine,
                    workspace=workspace,
                    timeline=timeline,
                    max_parse_failures=int(self.single_config.get("history", {}).get("max_parse_failures_per_history", 3)),
                    per_generation_timeout_sec=int(timeouts.get("per_generation_sec", 300)),
                    per_tool_timeout_sec=int(timeouts.get("per_tool_command_sec", 120)),
                    max_tokens=int(self.run_config["max_tokens"]),
                    generation_seed=int(self.run_config["evaluation_seed"]),
                )
                history1 = runtime.run_history(
                    "history1",
                    messages1,
                    max_turns=int(self.single_config.get("history", {}).get("history1_max_turns", 30)),
                )
                self._write_rollout_artifact(attempt_dir, history1)
                prompt_preparer.save_messages("history1_final", history1.messages)
                snapshot_workspace(workspace_dir, attempt_dir / "workspace_snapshots" / "after_history1")

                if not history1.completed:
                    result["history1_status"] = "rollout_failed"
                    result["history1_failure_class"] = history1.failure_class
                    result["history1_turns"] = history1.turns
                    self._skip_history2(attempt_dir, prompt_preparer, "history1 rollout did not complete", result)
                    result["final_status"] = "failed"
                    result["failure_stage"] = "history1_rollout"
                    return self._finalize_sample(attempt_dir, result, started)

                current_stage = "history1_verification"
                verifier = self._verifier(attempt_dir, docker_image)
                verification1 = verifier.verify("history1", attempt_dir / "workspace_snapshots" / "after_history1", sample.source_task)
                result["history1_status"] = "verification_passed" if verification1.passed else "verification_failed"
                result["history1_failure_class"] = verification1.failure_class
                result["history1_turns"] = history1.turns
                if not verification1.passed:
                    self._skip_history2(attempt_dir, prompt_preparer, verification1.failure_class or "history1 verification failed", result)
                    result["final_status"] = "failed"
                    result["failure_stage"] = "history1_verification"
                    return self._finalize_sample(attempt_dir, result, started)

                current_stage = "history2_prepare"
                messages2 = prompt_preparer.prepare_history2(sample, history1.messages)
                current_stage = "history2_rollout"
                history2 = runtime.run_history(
                    "history2",
                    messages2,
                    max_turns=int(self.single_config.get("history", {}).get("history2_max_turns", 30)),
                )
                self._write_rollout_artifact(attempt_dir, history2)
                prompt_preparer.save_messages("history2_final", history2.messages)
                snapshot_workspace(workspace_dir, attempt_dir / "workspace_snapshots" / "after_history2")

            if not history2.completed:
                result["history2_status"] = "rollout_failed"
                result["history2_failure_class"] = history2.failure_class
                result["history2_turns"] = history2.turns
                self._write_skipped_verification(attempt_dir, "history2", history2.failure_class or "history2.rollout.failed")
                CodexJudge(attempt_dir).write_skipped("history2 rollout did not complete")
                result["final_status"] = "failed"
                result["failure_stage"] = "history2_rollout"
                return self._finalize_sample(attempt_dir, result, started)

            current_stage = "history2_verification"
            verifier = self._verifier(attempt_dir, docker_image)
            verification2 = verifier.verify("history2", attempt_dir / "workspace_snapshots" / "after_history2", sample.source_task)
            result["history2_status"] = "verification_passed" if verification2.passed else "verification_failed"
            result["history2_failure_class"] = verification2.failure_class
            result["history2_turns"] = history2.turns
            if not verification2.passed:
                CodexJudge(attempt_dir).write_skipped("unit tests failed; judge skipped")
                result["final_status"] = "failed"
                result["failure_stage"] = "history2_verification"
                return self._finalize_sample(attempt_dir, result, started)

            current_stage = "judge"
            judge_result = self._run_judge(attempt_dir, verification2)
            result["judge_decision"] = judge_result.decision
            result["judge_failure_class"] = judge_result.failure_class
            if judge_result.failure_class == "judge.cli_error":
                result["final_status"] = "failed_infra"
                result["failure_stage"] = "judge"
            elif judge_result.decision == "pass":
                result["final_status"] = "succeeded"
                result["failure_stage"] = None
            elif judge_result.decision == "fail":
                result["final_status"] = "failed"
                result["failure_stage"] = "judge"
            else:
                result["final_status"] = "uncertain"
                result["failure_stage"] = "judge"
            return self._finalize_sample(attempt_dir, result, started)

        except DatasetMiningError as exc:
            failure_class = self._failure_class_for_exception(exc)
            result["final_status"] = "failed_infra"
            result["failure_stage"] = "infra"
            self._record_infra_failure_stage(result, current_stage, failure_class)
            result["error"] = str(exc)
            timeline.append("eval_sample.failed", str(exc), {"failure_class": failure_class})
            self._ensure_failure_placeholders(attempt_dir, failure_class)
            return self._finalize_sample(attempt_dir, result, started)
        except Exception as exc:  # noqa: BLE001 - preserve unexpected failure as an artifact.
            failure_class = "infra.unexpected_error"
            result["final_status"] = "failed_infra"
            result["failure_stage"] = "infra"
            self._record_infra_failure_stage(result, current_stage, failure_class)
            result["error"] = f"{exc.__class__.__name__}: {exc}"
            timeline.append("eval_sample.failed", result["error"], {"failure_class": failure_class})
            self._ensure_failure_placeholders(attempt_dir, failure_class)
            return self._finalize_sample(attempt_dir, result, started)

    def _initial_result(self, sample: AcceptedDatasetSample, attempt_id: str, attempt_dir: Path) -> dict[str, Any]:
        return {
            "eval_run_id": self.run_config["run_id"],
            "attempt_id": attempt_id,
            "sample_id": sample.sample_id,
            "task_id": sample.task_id,
            "source_dataset_version": sample.dataset_version,
            "source_dataset_row_number": sample.row_number,
            "source_artifact_path": str(sample.source_artifact_path),
            "target_model": self.run_config["model_id"],
            "evaluation_mode": EVALUATION_MODE,
            "history1_status": "not_started",
            "history1_failure_class": None,
            "history2_status": "not_started",
            "history2_failure_class": None,
            "judge_decision": "not_run",
            "judge_failure_class": None,
            "final_status": "failed_infra",
            "failure_stage": None,
            "history1_turns": 0,
            "history2_turns": 0,
            "prompt_token_counts": {"history1": [], "history2": [], "total": 0},
            "completion_token_counts": {"history1": [], "history2": [], "total": 0},
            "latency": {"history1_sec": 0.0, "history2_sec": 0.0, "total_generation_sec": 0.0, "sample_wall_sec": 0.0},
            "artifact_path": str(attempt_dir),
            "source_missing_artifacts": sample.missing_artifacts,
        }

    def _write_static_artifacts(self, sample: AcceptedDatasetSample, attempt_dir: Path) -> None:
        for rel in [
            "prepared_glm",
            "serving",
            "trajectories",
            "workspace_snapshots",
            "verification",
            "judge",
        ]:
            (attempt_dir / rel).mkdir(parents=True, exist_ok=True)
        dump_yaml(self._sample_eval_config(sample), attempt_dir / "eval_config.yaml")
        (attempt_dir / "source_dataset_row.json").write_text(
            json.dumps(sample.row, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (attempt_dir / "source_artifact_reference.json").write_text(
            json.dumps(self._source_reference(sample), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        sample.source_task.write_json(attempt_dir / "source_task.json")

    def _source_reference(self, sample: AcceptedDatasetSample) -> dict[str, Any]:
        return {
            "source_artifact_path": str(sample.source_artifact_path),
            "source_artifact_is_read_only_input": True,
            "source_dataset_version": sample.dataset_version,
            "source_accepted_path": str(sample.accepted_path),
            "source_attempt_id": sample.row.get("attempt_id"),
            "source_final_status": sample.row.get("final_status"),
            "missing_artifacts": sample.missing_artifacts,
        }

    def _sample_eval_config(self, sample: AcceptedDatasetSample) -> dict[str, Any]:
        return {
            "run_id": self.run_config["run_id"],
            "sample_id": sample.sample_id,
            "dataset_version": sample.dataset_version,
            "model_id": self.run_config["model_id"],
            "backend": "local_vllm_python_engine",
            "evaluation_mode": EVALUATION_MODE,
            "evaluation_seed": self.run_config["evaluation_seed"],
            "gpu_slot": self.gpu_slot,
            "tensor_parallel_size": self.run_config["tensor_parallel_size"],
            "pipeline_parallel_size": self.run_config["pipeline_parallel_size"],
            "decoding": self.run_config["decoding"],
            "prompt_rendering": self.run_config["prompt_rendering"],
            "source_artifact_path": str(sample.source_artifact_path),
        }

    def _create_initial_workspace(self, workspace_dir: Path) -> None:
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        initial_files = self.single_config.get("workspace", {}).get(
            "initial_files",
            {"solution.py": "# TODO: implement the required function.\n"},
        )
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
        return self.single_config.get("docker", {}).get("image", DEFAULT_DOCKER_IMAGE)

    def _verifier(self, attempt_dir: Path, docker_image: str) -> HiddenVerifier:
        docker_cfg = self.single_config.get("docker", {})
        verifier_cfg = self.single_config.get("verifier", {})
        return HiddenVerifier(
            attempt_dir=attempt_dir,
            docker_image=docker_image,
            cpus=docker_cfg.get("cpus", 8),
            memory=docker_cfg.get("memory", "8g"),
            test_command=verifier_cfg.get("test_command", "pytest -q tests/test_solution.py"),
            timeout_sec=int(docker_cfg.get("timeouts", {}).get("per_tool_command_sec", 120)),
        )

    def _engine(self) -> LocalVLLMEngine:
        return LocalVLLMEngine(
            model_id=str(self.run_config["model_id"]),
            gpu_devices=self.gpu_slot or [ALLOWED_CUDA_DEVICES[0]],
            tensor_parallel_size=int(self.run_config["tensor_parallel_size"]),
            pipeline_parallel_size=int(self.run_config["pipeline_parallel_size"]),
            max_model_len=int(self.run_config["max_model_len"]),
            gpu_memory_utilization=float(self.run_config["gpu_memory_utilization"]),
            max_num_seqs=1,
            enforce_eager=True,
        )

    def _run_judge(self, attempt_dir: Path, verification2: VerificationResult) -> Any:
        prepared = attempt_dir / "prepared_glm"
        return CodexJudge(attempt_dir).run(
            cline_rules_v1=(prepared / "cline_rules_v1.md").read_text(encoding="utf-8"),
            files_after_history1=read_text_files(attempt_dir / "workspace_snapshots" / "after_history1"),
            cline_rules_v2=(prepared / "cline_rules_v2.md").read_text(encoding="utf-8"),
            query2=(prepared / "query2.txt").read_text(encoding="utf-8"),
            files_after_history2=read_text_files(attempt_dir / "workspace_snapshots" / "after_history2"),
            history2_unit_test_summary=verification2.to_dict(),
        )

    def _write_rollout_artifact(self, attempt_dir: Path, result: RolloutResult) -> None:
        path = attempt_dir / "trajectories" / f"{result.history_name}_rollout_summary.json"
        path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def _skip_history2(
        self,
        attempt_dir: Path,
        prompt_preparer: EvalPromptPreparer,
        reason: str,
        result: dict[str, Any],
    ) -> None:
        prompt_preparer.write_skipped_history2(reason)
        result["history2_status"] = "skipped"
        result["history2_failure_class"] = "history2.skipped_after_history1_failure"
        (attempt_dir / "trajectories" / "history2_trajectory.jsonl").write_text("", encoding="utf-8")
        after_history1 = attempt_dir / "workspace_snapshots" / "after_history1"
        after_history2 = attempt_dir / "workspace_snapshots" / "after_history2"
        if after_history1.exists() and not after_history2.exists():
            shutil.copytree(after_history1, after_history2)
        self._write_skipped_verification(attempt_dir, "history2", result["history2_failure_class"])
        CodexJudge(attempt_dir).write_skipped(reason)

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
        prepared = attempt_dir / "prepared_glm"
        prepared.mkdir(parents=True, exist_ok=True)
        initial = attempt_dir / "workspace_snapshots" / "initial"
        after_history1 = attempt_dir / "workspace_snapshots" / "after_history1"
        after_history2 = attempt_dir / "workspace_snapshots" / "after_history2"
        if initial.exists() and not after_history1.exists():
            shutil.copytree(initial, after_history1)
        if after_history1.exists() and not after_history2.exists():
            shutil.copytree(after_history1, after_history2)
        for history in ["history1", "history2"]:
            messages_path = prepared / f"messages_{history}.json"
            if not messages_path.exists():
                messages_path.write_text(json.dumps({"failed_before_prepare": True, "failure_class": failure_class}, indent=2), encoding="utf-8")
            (attempt_dir / "trajectories" / f"{history}_trajectory.jsonl").parent.mkdir(parents=True, exist_ok=True)
            (attempt_dir / "trajectories" / f"{history}_trajectory.jsonl").touch(exist_ok=True)
            if not (attempt_dir / "verification" / history / "checklist.json").exists():
                self._write_skipped_verification(attempt_dir, history, failure_class or f"{history}.skipped")
        if not (attempt_dir / "judge" / "judge_decision.json").exists():
            CodexJudge(attempt_dir).write_skipped(failure_class or "failed before judge")

    def _failure_class_for_exception(self, exc: DatasetMiningError) -> str:
        if isinstance(exc, InfraError):
            text = str(exc).lower()
            if "docker" in text:
                return "infra.docker_start_failed"
            if "vllm" in text or "tokenizer" in text:
                return "infra.vllm_generation_failed"
        if isinstance(exc, ConfigError):
            return "infra.config_error"
        return "infra.filesystem_permission_error"

    def _record_infra_failure_stage(self, result: dict[str, Any], current_stage: str, failure_class: str) -> None:
        if current_stage.startswith("history2"):
            result["history2_status"] = "failed_infra"
            result["history2_failure_class"] = failure_class
            if result["history1_status"] == "not_started":
                result["history1_status"] = "skipped"
        elif current_stage == "judge":
            result["judge_decision"] = "uncertain"
            result["judge_failure_class"] = failure_class
        else:
            result["history1_status"] = "failed_infra"
            result["history1_failure_class"] = failure_class
            if result["history2_status"] == "not_started":
                result["history2_status"] = "skipped"
                result["history2_failure_class"] = "history2.skipped_after_infra_failure"

    def _finalize_sample(self, attempt_dir: Path, result: dict[str, Any], started: float) -> dict[str, Any]:
        self._sync_prompt_artifacts(attempt_dir)
        metrics = collect_trajectory_metrics(attempt_dir)
        result["prompt_token_counts"] = metrics["prompt_token_counts"]
        result["completion_token_counts"] = metrics["completion_token_counts"]
        result["latency"] = metrics["latency"]
        result["latency"]["sample_wall_sec"] = time.time() - started
        result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (attempt_dir / "eval_sample_summary.json").write_text(
            json.dumps(result, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return result

    def _sync_prompt_artifacts(self, attempt_dir: Path) -> None:
        serving_dir = attempt_dir / "serving"
        prepared_dir = attempt_dir / "prepared_glm"
        prepared_dir.mkdir(parents=True, exist_ok=True)
        for path in serving_dir.glob("history*_turn_*_rendered_prompt.txt"):
            parts = path.name.split("_")
            if len(parts) >= 4:
                history = parts[0]
                turn = parts[2]
                target = prepared_dir / f"rendered_prompt_{history}_turn_{turn}.txt"
                shutil.copy2(path, target)
        for path in serving_dir.glob("history*_turn_*_tokenization.json"):
            parts = path.name.split("_")
            if len(parts) >= 4:
                history = parts[0]
                turn = parts[2]
                target = prepared_dir / f"tokenization_{history}_turn_{turn}.json"
                shutil.copy2(path, target)


def collect_trajectory_metrics(attempt_dir: Path) -> dict[str, Any]:
    prompt_counts: dict[str, list[int]] = {"history1": [], "history2": []}
    completion_counts: dict[str, list[int]] = {"history1": [], "history2": []}
    latency: dict[str, float] = {"history1_sec": 0.0, "history2_sec": 0.0, "total_generation_sec": 0.0}
    for history in ["history1", "history2"]:
        trajectory = attempt_dir / "trajectories" / f"{history}_trajectory.jsonl"
        if not trajectory.exists():
            continue
        with trajectory.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("event") != "model_response":
                    continue
                prompt_meta = event.get("prompt_metadata") or {}
                generation_meta = event.get("generation_metadata") or {}
                if prompt_meta.get("token_count") is not None:
                    prompt_counts[history].append(int(prompt_meta["token_count"]))
                completion_count = generation_meta.get("completion_token_count")
                if completion_count is not None:
                    completion_counts[history].append(int(completion_count))
                elapsed = float(generation_meta.get("elapsed_sec") or 0.0)
                latency[f"{history}_sec"] += elapsed
                latency["total_generation_sec"] += elapsed
    return {
        "prompt_token_counts": {
            "history1": prompt_counts["history1"],
            "history2": prompt_counts["history2"],
            "total": sum(prompt_counts["history1"]) + sum(prompt_counts["history2"]),
        },
        "completion_token_counts": {
            "history1": completion_counts["history1"],
            "history2": completion_counts["history2"],
            "total": sum(completion_counts["history1"]) + sum(completion_counts["history2"]),
        },
        "latency": latency,
    }


def parse_gpu_slots(raw: str | None, workers: int, profile: str) -> list[list[int]]:
    if raw is None:
        raw = "4" if profile == "smoke" else "4,5,6,7"
    normalized = raw.replace(";", ",")
    devices = [int(item.strip()) for item in normalized.split(",") if item.strip()]
    if not devices:
        raise ConfigError("At least one GPU slot is required.")
    if workers > len(devices):
        raise ConfigError(f"workers={workers} requires at least {workers} GPU slots, got {devices}.")
    return [[device] for device in devices[:workers]]


def validate_eval_gpu_slots(slots: list[list[int]], workers: int) -> list[list[int]]:
    if workers < 1 or workers > 4:
        raise ConfigError("Evaluation workers must be between 1 and 4.")
    if len(slots) != workers:
        raise ConfigError(f"Worker count must match GPU slot count: workers={workers}, slots={slots}")
    allowed = set(ALLOWED_CUDA_DEVICES)
    seen: set[int] = set()
    for slot in slots:
        if len(slot) != 1:
            raise ConfigError(f"Evaluation uses tp=1/pp=1, so each worker must have one GPU: {slot}")
        device = int(slot[0])
        if device not in allowed:
            raise ConfigError(f"GPU {device} is not allowed for evaluation; allowed={sorted(allowed)}")
        if device in seen:
            raise ConfigError(f"GPU {device} appears in more than one evaluation worker slot.")
        seen.add(device)
    return slots


def apply_build_env_overrides() -> None:
    for key, value in BUILD_ENV_OVERRIDES.items():
        os.environ[key] = value


def _worker_main(
    worker_id: int,
    gpu_slot: list[int],
    run_config: dict[str, Any],
    job_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    apply_build_env_overrides()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in gpu_slot)
    engine = LocalVLLMEngine(
        model_id=str(run_config["model_id"]),
        gpu_devices=gpu_slot,
        tensor_parallel_size=int(run_config["tensor_parallel_size"]),
        pipeline_parallel_size=int(run_config["pipeline_parallel_size"]),
        max_model_len=int(run_config["max_model_len"]),
        gpu_memory_utilization=float(run_config["gpu_memory_utilization"]),
        max_num_seqs=1,
        enforce_eager=True,
    )
    evaluator = GLMSampleEvaluator(run_config, engine=engine, gpu_slot=gpu_slot)
    while True:
        job = job_queue.get()
        if job is None:
            return
        sample = job["sample"]
        try:
            result = evaluator.evaluate(sample)
        except Exception as exc:  # noqa: BLE001 - worker must not die without reporting a sample.
            attempt_dir = Path(run_config["run_root"]) / "per_sample" / sample.sample_id / f"attempt_worker_error_{uuid.uuid4().hex[:8]}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "eval_run_id": run_config["run_id"],
                "attempt_id": attempt_dir.name,
                "sample_id": sample.sample_id,
                "task_id": sample.task_id,
                "source_dataset_version": sample.dataset_version,
                "source_artifact_path": str(sample.source_artifact_path),
                "target_model": run_config["model_id"],
                "evaluation_mode": EVALUATION_MODE,
                "history1_status": "failed_infra",
                "history1_failure_class": "infra.worker_error",
                "history2_status": "skipped",
                "history2_failure_class": "history2.skipped_after_infra_failure",
                "judge_decision": "skipped",
                "final_status": "failed_infra",
                "failure_stage": "infra",
                "history1_turns": 0,
                "history2_turns": 0,
                "prompt_token_counts": {"history1": [], "history2": [], "total": 0},
                "completion_token_counts": {"history1": [], "history2": [], "total": 0},
                "latency": {"history1_sec": 0.0, "history2_sec": 0.0, "total_generation_sec": 0.0, "sample_wall_sec": 0.0},
                "artifact_path": str(attempt_dir),
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            (attempt_dir / "eval_sample_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        result_queue.put({"worker_id": worker_id, "result": result})


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    apply_build_env_overrides()
    ensure_model_evaluation_dirs()
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = args.run_id or f"{args.eval_name}_{timestamp}"
    run_root = MODEL_EVALUATION_RUNS_ROOT / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    results_path = run_root / "results.jsonl"
    results_path.write_text("", encoding="utf-8")

    single_config_path = Path(args.single_config)
    single_config = load_yaml(single_config_path)
    samples = load_accepted_samples(args.dataset_version)
    selected_samples = select_samples(samples, args.profile, args.sample_id, args.max_samples)
    workers = int(args.workers or (1 if args.profile == "smoke" else min(4, len(selected_samples))))
    workers = min(workers, max(1, len(selected_samples)))
    slots = validate_eval_gpu_slots(parse_gpu_slots(args.gpu_slots, workers, args.profile), workers)
    if int(args.tensor_parallel_size) != 1 or int(args.pipeline_parallel_size) != 1:
        raise ConfigError("This evaluation runner enforces tp=1 and pp=1.")

    run_config = build_run_config(args, run_id, run_root, single_config, single_config_path, selected_samples, slots)
    dump_yaml(run_config_for_yaml(run_config), run_root / "eval_config.yaml")
    write_dataset_audit(run_root, args.dataset_version, samples, selected_samples)

    dockerfile = Path(single_config.get("docker", {}).get("dockerfile", str(DEFAULT_DOCKERFILE)))
    if not dockerfile.is_absolute():
        dockerfile = REPO_ROOT / dockerfile
    docker_image = single_config.get("docker", {}).get("image", DEFAULT_DOCKER_IMAGE)
    if bool(single_config.get("docker", {}).get("build_if_missing", True)):
        DockerImageManager(docker_image, dockerfile).ensure_image(
            build_if_missing=True,
            timeout_sec=int(single_config.get("docker", {}).get("timeouts", {}).get("image_build_sec", 900)),
        )

    if workers == 1:
        engine = LocalVLLMEngine(
            model_id=str(run_config["model_id"]),
            gpu_devices=slots[0],
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            max_model_len=int(run_config["max_model_len"]),
            gpu_memory_utilization=float(run_config["gpu_memory_utilization"]),
            max_num_seqs=1,
            enforce_eager=True,
        )
        evaluator = GLMSampleEvaluator(run_config, engine=engine, gpu_slot=slots[0])
        results = []
        for index, sample in enumerate(selected_samples, start=1):
            result = evaluator.evaluate(sample)
            append_jsonl(results_path, result)
            results.append(result)
            print(f"[{index}/{len(selected_samples)}] {sample.sample_id}: {result['final_status']} ({result.get('failure_stage')})", flush=True)
    else:
        results = run_parallel(selected_samples, run_config, slots, results_path)

    summary = write_summary(run_root, run_config, results)
    return summary


def run_parallel(
    selected_samples: list[AcceptedDatasetSample],
    run_config: dict[str, Any],
    slots: list[list[int]],
    results_path: Path,
) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    job_queue: mp.Queue = ctx.Queue()
    result_queue: mp.Queue = ctx.Queue()
    workers: list[mp.Process] = []
    for worker_id, slot in enumerate(slots):
        proc = ctx.Process(target=_worker_main, args=(worker_id, slot, run_config, job_queue, result_queue), daemon=False)
        proc.start()
        workers.append(proc)
    for sample in selected_samples:
        job_queue.put({"sample": sample})
    for _ in workers:
        job_queue.put(None)

    results: list[dict[str, Any]] = []
    try:
        while len(results) < len(selected_samples):
            try:
                item = result_queue.get(timeout=5.0)
            except queue.Empty:
                if not any(proc.is_alive() for proc in workers):
                    raise InfraError("All evaluation workers exited before reporting all samples.")
                continue
            result = item["result"]
            append_jsonl(results_path, result)
            results.append(result)
            print(
                f"[{len(results)}/{len(selected_samples)}] worker={item['worker_id']} "
                f"{result['sample_id']}: {result['final_status']} ({result.get('failure_stage')})",
                flush=True,
            )
    finally:
        for proc in workers:
            proc.join(timeout=30)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=10)
    return results


def select_samples(
    samples: list[AcceptedDatasetSample],
    profile: str,
    sample_ids: list[str] | None,
    max_samples: int | None,
) -> list[AcceptedDatasetSample]:
    selected = samples
    if sample_ids:
        wanted = set(sample_ids)
        selected = [sample for sample in selected if sample.sample_id in wanted]
        missing = wanted - {sample.sample_id for sample in selected}
        if missing:
            raise ConfigError(f"Requested sample_id values were not in accepted.jsonl: {sorted(missing)}")
    if max_samples is not None:
        selected = selected[: int(max_samples)]
    elif profile == "smoke":
        selected = selected[:1]
    if not selected:
        raise ConfigError("No accepted samples selected for evaluation.")
    return selected


def build_run_config(
    args: argparse.Namespace,
    run_id: str,
    run_root: Path,
    single_config: dict[str, Any],
    single_config_path: Path,
    selected_samples: list[AcceptedDatasetSample],
    slots: list[list[int]],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "eval_name": args.eval_name,
        "profile": args.profile,
        "run_root": str(run_root),
        "dataset_version": args.dataset_version,
        "accepted_path": str(selected_samples[0].accepted_path),
        "accepted_count": len(load_accepted_samples(args.dataset_version)),
        "selected_count": len(selected_samples),
        "selected_sample_ids": [sample.sample_id for sample in selected_samples],
        "model_id": args.model_id,
        "backend": "local_vllm_python_engine",
        "workers": len(slots),
        "gpu_slots": slots,
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "pipeline_parallel_size": int(args.pipeline_parallel_size),
        "max_model_len": int(args.max_model_len),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_tokens": int(args.max_tokens),
        "evaluation_seed": int(args.evaluation_seed),
        "mining_seed": single_config.get("run", {}).get("mining_seed"),
        "random_seed": os.environ.get("RANDOM_SEED"),
        "decoding": {
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "mode": "greedy",
            "seed": int(args.evaluation_seed),
        },
        "prompt_profile": args.prompt_profile,
        "prompt_rendering": {
            "source": "stored_semantic_components_plus_glm_generated_history",
            "tokenizer_chat_template_model_id": args.model_id,
            "render_chat_template_inside_putpocket": True,
            "allow_vllm_internal_template": False,
            "qwen_rendered_prompts_used": False,
        },
        "judge": {
            "enabled": True,
            "backend": "codex_cli",
            "sandbox": "read-only",
            "approval": "never",
            "skip_if_unit_test_failed": True,
        },
        "single_config_path": str(single_config_path),
        "single_config": single_config,
        "build_env_overrides": BUILD_ENV_OVERRIDES,
    }


def run_config_for_yaml(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "single_config"}


def write_dataset_audit(
    run_root: Path,
    dataset_version: str,
    samples: list[AcceptedDatasetSample],
    selected_samples: list[AcceptedDatasetSample],
) -> None:
    audit = {
        "dataset_version": dataset_version,
        "dataset_root": str(samples[0].dataset_root if samples else ""),
        "accepted_path": str(samples[0].accepted_path if samples else ""),
        "accepted_count": len(samples),
        "selected_count": len(selected_samples),
        "accepted_row_schema": sorted(samples[0].row.keys()) if samples else [],
        "artifact_completeness": [
            {
                "sample_id": sample.sample_id,
                "artifact_path": str(sample.source_artifact_path),
                "missing_artifacts": sample.missing_artifacts,
            }
            for sample in samples
        ],
    }
    (run_root / "dataset_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")


def write_summary(run_root: Path, run_config: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    final_counts = Counter(result.get("final_status", "missing") for result in results)
    history1_counts = Counter(result.get("history1_status", "missing") for result in results)
    history2_counts = Counter(result.get("history2_status", "missing") for result in results)
    judge_counts = Counter(result.get("judge_decision", "missing") for result in results)
    failure_stage_counts = Counter(str(result.get("failure_stage") or "none") for result in results)
    failure_class_counts = Counter()
    for result in results:
        for key in ["history1_failure_class", "history2_failure_class", "judge_failure_class"]:
            if result.get(key):
                failure_class_counts[str(result[key])] += 1
    run_status = "blocked" if results and final_counts == Counter({"failed_infra": len(results)}) else "complete"
    summary = {
        "run_id": run_config["run_id"],
        "eval_name": run_config["eval_name"],
        "dataset_version": run_config["dataset_version"],
        "accepted_count": run_config["accepted_count"],
        "selected_count": len(results),
        "target_model": run_config["model_id"],
        "backend": run_config["backend"],
        "status": run_status,
        "results_path": str(run_root / "results.jsonl"),
        "summary_json_path": str(run_root / "summary.json"),
        "summary_md_path": str(run_root / "summary.md"),
        "counts": {
            "final_status": dict(final_counts),
            "history1_status": dict(history1_counts),
            "history2_status": dict(history2_counts),
            "judge_decision": dict(judge_counts),
            "failure_stage": dict(failure_stage_counts),
            "failure_class": dict(failure_class_counts),
        },
        "representative_success": next((result for result in results if result.get("final_status") == "succeeded"), None),
        "representative_failure": next((result for result in results if result.get("final_status") != "succeeded"), None),
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (run_root / "summary.md").write_text(render_summary_md(summary), encoding="utf-8")
    return summary


def render_summary_md(summary: dict[str, Any]) -> str:
    lines = [
        f"# GLM Evaluation Summary: {summary['run_id']}",
        "",
        f"- Dataset: {summary['dataset_version']}",
        f"- Target model: {summary['target_model']}",
        f"- Backend: {summary['backend']}",
        f"- Samples evaluated: {summary['selected_count']} of {summary['accepted_count']} accepted rows",
        "",
        "## Final Status Counts",
        "",
    ]
    for key, value in sorted(summary["counts"]["final_status"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Stage Counts", ""])
    for section in ["history1_status", "history2_status", "judge_decision", "failure_stage"]:
        lines.append(f"### {section}")
        for key, value in sorted(summary["counts"][section].items()):
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.append(f"Results JSONL: `{summary['results_path']}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m putpocket_dataset_mining.model_evaluation.glm_eval")
    parser.add_argument("--dataset-version", default="mbpp_stateful_working_v0")
    parser.add_argument("--model-id", default=GLM52_08B_MODEL_ID)
    parser.add_argument("--eval-name", default="eval_glm52_08b_on_mbpp_stateful_working_v0")
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--single-config", default="configs/dataset_mining/mbpp_stateful_single.yaml")
    parser.add_argument("--gpu-slots", default=None, help="Comma-separated physical GPU ids, e.g. 4,5,6,7.")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sample-id", action="append", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--evaluation-seed", type=int, default=20260721)
    parser.add_argument("--prompt-profile", choices=["compact", "full"], default="compact")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = run_evaluation(args)
    except Exception as exc:  # noqa: BLE001 - command-line failure should be explicit.
        print(json.dumps({"status": "failed", "error": f"{exc.__class__.__name__}: {exc}"}, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
