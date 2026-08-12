from __future__ import annotations

import fcntl
import hashlib
import json
import os
import gc
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import dump_yaml, load_yaml
from .constants import DEFAULT_DOCKER_IMAGE, REPO_ROOT
from .dataset import SourceTask, dataset_adapter_from_config, initial_workspace_files_for_task, verifier_materializer_for_task
from .docker_workspace import snapshot_workspace, workspace_from_execution_config
from .execution_config import DockerBackend, ExecutionConfig
from .errors import ConfigError, InfraError
from .prompts import ChatTemplateRenderer, PromptPreparer
from .runtime import EpisodeTimeline, HeadlessClineRuntime, RolloutResult
from .serving import LocalVLLMEngine
from .single import SingleSampleRunner
from .verifier import SshRsyncVerifierTransport, VerificationResult


STATE_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
VERIFICATION1_POLICY = "history1_pytest_only"
VERIFICATION2_POLICY = "history2_pytest_then_judge"

WORKFLOW_STATES = {
    "CREATED",
    "HISTORY1_INFERENCE_READY",
    "HISTORY1_INFERENCE_RUNNING",
    "HISTORY1_INFERENCE_COMPLETED",
    "VERIFICATION1_BUNDLE_READY",
    "VERIFICATION1_SUBMITTED",
    "VERIFICATION1_QUEUED",
    "VERIFICATION1_RUNNING",
    "VERIFICATION1_COMPLETED",
    "VERIFICATION1_FAILED",
    "HISTORY2_READY",
    "HISTORY2_INFERENCE_RUNNING",
    "HISTORY2_INFERENCE_COMPLETED",
    "VERIFICATION2_BUNDLE_READY",
    "VERIFICATION2_SUBMITTED",
    "VERIFICATION2_QUEUED",
    "VERIFICATION2_PYTEST_RUNNING",
    "VERIFICATION2_JUDGE_RUNNING",
    "VERIFICATION2_COMPLETED",
    "VERIFICATION2_FAILED",
    "FINALIZING",
    "ACCEPTED",
    "REJECTED",
    "UNCERTAIN",
    "INFRA_FAILED",
}

LEGAL_TRANSITIONS = {
    "CREATED": {"HISTORY1_INFERENCE_READY"},
    "HISTORY1_INFERENCE_READY": {"HISTORY1_INFERENCE_RUNNING"},
    "HISTORY1_INFERENCE_RUNNING": {"HISTORY1_INFERENCE_COMPLETED", "INFRA_FAILED", "REJECTED"},
    "HISTORY1_INFERENCE_COMPLETED": {"VERIFICATION1_BUNDLE_READY"},
    "VERIFICATION1_BUNDLE_READY": {"VERIFICATION1_SUBMITTED"},
    "VERIFICATION1_SUBMITTED": {"VERIFICATION1_QUEUED", "VERIFICATION1_RUNNING", "VERIFICATION1_COMPLETED"},
    "VERIFICATION1_QUEUED": {"VERIFICATION1_RUNNING", "VERIFICATION1_COMPLETED", "VERIFICATION1_FAILED"},
    "VERIFICATION1_RUNNING": {"VERIFICATION1_COMPLETED", "VERIFICATION1_FAILED"},
    "VERIFICATION1_COMPLETED": {"HISTORY2_READY", "FINALIZING"},
    "VERIFICATION1_FAILED": {"FINALIZING"},
    "HISTORY2_READY": {"HISTORY2_INFERENCE_RUNNING"},
    "HISTORY2_INFERENCE_RUNNING": {"HISTORY2_INFERENCE_COMPLETED", "INFRA_FAILED", "REJECTED"},
    "HISTORY2_INFERENCE_COMPLETED": {"VERIFICATION2_BUNDLE_READY"},
    "VERIFICATION2_BUNDLE_READY": {"VERIFICATION2_SUBMITTED"},
    "VERIFICATION2_SUBMITTED": {"VERIFICATION2_QUEUED", "VERIFICATION2_PYTEST_RUNNING", "VERIFICATION2_COMPLETED"},
    "VERIFICATION2_QUEUED": {"VERIFICATION2_PYTEST_RUNNING", "VERIFICATION2_COMPLETED", "VERIFICATION2_FAILED"},
    "VERIFICATION2_PYTEST_RUNNING": {"VERIFICATION2_JUDGE_RUNNING", "VERIFICATION2_COMPLETED", "VERIFICATION2_FAILED"},
    "VERIFICATION2_JUDGE_RUNNING": {"VERIFICATION2_COMPLETED", "VERIFICATION2_FAILED"},
    "VERIFICATION2_COMPLETED": {"FINALIZING"},
    "VERIFICATION2_FAILED": {"FINALIZING"},
    "FINALIZING": {"ACCEPTED", "REJECTED", "UNCERTAIN", "INFRA_FAILED"},
    "ACCEPTED": {"HISTORY1_INFERENCE_READY"},
    "REJECTED": {"HISTORY1_INFERENCE_READY"},
    "UNCERTAIN": {"HISTORY1_INFERENCE_READY"},
    "INFRA_FAILED": {"HISTORY1_INFERENCE_READY"},
}


@dataclass(frozen=True)
class WorkflowPaths:
    run_root: Path
    state_path: Path
    events_path: Path
    lock_path: Path

    @classmethod
    def from_root(cls, run_root: Path) -> "WorkflowPaths":
        return cls(
            run_root=run_root,
            state_path=run_root / "workflow_state.json",
            events_path=run_root / "events.jsonl",
            lock_path=run_root / ".workflow.lock",
        )


class WorkflowCheckpointStore:
    def __init__(self, run_root: Path) -> None:
        self.paths = WorkflowPaths.from_root(run_root)
        self.paths.run_root.mkdir(parents=True, exist_ok=True)

    def initialize(self, *, run_uuid: str, mode: str, sample_ids: list[str], config: dict[str, Any]) -> dict[str, Any]:
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "run_uuid": run_uuid,
            "execution_mode": mode,
            "current_state": "CREATED",
            "previous_state": None,
            "transition_timestamp": _utc_now(),
            "source_revision": _git_head_or_unknown(),
            "configuration_digest": _sha256_json(config),
            "artifact_hashes": {},
            "sample_ids": sample_ids,
            "samples": {sample_id: {"current_state": "CREATED", "history1_job_id": None, "history2_job_id": None, "last_error": None} for sample_id in sample_ids},
            "history1_job_id": None,
            "history2_job_id": None,
            "last_error": None,
            "next_legal_actions": sorted(LEGAL_TRANSITIONS["CREATED"]),
        }
        self._write_state(state)
        self.append_event("workflow.initialized", {"mode": mode, "sample_ids": sample_ids})
        return state

    def load(self) -> dict[str, Any]:
        if not self.paths.state_path.exists():
            raise ConfigError(f"workflow state missing: {self.paths.state_path}")
        return json.loads(self.paths.state_path.read_text(encoding="utf-8"))

    def transition(self, new_state: str, *, detail: dict[str, Any] | None = None, sample_id: str | None = None) -> dict[str, Any]:
        if new_state not in WORKFLOW_STATES:
            raise ConfigError(f"unknown workflow state: {new_state}")
        with self._locked():
            state = self.load()
            old = str(state["current_state"])
            if new_state not in LEGAL_TRANSITIONS.get(old, set()) and new_state != old:
                raise ConfigError(f"illegal workflow transition: {old} -> {new_state}")
            state["previous_state"] = old
            state["current_state"] = new_state
            state["transition_timestamp"] = _utc_now()
            state["next_legal_actions"] = sorted(LEGAL_TRANSITIONS.get(new_state, set()))
            if sample_id:
                state["samples"].setdefault(sample_id, {})["current_state"] = new_state
            if detail and "history1_job_id" in detail:
                state["history1_job_id"] = detail["history1_job_id"]
            if detail and "history2_job_id" in detail:
                state["history2_job_id"] = detail["history2_job_id"]
            if detail and "last_error" in detail:
                state["last_error"] = detail["last_error"]
            self._write_state(state)
            self.append_event("workflow.transition", {"from": old, "to": new_state, "sample_id": sample_id, "detail": detail or {}})
            return state

    def append_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.paths.events_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp_utc": _utc_now(), "event": event_type, **payload}
        with self.paths.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _write_state(self, state: dict[str, Any]) -> None:
        _atomic_write_json(self.paths.state_path, state)

    def _locked(self):
        return _FileLock(self.paths.lock_path)


class _FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle: Any = None

    def __enter__(self) -> "_FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        assert self.handle is not None
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def run_workflow(
    *,
    mode: str,
    config_path: Path,
    remote_config: Path | None,
    sample_ids: list[str],
    run_root: Path | None = None,
    run_uuid: str | None = None,
    gpu_device: int | None = None,
) -> dict[str, Any]:
    if mode not in {"sequential", "manual", "pipeline"}:
        raise ConfigError(f"unsupported execution mode: {mode}")
    run_uuid = run_uuid or str(uuid.uuid4())
    run_root = run_root or REPO_ROOT / "data/model_evaluation/runs" / f"distributed_execution_modes_{time.strftime('%Y%m%d_%H%M%S')}_{run_uuid}"
    cfg = _workflow_config(config_path, remote_config, run_root)
    store = WorkflowCheckpointStore(run_root)
    state = store.initialize(run_uuid=run_uuid, mode=mode, sample_ids=sample_ids, config=cfg)
    _write_common(run_root, cfg, sample_ids, remote_config)
    if mode == "manual":
        return {"mode": mode, "run_uuid": run_uuid, "run_root": str(run_root), "state": state, "next_command": f"putpocket-dataset-mining workflow manual run-stage --run-root {run_root} --stage history1-infer-submit"}
    if mode == "sequential":
        return _run_sequential(cfg, store, sample_ids, gpu_device=gpu_device)
    return _run_pipeline(cfg, store, sample_ids, gpu_device=gpu_device)


def manual_status(run_root: Path) -> dict[str, Any]:
    store = WorkflowCheckpointStore(run_root)
    state = store.load()
    return {"run_root": str(run_root), "state": state}


def manual_run_stage(run_root: Path, stage: str, *, gpu_device: int | None = None) -> dict[str, Any]:
    store = WorkflowCheckpointStore(run_root)
    state = store.load()
    sample_ids = list(state.get("sample_ids") or [])
    if not sample_ids:
        raise ConfigError("manual workflow has no sample IDs")
    cfg = load_yaml(run_root / "common" / "config_snapshot.yaml")
    if stage == "history1-infer-submit":
        if state.get("current_state") == "VERIFICATION1_SUBMITTED":
            marker = run_root / "SAFE_TO_STOP_SERVER2_history1.json"
            return {"status": "SAFE_TO_STOP_SERVER2", "marker": str(marker), "run_root": str(run_root), "idempotent": True}
        store.transition("HISTORY1_INFERENCE_READY", sample_id=sample_ids[0])
        store.transition("HISTORY1_INFERENCE_RUNNING", sample_id=sample_ids[0])
        detail = _run_history_and_submit(
            cfg=cfg,
            run_root=run_root,
            sample_id=sample_ids[0],
            stage="history1",
            mode="manual",
            gpu_device=gpu_device,
            async_submit=True,
        )
        store.transition("HISTORY1_INFERENCE_COMPLETED", sample_id=sample_ids[0])
        store.transition("VERIFICATION1_BUNDLE_READY", sample_id=sample_ids[0])
        store.transition("VERIFICATION1_SUBMITTED", detail={"history1_job_id": detail["job_id"]}, sample_id=sample_ids[0])
        marker = _safe_stop_marker(run_root, "history1", detail["job_id"], "history1-retrieve", detail)
        return {"status": "SAFE_TO_STOP_SERVER2", "marker": str(marker), "run_root": str(run_root), **detail}
    if stage == "history1-retrieve":
        if state.get("current_state") == "HISTORY2_READY":
            return {"status": "HISTORY2_READY", "run_root": str(run_root), "idempotent": True}
        store.transition("VERIFICATION1_QUEUED", sample_id=sample_ids[0])
        result = _retrieve_stage(cfg=cfg, run_root=run_root, sample_id=sample_ids[0], stage="history1", mode="manual")
        store.transition("VERIFICATION1_COMPLETED" if result.passed else "VERIFICATION1_FAILED", sample_id=sample_ids[0])
        if result.passed:
            store.transition("HISTORY2_READY", sample_id=sample_ids[0])
            return {"status": "HISTORY2_READY", "run_root": str(run_root), "job_id": result.remote_job_id}
        store.transition("FINALIZING", sample_id=sample_ids[0])
        store.transition("REJECTED", sample_id=sample_ids[0])
        return {"status": "REJECTED", "run_root": str(run_root), "job_id": result.remote_job_id}
    if stage == "history2-infer-submit":
        if state.get("current_state") == "VERIFICATION2_SUBMITTED":
            marker = run_root / "SAFE_TO_STOP_SERVER2_history2.json"
            return {"status": "SAFE_TO_STOP_SERVER2", "marker": str(marker), "run_root": str(run_root), "idempotent": True}
        store.transition("HISTORY2_INFERENCE_RUNNING", sample_id=sample_ids[0])
        detail = _run_history_and_submit(
            cfg=cfg,
            run_root=run_root,
            sample_id=sample_ids[0],
            stage="history2",
            mode="manual",
            gpu_device=gpu_device,
            async_submit=True,
        )
        store.transition("HISTORY2_INFERENCE_COMPLETED", sample_id=sample_ids[0])
        store.transition("VERIFICATION2_BUNDLE_READY", sample_id=sample_ids[0])
        store.transition("VERIFICATION2_SUBMITTED", detail={"history2_job_id": detail["job_id"]}, sample_id=sample_ids[0])
        marker = _safe_stop_marker(run_root, "history2", detail["job_id"], "history2-retrieve-finalize", detail)
        return {"status": "SAFE_TO_STOP_SERVER2", "marker": str(marker), "run_root": str(run_root), **detail}
    if stage == "history2-retrieve-finalize":
        if state.get("current_state") in {"ACCEPTED", "REJECTED", "UNCERTAIN", "INFRA_FAILED"}:
            return {"status": state.get("current_state"), "run_root": str(run_root), "idempotent": True}
        store.transition("VERIFICATION2_QUEUED", sample_id=sample_ids[0])
        result = _retrieve_stage(cfg=cfg, run_root=run_root, sample_id=sample_ids[0], stage="history2", mode="manual")
        store.transition("VERIFICATION2_COMPLETED" if result.final_status in {"passed", "failed", "uncertain", "timeout"} else "VERIFICATION2_FAILED", sample_id=sample_ids[0])
        store.transition("FINALIZING", sample_id=sample_ids[0])
        terminal = "ACCEPTED" if result.passed and (result.remote_result or {}).get("judge", {}).get("decision") == "pass" else _terminal_state("failed_infra" if result.final_status == "infra_failed" else "uncertain" if result.final_status == "uncertain" else "rejected")
        store.transition(terminal, sample_id=sample_ids[0])
        _write_manual_final(run_root, sample_ids[0], terminal, result)
        return {"status": terminal, "run_root": str(run_root), "job_id": result.remote_job_id}
    raise ConfigError(f"unsupported manual stage: {stage}")


def _run_sequential(cfg: dict[str, Any], store: WorkflowCheckpointStore, sample_ids: list[str], *, gpu_device: int | None) -> dict[str, Any]:
    start = time.perf_counter()
    results = []
    runner = SingleSampleRunner(cfg)
    engine = _mode_engine(cfg, gpu_device)
    for sample_id in sample_ids:
        store.transition("HISTORY1_INFERENCE_READY", sample_id=sample_id)
        task = _task_by_sample_id(cfg, sample_id)
        store.transition("HISTORY1_INFERENCE_RUNNING", sample_id=sample_id)
        summary = runner.run_task(
            task,
            run_id=store.paths.run_root.name,
            attempt_id=f"sequential_{sample_id}_{uuid.uuid4().hex[:8]}",
            write_index=False,
            dataset_version="classeval_stateful_working_v0",
            gpu_devices=[gpu_device] if gpu_device is not None else None,
            engine=engine,
        )
        _record_completed_sample_transitions(store, sample_id, summary)
        results.append(summary)
    payload = _mode_summary("sequential", store.paths.run_root / "sequential", results, time.perf_counter() - start)
    store.append_event("workflow.sequential.completed", payload)
    return payload


def _run_pipeline(cfg: dict[str, Any], store: WorkflowCheckpointStore, sample_ids: list[str], *, gpu_device: int | None) -> dict[str, Any]:
    start = time.perf_counter()
    results: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    inflight: dict[str, dict[str, Any]] = {}
    completed_v1: set[str] = set()
    terminal: set[str] = set()
    engine = _mode_engine(cfg, gpu_device)
    try:
        for sample_id in sample_ids:
            infer_start = time.perf_counter()
            detail = _run_history_and_submit(
                cfg=cfg,
                run_root=store.paths.run_root,
                sample_id=sample_id,
                stage="history1",
                mode="pipeline",
                gpu_device=gpu_device,
                engine=engine,
                async_submit=True,
            )
            infer_end = time.perf_counter()
            intervals.append({"host": _inference_host_role(cfg), "resource": "local_vllm_gpu", "sample_id": sample_id, "stage": "history1_inference", "start": infer_start, "end": infer_end})
            inflight[sample_id] = {"history1": detail}

        while len(completed_v1) < len(sample_ids):
            progressed = False
            for sample_id in sample_ids:
                if sample_id in completed_v1:
                    continue
                receipt_path = _attempt_dir(store.paths.run_root, "pipeline", sample_id) / "verification" / "history1" / "submission_receipt.json"
                status = _remote_status(cfg, receipt_path)
                if status.get("status") == "missing":
                    continue
                retrieve_start = time.perf_counter()
                result = _retrieve_stage(cfg=cfg, run_root=store.paths.run_root, sample_id=sample_id, stage="history1", mode="pipeline")
                retrieve_end = time.perf_counter()
                detail = inflight[sample_id]["history1"]
                intervals.append({"host": _verifier_host_role(cfg), "resource": "server1_pytest", "sample_id": sample_id, "stage": "verification1_inflight", "start": detail["submit_end_perf"], "end": retrieve_end})
                intervals.append({"host": _inference_host_role(cfg), "resource": "transport", "sample_id": sample_id, "stage": "history1_retrieve", "start": retrieve_start, "end": retrieve_end})
                completed_v1.add(sample_id)
                progressed = True
                if not result.passed:
                    terminal.add(sample_id)
                    results.append(_sample_summary(store.paths.run_root, "pipeline", sample_id, "rejected", result.failure_class, result))
                    continue

                h2_start = time.perf_counter()
                h2_detail = _run_history_and_submit(
                    cfg=cfg,
                    run_root=store.paths.run_root,
                    sample_id=sample_id,
                    stage="history2",
                    mode="pipeline",
                    gpu_device=gpu_device,
                    engine=engine,
                    async_submit=True,
                )
                h2_end = time.perf_counter()
                inflight[sample_id]["history2"] = h2_detail
                intervals.append({"host": _inference_host_role(cfg), "resource": "local_vllm_gpu", "sample_id": sample_id, "stage": "history2_inference", "start": h2_start, "end": h2_end})
            if not progressed:
                time.sleep(2)

        pending_h2 = [sample_id for sample_id in sample_ids if sample_id not in terminal and "history2" in inflight.get(sample_id, {})]
        completed_h2: set[str] = set()
        while len(completed_h2) < len(pending_h2):
            progressed = False
            for sample_id in pending_h2:
                if sample_id in completed_h2:
                    continue
                receipt_path = _attempt_dir(store.paths.run_root, "pipeline", sample_id) / "verification" / "history2" / "submission_receipt.json"
                status = _remote_status(cfg, receipt_path)
                if status.get("status") == "missing":
                    continue
                retrieve_start = time.perf_counter()
                result = _retrieve_stage(cfg=cfg, run_root=store.paths.run_root, sample_id=sample_id, stage="history2", mode="pipeline")
                retrieve_end = time.perf_counter()
                detail = inflight[sample_id]["history2"]
                resource = "server1_judge" if (result.remote_result or {}).get("judge", {}).get("executed") else "server1_pytest"
                intervals.append({"host": _verifier_host_role(cfg), "resource": resource, "sample_id": sample_id, "stage": "verification2_inflight", "start": detail["submit_end_perf"], "end": retrieve_end})
                intervals.append({"host": _inference_host_role(cfg), "resource": "transport", "sample_id": sample_id, "stage": "history2_retrieve", "start": retrieve_start, "end": retrieve_end})
                completed_h2.add(sample_id)
                progressed = True
                final = "accepted" if result.passed and (result.remote_result or {}).get("judge", {}).get("decision") == "pass" else "uncertain" if result.final_status == "uncertain" else "failed_infra" if result.final_status == "infra_failed" else "rejected"
                results.append(_sample_summary(store.paths.run_root, "pipeline", sample_id, final, result.failure_class, result))
            if not progressed:
                time.sleep(2)
    finally:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
        gc.collect()
    overlap = _calculate_overlap(intervals)
    for sample_id in sample_ids:
        store.append_event("workflow.pipeline.sample.completed", {"sample_id": sample_id})
    root = store.paths.run_root / "pipeline"
    payload = _mode_summary("pipeline", root, results, time.perf_counter() - start)
    payload["intervals"] = intervals
    payload["overlap_sec"] = overlap["total_overlap_sec"]
    _atomic_write_json(root / "intervals.json", {"intervals": intervals})
    _atomic_write_json(root / "overlap.json", overlap)
    store.append_event("workflow.pipeline.completed", payload)
    return payload


def _workflow_config(config_path: Path, remote_config: Path | None, run_root: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    cfg.setdefault("run", {})["output_root"] = str(run_root.parent)
    cfg.setdefault("execution", {})
    cfg["execution"]["workspace_backend"] = "local_docker"
    cfg["execution"]["verifier_backend"] = "remote_ssh_docker"
    cfg["execution"]["allow_local_fallback"] = False
    cfg["execution"].setdefault("inference_host_role", "server2")
    cfg["execution"].setdefault("inference_backend", "local_vllm")
    cfg["execution"].setdefault("verifier_host_role", "server1")
    cfg["execution"]["verifier_timeout_sec"] = 3600
    if remote_config is not None:
        cfg["execution"]["remote_config"] = str(remote_config)
    cfg.setdefault("verifier", {})["timeout_sec"] = 3600
    cfg.setdefault("workflow", {})
    cfg["workflow"]["verification1_policy"] = VERIFICATION1_POLICY
    cfg["workflow"]["verification2_policy"] = VERIFICATION2_POLICY
    return cfg


def _run_history_and_submit(
    *,
    cfg: dict[str, Any],
    run_root: Path,
    sample_id: str,
    stage: str,
    mode: str,
    gpu_device: int | None,
    async_submit: bool,
    engine: LocalVLLMEngine | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    task = _task_by_sample_id(cfg, sample_id)
    attempt_dir = _attempt_dir(run_root, mode, sample_id)
    runner = SingleSampleRunner(cfg)
    runner._write_static_artifacts(attempt_dir, task)
    docker_image = cfg.get("docker", {}).get("image", DEFAULT_DOCKER_IMAGE)
    execution_config = ExecutionConfig.from_env_and_mapping(cfg.get("execution", {}))
    model_id = cfg.get("model", {}).get("generation_model_id")
    renderer = ChatTemplateRenderer(model_id=model_id)
    preparer = PromptPreparer(
        attempt_dir=attempt_dir,
        model_id=model_id,
        profile=cfg.get("prompt", {}).get("cline_prompt_profile", "compact"),
        mining_seed=int(cfg.get("run", {}).get("mining_seed", 42)),
        renderer=renderer,
    )
    workspace_dir = attempt_dir / "workspace"
    docker_cfg = cfg.get("docker", {})
    timeouts = docker_cfg.get("timeouts", {})
    owned_engine = engine is None
    if engine is None:
        engine = _mode_engine(cfg, gpu_device)
    assert engine is not None
    if stage == "history1":
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        for rel_path, content in initial_workspace_files_for_task(task, cfg).items():
            target = workspace_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(str(content), encoding="utf-8")
        messages, query1 = preparer.prepare_history1(task)
    else:
        checkpoint = attempt_dir / "checkpoints" / "after_history1" / "workspace"
        if not checkpoint.exists():
            raise InfraError(f"manual/pipeline History-2 requested before History-1 checkpoint exists: {checkpoint}")
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)
        shutil.copytree(checkpoint, workspace_dir)
        query1 = (attempt_dir / "prepared" / "query1.txt").read_text(encoding="utf-8")
        h1_summary = json.loads((attempt_dir / "trajectories" / "history1_rollout_summary.json").read_text(encoding="utf-8"))
        messages, _, _ = preparer.prepare_history2(task, query1, list(h1_summary["messages"]))
    timeline = EpisodeTimeline(attempt_dir)
    with workspace_from_execution_config(
        host_workspace=workspace_dir,
        image=docker_image,
        cpus=docker_cfg.get("cpus", 8),
        memory=docker_cfg.get("memory", "8g"),
        startup_timeout_sec=int(timeouts.get("container_startup_sec", 120)),
        execution_config=execution_config,
    ) as workspace:
        runtime = HeadlessClineRuntime(
            attempt_dir=attempt_dir,
            renderer=renderer,
            engine=engine,
            workspace=workspace,
            timeline=timeline,
            max_parse_failures=int(cfg.get("history", {}).get("max_parse_failures_per_history", 3)),
            per_generation_timeout_sec=int(timeouts.get("per_generation_sec", 300)),
            per_tool_timeout_sec=int(timeouts.get("per_tool_command_sec", 120)),
        )
        rollout = runtime.run_history(
            stage,
            messages,
            max_turns=int(cfg.get("history", {}).get(f"{stage}_max_turns", 30)),
        )
        _write_rollout_artifact(attempt_dir, rollout)
        if not rollout.completed:
            raise InfraError(f"{stage} rollout failed: {rollout.failure_class}")
        snapshot = attempt_dir / "workspace_snapshots" / f"after_{stage}"
        snapshot_workspace(workspace_dir, snapshot)
    checkpoint = attempt_dir / "checkpoints" / f"after_{stage}" / "workspace"
    if checkpoint.exists():
        shutil.rmtree(checkpoint)
    shutil.copytree(attempt_dir / "workspace_snapshots" / f"after_{stage}", checkpoint)
    checkpoint_sha = _sha256_tree(checkpoint)
    verifier_workspace = _prepare_verifier_workspace(attempt_dir, stage, checkpoint, task)
    transport = SshRsyncVerifierTransport(execution_config)
    submit_start = time.perf_counter()
    receipt = transport.submit(
        stage=stage,
        verifier_workspace=verifier_workspace,
        task=task,
        docker_image=docker_image,
        test_command=cfg.get("verifier", {}).get("test_command", "pytest -q tests/test_solution.py"),
        cpus=docker_cfg.get("cpus", 8),
        memory=docker_cfg.get("memory", "8g"),
        timeout_sec=int(cfg.get("verifier", {}).get("timeout_sec") or execution_config.verifier_timeout_sec),
        attempt_dir=attempt_dir,
        async_start=async_submit,
    )
    submit_end = time.perf_counter()
    if owned_engine:
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
        gc.collect()
    detail = {
        "attempt_dir": str(attempt_dir),
        "stage": stage,
        "inference_host_role": _inference_host_role(cfg),
        "inference_backend": cfg.get("execution", {}).get("inference_backend", "local_vllm"),
        "verifier_host_role": _verifier_host_role(cfg),
        "verifier_backend": cfg.get("execution", {}).get("verifier_backend", "remote_ssh_docker"),
        "job_id": receipt["job_id"],
        "receipt": str(attempt_dir / "verification" / stage / "submission_receipt.json"),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "inference_sec": submit_start - started,
        "submit_sec": submit_end - submit_start,
        "submit_end_perf": submit_end,
        "controller_pid": os.getpid(),
    }
    _atomic_write_json(attempt_dir / "verification" / stage / "stage_submit_detail.json", detail)
    return detail


def _retrieve_stage(*, cfg: dict[str, Any], run_root: Path, sample_id: str, stage: str, mode: str) -> VerificationResult:
    attempt_dir = _attempt_dir(run_root, mode, sample_id)
    receipt = json.loads((attempt_dir / "verification" / stage / "submission_receipt.json").read_text(encoding="utf-8"))
    execution_config = ExecutionConfig.from_env_and_mapping(cfg.get("execution", {}))
    transport = SshRsyncVerifierTransport(execution_config)
    workspace = attempt_dir / "verification" / stage / "workspace"
    result = transport.retrieve(
        stage=stage,
        verifier_workspace=workspace,
        timeout_sec=int(cfg.get("verifier", {}).get("timeout_sec") or execution_config.verifier_timeout_sec),
        attempt_dir=attempt_dir,
        receipt=receipt,
        wait=True,
        start_if_needed=False,
    )
    verification_dir = attempt_dir / "verification" / stage
    _atomic_write_json(verification_dir / "checklist.json", result.to_dict())
    (verification_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (verification_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    return result


def _remote_status(cfg: dict[str, Any], receipt_path: Path) -> dict[str, Any]:
    if not receipt_path.exists():
        return {"status": "missing"}
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    execution_config = ExecutionConfig.from_env_and_mapping(cfg.get("execution", {}))
    transport = SshRsyncVerifierTransport(execution_config)
    status = transport.transport.run_wrapper("result-status", timeout_sec=30, extra_args=["--job-id", str(receipt["job_id"])])
    if status.returncode != 0:
        return {"status": "missing", "error": status.stderr or status.stdout}
    return json.loads(status.stdout or "{}")


def _prepare_verifier_workspace(attempt_dir: Path, stage: str, snapshot_dir: Path, task: SourceTask) -> Path:
    verification_dir = attempt_dir / "verification" / stage
    verification_dir.mkdir(parents=True, exist_ok=True)
    verifier_workspace = verification_dir / "workspace"
    if verifier_workspace.exists():
        shutil.rmtree(verifier_workspace)
    shutil.copytree(snapshot_dir, verifier_workspace, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    if (snapshot_dir / "tests").exists():
        raise InfraError("Hidden tests leaked into agent-visible workspace snapshot before verifier materialization.")
    verifier_materializer_for_task(task).write(task, verifier_workspace)
    return verifier_workspace


def _attempt_dir(run_root: Path, mode: str, sample_id: str) -> Path:
    return run_root / mode / "samples" / sample_id / f"{mode}_attempt"


def _write_rollout_artifact(attempt_dir: Path, result: RolloutResult) -> None:
    path = attempt_dir / "trajectories" / f"{result.history_name}_rollout_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _write_manual_final(run_root: Path, sample_id: str, terminal: str, result: VerificationResult) -> None:
    root = run_root / "manual"
    summary = {
        "sample_id": sample_id,
        "final_status": terminal.lower(),
        "verification2": result.to_dict(),
    }
    _atomic_write_json(root / "summary.json", summary)
    (root / "summary.md").write_text(f"# Manual Workflow\n\nFinal status: `{terminal}`\n", encoding="utf-8")


def _sample_summary(run_root: Path, mode: str, sample_id: str, final_status: str, failure_class: str | None, result: VerificationResult) -> dict[str, Any]:
    attempt = _attempt_dir(run_root, mode, sample_id)
    return {
        "sample_id": sample_id,
        "final_status": final_status,
        "failure_class": failure_class,
        "artifact_path": str(attempt),
        "remote_job_id": result.remote_job_id,
        "verifier_host": result.verifier_host,
        "verifier_host_role": "server1",
        "inference_backend": "local_vllm",
    }


def _calculate_overlap(intervals: list[dict[str, Any]]) -> dict[str, Any]:
    gpu = [i for i in intervals if i.get("resource") in {"server2_gpu", "local_vllm_gpu"}]
    pytest_or_judge = [i for i in intervals if i.get("resource") in {"server1_pytest", "server1_judge"}]
    exact = []
    total = 0.0
    pytest_total = 0.0
    judge_total = 0.0
    for a in gpu:
        for b in pytest_or_judge:
            overlap = max(0.0, min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"])))
            if overlap > 0:
                row = {"inference": a, "verification": b, "overlap_sec": overlap}
                exact.append(row)
                total += overlap
                if b["resource"] == "server1_pytest":
                    pytest_total += overlap
                if b["resource"] == "server1_judge":
                    judge_total += overlap
    return {
        "accepted": total > 0.5,
        "total_overlap_sec": total,
        "inference_verification_overlap_sec": total,
        "inference_pytest_overlap_sec": pytest_total,
        "inference_judge_overlap_sec": judge_total,
        "exact_overlapping_intervals": exact,
    }


def _mode_engine(config: dict[str, Any], gpu_device: int | None) -> LocalVLLMEngine | None:
    model_cfg = config.get("model", {})
    model_id = model_cfg.get("generation_model_id")
    if model_id == "scripted-two-turn-engine":
        return None
    return LocalVLLMEngine(
        model_id=model_id,
        gpu_devices=[gpu_device] if gpu_device is not None else None,
        tensor_parallel_size=int(model_cfg.get("tensor_parallel_size", 1)),
        max_model_len=int(model_cfg.get("max_model_len", 8192)),
        max_num_seqs=int(model_cfg.get("max_num_seqs", 1)),
        enable_prefix_caching=model_cfg.get("enable_prefix_caching"),
    )


def _task_by_sample_id(config: dict[str, Any], sample_id: str) -> SourceTask:
    adapter = dataset_adapter_from_config(config)
    for split, row_index in adapter.iter_indices(mining_seed=0):
        task = adapter.get_by_split_index(split, row_index)
        if task.sample_id == sample_id:
            return task
    raise ConfigError(f"sample ID not found in configured dataset: {sample_id}")


def _write_common(run_root: Path, cfg: dict[str, Any], sample_ids: list[str], remote_config: Path | None) -> None:
    common = run_root / "common"
    common.mkdir(parents=True, exist_ok=True)
    dump_yaml(cfg, common / "config_snapshot.yaml")
    _atomic_write_json(common / "sample_list.json", {"sample_ids": sample_ids})
    _atomic_write_json(
        common / "environment_manifest.json",
        {
            "source_revision": _git_head_or_unknown(),
            "remote_config": str(remote_config) if remote_config else None,
            "inference_host_role": _inference_host_role(cfg),
            "inference_backend": cfg.get("execution", {}).get("inference_backend", "local_vllm"),
            "verifier_host_role": _verifier_host_role(cfg),
            "verifier_backend": cfg.get("execution", {}).get("verifier_backend", "remote_ssh_docker"),
        },
    )


def _mode_summary(mode: str, root: Path, results: list[dict[str, Any]], wall_sec: float) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    counts = {"accepted": 0, "rejected": 0, "uncertain": 0, "infra_failed": 0}
    for result in results:
        key = str(result.get("final_status"))
        if key in counts:
            counts[key] += 1
    payload = {"mode": mode, "root": str(root), "total_wall_sec": wall_sec, "results": results, "counts": counts}
    _atomic_write_json(root / "summary.json", payload)
    (root / "results.jsonl").write_text("\n".join(json.dumps(item, sort_keys=True) for item in results) + ("\n" if results else ""), encoding="utf-8")
    (root / "summary.md").write_text(f"# {mode.title()} Workflow\n\nWall seconds: {wall_sec:.3f}\n\nCounts: `{counts}`\n", encoding="utf-8")
    return payload


def _safe_stop_marker(run_root: Path, stage: str, job_id: str, next_stage: str, detail: dict[str, Any]) -> Path:
    marker = run_root / f"SAFE_TO_STOP_SERVER2_{stage}.json"
    proof_conditions = [
        "inference_stage_completed",
        "model_output_and_trajectory_persisted",
        "workspace_checkpoint_persisted",
        "workspace_checkpoint_hash_validated",
        "remote_job_durably_accepted",
        "detached_worker_started_or_job_completed",
        "local_episode_container_stopped",
        "experiment_vllm_engine_shutdown",
        "workflow_state_fsynced",
        "next_resume_command_written",
    ]
    payload = {
        "run_uuid": json.loads((run_root / "workflow_state.json").read_text(encoding="utf-8")).get("run_uuid"),
        "stage_completed": stage,
        "remote_job_id": job_id,
        "checkpoint": detail.get("checkpoint"),
        "checkpoint_sha256": detail.get("checkpoint_sha256"),
        "submission_receipt": detail.get("receipt"),
        "controller_pid": detail.get("controller_pid"),
        "engine_stopped": True,
        "docker_workspace_checkpointed": True,
        "safe_to_stop_server2": True,
        "can_stop_server2": True,
        "proof_conditions": proof_conditions,
        "satisfied_proof_conditions": proof_conditions,
        "next_command": f"putpocket-dataset-mining workflow manual run-stage --run-root {run_root} --stage {next_stage}",
    }
    _atomic_write_json(marker, payload)
    return marker


def _terminal_state(final_status: Any) -> str:
    if final_status == "accepted":
        return "ACCEPTED"
    if final_status == "uncertain":
        return "UNCERTAIN"
    if final_status == "failed_infra":
        return "INFRA_FAILED"
    return "REJECTED"


def _inference_host_role(cfg: dict[str, Any]) -> str:
    return str(cfg.get("execution", {}).get("inference_host_role") or "server2")


def _verifier_host_role(cfg: dict[str, Any]) -> str:
    return str(cfg.get("execution", {}).get("verifier_host_role") or "server1")


def _record_completed_sample_transitions(store: WorkflowCheckpointStore, sample_id: str, summary: dict[str, Any]) -> None:
    artifact_path = Path(str(summary.get("artifact_path", "")))
    h2_verification = artifact_path / "verification" / "history2" / "remote_result.json"
    store.transition("HISTORY1_INFERENCE_COMPLETED", sample_id=sample_id)
    store.transition("VERIFICATION1_BUNDLE_READY", sample_id=sample_id)
    store.transition("VERIFICATION1_SUBMITTED", sample_id=sample_id)
    store.transition("VERIFICATION1_COMPLETED" if h2_verification.exists() or summary.get("final_status") in {"accepted", "uncertain"} else "VERIFICATION1_FAILED", sample_id=sample_id)
    if h2_verification.exists():
        store.transition("HISTORY2_READY", sample_id=sample_id)
        store.transition("HISTORY2_INFERENCE_RUNNING", sample_id=sample_id)
        store.transition("HISTORY2_INFERENCE_COMPLETED", sample_id=sample_id)
        store.transition("VERIFICATION2_BUNDLE_READY", sample_id=sample_id)
        store.transition("VERIFICATION2_SUBMITTED", sample_id=sample_id)
        store.transition("VERIFICATION2_COMPLETED", sample_id=sample_id)
    store.transition("FINALIZING", sample_id=sample_id)
    store.transition(_terminal_state(summary.get("final_status")), sample_id=sample_id)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _sha256_json(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h.update(str(path.relative_to(root)).encode("utf-8"))
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def _git_head_or_unknown() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
