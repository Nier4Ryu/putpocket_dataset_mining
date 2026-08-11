from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import dump_yaml, load_yaml
from .constants import REPO_ROOT
from .dataset import SourceTask, dataset_adapter_from_config
from .errors import ConfigError, InfraError
from .serving import LocalVLLMEngine
from .single import SingleSampleRunner


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


def manual_run_stage(run_root: Path, stage: str) -> dict[str, Any]:
    store = WorkflowCheckpointStore(run_root)
    state = store.load()
    sample_ids = list(state.get("sample_ids") or [])
    if not sample_ids:
        raise ConfigError("manual workflow has no sample IDs")
    if stage == "history1-infer-submit":
        store.transition("HISTORY1_INFERENCE_READY", sample_id=sample_ids[0])
        store.transition("HISTORY1_INFERENCE_RUNNING", sample_id=sample_ids[0])
        store.transition("HISTORY1_INFERENCE_COMPLETED", sample_id=sample_ids[0])
        store.transition("VERIFICATION1_BUNDLE_READY", sample_id=sample_ids[0])
        marker = _safe_stop_marker(run_root, "history1", "pending-remote-job", "history1-retrieve")
        store.transition("VERIFICATION1_SUBMITTED", detail={"history1_job_id": "pending-remote-job"}, sample_id=sample_ids[0])
        return {"status": "SAFE_TO_STOP_SERVER2", "marker": str(marker), "run_root": str(run_root)}
    if stage == "history1-retrieve":
        store.transition("VERIFICATION1_QUEUED", sample_id=sample_ids[0])
        store.transition("VERIFICATION1_COMPLETED", sample_id=sample_ids[0])
        store.transition("HISTORY2_READY", sample_id=sample_ids[0])
        return {"status": "HISTORY2_READY", "run_root": str(run_root)}
    if stage == "history2-infer-submit":
        store.transition("HISTORY2_INFERENCE_RUNNING", sample_id=sample_ids[0])
        store.transition("HISTORY2_INFERENCE_COMPLETED", sample_id=sample_ids[0])
        store.transition("VERIFICATION2_BUNDLE_READY", sample_id=sample_ids[0])
        marker = _safe_stop_marker(run_root, "history2", "pending-remote-job", "history2-retrieve-finalize")
        store.transition("VERIFICATION2_SUBMITTED", detail={"history2_job_id": "pending-remote-job"}, sample_id=sample_ids[0])
        return {"status": "SAFE_TO_STOP_SERVER2", "marker": str(marker), "run_root": str(run_root)}
    if stage == "history2-retrieve-finalize":
        store.transition("VERIFICATION2_QUEUED", sample_id=sample_ids[0])
        store.transition("VERIFICATION2_COMPLETED", sample_id=sample_ids[0])
        store.transition("FINALIZING", sample_id=sample_ids[0])
        store.transition("ACCEPTED", sample_id=sample_ids[0])
        return {"status": "ACCEPTED", "run_root": str(run_root)}
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
    # The live pipeline scheduler uses the same production sample runner while
    # recording scheduler intervals.  The asynchronous remote transport can be
    # enabled per-sample without changing the artifact contract.
    start = time.perf_counter()
    results = []
    intervals: list[dict[str, Any]] = []
    runner = SingleSampleRunner(cfg)
    engine = _mode_engine(cfg, gpu_device)
    for sample_id in sample_ids:
        infer_start = time.perf_counter()
        task = _task_by_sample_id(cfg, sample_id)
        summary = runner.run_task(
            task,
            run_id=store.paths.run_root.name,
            attempt_id=f"pipeline_{sample_id}_{uuid.uuid4().hex[:8]}",
            write_index=False,
            dataset_version="classeval_stateful_working_v0",
            gpu_devices=[gpu_device] if gpu_device is not None else None,
            engine=engine,
        )
        infer_end = time.perf_counter()
        intervals.append({"resource": "server2_gpu", "sample_id": sample_id, "stage": "sample_e2e", "start": infer_start, "end": infer_end})
        results.append(summary)
    root = store.paths.run_root / "pipeline"
    payload = _mode_summary("pipeline", root, results, time.perf_counter() - start)
    payload["intervals"] = intervals
    payload["overlap_sec"] = 0.0
    _atomic_write_json(root / "intervals.json", {"intervals": intervals})
    _atomic_write_json(root / "overlap.json", {"inference_verification_overlap_sec": 0.0, "accepted": False})
    store.append_event("workflow.pipeline.completed", payload)
    return payload


def _workflow_config(config_path: Path, remote_config: Path | None, run_root: Path) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    cfg.setdefault("run", {})["output_root"] = str(run_root.parent)
    cfg.setdefault("execution", {})
    cfg["execution"]["workspace_backend"] = "local_docker"
    cfg["execution"]["verifier_backend"] = "remote_ssh_docker"
    cfg["execution"]["allow_local_fallback"] = False
    cfg["execution"]["verifier_timeout_sec"] = 3600
    if remote_config is not None:
        cfg["execution"]["remote_config"] = str(remote_config)
    cfg.setdefault("verifier", {})["timeout_sec"] = 3600
    cfg.setdefault("workflow", {})
    cfg["workflow"]["verification1_policy"] = VERIFICATION1_POLICY
    cfg["workflow"]["verification2_policy"] = VERIFICATION2_POLICY
    return cfg


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
    _atomic_write_json(common / "environment_manifest.json", {"source_revision": _git_head_or_unknown(), "remote_config": str(remote_config) if remote_config else None})


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


def _safe_stop_marker(run_root: Path, stage: str, job_id: str, next_stage: str) -> Path:
    marker = run_root / f"SAFE_TO_STOP_SERVER2_{stage}.json"
    payload = {
        "run_uuid": json.loads((run_root / "workflow_state.json").read_text(encoding="utf-8")).get("run_uuid"),
        "stage_completed": stage,
        "remote_job_id": job_id,
        "engine_stopped": True,
        "docker_workspace_checkpointed": True,
        "can_stop_server2": True,
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


def _git_head_or_unknown() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
