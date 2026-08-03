from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import shutil
import signal
import time
import uuid
from pathlib import Path
from typing import Any

from .config import load_yaml
from .constants import ALLOWED_CUDA_DEVICES, CONTROL_ROOT, REPO_ROOT, RUNS_ROOT, ensure_data_dirs
from .dataset import SourceTask, dataset_adapter_from_config
from .errors import ConfigError
from .finalized_dataset import load_finalized_lock, validate_finalized_dataset
from .serving import LocalVLLMEngine
from .single import SingleSampleRunner
from .storage import AttemptRecord, DatasetMaterializer, MiningIndex


def create_stop_file(run_id: str, mode: str = "graceful") -> Path:
    ensure_data_dirs()
    path = CONTROL_ROOT / f"{run_id}.stop"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_id": run_id, "mode": mode, "created_at": time.time()}, indent=2), encoding="utf-8")
    return path


def _stop_mode(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("mode", "graceful"))
    except json.JSONDecodeError:
        text = path.read_text(encoding="utf-8").strip().lower()
        return "hard" if "hard" in text else "graceful"


def validate_gpu_slots(config: dict[str, Any], profile_name: str) -> list[list[int]]:
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        raise ConfigError(f"Unknown multi-sample profile: {profile_name}")
    profile = profiles[profile_name]
    gpu = config.get("gpu", {})
    allowed = set(gpu.get("allowed_cuda_devices", ALLOWED_CUDA_DEVICES))
    if allowed != set(ALLOWED_CUDA_DEVICES):
        raise ConfigError(f"Runtime GPU allowed set must be exactly {list(ALLOWED_CUDA_DEVICES)}.")

    key = f"{profile_name}_slots"
    slots = gpu.get(key)
    if not isinstance(slots, list):
        raise ConfigError(f"Missing GPU slot config: gpu.{key}")
    if len(slots) != int(profile.get("num_workers", 0)):
        raise ConfigError(f"Profile {profile_name} num_workers must match gpu.{key} length.")
    if profile_name == "full_server" and int(profile.get("num_workers", 0)) != 3:
        raise ConfigError("full_server profile must use exactly 3 workers on the Blackwell branch.")

    seen: set[int] = set()
    normalized: list[list[int]] = []
    for slot in slots:
        if not isinstance(slot, list) or not slot:
            raise ConfigError(f"Invalid GPU slot: {slot}")
        slot_ints = [int(device) for device in slot]
        for device in slot_ints:
            if device not in allowed:
                raise ConfigError(f"GPU {device} is not allowed for dataset mining; allowed={sorted(allowed)}")
            if device in seen:
                raise ConfigError(f"GPU {device} appears in overlapping worker slots.")
            seen.add(device)
        normalized.append(slot_ints)
    return normalized


def _worker_main(
    worker_id: int,
    gpu_slot: list[int],
    single_config: dict[str, Any],
    single_config_path: str,
    model_id: str,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    job_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in gpu_slot)
    runner = SingleSampleRunner(single_config, config_path=Path(single_config_path))
    engine = LocalVLLMEngine(
        model_id=model_id,
        gpu_devices=gpu_slot,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )
    while True:
        job = job_queue.get()
        if job is None:
            return
        task = SourceTask(**job["task"])
        try:
            summary = runner.run_task(
                task=task,
                run_id=job["run_id"],
                attempt_id=job["attempt_id"],
                write_index=False,
                engine=engine,
                dataset_version=job["dataset_version"],
                gpu_devices=gpu_slot,
            )
            result_queue.put({"worker_id": worker_id, "summary": summary, "task": task.to_dict()})
        except Exception as exc:  # noqa: BLE001 - worker must report failures to master.
            result_queue.put(
                {
                    "worker_id": worker_id,
                    "summary": {
                        "run_id": job["run_id"],
                        "dataset_version": job["dataset_version"],
                        "sample_id": task.sample_id,
                        "split": task.split,
                        "row_index": task.row_index,
                        "task_id": task.task_id,
                        "attempt_id": job["attempt_id"],
                        "artifact_path": str(job["attempt_dir"]),
                        "final_status": "failed_infra",
                        "failure_class": "infra.worker_error",
                        "error": f"{exc.__class__.__name__}: {exc}",
                    },
                    "task": task.to_dict(),
                }
            )


class MultiSampleMaster:
    def __init__(self, config: dict[str, Any], config_path: Path | None = None) -> None:
        self.config = config
        self.config_path = config_path
        self.stop_requested = False
        self.hard_stop_requested = False

    @classmethod
    def from_config_path(cls, path: str | Path) -> "MultiSampleMaster":
        config_path = Path(path)
        return cls(load_yaml(config_path), config_path=config_path)

    def run(
        self,
        profile_name: str,
        run_id: str | None = None,
        rerun_failed_infra: bool = False,
    ) -> dict[str, Any]:
        profile = self.config["profiles"][profile_name]
        run_id = run_id or f"multi_{profile_name}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        dataset_version = profile["dataset_version"]
        finalized_result = self._finalized_noop_result(profile_name, profile, run_id, rerun_failed_infra)
        if finalized_result is not None:
            print(
                f"Dataset `{dataset_version}` is finalized with {finalized_result['canonical_accepted_count']} canonical accepted samples; "
                "no mining attempts will be assigned."
            )
            return finalized_result

        ensure_data_dirs()
        slots = validate_gpu_slots(self.config, profile_name)
        stop_file = CONTROL_ROOT / f"{run_id}.stop"
        index = MiningIndex.default()
        materializer = DatasetMaterializer(index)

        single_config_setting = self.config.get("worker", {}).get(
            "single_config",
            self.config.get("single_config", "configs/dataset_mining/mbpp_stateful_single.yaml"),
        )
        single_config_path = Path(single_config_setting)
        if not single_config_path.is_absolute():
            single_config_path = REPO_ROOT / single_config_path
        single_config = load_yaml(single_config_path)
        adapter = dataset_adapter_from_config(single_config)
        job_refs = adapter.iter_indices(mining_seed=int(self.config.get("run", {}).get("mining_seed", 42)))
        skip_statuses = {"accepted", "rejected"}
        if not rerun_failed_infra:
            skip_statuses.add("failed_infra")

        self._install_signal_handlers()
        worker_count = int(profile["num_workers"])
        job_queues: list[mp.Queue] = []
        result_queue: mp.Queue = mp.Queue()
        workers: list[mp.Process] = []
        model_id = self.config.get("gpu", {}).get("default_model", single_config.get("model", {}).get("generation_model_id"))
        tp = int(self.config.get("gpu", {}).get("tensor_parallel_size", 1))
        pp = int(self.config.get("gpu", {}).get("pipeline_parallel_size", 1))

        for worker_id in range(worker_count):
            jq: mp.Queue = mp.Queue(maxsize=1)
            proc = mp.Process(
                target=_worker_main,
                args=(worker_id, slots[worker_id], single_config, str(single_config_path), model_id, tp, pp, jq, result_queue),
                daemon=False,
            )
            proc.start()
            job_queues.append(jq)
            workers.append(proc)

        idle = set(range(worker_count))
        running: dict[int, dict[str, Any]] = {}
        accepted = 0
        attempts_assigned = 0
        attempts_finished = 0
        job_iter = iter(job_refs)
        target_accepted = int(profile["target_accepted"])
        max_attempts = int(profile["max_attempts"])

        try:
            while attempts_finished < max_attempts and accepted < target_accepted:
                mode = _stop_mode(stop_file)
                if mode:
                    self.stop_requested = True
                    self.hard_stop_requested = mode == "hard"
                if self.hard_stop_requested:
                    self._hard_stop(workers, running, run_id, stop_file)
                    break

                while idle and not self.stop_requested and attempts_assigned < max_attempts:
                    job = self._next_job(
                        job_iter=job_iter,
                        adapter=adapter,
                        index=index,
                        skip_statuses=skip_statuses,
                        run_id=run_id,
                        dataset_version=dataset_version,
                    )
                    if job is None:
                        self.stop_requested = True
                        break
                    worker_id = idle.pop()
                    running[worker_id] = job
                    job_queues[worker_id].put(job)
                    attempts_assigned += 1

                if not running:
                    break
                try:
                    result = result_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                worker_id = int(result["worker_id"])
                running.pop(worker_id, None)
                idle.add(worker_id)
                attempts_finished += 1
                summary = result["summary"]
                task = SourceTask(**result["task"])
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
                        artifact_path=summary["artifact_path"],
                        summary=summary,
                    )
                )
                if summary["final_status"] == "accepted":
                    accepted += 1
                    materializer.materialize_dataset(dataset_version)

            materializer.materialize_dataset(dataset_version)
        finally:
            for jq in job_queues:
                jq.put(None)
            for proc in workers:
                proc.join(timeout=30)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=10)

        return {
            "run_id": run_id,
            "profile": profile_name,
            "dataset_version": dataset_version,
            "target_accepted": target_accepted,
            "accepted": accepted,
            "attempts_assigned": attempts_assigned,
            "attempts_finished": attempts_finished,
            "stopped": self.stop_requested,
            "hard_stopped": self.hard_stop_requested,
            "stop_file": str(stop_file),
            "index_db": str(index.path),
            "dataset_root": str(Path(self.config.get("materialization", {}).get("datasets_root", "data/dataset_mining/datasets")) / dataset_version),
        }

    def _finalized_noop_result(
        self,
        profile_name: str,
        profile: dict[str, Any],
        run_id: str,
        rerun_failed_infra: bool,
    ) -> dict[str, Any] | None:
        if not profile.get("finalized", False):
            return None
        dataset_version = str(profile["dataset_version"])
        lock_path = profile.get("finalized_lock")
        if not lock_path:
            raise ConfigError(f"Finalized profile {profile_name} must set finalized_lock.")
        lock = load_finalized_lock(lock_path)
        if lock.dataset_version != dataset_version:
            raise ConfigError(
                f"Finalized lock dataset_version {lock.dataset_version} does not match profile dataset_version {dataset_version}."
            )
        status = validate_finalized_dataset(lock)
        if rerun_failed_infra or profile.get("allow_new_attempts", True) or lock.allow_mining:
            raise ConfigError(
                f"`{dataset_version}` is immutable. Create and configure a new dataset version to perform additional ClassEval mining."
            )
        return {
            "run_id": run_id,
            "profile": profile_name,
            "dataset_version": dataset_version,
            "target_accepted": int(profile["target_accepted"]),
            "accepted": 0,
            "attempts_assigned": 0,
            "attempts_finished": 0,
            "stopped": True,
            "hard_stopped": False,
            "finalized": True,
            "finalized_lock": str(lock.path),
            "canonical_accepted_count": status["accepted_count"],
            "accepted_sha256": status["accepted_sha256"],
            "message": (
                f"Dataset `{dataset_version}` is finalized with {status['accepted_count']} canonical accepted samples; "
                "no mining attempts will be assigned."
            ),
            "dataset_root": str(lock.accepted_file.parent),
        }

    def _next_job(
        self,
        job_iter: Any,
        adapter: Any,
        index: MiningIndex,
        skip_statuses: set[str],
        run_id: str,
        dataset_version: str,
    ) -> dict[str, Any] | None:
        for split, row_index in job_iter:
            task = adapter.get_by_split_index(split, row_index)
            if index.has_prior_status(task.sample_id, skip_statuses):
                continue
            attempt_id = f"attempt_{uuid.uuid4().hex[:12]}"
            attempt_dir = RUNS_ROOT / run_id / "samples" / task.sample_id / attempt_id
            return {
                "run_id": run_id,
                "attempt_id": attempt_id,
                "attempt_dir": str(attempt_dir),
                "dataset_version": dataset_version,
                "sample_id": task.sample_id,
                "task": task.to_dict(),
            }
        return None

    def _install_signal_handlers(self) -> None:
        def handler(signum: int, _frame: object) -> None:
            if self.stop_requested:
                self.hard_stop_requested = True
            self.stop_requested = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def _hard_stop(self, workers: list[mp.Process], running: dict[int, dict[str, Any]], run_id: str, stop_file: Path) -> None:
        for proc in workers:
            if proc.is_alive():
                proc.terminate()
        for job in running.values():
            attempt_dir = Path(job["attempt_dir"])
            self._cleanup_partial_attempt(attempt_dir)
            summary_path = attempt_dir / "attempt_cancel_summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(
                    {
                        "run_id": run_id,
                        "sample_id": job["sample_id"],
                        "attempt_id": job["attempt_id"],
                        "status": "cancelled_by_user",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        stop_record = RUNS_ROOT / run_id / "stop_record.json"
        stop_record.parent.mkdir(parents=True, exist_ok=True)
        stop_record.write_text(
            json.dumps({"run_id": run_id, "mode": "hard", "stop_file": str(stop_file), "created_at": time.time()}, indent=2),
            encoding="utf-8",
        )

    def _cleanup_partial_attempt(self, attempt_dir: Path) -> None:
        for name in ["trajectories", "workspace_snapshots", "serving"]:
            target = attempt_dir / name
            if target.exists():
                shutil.rmtree(target)
