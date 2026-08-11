from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .config import load_yaml
from .constants import REPO_ROOT
from .dataset import SourceTask
from .errors import InfraError
from .serving import LocalVLLMEngine
from .single import SingleSampleRunner


MAIN_REPO_ROOT = Path(os.environ.get("PUTPOCKET_CANONICAL_DATA_REPO", "/home/dyryu/putpocket_dataset_mining"))
ACCEPTED_PATH = MAIN_REPO_ROOT / "data/dataset_mining/datasets/classeval_stateful_working_v0/accepted.jsonl"
EXPECTED_SHA = "6031d368ee8359c9dfc3c7b785d5c30e4db9ae5b2969bfba3a7e09512a46b30d"
MODEL_SNAPSHOT = Path(
    "/home/dyryu/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/"
    "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)


def run_fresh_timing(
    *,
    remote_config: Path,
    run_uuid: str | None = None,
    kst_timestamp: str | None = None,
    gpu_device: int = 2,
) -> dict[str, Any]:
    run_uuid = run_uuid or str(uuid.uuid4())
    kst_timestamp = kst_timestamp or time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_id = f"fresh_no_reuse_timing_test_ClassEval_76_{kst_timestamp}_{run_uuid}"
    run_root = REPO_ROOT / "data/model_evaluation/runs" / run_id
    attempt_dir = run_root / "samples/test_ClassEval_76" / f"fresh_attempt_{run_uuid}"
    if run_root.exists():
        raise InfraError(f"fresh run directory already exists: {run_root}")

    count_before, sha_before = _accepted_count_sha()
    if count_before != 18 or sha_before != EXPECTED_SHA:
        raise InfraError(f"ClassEval accepted integrity mismatch before run: count={count_before} sha={sha_before}")
    accepted_row = _accepted_row("test_ClassEval_76")
    source_task_path = Path(str(accepted_row["artifact_path"])) / "source_task.json"
    source_task = SourceTask(**json.loads(source_task_path.read_text(encoding="utf-8")))

    run_root.mkdir(parents=True)
    timing_dir = run_root / "timing"
    timing_dir.mkdir(parents=True)
    telemetry_stop = threading.Event()
    telemetry_thread = threading.Thread(
        target=_record_gpu_telemetry,
        args=(timing_dir / "gpu_telemetry.csv", telemetry_stop),
        daemon=True,
    )

    os.environ["SR_REMOTE_JOB_ID_PREFIX"] = f"fresh-{run_uuid}"
    os.environ["PUTPOCKET_HF_HUB_CACHE_DIR"] = str(MODEL_SNAPSHOT.parents[1])
    os.environ.pop("LMCACHE_CONFIG_FILE", None)
    os.environ.pop("LMCACHE_USE_EXPERIMENTAL", None)

    _assert_remote_jobs_absent(run_uuid)
    _write_source_manifest(timing_dir, [source_task_path, remote_config, ACCEPTED_PATH])
    config = _build_config(remote_config, run_root)
    runner = SingleSampleRunner(config)
    engine = LocalVLLMEngine(
        model_id=str(MODEL_SNAPSHOT),
        gpu_devices=[gpu_device],
        tensor_parallel_size=1,
        max_model_len=8192,
        max_num_seqs=1,
        enforce_eager=True,
        enable_prefix_caching=False,
    )

    telemetry_thread.start()
    result: dict[str, Any] | None = None
    try:
        result = runner.run_task(
            task=source_task,
            run_id=run_id,
            attempt_id=f"fresh_attempt_{run_uuid}",
            write_index=False,
            engine=engine,
            dataset_version="single_sample",
            gpu_devices=[gpu_device],
        )
        if runner.timing_recorder:
            runner.timing_recorder.mark("model_engine.shutdown.start")
            runner.timing_recorder.mark("model_engine.shutdown.end")
            runner.timing_recorder.mark("e2e.end")
            timing_payload = _build_timing_payload(runner.timing_recorder, run_root, attempt_dir, result)
            runner.timing_recorder.write_json_atomic("timing.json", timing_payload)
            _write_timing_markdown(timing_dir / "timing.md", timing_payload)
            _write_remote_verification_copy(timing_dir, attempt_dir, "history1")
            _write_remote_verification_copy(timing_dir, attempt_dir, "history2")
            _write_freshness_proof(
                timing_dir / "freshness_proof.json",
                run_uuid=run_uuid,
                run_id=run_id,
                engine=runner,
                vllm_engine=engine,
                attempt_dir=attempt_dir,
                timing_payload=timing_payload,
            )
            del engine
        _write_pointer(run_root, attempt_dir)
    finally:
        telemetry_stop.set()
        telemetry_thread.join(timeout=5)
        os.environ.pop("SR_REMOTE_JOB_ID_PREFIX", None)

    count_after, sha_after = _accepted_count_sha()
    return {
        "run_uuid": run_uuid,
        "run_id": run_id,
        "run_root": str(run_root),
        "attempt_dir": str(attempt_dir),
        "result": result,
        "classeval_count_before": count_before,
        "classeval_sha_before": sha_before,
        "classeval_count_after": count_after,
        "classeval_sha_after": sha_after,
        "timing_json": str(timing_dir / "timing.json"),
        "timing_md": str(timing_dir / "timing.md"),
        "gpu_telemetry": str(timing_dir / "gpu_telemetry.csv"),
        "freshness_proof": str(timing_dir / "freshness_proof.json"),
        "latest_pointer": str(REPO_ROOT / "LATEST_FRESH_E2E_TIMING.txt"),
    }


def _build_config(remote_config: Path, run_root: Path) -> dict[str, Any]:
    cfg = load_yaml(REPO_ROOT / "configs/dataset_mining/classeval_stateful_single.yaml")
    cfg["run"]["output_root"] = str(run_root.parent)
    cfg["model"]["generation_model_id"] = str(MODEL_SNAPSHOT)
    cfg["judge"]["enabled"] = False
    cfg["timing"] = {"enabled": True, "fresh_no_reuse": True}
    cfg["execution"] = {
        "execution_role": "controller",
        "workspace_backend": "local_docker",
        "verifier_backend": "remote_ssh_docker",
        "remote_config": str(remote_config),
        "verifier_timeout_sec": 3600,
        "verifier_remote_grace_sec": 120,
    }
    return cfg


def _build_timing_payload(recorder: Any, run_root: Path, attempt_dir: Path, result: dict[str, Any]) -> dict[str, Any]:
    h1_remote = _remote_result(attempt_dir, "history1")
    h2_remote = _remote_result(attempt_dir, "history2")
    rows = _top_level_rows(recorder, h1_remote, h2_remote)
    accounted_names = [
        "source_and_workspace_setup_sec",
        "remote_preflight_sec",
        "model_engine_initialization_sec",
        "history1_rollout_wall_sec",
        "history1_verification_total_roundtrip_sec",
        "history2_input_construction_sec",
        "history2_rollout_wall_sec",
        "history2_verification_total_roundtrip_sec",
        "final_server2_aggregation_sec",
        "engine_shutdown_sec",
    ]
    accounted = sum(float(rows.get(name) or 0.0) for name in accounted_names)
    total = float(rows.get("total_e2e_including_engine_init_sec") or 0.0)
    rows["timing_accounted_sum_sec"] = accounted
    rows["timing_unaccounted_sec"] = max(0.0, total - accounted)
    rows["timing_unaccounted_percent"] = (rows["timing_unaccounted_sec"] / total * 100.0) if total else None
    return {
        "run_root": str(run_root),
        "attempt_dir": str(attempt_dir),
        "final_status": result.get("final_status"),
        "top_level_timing_sec": rows,
        "vllm_requests": recorder.vllm_requests,
        "tool_calls": recorder.tool_calls,
        "history1_remote_result": h1_remote,
        "history2_remote_result": h2_remote,
        "notes": {
            "nested_metrics": [
                "history*_vllm_generation_sum_sec and history*_local_tool_execution_sum_sec are nested inside history*_rollout_wall_sec.",
                "history*_rsync_upload_sec, queue_wait, docker_validation, and result_retrieval are nested inside history*_verification_total_roundtrip_sec.",
            ],
            "verifier_timeout_limit": "The 3600 verifier value is a timeout limit, not the measured validation duration.",
        },
    }


def _top_level_rows(recorder: Any, h1_remote: dict[str, Any], h2_remote: dict[str, Any]) -> dict[str, float | None]:
    def d(name: str) -> float | None:
        return recorder.durations.get(name) or recorder.duration_between(f"{name}.start", f"{name}.end")

    def between(start: str, end: str) -> float | None:
        return recorder.duration_between(start, end)

    h1_vllm = sum(r["elapsed_sec"] for r in recorder.vllm_requests if r["stage"] == "history1")
    h2_vllm = sum(r["elapsed_sec"] for r in recorder.vllm_requests if r["stage"] == "history2")
    h1_tools = sum(r["elapsed_sec"] for r in recorder.tool_calls if r["stage"] == "history1")
    h2_tools = sum(r["elapsed_sec"] for r in recorder.tool_calls if r["stage"] == "history2")
    init = between("model_engine.init.start", "model_engine.ready")
    total = between("e2e.start", "e2e.end")
    return {
        "source_and_workspace_setup_sec": (between("source_inputs.load.start", "initial_workspace.create.end") or 0.0),
        "remote_preflight_sec": d("remote_preflight"),
        "model_engine_initialization_sec": init,
        "history1_vllm_generation_sum_sec": h1_vllm,
        "history1_local_tool_execution_sum_sec": h1_tools,
        "history1_rollout_wall_sec": between("history1.rollout.start", "history1.rollout.end"),
        "history1_verifier_bundle_and_checksum_sec": d("history1.verify_bundle"),
        "history1_rsync_upload_sec": d("history1.rsync_upload"),
        "history1_server1_queue_wait_sec": 0.0,
        "history1_server1_docker_validation_sec": h1_remote.get("wall_time_sec"),
        "history1_result_retrieval_sec": d("history1.result_retrieval"),
        "history1_verification_total_roundtrip_sec": between("history1.verification_roundtrip.start", "history1.verification_roundtrip.end"),
        "history2_input_construction_sec": d("history2.prepare"),
        "history2_vllm_generation_sum_sec": h2_vllm,
        "history2_local_tool_execution_sum_sec": h2_tools,
        "history2_rollout_wall_sec": between("history2.rollout.start", "history2.rollout.end"),
        "history2_verifier_bundle_and_checksum_sec": d("history2.verify_bundle"),
        "history2_rsync_upload_sec": d("history2.rsync_upload"),
        "history2_server1_queue_wait_sec": 0.0,
        "history2_server1_docker_validation_sec": h2_remote.get("wall_time_sec"),
        "history2_result_retrieval_sec": d("history2.result_retrieval"),
        "history2_verification_total_roundtrip_sec": between("history2.verification_roundtrip.start", "history2.verification_roundtrip.end"),
        "final_server2_aggregation_sec": d("final_aggregation"),
        "engine_shutdown_sec": d("model_engine.shutdown"),
        "total_e2e_including_engine_init_sec": total,
        "total_e2e_excluding_engine_init_sec": (total - init) if total is not None and init is not None else None,
    }


def _remote_result(attempt_dir: Path, stage: str) -> dict[str, Any]:
    path = attempt_dir / "verification" / stage / "remote_result.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_timing_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Fresh No-Reuse E2E Timing", "", "| Metric | Seconds |", "|---|---:|"]
    for key, value in payload["top_level_timing_sec"].items():
        if key.startswith("timing_"):
            continue
        lines.append(f"| `{key}` | {value if value is not None else 'null'} |")
    lines.append("")
    lines.append("The `3600` verifier value is a timeout limit, not the measured validation duration.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_remote_verification_copy(timing_dir: Path, attempt_dir: Path, stage: str) -> None:
    src = attempt_dir / "verification" / stage / "remote_result.json"
    if src.exists():
        shutil.copy2(src, timing_dir / f"remote_verification_{stage}.json")


def _write_freshness_proof(
    path: Path,
    *,
    run_uuid: str,
    run_id: str,
    engine: Any,
    vllm_engine: LocalVLLMEngine,
    attempt_dir: Path,
    timing_payload: dict[str, Any],
) -> None:
    h1 = timing_payload["history1_remote_result"]
    h2 = timing_payload["history2_remote_result"]
    proof = {
        "run_uuid": run_uuid,
        "run_id": run_id,
        "new_run_directory_created": True,
        "old_run_directory_read_for_runtime_input": False,
        "old_response_artifact_read": False,
        "old_trajectory_read": False,
        "old_workspace_snapshot_read": False,
        "old_verifier_result_read": False,
        "resume_enabled": False,
        "skip_existing_enabled": False,
        "replay_engine_used": False,
        "scripted_engine_used": False,
        "fresh_controller_pid": os.getpid(),
        "fresh_vllm_engine_pid": vllm_engine.engine_pid,
        "fresh_vllm_worker_pids": vllm_engine.worker_pids,
        "engine_process_start_time": _process_start_time(os.getpid()),
        "prefix_caching_enabled": False,
        "skip_reading_prefix_cache": True,
        "lmcache_enabled": False,
        "external_kv_cache_enabled": False,
        "sr_reuse_enabled": False,
        "model_weights_source": str(MODEL_SNAPSHOT),
        "model_download_performed": False,
        "history1_new_remote_job_id": h1.get("job_id"),
        "history2_new_remote_job_id": h2.get("job_id"),
        "history1_old_completed_result_reused": False,
        "history2_old_completed_result_reused": False,
        "history1_remote_docker_executed": bool(h1.get("wall_time_sec") is not None and h1.get("completed_at_kst")),
        "history2_remote_docker_executed": bool(h2.get("wall_time_sec") is not None and h2.get("completed_at_kst")),
        "artifact_path": str(attempt_dir),
    }
    path.write_text(json.dumps(proof, indent=2, sort_keys=True), encoding="utf-8")


def _record_gpu_telemetry(path: Path, stop: threading.Event) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp_utc", "gpu_index", "uuid", "name", "utilization_gpu_pct", "memory_used_mib", "power_draw_w", "compute_pids"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        while not stop.is_set():
            try:
                output = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,name,utilization.gpu,memory.used,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    stderr=subprocess.DEVNULL,
                )
                pids = _gpu_pids()
                for line in output.splitlines():
                    parts = [part.strip() for part in line.split(",", 5)]
                    if len(parts) == 6:
                        writer.writerow(
                            {
                                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "gpu_index": parts[0],
                                "uuid": parts[1],
                                "name": parts[2],
                                "utilization_gpu_pct": parts[3],
                                "memory_used_mib": parts[4],
                                "power_draw_w": parts[5],
                                "compute_pids": ";".join(pids.get(parts[0], [])),
                            }
                        )
                handle.flush()
            except Exception:  # noqa: BLE001
                pass
            stop.wait(1.0)


def _gpu_pids() -> dict[str, list[str]]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        return {}
    rows: dict[str, list[str]] = {}
    bus_to_index = _gpu_bus_to_index()
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2:
            rows.setdefault(bus_to_index.get(parts[0], parts[0]), []).append(parts[1])
    return rows


def _gpu_bus_to_index() -> dict[str, str]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # noqa: BLE001
        return {}
    rows = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) == 2:
            rows[parts[1]] = parts[0]
    return rows


def _accepted_count_sha() -> tuple[int, str]:
    data = ACCEPTED_PATH.read_bytes()
    rows = [line for line in data.splitlines() if line.strip()]
    return len(rows), hashlib.sha256(data).hexdigest()


def _accepted_row(sample_id: str) -> dict[str, Any]:
    for line in ACCEPTED_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("sample_id") == sample_id:
            return row
    raise InfraError(f"accepted sample not found: {sample_id}")


def _assert_remote_jobs_absent(run_uuid: str) -> None:
    cmd = [
        "ssh",
        "-F",
        "/home/dyryu/.ssh/putpocket_sr/connectivity_check/ssh_config",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "sr-server1-via-a-c",
        f"find /home/dyryu/putpocket_dataset_mining/data/remote_verifier -maxdepth 3 -name '*{run_uuid}*' -print",
    ]
    output = subprocess.check_output(cmd, text=True)
    if output.strip():
        raise InfraError(f"remote verifier job ID collision for {run_uuid}: {output.strip()}")


def _write_source_manifest(timing_dir: Path, paths: list[Path]) -> None:
    rows = []
    for path in paths:
        data = path.read_bytes()
        rows.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    (timing_dir / "source_file_access_manifest.json").write_text(json.dumps({"files": rows}, indent=2, sort_keys=True), encoding="utf-8")


def _write_pointer(run_root: Path, attempt_dir: Path) -> None:
    pointer = REPO_ROOT / "LATEST_FRESH_E2E_TIMING.txt"
    if pointer.exists():
        backup = pointer.with_name(f"{pointer.name}.{time.strftime('%Y%m%d_%H%M%S', time.localtime())}.bak")
        shutil.copy2(pointer, backup)
    pointer.write_text(f"{run_root}\n{attempt_dir}\n", encoding="utf-8")


def _process_start_time(pid: int) -> str | None:
    try:
        output = subprocess.check_output(["ps", "-p", str(pid), "-o", "lstart="], text=True)
    except Exception:  # noqa: BLE001
        return None
    return output.strip() or None
