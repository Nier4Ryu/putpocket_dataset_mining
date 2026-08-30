from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .timing import utc_now_iso


TASK_ID = "T20260830-001__glm53-montblanc-deployment"
LOCK_RELATIVE_PATH = Path("configs/models/glm53_flash_nvfp4_montblanc.lock.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class GLM53DeploymentError(RuntimeError):
    """Fail-closed error raised by the GLM-5.3 deployment tooling."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_lock_path() -> Path:
    return repository_root() / LOCK_RELATIVE_PATH


def load_model_lock(path: Path | None = None) -> dict[str, Any]:
    lock_path = (path or default_lock_path()).resolve()
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GLM53DeploymentError(f"cannot load model lock {lock_path}: {exc}") from exc
    validate_model_lock(payload)
    return payload


def validate_model_lock(lock: dict[str, Any]) -> None:
    if lock.get("schema_version") != 1 or lock.get("task_id") != TASK_ID:
        raise GLM53DeploymentError("unexpected GLM-5.3 lock schema or task identity")
    selection = _mapping(lock, "selection")
    if selection.get("family") != "GLM-5.3" or selection.get("variant") != "GLM-5.3-Flash":
        raise GLM53DeploymentError("selected checkpoint is not GLM-5.3-Flash")
    if not GIT_SHA_RE.fullmatch(str(selection.get("revision", ""))):
        raise GLM53DeploymentError("checkpoint revision must be an immutable 40-hex Git SHA")
    if selection.get("weight_files_include_mtp") is not False:
        raise GLM53DeploymentError("the bounded smoke package must explicitly exclude MTP")

    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise GLM53DeploymentError("model lock has no selected files")
    seen: set[str] = set()
    total = 0
    weight_count = 0
    for item in files:
        if not isinstance(item, dict):
            raise GLM53DeploymentError("model file entry is not an object")
        relative = _safe_relative(str(item.get("path", "")))
        if relative in seen:
            raise GLM53DeploymentError(f"duplicate model file entry: {relative}")
        seen.add(relative)
        if relative == "model_mtp.safetensors":
            raise GLM53DeploymentError("MTP weights must not enter the bounded checkpoint set")
        size = item.get("size")
        if not isinstance(size, int) or size <= 0:
            raise GLM53DeploymentError(f"invalid file size for {relative}")
        if not SHA256_RE.fullmatch(str(item.get("sha256", ""))):
            raise GLM53DeploymentError(f"invalid SHA-256 for {relative}")
        total += size
        if re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", relative):
            weight_count += 1
    if total != lock.get("selected_files_total_bytes"):
        raise GLM53DeploymentError("selected file byte total does not match lock")
    if weight_count != 10 or "model.safetensors.index.json" not in seen:
        raise GLM53DeploymentError("expected ten main weight shards and one index")

    config = _mapping(lock, "model_config_contract")
    plan = _mapping(lock, "capacity_plan")
    parallel = _mapping(plan, "parallelism")
    heads = int(config.get("num_attention_heads", 0))
    experts = int(config.get("n_routed_experts", 0))
    if heads % 3 == 0:
        raise GLM53DeploymentError("lock no longer proves TP=3 illegal")
    if experts % 3 != 0:
        raise GLM53DeploymentError("expert count is not divisible by EP=3")
    expected_parallel = {
        "tensor_parallel_size": 1,
        "data_parallel_size": 3,
        "expert_parallel": True,
        "expert_parallel_size": 3,
        "enable_ep_weight_filter": True,
        "pipeline_parallel_size": 1,
    }
    if parallel != expected_parallel:
        raise GLM53DeploymentError("parallel layout must be TP1/DP3/EP3 with weight filtering")
    if int(config.get("effective_sparse_topk", 0)) != 2176:
        raise GLM53DeploymentError("SM120 NoPE runtime requires effective sparse top-k 2176")

    runtime = _mapping(lock, "runtime")
    for key in ("vllm_commit", "flashinfer_commit"):
        if not GIT_SHA_RE.fullmatch(str(runtime.get(key, ""))):
            raise GLM53DeploymentError(f"{key} must be an immutable 40-hex Git SHA")
    if runtime.get("attention_backend") != "FLASHINFER_MLA_SPARSE_SM120":
        raise GLM53DeploymentError("runtime must select the SM120 sparse MLA backend")
    if runtime.get("moe_backend") != "marlin":
        raise GLM53DeploymentError("SM120 compressed-tensors smoke must use Marlin MoE")
    if runtime.get("enable_mtp") is not False or runtime.get("enable_prefix_caching") is not False:
        raise GLM53DeploymentError("smoke runtime must disable MTP and prefix caching")

    package_files = lock.get("package_files")
    if not isinstance(package_files, list) or not package_files:
        raise GLM53DeploymentError("model lock has no reproducibility-critical package files")
    package_seen: set[str] = set()
    for item in package_files:
        if not isinstance(item, dict):
            raise GLM53DeploymentError("package file entry is not an object")
        relative = _safe_relative(str(item.get("path", "")))
        if relative in package_seen:
            raise GLM53DeploymentError(f"duplicate package file entry: {relative}")
        package_seen.add(relative)
        if not isinstance(item.get("size"), int) or item["size"] <= 0:
            raise GLM53DeploymentError(f"invalid package file size for {relative}")
        if not SHA256_RE.fullmatch(str(item.get("sha256", ""))):
            raise GLM53DeploymentError(f"invalid package SHA-256 for {relative}")


def selected_model_paths(lock: dict[str, Any]) -> list[str]:
    validate_model_lock(lock)
    return [str(item["path"]) for item in lock["files"]]


def verify_package_files(lock: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    validate_model_lock(lock)
    package_root = (root or repository_root()).resolve()
    failures: list[str] = []
    files: list[dict[str, Any]] = []
    for item in lock["package_files"]:
        relative = _safe_relative(item["path"])
        path = package_root / relative
        actual_size = path.stat().st_size if path.is_file() else None
        actual_sha256 = sha256_file(path) if path.is_file() else None
        status = "ok"
        if actual_size != item["size"]:
            status = "size_mismatch" if actual_size is not None else "missing"
            failures.append(f"{status}:{relative}")
        elif actual_sha256 != item["sha256"]:
            status = "hash_mismatch"
            failures.append(f"hash_mismatch:{relative}")
        files.append(
            {
                "path": relative,
                "status": status,
                "expected_size": item["size"],
                "actual_size": actual_size,
                "expected_sha256": item["sha256"],
                "actual_sha256": actual_sha256,
            }
        )
    return {"status": "ok" if not failures else "failed", "failures": failures, "files": files}


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_directory(
    model_dir: Path,
    lock: dict[str, Any],
    *,
    verify_hashes: bool = True,
) -> dict[str, Any]:
    validate_model_lock(lock)
    model_dir = model_dir.resolve()
    results: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in lock["files"]:
        relative = _safe_relative(item["path"])
        path = model_dir / relative
        result: dict[str, Any] = {
            "path": relative,
            "expected_size": item["size"],
            "expected_sha256": item["sha256"],
        }
        if not path.is_file():
            result["status"] = "missing"
            failures.append(f"missing:{relative}")
        else:
            actual_size = path.stat().st_size
            result["actual_size"] = actual_size
            if actual_size != item["size"]:
                result["status"] = "size_mismatch"
                failures.append(f"size:{relative}")
            elif verify_hashes:
                actual_sha256 = sha256_file(path)
                result["actual_sha256"] = actual_sha256
                if actual_sha256 != item["sha256"]:
                    result["status"] = "hash_mismatch"
                    failures.append(f"sha256:{relative}")
                else:
                    result["status"] = "ok"
            else:
                result["status"] = "size_ok_hash_skipped"
        results.append(result)

    config_check = _verify_local_model_config(model_dir, lock)
    if config_check["status"] != "ok":
        failures.extend(config_check["failures"])
    index_check = _verify_local_weight_index(model_dir, lock)
    if index_check["status"] != "ok":
        failures.extend(index_check["failures"])
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "checked_at_utc": utc_now_iso(),
        "model_dir": str(model_dir),
        "hashes_verified": verify_hashes,
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "files": results,
        "config": config_check,
        "weight_index": index_check,
    }


def host_doctor(
    lock: dict[str, Any],
    *,
    require_idle_gpus: bool = True,
    model_dir: Path | None = None,
) -> dict[str, Any]:
    validate_model_lock(lock)
    failures: list[str] = []
    gpu_rows = _nvidia_smi_rows()
    plan = lock["capacity_plan"]
    if len(gpu_rows) != plan["gpu_count"]:
        failures.append(f"gpu_count:{len(gpu_rows)}")
    for row in gpu_rows:
        if row["name"] != "NVIDIA RTX PRO 6000 Blackwell Server Edition":
            failures.append(f"gpu_name:{row['index']}:{row['name']}")
        if row["memory_total_bytes"] < plan["minimum_memory_bytes_per_gpu"]:
            failures.append(f"gpu_memory:{row['index']}")
        if row["compute_capability"] != plan["minimum_compute_capability"]:
            failures.append(f"gpu_compute_capability:{row['index']}")
    processes = _nvidia_compute_processes()
    if require_idle_gpus and processes:
        failures.append("gpu_compute_processes_present")

    disk = shutil.disk_usage(Path.home())
    required_download = int(lock["selected_files_total_bytes"])
    if model_dir is not None:
        required_download = sum(
            item["size"]
            for item in lock["files"]
            if not (model_dir / item["path"]).is_file()
            or (model_dir / item["path"]).stat().st_size != item["size"]
        )
    reserve = 40 * 1024**3
    if disk.free < required_download + reserve:
        failures.append("insufficient_disk_for_checkpoint_plus_40gib_reserve")
    shm = shutil.disk_usage(Path("/dev/shm"))
    if shm.total < 16 * 1024**3:
        failures.append("shared_memory_below_16gib")
    memory = _proc_memory_summary()
    statvfs = os.statvfs(Path.home())
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return {
        "schema_version": 1,
        "task_id": TASK_ID,
        "checked_at_utc": utc_now_iso(),
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "gpus": gpu_rows,
        "compute_processes": processes,
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "remaining_model_download_bytes": required_download,
        "required_disk_reserve_bytes": reserve,
        "shm": {"total_bytes": shm.total, "free_bytes": shm.free},
        "memory": memory,
        "inodes": {
            "total": statvfs.f_files,
            "free": statvfs.f_ffree,
        },
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "toolchain": {
            "docker": _command_output(["docker", "version", "--format", "{{.Server.Version}}"]),
            "gcc": shutil.which("gcc"),
            "g++": shutil.which("g++"),
            "nvcc": shutil.which("nvcc"),
            "cmake": shutil.which("cmake"),
            "ninja": shutil.which("ninja"),
        },
        "gpu_topology": _command_output(["nvidia-smi", "topo", "-m"]),
        "gpu_p2p_read": _command_output(["nvidia-smi", "topo", "-p2p", "r"]),
        "gpu_p2p_write": _command_output(["nvidia-smi", "topo", "-p2p", "w"]),
        "huggingface_access": {
            "hf_token_environment_present": bool(os.environ.get("HF_TOKEN")),
            "cached_token_file_present": (hf_home / "token").is_file(),
            "hf_home": str(hf_home),
        },
        "driver": _command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[0],
    }


def inspect_runtime_image(image: str, lock: dict[str, Any]) -> dict[str, Any]:
    validate_model_lock(lock)
    inspect = _run_json_command(["docker", "image", "inspect", image])
    if not isinstance(inspect, list) or len(inspect) != 1:
        raise GLM53DeploymentError(f"docker image inspect returned no unique image for {image}")
    image_info = inspect[0]
    labels = ((image_info.get("Config") or {}).get("Labels") or {})
    runtime = lock["runtime"]
    expected_labels = {
        "putpocket.task_id": TASK_ID,
        "putpocket.vllm.commit": runtime["vllm_commit"],
        "putpocket.flashinfer.commit": runtime["flashinfer_commit"],
    }
    failures = [
        f"label:{key}"
        for key, value in expected_labels.items()
        if labels.get(key) != value
    ]
    probe = _runtime_import_probe(image)
    if probe.get("status") != "ok":
        failures.extend(f"runtime_probe:{item}" for item in probe.get("failures", []))
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "image": image,
        "image_id": image_info.get("Id"),
        "repo_digests": image_info.get("RepoDigests") or [],
        "labels": {key: labels.get(key) for key in expected_labels},
        "runtime_probe": probe,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _verify_local_model_config(model_dir: Path, lock: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "failed", "failures": [f"config_unreadable:{exc}"]}
    text = config.get("text_config") or config
    expected = lock["model_config_contract"]
    actual = {
        "wrapper_model_type": config.get("model_type"),
        "model_type": text.get("model_type"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_attention_heads": text.get("num_attention_heads"),
        "n_routed_experts": text.get("n_routed_experts"),
        "num_experts_per_tok": text.get("num_experts_per_tok"),
        "index_topk": text.get("index_topk"),
        "index_kpool": text.get("index_kpool"),
        "qk_rope_head_dim": text.get("qk_rope_head_dim"),
        "kv_lora_rank": text.get("kv_lora_rank"),
        "quant_method": (config.get("quantization_config") or {}).get("quant_method"),
        "quantization_format": (config.get("quantization_config") or {}).get("format"),
    }
    for key, value in actual.items():
        if key == "quantization_format" and value is None:
            value = (config.get("quantization_config") or {}).get("config_groups") and "mixed-precision"
            actual[key] = value
        if value != expected[key]:
            failures.append(f"config:{key}:{value!r}!={expected[key]!r}")
    architectures = config.get("architectures") or []
    if lock["selection"]["architecture"] not in architectures:
        failures.append("config:architecture")
    return {"status": "ok" if not failures else "failed", "failures": failures, "actual": actual}


def _verify_local_weight_index(model_dir: Path, lock: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    try:
        index = json.loads(
            (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "failed", "failures": [f"index_unreadable:{exc}"]}
    referenced = set((index.get("weight_map") or {}).values())
    selected = {
        item["path"]
        for item in lock["files"]
        if item["path"].endswith(".safetensors")
    }
    if "model_mtp.safetensors" in referenced:
        failures.append("index_references_excluded_mtp")
    if referenced != selected:
        failures.append("index_shard_set_mismatch")
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "referenced_shards": sorted(referenced),
    }


def _nvidia_smi_rows() -> list[dict[str, Any]]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) != 4:
            raise GLM53DeploymentError(f"unexpected nvidia-smi GPU row: {line!r}")
        rows.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(parts[2]),
                "memory_total_bytes": int(parts[2]) * 1024 * 1024,
                "compute_capability": parts[3],
            }
        )
    return rows


def _proc_memory_summary() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, _, raw = line.partition(":")
        if key not in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
            continue
        number, unit = raw.strip().split()[:2]
        if unit != "kB":
            raise GLM53DeploymentError(f"unexpected /proc/meminfo unit for {key}: {unit}")
        values[f"{key.lower()}_bytes"] = int(number) * 1024
    return values


def _nvidia_compute_processes() -> list[dict[str, Any]]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise GLM53DeploymentError(f"nvidia-smi process query failed: {proc.stderr.strip()}")
    rows = []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        parts = [item.strip() for item in line.split(",", maxsplit=3)]
        if len(parts) != 4:
            raise GLM53DeploymentError(f"unexpected nvidia-smi process row: {line!r}")
        rows.append(
            {
                "gpu_uuid": parts[0],
                "pid": int(parts[1]),
                "process_name": parts[2],
                "used_memory_mib": int(parts[3]),
            }
        )
    return rows


def _runtime_import_probe(image: str) -> dict[str, Any]:
    code = r'''
import json
failures=[]
values={}
try:
 import torch, vllm, flashinfer
 values.update(torch=torch.__version__,torch_cuda=torch.version.cuda,vllm=vllm.__version__,flashinfer=flashinfer.__version__)
 from vllm.models.glm5next.nvidia.model import Glm5NextForConditionalGeneration
 from vllm.v1.attention.backends.mla.flashinfer_mla_sparse_sm120 import FlashInferMLASparseSM120Impl
 from vllm.model_executor.layers.sparse_attn_indexer_kpool import SparseAttnIndexerKpool
 from vllm.engine.arg_utils import EngineArgs
 from flashinfer.mla._sparse_mla_sm120_plan import _DECODE_GLM53_NOPE_DISPATCH
 values['architecture']=Glm5NextForConditionalGeneration.__name__
 values['attention_backend']=FlashInferMLASparseSM120Impl.__name__
 values['indexer']=SparseAttnIndexerKpool.__name__
 values['ep_weight_filter_arg']=hasattr(EngineArgs,'enable_ep_weight_filter')
 values['glm53_nope_dispatch']=(64,2176) in _DECODE_GLM53_NOPE_DISPATCH
except Exception as exc:
 failures.append(type(exc).__name__+':'+str(exc))
print(json.dumps({'status':'ok' if not failures else 'failed','failures':failures,'values':values},sort_keys=True))
'''
    proc = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", code],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return {"status": "failed", "failures": [f"container_exit_{proc.returncode}", proc.stderr[-2000:]]}
    try:
        payload = json.loads(proc.stdout.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {"status": "failed", "failures": [f"unparseable_probe:{exc}"]}
    return payload


def _safe_relative(value: str) -> str:
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise GLM53DeploymentError(f"unsafe model-relative path: {value!r}")
    return value


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise GLM53DeploymentError(f"{key} must be an object")
    return value


def _command_output(command: Iterable[str]) -> str:
    proc = subprocess.run(list(command), text=True, capture_output=True)
    if proc.returncode != 0:
        raise GLM53DeploymentError(
            f"command failed ({proc.returncode}): {' '.join(command)}: {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def _run_json_command(command: list[str]) -> Any:
    output = _command_output(command)
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise GLM53DeploymentError(f"command returned invalid JSON: {' '.join(command)}") from exc


def _emit(payload: dict[str, Any], output: Path | None) -> int:
    if output is not None:
        write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") == "ok" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail-closed GLM-5.3 deployment checks")
    parser.add_argument("--lock", type=Path, default=default_lock_path())
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-lock")
    validate.add_argument("--output", type=Path)

    host = subparsers.add_parser("host-doctor")
    host.add_argument("--allow-task-server", action="store_true")
    host.add_argument("--model-dir", type=Path)
    host.add_argument("--output", type=Path)

    model = subparsers.add_parser("verify-model")
    model.add_argument("--model-dir", type=Path, required=True)
    model.add_argument("--sizes-only", action="store_true")
    model.add_argument("--output", type=Path)

    image = subparsers.add_parser("inspect-image")
    image.add_argument("--image", required=True)
    image.add_argument("--output", type=Path)

    lock_run = subparsers.add_parser(
        "lock-run", help="run a mutating bootstrap command while holding the advisory build lock"
    )
    lock_run.add_argument("command_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        lock = load_model_lock(args.lock)
        if args.command == "validate-lock":
            package_report = verify_package_files(lock)
            return _emit(
                {
                    "schema_version": 1,
                    "task_id": TASK_ID,
                    "status": package_report["status"],
                    "lock": str(args.lock.resolve()),
                    "selected_files_total_bytes": lock["selected_files_total_bytes"],
                    "package_files": package_report,
                },
                args.output,
            )
        if args.command == "host-doctor":
            return _emit(
                host_doctor(
                    lock,
                    require_idle_gpus=not args.allow_task_server,
                    model_dir=args.model_dir,
                ),
                args.output,
            )
        if args.command == "verify-model":
            return _emit(
                verify_model_directory(
                    args.model_dir, lock, verify_hashes=not args.sizes_only
                ),
                args.output,
            )
        if args.command == "inspect-image":
            return _emit(inspect_runtime_image(args.image, lock), args.output)
        if args.command == "lock-run":
            if not args.command_args:
                raise GLM53DeploymentError("lock-run requires a command after --")
            command = list(args.command_args)
            if command[0] == "--":
                command = command[1:]
            if not command:
                raise GLM53DeploymentError("lock-run requires a non-empty command")
            from .agent_control import AgentConfig, acquire_agent_locks

            env = os.environ.copy()
            env["PUTPOCKET_GLM53_BUILD_LOCK_HELD"] = "1"
            with acquire_agent_locks(
                AgentConfig.load(),
                ["build"],
                operation="GLM-5.3 task-local bootstrap",
                wait_seconds=0,
            ):
                return subprocess.run(command, env=env).returncode
    except GLM53DeploymentError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True))
        return 1
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
