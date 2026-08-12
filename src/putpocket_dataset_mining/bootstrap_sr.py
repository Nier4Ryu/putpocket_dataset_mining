from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - bootstrap can still emit JSON only.
    yaml = None  # type: ignore[assignment]

from .constants import REPO_ROOT
from .errors import ConfigError
from .execution_config import (
    DOCKER_DISABLED_FOR_STATIC_ONLY,
    HardwareProfile,
    ServerProfile,
    cuda_arch_for_profile,
    default_hardware_for_server,
    ExecutionConfig,
)


CANONICAL_SERVER2_ROOT = Path(os.environ.get("PUTPOCKET_CANONICAL_SERVER2_ROOT", "/home/dyryu/putpocket_dataset_mining"))
SERVER2_ENV = CANONICAL_SERVER2_ROOT / "Putpocket_env"
SERVER2_EXTERNALS = CANONICAL_SERVER2_ROOT / "externals"
SERVER2_LOCK = REPO_ROOT / "configs" / "env" / "server2_blackwell.lock.yaml"
LEGACY_ENV_NAMES = ("Putpocket_env_glm52", "Putpocket_env_glm52_v025")


@dataclass(frozen=True)
class BootstrapRun:
    repo_root: Path
    log_dir: Path
    dry_run: bool
    doctor_only: bool

    def path(self, name: str) -> Path:
        return self.log_dir / name


def run_bootstrap(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.preset == "server2":
        return _run_server2_preset(args)
    return _run_static_multihost(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bootstrap_sr")
    parser.add_argument("--preset", choices=["server2"], default=None, help="Canonical preset. server2 provisions/validates Putpocket_env.")
    parser.add_argument("--doctor-only", action="store_true", help="Validate the selected preset without install/build mutation.")
    parser.add_argument("--force-vllm-build", action="store_true")
    parser.add_argument("--force-docker-build", action="store_true")
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-gpu-smoke", action="store_true")
    parser.add_argument("--phase", choices=["cpu", "gpu", "all"], default=None, help="Compatibility alias for --stage core|validate|all.")
    parser.add_argument("--stage", choices=["preflight", "system", "core", "verifier", "vllm_source", "vllm_build", "validate", "all"], default=None)
    parser.add_argument("--server-profile", choices=[x.value for x in ServerProfile], default="custom")
    parser.add_argument("--hardware-profile", choices=[x.value for x in HardwareProfile], default="auto")
    parser.add_argument("--role", choices=["controller", "verifier", "model_server", "development"], default=None)
    parser.add_argument("--execution-role", choices=["local_controller", "cloud_controller", "verifier_host"], default=None)
    parser.add_argument("--workspace-backend", choices=["local_docker", "remote_ssh_docker"], default=None)
    parser.add_argument("--verifier-backend", choices=["local_docker", "remote_ssh_docker", "disabled"], default=None)
    parser.add_argument("--vllm-profile", choices=["clean", "patched", "skip"], default="patched")
    parser.add_argument("--build-vllm", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--allow-system-install", action="store_true")
    parser.add_argument("--allow-docker-build", action="store_true")
    parser.add_argument("--allow-vllm-build", action="store_true")
    parser.add_argument("--runtime-checks", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manifest-dir", default="logs/bootstrap_sr")
    return parser


def _run_server2_preset(args: argparse.Namespace) -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = REPO_ROOT / "logs" / "env_setup" / stamp
    run = BootstrapRun(REPO_ROOT, log_dir, bool(args.dry_run), bool(args.doctor_only))
    planned = [
        "preflight",
        "uv",
        "python",
        "environment",
        "project",
        "externals",
        "qwen-runtime",
        "docker",
        "doctor",
        "tests",
        "manifest",
    ]
    plan = {
        "schema_version": 1,
        "preset": "server2",
        "repo_root": str(REPO_ROOT),
        "environment": str(SERVER2_ENV),
        "manager": "uv",
        "doctor_only": args.doctor_only,
        "dry_run": args.dry_run,
        "force_vllm_build": args.force_vllm_build,
        "force_docker_build": args.force_docker_build,
        "skip_docker": args.skip_docker,
        "skip_gpu_smoke": args.skip_gpu_smoke,
        "stages": planned,
        "mutations": _server2_mutations(args),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    log_dir.mkdir(parents=True, exist_ok=True)
    latest = REPO_ROOT / "logs" / "env_setup" / "latest"
    latest.parent.mkdir(parents=True, exist_ok=True)
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(log_dir.name)
    _write_json(run.path("plan.json"), plan)
    _write_text(run.path("bootstrap.log"), "bootstrap_sr --preset server2\n")
    before = _environment_manifest()
    _write_json(run.path("environment_before.json"), before)
    _write_before_snapshot()
    if not args.doctor_only:
        _ensure_uv_available(run)
        _ensure_python_environment(run)
        _ensure_project_editable(run)
        _ensure_externals(run, force_vllm=args.force_vllm_build)
        if not args.skip_docker:
            _validate_docker(run, force=args.force_docker_build)
    doctor = _doctor(run, skip_docker=args.skip_docker)
    _write_json(run.path("doctor.json"), doctor)
    after = _environment_manifest()
    _write_json(run.path("environment_after.json"), after)
    _write_contract(after)
    _write_legacy_environment_manifest()
    _write_summary(run, plan, before, after, doctor)
    if doctor["status"] != "passed":
        return 1
    return 0


def _server2_mutations(args: argparse.Namespace) -> list[str]:
    if args.doctor_only:
        return []
    mutations = [
        "create Putpocket_env only when missing",
        "install project editable only when console script/import is missing",
        "validate external source revisions",
    ]
    if args.force_vllm_build:
        mutations.append("force editable vLLM reinstall")
    if args.force_docker_build:
        mutations.append("force Docker image rebuild")
    return mutations


def _ensure_uv_available(run: BootstrapRun) -> None:
    uv = shutil.which("uv") or str(SERVER2_ENV / "bin" / "uv")
    if not uv or not Path(uv).exists():
        raise ConfigError("uv is required for --preset server2; install uv or provide it on PATH.")
    _write_text(run.path("uv_version.txt"), _command([uv, "--version"], check=False)["stdout"])


def _ensure_python_environment(run: BootstrapRun) -> None:
    if (SERVER2_ENV / "bin" / "python").exists():
        return
    uv = shutil.which("uv")
    if not uv:
        raise ConfigError("Cannot create Putpocket_env because uv is missing.")
    _command([uv, "venv", "--python", "3.13", str(SERVER2_ENV)], check=True, log=run.path("environment_create.log"))


def _ensure_project_editable(run: BootstrapRun) -> None:
    py = SERVER2_ENV / "bin" / "python"
    if not py.exists():
        raise ConfigError(f"Missing canonical Python: {py}")
    _command([str(py), "-m", "pip", "install", "setuptools==80.9.0"], check=True, log=run.path("package_sync.log"))
    probe = _command([str(py), "-c", "import putpocket_dataset_mining; print(putpocket_dataset_mining.__file__)"], check=False)
    if probe["returncode"] == 0 and str(REPO_ROOT / "src") in probe["stdout"]:
        return
    _command([str(py), "-m", "pip", "install", "-e", f"{REPO_ROOT}[dev]"], check=True, log=run.path("project_install.log"))


def _ensure_externals(run: BootstrapRun, *, force_vllm: bool) -> None:
    py = SERVER2_ENV / "bin" / "python"
    for name in ("vllm", "lmcache", "cline"):
        path = SERVER2_EXTERNALS / name
        if not path.exists():
            raise ConfigError(f"Missing external source: {path}")
        if (path / ".git").exists() and _command(["git", "-C", str(path), "status", "--porcelain"], check=False)["stdout"].strip():
            raise ConfigError(f"External source has uncommitted changes: {path}")
    if force_vllm:
        _command([str(py), "-m", "pip", "install", "--no-build-isolation", "-e", str(SERVER2_EXTERNALS / "vllm")], check=True, log=run.path("externals_install.log"))


def _validate_docker(run: BootstrapRun, *, force: bool) -> None:
    dockerfile = REPO_ROOT / "docker" / "default_python" / "Dockerfile"
    payload = {"docker_available": bool(shutil.which("docker")), "dockerfile": str(dockerfile), "force_requested": force}
    if dockerfile.exists():
        payload["dockerfile_sha256"] = _sha256(dockerfile)
    if shutil.which("docker"):
        result = _command(["docker", "image", "inspect", "putpocket-default-python:ubuntu22.04-py313-v1"], check=False)
        payload["default_image_present"] = result["returncode"] == 0
    _write_json(run.path("docker_images.json"), payload)


def _doctor(run: BootstrapRun, *, skip_docker: bool) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    py = SERVER2_ENV / "bin" / "python"
    checks["python"] = _command([str(py), "-V"], check=False)
    uv = SERVER2_ENV / "bin" / "uv"
    checks["uv_pip_check"] = _command([str(uv), "pip", "check"], check=False) if uv.exists() else _command([str(py), "-m", "pip", "check"], check=False)
    imports = (
        "import torch, ray, datasets, transformers, vllm, lmcache, putpocket_dataset_mining\n"
        "print(torch.__version__)\n"
        "print(getattr(torch.version, 'cuda', None))\n"
        "print(ray.__version__)\n"
        "print(datasets.__version__)\n"
        "print(transformers.__version__)\n"
        "print(vllm.__version__)\n"
        "print(vllm.__file__)\n"
        "print(lmcache.__file__)\n"
    )
    checks["imports"] = _command([str(py), "-c", imports], check=False)
    checks["cli_doctor"] = _command([str(SERVER2_ENV / "bin" / "putpocket-dataset-mining"), "doctor", "--json"], check=False)
    if not skip_docker:
        checks["docker"] = _command(["docker", "image", "inspect", "putpocket-default-python:ubuntu22.04-py313-v1"], check=False)
    status = "passed" if all(item["returncode"] == 0 for item in checks.values()) else "failed"
    _write_text(run.path("uv_pip_check.txt"), checks["uv_pip_check"]["stdout"] + checks["uv_pip_check"]["stderr"])
    return {"schema_version": 1, "status": status, "checks": checks}


def _environment_manifest() -> dict[str, Any]:
    py = SERVER2_ENV / "bin" / "python"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "environment": str(SERVER2_ENV),
        "exists": py.exists(),
        "git_head": _git(["rev-parse", "HEAD"]),
        "externals": _external_revisions(),
    }
    if py.exists():
        script = r"""
import importlib, json, os, sys
mods = ["torch", "ray", "datasets", "transformers", "vllm", "lmcache", "putpocket_dataset_mining"]
payload = {"python": sys.version.split()[0], "executable": sys.executable, "prefix": sys.prefix, "sys_path": sys.path, "modules": {}}
for name in mods:
    try:
        mod = importlib.import_module(name)
        item = {"version": getattr(mod, "__version__", "unknown"), "file": getattr(mod, "__file__", "")}
        if name == "torch":
            item["cuda"] = getattr(mod.version, "cuda", None)
        payload["modules"][name] = item
    except Exception as exc:
        payload["modules"][name] = {"error": f"{type(exc).__name__}: {exc}"}
print(json.dumps(payload, sort_keys=True))
"""
        result = _command([str(py), "-c", script], check=False)
        if result["returncode"] == 0:
            payload.update(json.loads(result["stdout"]))
        else:
            payload["probe_error"] = result
    return payload


def _external_revisions() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("vllm", "lmcache", "cline"):
        path = SERVER2_EXTERNALS / name
        item: dict[str, Any] = {"path": str(path), "exists": path.exists()}
        if (path / ".git").exists():
            item["head"] = _git(["-C", str(path), "rev-parse", "HEAD"])
            item["remote"] = _git(["-C", str(path), "remote", "get-url", "origin"])
            item["dirty"] = bool(_command(["git", "-C", str(path), "status", "--porcelain"], check=False)["stdout"].strip())
        out[name] = item
    return out


def _write_before_snapshot() -> None:
    root = REPO_ROOT / "logs" / "env_consolidation" / "before"
    root.mkdir(parents=True, exist_ok=True)
    manifest = _environment_manifest()
    _write_json(root / "environment_before.json", manifest)
    _write_text(root / "python_version.txt", f"{manifest.get('python', 'unknown')} {manifest.get('executable', '')}\n")
    freeze = _command([str(SERVER2_ENV / "bin" / "python"), "-m", "pip", "freeze"], check=False)
    _write_text(root / "pip_freeze.txt", freeze["stdout"] + freeze["stderr"])
    _write_json(root / "external_revisions.json", manifest.get("externals", {}))


def _write_contract(manifest: dict[str, Any]) -> None:
    modules = manifest.get("modules", {})
    payload = {
        "schema_version": 1,
        "environment": {
            "name": "server2",
            "path": str(SERVER2_ENV),
            "manager": "uv",
            "python": manifest.get("python", "unknown"),
        },
        "hardware": {
            "profile": "server2_blackwell",
            "cuda_home": os.environ.get("CUDA_HOME", "/usr/local/cuda-12.9"),
            "torch_cuda_arch_list": "12.0",
        },
        "python_packages": {
            name: modules.get(name, {}).get("version", "unknown")
            for name in ("torch", "ray", "datasets", "transformers")
        },
        "externals": manifest.get("externals", {}),
        "docker": {"episode_images": [{"tag": "putpocket-default-python:ubuntu22.04-py313-v1"}]},
        "cache": {"huggingface_home": os.environ.get("PUTPOCKET_HF_HUB_CACHE_DIR", "")},
        "build": {
            "max_jobs": os.environ.get("MAX_JOBS", "16"),
            "cmake_parallel_level": os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "16"),
            "cargo_build_jobs": os.environ.get("CARGO_BUILD_JOBS", "16"),
            "nvcc_threads": os.environ.get("NVCC_THREADS", "1"),
        },
    }
    SERVER2_LOCK.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        SERVER2_LOCK.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        SERVER2_LOCK.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_legacy_environment_manifest() -> None:
    items = []
    for name in LEGACY_ENV_NAMES:
        path = REPO_ROOT / name
        item = {"path": str(path), "classification": "LEGACY_BACKUP_NOT_ACTIVE", "exists": path.exists()}
        py = path / "bin" / "python"
        if py.exists():
            item["python"] = _command([str(py), "-V"], check=False)["stdout"].strip()
        items.append(item)
    out = REPO_ROOT / "logs" / "env_consolidation" / "legacy_environments.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(out, {"schema_version": 1, "legacy_environments": items})


def _write_summary(run: BootstrapRun, plan: dict[str, Any], before: dict[str, Any], after: dict[str, Any], doctor: dict[str, Any]) -> None:
    summary = {
        "schema_version": 1,
        "status": "passed" if doctor["status"] == "passed" else "failed",
        "preset": "server2",
        "environment": str(SERVER2_ENV),
        "log_dir": str(run.log_dir),
        "plan": plan,
        "before_python": before.get("python"),
        "after_python": after.get("python"),
        "doctor_status": doctor["status"],
        "lock": str(SERVER2_LOCK),
    }
    _write_json(run.path("summary.json"), summary)
    lines = [
        "# Server-2 Bootstrap Summary",
        "",
        f"- status: {summary['status']}",
        f"- environment: `{SERVER2_ENV}`",
        f"- lock: `{SERVER2_LOCK}`",
        f"- doctor: {doctor['status']}",
    ]
    _write_text(run.path("summary.md"), "\n".join(lines) + "\n")


def _run_static_multihost(args: argparse.Namespace) -> int:
    mapping: dict[str, Any] = {}
    if args.role and not args.execution_role:
        mapping["execution_role"] = args.role
    for key in ("execution_role", "workspace_backend", "verifier_backend", "vllm_profile"):
        value = getattr(args, key)
        if value:
            mapping[key] = value
    server = ServerProfile(args.server_profile)
    hardware = HardwareProfile(args.hardware_profile)
    if hardware == HardwareProfile.AUTO:
        hardware = default_hardware_for_server(server)
    if "TORCH_CUDA_ARCH_LIST" in os.environ:
        mapping["cuda_arch_list"] = os.environ["TORCH_CUDA_ARCH_LIST"]
    mapping["hardware_profile"] = hardware.value
    mapping["server_profile"] = server.value
    config = ExecutionConfig.from_env_and_mapping(mapping)
    stages = _stages_from_args(args)
    for stage in stages:
        if stage in {"preflight", "system", "core", "verifier", "vllm_source", "vllm_build"}:
            _stage_manifest(stage, config, args)
        elif stage == "validate":
            if args.runtime_checks:
                _gpu_phase(config, args)
            else:
                _stage_manifest("validate", config, args, extra={"runtime_checks": "disabled"})
    return 0


def _stage_manifest(stage: str, config: ExecutionConfig, args: argparse.Namespace, extra: dict[str, Any] | None = None) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        gpu_state = "hidden"
    else:
        gpu_state = "not_required"
    if config.execution_role.value in {"cloud_controller", "model_server"} and config.verifier_backend.value == "disabled":
        docker_state = "LOCAL_DOCKER_SKIPPED_EXPECTED"
    elif config.verifier_backend.value == "disabled":
        docker_state = DOCKER_DISABLED_FOR_STATIC_ONLY
    else:
        config.guard_cloud_local_docker()
        docker_state = "configured"
    try:
        from .finalized_dataset import load_finalized_lock, validate_finalized_dataset

        lock_status = validate_finalized_dataset(load_finalized_lock("configs/dataset_mining/classeval_stateful_working_v0.lock.yaml"))
    except Exception as exc:  # noqa: BLE001
        lock_status = {"status": "skipped_or_failed", "error": f"{exc.__class__.__name__}: {exc}"}
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "timestamp": time.time(),
        "repo": str(REPO_ROOT),
        "git_head": _git(["rev-parse", "HEAD"]),
        "execution": _execution_dict(config),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_state": gpu_state,
        "docker_state": docker_state,
        "class_eval_lock": lock_status,
        "vllm_build_requested": args.build_vllm,
        "vllm_profile": args.vllm_profile,
        "allow_system_install": args.allow_system_install,
        "allow_docker_build": args.allow_docker_build,
        "allow_vllm_build": args.allow_vllm_build,
        "runtime_checks": args.runtime_checks,
        "tools": _tool_report(),
        "cuda_arch_list": config.cuda_arch_list or cuda_arch_for_profile(config.hardware_profile),
        "dry_run": args.dry_run,
    }
    if extra:
        manifest.update(extra)
    _write_manifest(args.manifest_dir, stage, manifest)


def _gpu_phase(config: ExecutionConfig, args: argparse.Namespace) -> None:
    if config.hardware_profile == HardwareProfile.CPU:
        raise ConfigError("GPU bootstrap phase requires a non-cpu hardware profile.")
    actual = _nvidia_smi_query()
    arch = cuda_arch_for_profile(config.hardware_profile)
    manifest = {
        "schema_version": 1,
        "stage": "validate",
        "timestamp": time.time(),
        "git_head": _git(["rev-parse", "HEAD"]),
        "hardware_profile": config.hardware_profile.value,
        "server_profile": config.server_profile.value,
        "target_cuda_arch_list": config.cuda_arch_list or arch,
        "nvidia_smi": actual,
        "vllm_profile": args.vllm_profile,
        "build_vllm": args.build_vllm,
        "dry_run": args.dry_run,
    }
    _write_manifest(args.manifest_dir, "gpu", manifest)


def _stages_from_args(args: argparse.Namespace) -> list[str]:
    if args.stage:
        if args.stage == "all":
            return ["preflight", "system", "core", "verifier", "vllm_source", "vllm_build", "validate"]
        return [args.stage]
    if args.phase == "cpu":
        return ["core"]
    if args.phase == "gpu":
        return ["validate"]
    if args.phase == "all":
        return ["core", "validate"]
    raise SystemExit("--stage or --preset is required")


def _write_manifest(root: str, phase: str, manifest: dict[str, Any]) -> None:
    path = Path(root) / f"{phase}_bootstrap_manifest_{int(time.time())}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, manifest)
    print(path)


def _tool_report() -> dict[str, Any]:
    return {name: {"path": shutil.which(name)} for name in ["git", "ssh", "rsync", "docker", "nvcc", "nvidia-smi", "uv"]}


def _execution_dict(config: ExecutionConfig) -> dict[str, Any]:
    remote = config.remote
    return {
        "execution_role": config.execution_role.value,
        "workspace_backend": config.workspace_backend.value,
        "verifier_backend": config.verifier_backend.value,
        "hardware_profile": config.hardware_profile.value,
        "server_profile": config.server_profile.value,
        "model_id": config.model_id,
        "model_path": config.model_path,
        "vllm_profile": config.vllm_profile,
        "cuda_arch_list": config.cuda_arch_list,
        "remote": {
            "host": remote.host,
            "user": remote.user,
            "port": remote.port,
            "route": remote.route.value,
            "jump_hosts": [host.__dict__ for host in remote.jump_hosts],
            "repository_root": remote.repository_root,
            "job_root": remote.job_root,
            "strict_host_key_checking": remote.strict_host_key_checking,
            "docker_image": remote.docker_image,
            "max_concurrent_jobs": remote.max_concurrent_jobs,
        },
    }


def _command(cmd: list[str], *, check: bool, log: Path | None = None) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_ROOT)
    result = {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    if log:
        _write_text(log, "$ " + " ".join(cmd) + "\n" + proc.stdout + proc.stderr)
    if check and proc.returncode != 0:
        raise ConfigError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr}")
    return result


def _git(args: list[str]) -> str:
    result = _command(["git", *args], check=False)
    return result["stdout"].strip() if result["returncode"] == 0 else "unknown"


def _nvidia_smi_query() -> list[dict[str, str]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader"],
            text=True,
        )
    except Exception:  # noqa: BLE001
        return []
    rows = []
    for line in out.splitlines():
        name, cap, memory = [x.strip() for x in line.split(",", 2)]
        rows.append({"name": name, "compute_capability": cap, "memory_total": memory})
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(run_bootstrap())
