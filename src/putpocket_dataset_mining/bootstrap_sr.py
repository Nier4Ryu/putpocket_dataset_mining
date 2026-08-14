from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import re
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
from .agent_control import AgentConfig, acquire_agent_locks
from .externals import checkout_external
from .runpod_runtime import (
    ARCH_PROFILES,
    build_manifest,
    build_runpod_plan,
    docker_build_args,
    runtime_fingerprint,
    resolve_cuda_arch_contract,
    validate_base_image_contract,
    validate_network_volume,
    validate_torch_contract,
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
    if args.preset == "runpod-dev":
        return _run_runpod_dev_preset(args)
    return _run_static_multihost(args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bootstrap_sr")
    profile_help = "CUDA architecture profile. portable-nvidia = 8.6 9.0 10.0 12.0; rtx3090 = 8.6; hopper = 9.0; blackwell-datacenter = 10.0; blackwell-rtx = 12.0; native = explicit visible-GPU detection."
    parser.add_argument("--preset", choices=["server2", "runpod-dev"], default=None, help="Canonical preset. server2 provisions/validates Putpocket_env; runpod-dev plans/validates the editable Hopper development runtime.")
    parser.add_argument("--doctor-only", action="store_true", help="Validate the selected preset without install/build mutation.")
    parser.add_argument("--force-vllm-build", action="store_true")
    parser.add_argument("--force-docker-build", action="store_true")
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--skip-vllm-build", action="store_true", help="For runpod-dev, do not perform editable vLLM native build.")
    parser.add_argument("--skip-gpu-smoke", action="store_true")
    parser.add_argument("--persistent-root", default=None, help="RunPod repository root. Defaults to /workspace/putpocket_dataset_mining.")
    parser.add_argument("--storage-kind", choices=["network-volume", "local", "ephemeral"], default=None)
    parser.add_argument("--cuda-arch-profile", choices=[*ARCH_PROFILES.keys(), "native"], default=None, help=profile_help)
    parser.add_argument("--cuda-arch-list", default=None, help="Explicit CUDA arch list. Highest precedence; example: '8.6 9.0 10.0 12.0'.")
    parser.add_argument("--build-jobs", type=int, default=None, help="Native build jobs. For runpod-dev: CLI > PUTPOCKET_BUILD_JOBS > nproc.")
    parser.add_argument("--base-image-contract", default=None)
    parser.add_argument("--phase", choices=["cpu", "gpu", "all"], default=None, help="Compatibility alias for --stage core|validate|all.")
    parser.add_argument("--stage", choices=["preflight", "system", "core", "verifier", "vllm_source", "vllm_build", "validate", "all"], default=None)
    parser.add_argument("--server-profile", choices=["server1_rtx3090", "server2_rtxpro6000_blackwell", "server2_blackwell", "runpod_hopper", "custom"], default="custom")
    parser.add_argument("--hardware-profile", choices=["auto", "cpu", "sm86", "sm90", "sm100", "sm120", "rtx3090", "blackwell", "hopper"], default="auto")
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


def _run_runpod_dev_preset(args: argparse.Namespace) -> int:
    plan = build_runpod_plan(
        repo_root=REPO_ROOT,
        persistent_root=args.persistent_root,
        storage_kind=args.storage_kind,
        cuda_arch_profile=args.cuda_arch_profile,
        cuda_arch_list=args.cuda_arch_list,
        base_image_contract=args.base_image_contract,
        dry_run=bool(args.dry_run),
        doctor_only=bool(args.doctor_only),
        skip_vllm_build=bool(args.skip_vllm_build),
        force_vllm_build=bool(args.force_vllm_build),
        skip_gpu_smoke=bool(args.skip_gpu_smoke),
        build_jobs=args.build_jobs,
    )
    payload = plan.as_dict()
    if args.dry_run:
        base = validate_base_image_contract(plan.base_image_contract)
        torch_contract = validate_torch_contract(plan.torch_contract, require_resolved=False)
        payload["base_image_contract_status"] = "passed"
        payload["torch_contract_status"] = torch_contract.get("provenance_status", "unknown")
        payload["runtime_fingerprint"] = runtime_fingerprint(plan, base, torch_contract)
        payload["build_manifest"] = build_manifest(plan, payload["runtime_fingerprint"])
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    return _run_runpod_dev_preset_locked(args, plan, payload)


def _run_runpod_dev_preset_locked(args: argparse.Namespace, plan: Any, payload: dict[str, Any]) -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = REPO_ROOT / "logs" / "env_setup" / f"runpod_dev_{stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    _write_json(log_dir / "plan.json", payload)
    try:
        base = validate_base_image_contract(plan.base_image_contract)
        torch_contract = validate_torch_contract(plan.torch_contract, require_resolved=not args.doctor_only)
        storage: dict[str, Any] | None = None
        if not args.doctor_only:
            storage = validate_network_volume(plan)
            with acquire_agent_locks(
                AgentConfig.load(),
                ["canonical-runtime", "build"],
                operation="bootstrap runpod-dev build",
            ):
                _ensure_runpod_runtime(plan, torch_contract, log_dir)
                fingerprint = runtime_fingerprint(plan, base, torch_contract)
                manifest = build_manifest(plan, fingerprint)
                if not args.skip_vllm_build:
                    _run_runpod_vllm_build(plan, manifest, log_dir)
                _run_runpod_lmcache_build(plan, manifest, log_dir)
        else:
            _validate_installed_runpod_runtime(plan, torch_contract)
            fingerprint = runtime_fingerprint(plan, base, torch_contract)
            manifest = build_manifest(plan, fingerprint)
        _write_json(log_dir / "runtime_fingerprint.json", fingerprint)
        _write_json(log_dir / "build_manifest.json", manifest)
        summary = {
            "schema_version": 1,
            "preset": "runpod-dev",
            "status": "passed",
            "log_dir": str(log_dir),
            "storage": storage,
            "torch_provenance": torch_contract.get("provenance_status"),
            "uv_bootstrap_ready": torch_contract.get("provenance_status") == "resolved",
            "detected_cpus": plan.cpu_count_detected,
            "vllm_build_jobs": plan.build_jobs_effective,
            "cmake_parallel_level": plan.build_jobs_effective,
            "nvcc_threads": plan.nvcc_threads,
            "cpu_count_detected": plan.cpu_count_detected,
            "build_jobs_requested": plan.build_jobs_requested,
            "build_jobs_effective": plan.build_jobs_effective,
            "max_jobs": plan.build_jobs_effective,
            "cmake_build_parallel_level": plan.build_jobs_effective,
            "vllm_native_build_wall_time_seconds": manifest["vllm_editable_build"]["build_wall_time_seconds"],
            "memory_related_build_fallback_used": manifest["vllm_editable_build"]["memory_related_build_fallback_used"],
            "fallback_effective_jobs": manifest["vllm_editable_build"]["fallback_effective_jobs"],
        }
        _write_json(log_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except ConfigError as exc:
        summary = {"schema_version": 1, "preset": "runpod-dev", "status": "failed", "error": str(exc), "log_dir": str(log_dir)}
        _write_json(log_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 2


def _run_runpod_vllm_build(plan: Any, manifest: dict[str, Any], log_dir: Path) -> None:
    """Execute the canonical editable vLLM build with recorded resource evidence."""

    vllm_root = plan.repo_root / "externals" / "vllm"
    env_python = plan.env_path / "bin" / "python"
    uv = Path(shutil.which("uv") or "")
    if not vllm_root.is_dir():
        raise ConfigError(f"Missing editable vLLM source: {vllm_root}")
    if not env_python.exists() or not uv.is_file():
        raise ConfigError(f"Missing runpod-dev environment tools under: {plan.env_path}")

    evidence = {
        "nproc": _command(["nproc"], check=True),
        "free_h": _command(["free", "-h"], check=True),
        "df_workspace": _command(["df", "-h", "/workspace"], check=True),
        "df_tmp": _command(["df", "-h", "/tmp"], check=True),
    }
    _write_json(log_dir / "native_build_resources_before.json", evidence)
    build = manifest["vllm_editable_build"]
    build["ram_before_build"] = evidence["free_h"]["stdout"]
    build["build_start_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started = time.monotonic()
    build_env = os.environ.copy()
    build_env.update(plan.environment())
    command = (
        f"cd {vllm_root} && "
        f"{uv} pip install --python {env_python} -r requirements/build.txt && "
        f"CCACHE_NOHASHDIR=true {uv} pip install --python {env_python} --no-build-isolation --no-deps -e ."
    )
    result = subprocess.run(["bash", "-lc", command], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=build_env)
    build["build_end_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    build["build_wall_time_seconds"] = round(time.monotonic() - started, 3)
    after = _command(["free", "-h"], check=True)
    build["ram_after_build"] = after["stdout"]
    _write_text(log_dir / "vllm_native_build.log", "$ " + command + "\n" + result.stdout + result.stderr)
    _write_json(log_dir / "build_manifest.json", manifest)
    if result.returncode != 0:
        raise ConfigError(
            "editable vLLM build failed; no automatic job-count fallback was used. "
            f"Requested/effective jobs: {plan.build_jobs_requested}/{plan.build_jobs_effective}. "
            f"See {log_dir / 'vllm_native_build.log'}"
        )


def _ensure_runpod_runtime(plan: Any, contract: dict[str, Any], log_dir: Path) -> None:
    uv = shutil.which("uv")
    if not uv:
        raise ConfigError("uv is required for runpod-dev")
    for path in (
        plan.repo_root / ".cache" / "uv",
        plan.repo_root / ".cache" / "vllm",
        plan.repo_root / ".cache" / "torch",
        plan.repo_root / "builds",
        plan.repo_root / "builds" / "tmp",
        plan.repo_root / "models" / "hf",
    ):
        path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(plan.environment())
    if not (plan.env_path / "bin" / "python").exists():
        _command([uv, "venv", "--python", contract["python"]["version"], str(plan.env_path)], check=True, log=log_dir / "uv_venv.log", env=env)
    py = plan.env_path / "bin" / "python"
    torch = contract["torch"]
    torch_requirement = f"{torch['wheel_url']}#sha256={torch['wheel_sha256']}"
    _command([uv, "pip", "install", "--python", str(py), torch_requirement], check=True, log=log_dir / "torch_install.log", env=env)
    _command([uv, "pip", "install", "--python", str(py), "-e", f"{plan.repo_root}[dev]"], check=True, log=log_dir / "project_install.log", env=env)
    for name in ("vllm", "lmcache"):
        item = contract[name]
        path = plan.repo_root / "externals" / name
        _ensure_pinned_external(path, item, log_dir)
    # Resolve runtime dependencies before native editable builds while forcing
    # the reproducible torch wheel selected by this RunPod contract.
    _command(
        [uv, "pip", "install", "--python", str(py), "-r", str(plan.repo_root / "externals" / "vllm" / "requirements" / "cuda.txt"), torch_requirement],
        check=True,
        log=log_dir / "vllm_runtime_dependencies.log",
        env=env,
    )
    _command(
        [uv, "pip", "install", "--python", str(py), "-r", str(plan.repo_root / "externals" / "lmcache" / "requirements" / "cuda.txt")],
        check=True,
        log=log_dir / "lmcache_runtime_dependencies.log",
        env=env,
    )


def _ensure_pinned_external(path: Path, item: dict[str, Any], log_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        clone = ["git", "clone", item["url"], str(path)]
        if item.get("branch"):
            clone = ["git", "clone", "--branch", item["branch"], "--single-branch", item["url"], str(path)]
        _command(clone, check=True, log=log_dir / f"{path.name}_clone.log")
    if not (path / ".git").exists():
        raise ConfigError(f"External path is not a Git checkout: {path}")
    dirty = _command(["git", "-C", str(path), "status", "--porcelain"], check=True)["stdout"].strip()
    if dirty:
        raise ConfigError(f"External source has local changes: {path}")
    remote_url = _command(["git", "-C", str(path), "remote", "get-url", "origin"], check=True)["stdout"].strip()
    if remote_url != item["url"]:
        _command(["git", "-C", str(path), "remote", "set-url", "origin", item["url"]], check=True, log=log_dir / f"{path.name}_remote_set_url.log")
    ref = item.get("branch") or item.get("ref") or item.get("tag")
    if ref:
        _command(["git", "-C", str(path), "fetch", "origin", ref], check=True, log=log_dir / f"{path.name}_fetch.log")
    else:
        _command(["git", "-C", str(path), "fetch", "origin"], check=True, log=log_dir / f"{path.name}_fetch.log")
    head = _command(["git", "-C", str(path), "rev-parse", "HEAD"], check=True)["stdout"].strip()
    if head != item["sha"]:
        _command(["git", "-C", str(path), "switch", "--detach", item["sha"]], check=True, log=log_dir / f"{path.name}_checkout.log")
        head = _command(["git", "-C", str(path), "rev-parse", "HEAD"], check=True)["stdout"].strip()
    if head != item["sha"]:
        raise ConfigError(f"Pinned {path.name} SHA mismatch: expected {item['sha']}, found {head}")


def _run_runpod_lmcache_build(plan: Any, manifest: dict[str, Any], log_dir: Path) -> None:
    uv = shutil.which("uv")
    py = plan.env_path / "bin" / "python"
    root = plan.repo_root / "externals" / "lmcache"
    if not uv or not py.exists() or not root.exists():
        raise ConfigError("LMCache editable build prerequisites are missing")
    env = os.environ.copy()
    env.update(plan.environment())
    started = time.monotonic()
    start_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    result = _command([uv, "pip", "install", "--python", str(py), "--no-build-isolation", "--no-deps", "-e", str(root)], check=False, log=log_dir / "lmcache_native_build.log", env=env)
    manifest["lmcache_editable_build"] = {
        "build_start_time": start_time,
        "build_end_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "build_wall_time_seconds": round(time.monotonic() - started, 3),
        "returncode": result["returncode"],
    }
    if result["returncode"] != 0:
        raise ConfigError(f"editable LMCache build failed; see {log_dir / 'lmcache_native_build.log'}")


def _validate_installed_runpod_runtime(plan: Any, contract: dict[str, Any]) -> None:
    py = plan.env_path / "bin" / "python"
    if not py.exists():
        raise ConfigError(f"RunPod environment is missing: {plan.env_path}")
    probe = "import torch,vllm,lmcache,putpocket_dataset_mining; print(torch.__version__); print(torch.version.cuda)"
    _command([str(py), "-c", probe], check=True)
    for name in ("vllm", "lmcache"):
        path = plan.repo_root / "externals" / name
        head = _command(["git", "-C", str(path), "rev-parse", "HEAD"], check=True)["stdout"].strip()
        if head != contract[name]["sha"]:
            raise ConfigError(f"Installed {name} SHA does not match RunPod lock")


def _run_server2_preset(args: argparse.Namespace) -> int:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_dir = REPO_ROOT / "logs" / "env_setup" / stamp
    run = BootstrapRun(REPO_ROOT, log_dir, bool(args.dry_run), bool(args.doctor_only))
    resolved_profile, resolved_arch = resolve_cuda_arch_contract(
        preset_default_profile="blackwell-rtx",
        cli_arch_profile=args.cuda_arch_profile,
        cli_arch_list=args.cuda_arch_list,
        env=os.environ,
    )
    effective_build_jobs = str(args.build_jobs or os.environ.get("PUTPOCKET_BUILD_JOBS") or "16")
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
        "cuda_arch_profile": resolved_profile,
        "requested_cuda_arch_list": resolved_arch.split(),
        "torch_cuda_arch_list": resolved_arch,
        "build_jobs_requested": args.build_jobs,
        "build_jobs_effective": int(effective_build_jobs),
        "max_jobs": int(effective_build_jobs),
        "cmake_build_parallel_level": int(effective_build_jobs),
        "nvcc_threads": int(os.environ.get("NVCC_THREADS", "1")),
        "docker_build_args": docker_build_args(resolved_arch),
        "stages": planned,
        "mutations": _server2_mutations(args),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    lock_context = nullcontext()
    if not args.doctor_only:
        lock_context = acquire_agent_locks(
            AgentConfig.load(),
            ["canonical-runtime", "build"],
            operation="bootstrap server2 build",
        )
    with lock_context:
        return _run_server2_preset_locked(args, run, plan, resolved_arch)


def _run_server2_preset_locked(args: argparse.Namespace, run: BootstrapRun, plan: dict[str, Any], resolved_arch: str) -> int:
    log_dir = run.log_dir
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
        build_env = _server2_build_env(resolved_arch, args.build_jobs)
        _ensure_uv_available(run)
        _ensure_python_environment(run)
        _ensure_server2_external_sources(run)
        _ensure_server2_runtime_packages(run, build_env)
        _ensure_project_editable(run)
        _ensure_externals(run, force_vllm=args.force_vllm_build, cuda_arch_list=resolved_arch, env=build_env)
        if not args.skip_docker:
            _validate_docker(run, force=args.force_docker_build)
    doctor = _doctor(run, skip_docker=args.skip_docker)
    _write_json(run.path("doctor.json"), doctor)
    after = _environment_manifest()
    _write_json(run.path("environment_after.json"), after)
    if not args.doctor_only:
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
    python_version = str(_server2_lock_data().get("environment", {}).get("python", "3.13.14"))
    _command([uv, "venv", "--python", python_version, str(SERVER2_ENV)], check=True, log=run.path("environment_create.log"))


def _server2_build_env(cuda_arch_list: str | None, build_jobs: int | None) -> dict[str, str]:
    env = os.environ.copy()
    jobs = str(build_jobs or env.get("PUTPOCKET_BUILD_JOBS") or "16")
    cuda_home = env.get("CUDA_HOME") or str(_server2_lock_data().get("hardware", {}).get("cuda_home", "/usr/local/cuda-12.9"))
    env["CUDA_HOME"] = cuda_home
    env["PATH"] = f"{cuda_home}/bin:{env.get('PATH', '')}"
    env["LD_LIBRARY_PATH"] = _prepend_env_path(env.get("LD_LIBRARY_PATH"), f"{cuda_home}/lib64")
    if cuda_arch_list:
        env["PUTPOCKET_CUDA_ARCH_LIST"] = cuda_arch_list
        env["TORCH_CUDA_ARCH_LIST"] = cuda_arch_list
    env["PUTPOCKET_BUILD_JOBS"] = jobs
    env["MAX_JOBS"] = jobs
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = jobs
    env["CARGO_BUILD_JOBS"] = jobs
    env["NVCC_THREADS"] = env.get("NVCC_THREADS", "1")
    env["CCACHE_NOHASHDIR"] = env.get("CCACHE_NOHASHDIR", "true")
    return env


def _server2_lock_data() -> dict[str, Any]:
    if yaml is None:
        return json.loads(SERVER2_LOCK.read_text(encoding="utf-8"))
    return yaml.safe_load(SERVER2_LOCK.read_text(encoding="utf-8")) or {}


def _ensure_server2_external_sources(run: BootstrapRun) -> None:
    for name in ("vllm", "lmcache", "cline"):
        try:
            checkout_external(name)
        except Exception as exc:  # noqa: BLE001
            raise ConfigError(f"failed to reconcile external source {name}: {exc}") from exc
    _write_json(run.path("external_revisions.json"), _external_revisions())


def _ensure_server2_runtime_packages(run: BootstrapRun, env: dict[str, str]) -> None:
    py = SERVER2_ENV / "bin" / "python"
    if not py.exists():
        raise ConfigError(f"Missing canonical Python: {py}")
    lock = _server2_lock_data()
    packages = lock.get("python_packages", {})
    install_specs: list[str] = []
    torch_spec = _server2_torch_requirement()
    if torch_spec:
        install_specs.append(torch_spec)
    elif packages.get("torch"):
        install_specs.append(f"torch=={packages['torch']}")
    for name in ("ray", "datasets", "transformers"):
        if packages.get(name):
            install_specs.append(f"{name}=={packages[name]}")
    if install_specs:
        _command(_pip_install_cmd(py, install_specs), check=True, log=run.path("runtime_package_sync.log"), env=env)
    _validate_server2_torch_cuda_contract(run, env=env, phase="runtime_package_sync")


def _ensure_project_editable(run: BootstrapRun) -> None:
    py = SERVER2_ENV / "bin" / "python"
    if not py.exists():
        raise ConfigError(f"Missing canonical Python: {py}")
    _command(_pip_install_cmd(py, ["setuptools==80.9.0"]), check=True, log=run.path("package_sync.log"))
    probe = _command([str(py), "-c", "import putpocket_dataset_mining; print(putpocket_dataset_mining.__file__)"], check=False)
    console_scripts = ["putpocket-dataset-mining", "putpocket-remote-verifier", "putpocket-agent"]
    console_ok = all((SERVER2_ENV / "bin" / name).exists() for name in console_scripts)
    if probe["returncode"] == 0 and str(REPO_ROOT / "src") in probe["stdout"] and console_ok:
        return
    _command(_pip_install_cmd(py, ["-e", f"{REPO_ROOT}[dev]"]), check=True, log=run.path("project_install.log"))


def _ensure_externals(run: BootstrapRun, *, force_vllm: bool, cuda_arch_list: str | None = None, env: dict[str, str] | None = None) -> None:
    py = SERVER2_ENV / "bin" / "python"
    for name in ("vllm", "lmcache", "cline"):
        path = SERVER2_EXTERNALS / name
        if not path.exists():
            raise ConfigError(f"Missing external source: {path}")
        if (path / ".git").exists() and _command(["git", "-C", str(path), "status", "--porcelain"], check=False)["stdout"].strip():
            raise ConfigError(f"External source has uncommitted changes: {path}")
    import_probe = _command([str(py), "-c", "import vllm, lmcache"], check=False)
    if force_vllm or import_probe["returncode"] != 0:
        build_env = env or _server2_build_env(cuda_arch_list, None)
        _validate_server2_torch_cuda_contract(run, env=build_env, phase="before_vllm_requirements")
        vllm_root = SERVER2_EXTERNALS / "vllm"
        cuda_requirements = vllm_root / "requirements" / "cuda.txt"
        build_requirements = vllm_root / "requirements" / "build" / "cuda.txt"
        if not build_requirements.exists():
            build_requirements = vllm_root / "requirements" / "build.txt"
        for label, req in (("runtime", cuda_requirements), ("build", build_requirements)):
            if req.exists():
                _command(_pip_install_cmd(py, ["-r", str(req)]), check=True, log=run.path(f"vllm_{label}_requirements.log"), env=build_env)
                _reinstall_server2_torch(run, build_env, log_name=f"torch_after_vllm_{label}_requirements.log")
                _validate_server2_torch_cuda_contract(run, env=build_env, phase=f"after_vllm_{label}_requirements")
        _validate_cuda12_runtime_library(run, build_env)
        _validate_server2_torch_cuda_contract(run, env=build_env, phase="before_vllm_native_build")
        _command(
            _pip_install_cmd(py, ["--no-build-isolation", "--no-deps", "-e", str(SERVER2_EXTERNALS / "vllm")]),
            check=True,
            log=run.path("externals_install.log"),
            env=build_env,
        )
        _validate_server2_torch_cuda_contract(run, env=build_env, phase="after_vllm_native_build")
        _validate_cuda12_runtime_library(run, build_env)
        lmcache_root = SERVER2_EXTERNALS / "lmcache"
        lmcache_common = lmcache_root / "requirements" / "common.txt"
        lmcache_cuda_core = lmcache_root / "requirements" / "cuda_core.txt"
        lmcache_deps = [item for item in (lmcache_common, lmcache_cuda_core) if item.exists()]
        if lmcache_deps:
            dep_args: list[str] = []
            for req in lmcache_deps:
                dep_args.extend(["-r", str(req)])
            requirement = _server2_torch_requirement()
            if requirement:
                dep_args.append(requirement)
            _command(_pip_install_cmd(py, dep_args), check=True, log=run.path("lmcache_runtime_requirements.log"), env=build_env)
            _reinstall_server2_torch(run, build_env, log_name="torch_after_lmcache_requirements.log")
            _validate_server2_torch_cuda_contract(run, env=build_env, phase="after_lmcache_requirements")
        _validate_server2_torch_cuda_contract(run, env=build_env, phase="before_lmcache_native_build")
        _command(
            _pip_install_cmd(py, ["--no-build-isolation", "--no-deps", "-e", str(SERVER2_EXTERNALS / "lmcache")]),
            check=True,
            log=run.path("lmcache_install.log"),
            env=build_env,
        )
        _validate_server2_torch_cuda_contract(run, env=build_env, phase="after_lmcache_native_build")


def _prepend_env_path(existing: str | None, new_item: str) -> str:
    items = [item for item in (existing or "").split(os.pathsep) if item]
    items = [item for item in items if item != new_item]
    return os.pathsep.join([new_item, *items])


def _server2_torch_requirement() -> str | None:
    packages = _server2_lock_data().get("python_packages", {})
    torch_wheel = packages.get("torch_wheel", {})
    if isinstance(torch_wheel, dict) and torch_wheel.get("url") and torch_wheel.get("sha256"):
        return f"{torch_wheel['url']}#sha256={torch_wheel['sha256']}"
    if packages.get("torch"):
        return f"torch=={packages['torch']}"
    return None


def _reinstall_server2_torch(run: BootstrapRun, env: dict[str, str], *, log_name: str) -> None:
    requirement = _server2_torch_requirement()
    if not requirement:
        raise ConfigError("Server-2 lock is missing an exact torch requirement")
    py = SERVER2_ENV / "bin" / "python"
    _command(_pip_install_cmd(py, [requirement]), check=True, log=run.path(log_name), env=env)


def _validate_server2_torch_cuda_contract(run: BootstrapRun, *, env: dict[str, str], phase: str) -> None:
    py = SERVER2_ENV / "bin" / "python"
    lock = _server2_lock_data()
    packages = lock.get("python_packages", {})
    torch_wheel = packages.get("torch_wheel", {}) if isinstance(packages.get("torch_wheel"), dict) else {}
    expected_torch = str(packages.get("torch", "2.11.0+cu129"))
    expected_torch_cuda = str(torch_wheel.get("torch_cuda", "12.9"))
    expected_nvcc = str(lock.get("hardware", {}).get("cuda_home", "/usr/local/cuda-12.9"))
    script = r"""
import importlib.metadata as md
import json
import sys
import torch

direct = None
try:
    dist = md.distribution("torch")
    text = dist.read_text("direct_url.json")
    direct = json.loads(text) if text else None
except Exception:
    direct = None

payload = {
    "python": sys.executable,
    "torch_version": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "direct_url": direct,
}
print(json.dumps(payload, sort_keys=True))
"""
    torch_result = _command([str(py), "-c", script], check=False, env=env)
    nvcc_result = _command(["nvcc", "--version"], check=False, env=env)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "phase": phase,
        "expected_torch": expected_torch,
        "expected_torch_cuda": expected_torch_cuda,
        "expected_cuda_home": expected_nvcc,
        "torch_probe": torch_result,
        "nvcc_probe": nvcc_result,
        "torch_source": torch_wheel.get("index_url") or torch_wheel.get("url"),
    }
    _write_json(run.path(f"torch_cuda_contract_{phase}.json"), payload)
    if torch_result["returncode"] != 0 or nvcc_result["returncode"] != 0:
        raise ConfigError("TORCH_SYSTEM_CUDA_CONTRACT_MISMATCH: unable to inspect torch or nvcc")
    observed = json.loads(torch_result["stdout"])
    nvcc_version = _parse_nvcc_cuda_version(nvcc_result["stdout"])
    if (
        observed.get("torch_version") != expected_torch
        or str(observed.get("torch_cuda")) != expected_torch_cuda
        or nvcc_version != expected_torch_cuda
    ):
        raise ConfigError(
            "TORCH_SYSTEM_CUDA_CONTRACT_MISMATCH: "
            f"phase={phase}; torch={observed.get('torch_version')}; "
            f"torch_cuda={observed.get('torch_cuda')}; nvcc={nvcc_version}; "
            f"torch_source={payload['torch_source']}"
        )


def _parse_nvcc_cuda_version(output: str) -> str:
    match = re.search(r"release\s+(\d+\.\d+)", output)
    if not match:
        raise ConfigError(f"Unable to parse nvcc CUDA version from: {output}")
    return match.group(1)


def _validate_cuda12_runtime_library(run: BootstrapRun, env: dict[str, str]) -> None:
    script = r"""
import ctypes
import ctypes.util
import json
import os

candidate = ctypes.util.find_library("cudart")
loaded = None
error = None
try:
    handle = ctypes.CDLL("libcudart.so.12")
    loaded = getattr(handle, "_name", "libcudart.so.12")
except Exception as exc:
    error = f"{type(exc).__name__}: {exc}"

print(json.dumps({
    "candidate": candidate,
    "loaded": loaded,
    "error": error,
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
}, sort_keys=True))
"""
    py = SERVER2_ENV / "bin" / "python"
    result = _command([str(py), "-c", script], check=False, env=env)
    _write_json(run.path("cuda12_runtime_library.json"), {"schema_version": 1, "probe": result})
    if result["returncode"] != 0:
        raise ConfigError("CUDA12_RUNTIME_LIBRARY_UNAVAILABLE: library inspection failed")
    payload = json.loads(result["stdout"])
    if payload.get("error"):
        raise ConfigError(f"CUDA12_RUNTIME_LIBRARY_UNAVAILABLE: {payload['error']}")


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
    checks["uv_pip_check"] = _command(_pip_check_cmd(py), check=False)
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
    py = SERVER2_ENV / "bin" / "python"
    if py.exists():
        freeze = _command(_pip_freeze_cmd(py), check=False)
        _write_text(root / "pip_freeze.txt", freeze["stdout"] + freeze["stderr"])
    else:
        _write_text(root / "pip_freeze.txt", "environment_missing\n")
    _write_json(root / "external_revisions.json", manifest.get("externals", {}))


def _uv_executable() -> str | None:
    uv = shutil.which("uv")
    return uv if uv else None


def _pip_install_cmd(py: Path, args: list[str]) -> list[str]:
    uv = _uv_executable()
    if uv:
        return [uv, "pip", "install", "--python", str(py), *args]
    return [str(py), "-m", "pip", "install", *args]


def _pip_check_cmd(py: Path) -> list[str]:
    uv = _uv_executable()
    if uv:
        return [uv, "pip", "check", "--python", str(py)]
    return [str(py), "-m", "pip", "check"]


def _pip_freeze_cmd(py: Path) -> list[str]:
    uv = _uv_executable()
    if uv:
        return [uv, "pip", "freeze", "--python", str(py)]
    return [str(py), "-m", "pip", "freeze"]


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
    out = REPO_ROOT / "logs" / "env_consolidation" / "server2_blackwell.effective.lock.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    if yaml is not None:
        out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    else:
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    from .execution_config import ExecutionConfig, HardwareProfile, ServerProfile, default_hardware_for_server

    mapping: dict[str, Any] = {}
    if args.role and not args.execution_role:
        mapping["execution_role"] = args.role
    for key in ("execution_role", "workspace_backend", "verifier_backend", "vllm_profile"):
        value = getattr(args, key)
        if value:
            mapping[key] = value
    server = ServerProfile(args.server_profile)
    hardware_aliases = {"rtx3090": "sm86", "hopper": "sm90", "blackwell": "sm120"}
    hardware = HardwareProfile(hardware_aliases.get(args.hardware_profile, args.hardware_profile))
    if hardware == HardwareProfile.AUTO:
        hardware = default_hardware_for_server(server)
    preset_profile = {
        ServerProfile.SERVER1_RTX3090: "rtx3090",
        ServerProfile.RUNPOD_HOPPER: "hopper",
        ServerProfile.SERVER2_BLACKWELL: "blackwell-rtx",
        ServerProfile.SERVER2_RTXPRO6000_BLACKWELL: "blackwell-rtx",
        ServerProfile.CUSTOM: "portable-nvidia",
    }[server]
    resolved_profile, resolved_arch = resolve_cuda_arch_contract(
        preset_default_profile=preset_profile,
        cli_arch_profile=args.cuda_arch_profile,
        cli_arch_list=args.cuda_arch_list,
        env=os.environ,
    )
    mapping["cuda_arch_list"] = resolved_arch
    mapping["hardware_profile"] = hardware.value
    mapping["server_profile"] = server.value
    mapping["cuda_arch_profile"] = resolved_profile
    setattr(args, "_resolved_cuda_arch_profile", resolved_profile)
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
    from .execution_config import DOCKER_DISABLED_FOR_STATIC_ONLY, cuda_arch_for_profile

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
    resolved_arch = config.cuda_arch_list or cuda_arch_for_profile(config.hardware_profile)
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
        "cuda_arch_profile": getattr(args, "_resolved_cuda_arch_profile", None) or getattr(args, "cuda_arch_profile", None) or "resolved",
        "cuda_arch_list": resolved_arch,
        "torch_cuda_arch_list": resolved_arch,
        "docker_build_args": docker_build_args(resolved_arch) if resolved_arch else [],
        "dry_run": args.dry_run,
    }
    if extra:
        manifest.update(extra)
    _write_manifest(args.manifest_dir, stage, manifest)


def _gpu_phase(config: ExecutionConfig, args: argparse.Namespace) -> None:
    from .execution_config import HardwareProfile, cuda_arch_for_profile

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
        "torch_cuda_arch_list": config.cuda_arch_list or arch,
        "cuda_arch_profile": getattr(args, "_resolved_cuda_arch_profile", None) or getattr(args, "cuda_arch_profile", None) or "resolved",
        "docker_build_args": docker_build_args(config.cuda_arch_list or arch),
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


def _command(cmd: list[str], *, check: bool, log: Path | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=REPO_ROOT, env=env)
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
