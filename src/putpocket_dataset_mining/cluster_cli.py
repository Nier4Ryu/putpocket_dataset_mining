from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .cluster_config import (
    PROFILE_DIR,
    load_cluster_profile,
    load_cluster_site,
    validate_environment_lock,
)
from .cluster_environment import bootstrap_plan_json, build_bootstrap_plan, execute_bootstrap_plan
from .cluster_manifest import capture_run_manifest, execute_guarded, write_manifest
from .cluster_readiness import run_readiness
from .cluster_safety import require_slurm_allocation, safe_absolute_path
from .cluster_slurm import SLURM_JOB_KINDS, render_slurm_job
from .errors import ConfigError


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(args_list)
    try:
        if args.command == "profiles":
            return _profiles_validate(args)
        if args.command == "render":
            return _render(args)
        if args.command == "allocation-check":
            print(json.dumps({"schema_version": 1, "allocation": require_slurm_allocation()}, indent=2, sort_keys=True))
            return 0
        if args.command == "env-bootstrap":
            return _env_bootstrap(args)
        if args.command == "readiness":
            return _readiness(args, args_list)
        if args.command == "run-guarded":
            return _run_guarded(args)
    except ConfigError as exc:
        print(json.dumps({"schema_version": 1, "status": "failed", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    parser.error("a command is required")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="putpocket-cluster")
    commands = parser.add_subparsers(dest="command")

    profiles = commands.add_parser("profiles", help="Validate committed Cluster profiles and lock")
    profiles.add_argument("action", choices=["validate"])
    profiles.add_argument("--profile", action="append", default=[])

    render = commands.add_parser("render", help="Render a Slurm script; never submits it")
    render.add_argument("--profile", required=True)
    render.add_argument("--site", required=True)
    render.add_argument("--job", choices=SLURM_JOB_KINDS, required=True)
    render.add_argument("--output", default=None)

    commands.add_parser("allocation-check", help="Refuse unless explicit Slurm allocation variables are valid")

    bootstrap = commands.add_parser("env-bootstrap", help="Render or execute the allocation-guarded SM90 preset")
    bootstrap.add_argument("--lock", required=True)
    bootstrap.add_argument("--repository-root", required=True)
    bootstrap.add_argument("--environment-root", required=True)
    bootstrap.add_argument("--vllm-source-root", required=True)
    bootstrap.add_argument("--cache-root", required=True)
    bootstrap.add_argument("--python-executable", required=True)
    bootstrap.add_argument("--uv-executable", required=True)
    bootstrap.add_argument("--git-executable", required=True)
    bootstrap.add_argument("--build-jobs", type=int, required=True)
    bootstrap.add_argument("--execute", action="store_true")

    readiness = commands.add_parser("readiness", help="Run staged Cluster readiness checks")
    readiness.add_argument("--profile", required=True)
    readiness.add_argument(
        "--stage",
        choices=["static", "allocation", "gpu", "imports", "checkpoint", "model-load", "generation-handoff", "all"],
        required=True,
    )
    readiness.add_argument("--model-path", default=None)
    readiness.add_argument("--model-revision", default=None)
    readiness.add_argument("--artifact-root", default=None)
    readiness.add_argument("--git-executable", default=None)
    readiness.add_argument("--nvidia-smi-executable", default=None)
    readiness.add_argument("--nvcc-executable", default=None)

    guarded = commands.add_parser("run-guarded", help="Execute a heavy command only inside a Slurm allocation")
    guarded.add_argument(
        "--action",
        choices=[
            "environment-build",
            "dependency-install",
            "checkpoint-stage",
            "gpu-smoke",
            "model-load",
            "benchmark",
            "one-shot-generation",
        ],
        required=True,
    )
    guarded.add_argument("--profile", required=True)
    guarded.add_argument("--artifact-root", required=True)
    guarded.add_argument("--git-executable", required=True)
    guarded.add_argument("--nvidia-smi-executable", required=True)
    guarded.add_argument("--nvcc-executable", required=True)
    guarded.add_argument("--model-revision", default=None)
    guarded.add_argument("heavy_command", nargs=argparse.REMAINDER)
    return parser


def _profiles_validate(args: argparse.Namespace) -> int:
    validate_environment_lock()
    paths = [Path(value) for value in args.profile] if args.profile else sorted(PROFILE_DIR.glob("*.yaml"))
    profiles = [load_cluster_profile(path) for path in paths]
    payload = {
        "schema_version": 1,
        "status": "passed",
        "profile_count": len(profiles),
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "model_id": profile.model_id,
                "gpus": profile.world_size,
                "tp": profile.tensor_parallel_size,
                "pcp": profile.prefill_context_parallel_size,
                "ep": profile.expert_parallel,
            }
            for profile in profiles
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _render(args: argparse.Namespace) -> int:
    profile = load_cluster_profile(args.profile)
    site = load_cluster_site(args.site)
    rendered = render_slurm_job(profile, site, args.job)
    if args.output:
        output = Path(args.output)
        if output.suffix != ".sbatch":
            raise ConfigError("Rendered Slurm output must use the .sbatch suffix")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(json.dumps({"status": "rendered", "submitted": False, "output": str(output)}, indent=2))
    else:
        print(rendered, end="")
    return 0


def _env_bootstrap(args: argparse.Namespace) -> int:
    plan = build_bootstrap_plan(
        lock_path=args.lock,
        repository_root=args.repository_root,
        environment_root=args.environment_root,
        vllm_source_root=args.vllm_source_root,
        cache_root=args.cache_root,
        python_executable=args.python_executable,
        uv_executable=args.uv_executable,
        git_executable=args.git_executable,
        build_jobs=args.build_jobs,
    )
    print(bootstrap_plan_json(plan))
    if args.execute:
        execute_bootstrap_plan(plan)
    return 0


def _readiness(args: argparse.Namespace, args_list: list[str]) -> int:
    profile = load_cluster_profile(args.profile)
    report = run_readiness(
        profile,
        stage=args.stage,
        model_path=args.model_path,
        model_revision=args.model_revision or None,
        nvidia_smi_executable=args.nvidia_smi_executable,
    )
    payload = report.as_dict()
    if args.stage != "static" and args.artifact_root:
        root = safe_absolute_path(args.artifact_root, "artifact_root")
        if not args.git_executable or not args.nvidia_smi_executable or not args.nvcc_executable:
            raise ConfigError("non-static readiness artifact capture requires explicit git, nvidia-smi, and nvcc executables")
        command = [sys.executable, "-m", "putpocket_dataset_mining.cluster_cli", *args_list]
        manifest = capture_run_manifest(
            profile=profile,
            command=command,
            artifact_root=root,
            git_executable=args.git_executable,
            nvidia_smi_executable=args.nvidia_smi_executable,
            nvcc_executable=args.nvcc_executable,
            model_revision=args.model_revision or None,
        )
        manifest["action"] = "readiness"
        manifest["status"] = report.status
        write_manifest(root / "readiness_manifest.json", manifest)
        write_manifest(root / "readiness_report.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.succeeded else 2


def _run_guarded(args: argparse.Namespace) -> int:
    command = list(args.heavy_command)
    if command and command[0] == "--":
        command = command[1:]
    profile = load_cluster_profile(args.profile)
    return execute_guarded(
        action=args.action,
        profile=profile,
        command=command,
        artifact_root=args.artifact_root,
        git_executable=args.git_executable,
        nvidia_smi_executable=args.nvidia_smi_executable,
        nvcc_executable=args.nvcc_executable,
        model_revision=args.model_revision or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
