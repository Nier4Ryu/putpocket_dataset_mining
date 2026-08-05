from __future__ import annotations

import argparse
import json
import sys

from .doctor import collect_doctor_report, format_doctor_report
from .errors import ConfigError, InfraError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="putpocket-dataset-mining")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check local runtime dependencies and config paths.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    docker = sub.add_parser("docker", help="Docker image helpers.")
    docker_sub = docker.add_subparsers(dest="docker_command", required=True)
    ensure_image = docker_sub.add_parser("ensure-image", help="Build a configured image if missing.")
    ensure_image.add_argument("--config", default=None, help="Dataset mining config containing docker.image/dockerfile.")

    externals = sub.add_parser("externals", help="External checkout/install helpers.")
    externals_sub = externals.add_subparsers(dest="externals_command", required=True)
    checkout = externals_sub.add_parser("checkout", help="Clone one or all externals if missing.")
    checkout.add_argument("name", choices=["vllm", "lmcache", "cline", "all"])
    install = externals_sub.add_parser("install-editable", help="Editable-install vLLM/LMCache with capped build threads.")
    install.add_argument("name", choices=["vllm", "lmcache"])
    install.add_argument("--python", default="python", help="Python executable inside the target env.")

    single = sub.add_parser("single", help="Run one single-sample mining attempt.")
    single.add_argument("--config", default="configs/dataset_mining/mbpp_stateful_single.yaml")
    single.add_argument("--sample-index", type=int, default=0)
    single.add_argument("--split", default=None)
    single.add_argument("--run-id", default=None)
    single.add_argument("--attempt-id", default=None)
    single.add_argument("--no-index", action="store_true", help="Do not write the SQLite index.")

    multi = sub.add_parser("multi", help="Run multi-sample master-worker mining.")
    multi.add_argument("--config", default="configs/dataset_mining/mbpp_stateful_multi.yaml")
    multi.add_argument("--profile", choices=["debug", "first_parallel", "full_server"], default="debug")
    multi.add_argument("--run-id", default=None)
    multi.add_argument("--rerun-failed-infra", action="store_true")

    stop = sub.add_parser("stop", help="Create a graceful stop file for a run id.")
    stop.add_argument("run_id")

    materialize = sub.add_parser("materialize", help="Rebuild a local materialized dataset view from the SQLite index.")
    materialize.add_argument("dataset_version")

    preflight = sub.add_parser("remote-preflight", help="Check an SSH/rsync remote Docker worker.")
    preflight.add_argument("--config", default=None, help="Remote verifier config YAML.")
    preflight.add_argument("--docker-image", default=None)

    remote_test = sub.add_parser("remote-test", help="Run disposable remote verifier pass/fail/timeout fixtures.")
    remote_test.add_argument("--config", required=True, help="Remote verifier config YAML.")
    remote_test.add_argument("--fixtures", default="pass,fail,timeout")
    remote_test.add_argument("--timeout-fixture-sec", type=int, default=2)
    remote_test.add_argument("--output-dir", default=None)
    remote_test.add_argument("--dry-run", action="store_true")

    sync = sub.add_parser("sync-artifacts", help="Build or copy a selective artifact replication manifest.")
    sync.add_argument("--source-root", required=True)
    sync.add_argument("--destination-root", default=None)
    sync.add_argument("--profile", choices=["analysis_minimal", "analysis_with_workspaces", "analysis_with_selected_kv", "verifier_input", "verifier_output"], default="analysis_minimal")
    sync.add_argument("--manifest-out", default=None)
    sync.add_argument("--dry-run", action="store_true")

    bootstrap = sub.add_parser("bootstrap-sr", help="Run staged SR bootstrap checks.")
    bootstrap.add_argument("--phase", choices=["cpu", "gpu", "all"], default=None)
    bootstrap.add_argument("--stage", choices=["preflight", "system", "core", "verifier", "vllm_source", "vllm_build", "validate", "all"], default=None)
    bootstrap.add_argument("--server-profile", choices=["server1_rtx3090", "server2_rtxpro6000_blackwell", "server2_blackwell", "runpod_hopper", "custom"], default="custom")
    bootstrap.add_argument("--hardware-profile", choices=["cpu", "sm86", "sm90", "sm120", "auto"], default="auto")
    bootstrap.add_argument("--role", choices=["controller", "verifier", "model_server", "development"], default=None)
    bootstrap.add_argument("--execution-role", choices=["local_controller", "cloud_controller", "verifier_host"], default=None)
    bootstrap.add_argument("--workspace-backend", choices=["local_docker", "remote_ssh_docker"], default=None)
    bootstrap.add_argument("--verifier-backend", choices=["local_docker", "remote_ssh_docker", "disabled"], default=None)
    bootstrap.add_argument("--vllm-profile", choices=["clean", "patched", "skip"], default="patched")
    bootstrap.add_argument("--build-vllm", choices=["auto", "yes", "no"], default="auto")
    bootstrap.add_argument("--allow-system-install", action="store_true")
    bootstrap.add_argument("--allow-docker-build", action="store_true")
    bootstrap.add_argument("--allow-vllm-build", action="store_true")
    bootstrap.add_argument("--runtime-checks", action="store_true")
    bootstrap.add_argument("--dry-run", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "doctor":
        report = collect_doctor_report()
        print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_doctor_report(report))
        return 0

    if args.command == "docker":
        from .docker_workspace import DockerImageManager
        from .config import load_yaml

        try:
            if args.config:
                from .constants import REPO_ROOT
                from pathlib import Path

                cfg = load_yaml(args.config)
                docker_cfg = cfg.get("docker", {})
                dockerfile = Path(docker_cfg.get("dockerfile", "docker/default_python/Dockerfile"))
                if not dockerfile.is_absolute():
                    dockerfile = REPO_ROOT / dockerfile
                DockerImageManager(docker_cfg.get("image", "putpocket-default-python:ubuntu22.04-py313-v1"), dockerfile).ensure_image(
                    timeout_sec=int(docker_cfg.get("timeouts", {}).get("image_build_sec", 900))
                )
            else:
                DockerImageManager.from_default().ensure_image()
        except InfraError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        return 0

    if args.command == "externals":
        from .externals import EXTERNALS, checkout_external, editable_install

        if args.externals_command == "checkout":
            names = list(EXTERNALS) if args.name == "all" else [args.name]
            for name in names:
                repo = checkout_external(name)
                print(f"{name}: {repo.path}")
            return 0
        if args.externals_command == "install-editable":
            editable_install(EXTERNALS[args.name], python=args.python)
            return 0

    if args.command == "single":
        from .single import SingleSampleRunner

        result = SingleSampleRunner.from_config_path(args.config).run(
            sample_index=args.sample_index,
            split=args.split,
            run_id=args.run_id,
            attempt_id=args.attempt_id,
            write_index=not args.no_index,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["final_status"] in {"accepted", "rejected", "uncertain"} else 2

    if args.command == "multi":
        from .multi import MultiSampleMaster

        try:
            result = MultiSampleMaster.from_config_path(args.config).run(
                profile_name=args.profile,
                run_id=args.run_id,
                rerun_failed_infra=args.rerun_failed_infra,
            )
        except ConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "stop":
        from .multi import create_stop_file

        path = create_stop_file(args.run_id)
        print(str(path))
        return 0

    if args.command == "materialize":
        from .storage import MiningIndex, DatasetMaterializer

        index = MiningIndex.default()
        materializer = DatasetMaterializer(index)
        try:
            path = materializer.materialize_dataset(args.dataset_version)
        except ConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(str(path))
        return 0

    if args.command == "remote-preflight":
        from .execution_config import ExecutionConfig
        from .ssh_transport import SshRsyncTransport

        try:
            cfg = _load_execution_config(args.config) if args.config else ExecutionConfig.from_env_and_mapping()
            transport = SshRsyncTransport(cfg.remote)
            result = transport.lightweight_preflight(args.docker_image)
        except (ConfigError, InfraError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(result.__dict__, indent=2, sort_keys=True))
        return 0 if result.status == "REMOTE_DOCKER_PREFLIGHT_PASSED" else 2

    if args.command == "remote-test":
        from pathlib import Path
        from .remote_fixtures import run_remote_verifier_fixtures

        try:
            cfg = _load_execution_config(args.config)
            result = run_remote_verifier_fixtures(
                execution_config=cfg,
                fixtures=[item.strip() for item in args.fixtures.split(",") if item.strip()],
                timeout_fixture_sec=args.timeout_fixture_sec,
                output_dir=Path(args.output_dir) if args.output_dir else None,
                dry_run=args.dry_run,
            )
        except (ConfigError, InfraError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    if args.command == "sync-artifacts":
        from pathlib import Path
        from .artifact_sync import build_sync_manifest, copy_from_manifest

        source = Path(args.source_root)
        manifest = build_sync_manifest(source, args.profile)
        if args.manifest_out:
            Path(args.manifest_out).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        if args.destination_root:
            result = copy_from_manifest(source, Path(args.destination_root), manifest, dry_run=args.dry_run)
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if args.command == "bootstrap-sr":
        from .bootstrap_sr import run_bootstrap

        forwarded = ["--server-profile",
            args.server_profile,
            "--hardware-profile",
            args.hardware_profile,
            "--vllm-profile",
            args.vllm_profile,
            "--build-vllm",
            args.build_vllm,
        ]
        if args.phase:
            forwarded.extend(["--phase", args.phase])
        if args.stage:
            forwarded.extend(["--stage", args.stage])
        if args.role:
            forwarded.extend(["--role", args.role])
        if args.execution_role:
            forwarded.extend(["--execution-role", args.execution_role])
        if args.workspace_backend:
            forwarded.extend(["--workspace-backend", args.workspace_backend])
        if args.verifier_backend:
            forwarded.extend(["--verifier-backend", args.verifier_backend])
        if args.dry_run:
            forwarded.append("--dry-run")
        for flag in ["allow_system_install", "allow_docker_build", "allow_vllm_build", "runtime_checks"]:
            if getattr(args, flag):
                forwarded.append("--" + flag.replace("_", "-"))
        return run_bootstrap(forwarded)

    return 1


def _load_execution_config(path: str):
    from .config import load_yaml
    from .execution_config import ExecutionConfig

    return ExecutionConfig.from_remote_verifier_mapping(load_yaml(path))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
