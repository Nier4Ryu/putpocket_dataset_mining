from __future__ import annotations

import argparse
import json
import sys

from .doctor import collect_doctor_report, format_doctor_report
from .errors import InfraError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="putpocket-dataset-mining")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="Check local runtime dependencies and config paths.")
    doctor.add_argument("--json", action="store_true", help="Print machine-readable JSON.")

    docker = sub.add_parser("docker", help="Docker image helpers.")
    docker_sub = docker.add_subparsers(dest="docker_command", required=True)
    docker_sub.add_parser("ensure-image", help="Build the default image if missing.")

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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "doctor":
        report = collect_doctor_report()
        print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_doctor_report(report))
        return 0

    if args.command == "docker":
        from .docker_workspace import DockerImageManager

        try:
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

        result = MultiSampleMaster.from_config_path(args.config).run(
            profile_name=args.profile,
            run_id=args.run_id,
            rerun_failed_infra=args.rerun_failed_infra,
        )
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
        path = materializer.materialize_dataset(args.dataset_version)
        print(str(path))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
