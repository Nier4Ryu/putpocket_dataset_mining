from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .errors import ConfigError
from .glm52_runpod import (
    PACKAGE_LOCK,
    SCHEDULE,
    analyze_probe,
    capture_probe,
    load_package_lock,
    prepare_probe,
    run_doctor,
    validate_project_artifacts,
    validate_schedule,
    validate_vllm_tree,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="putpocket-glm52-runpod")
    sub = parser.add_subparsers(dest="command", required=True)

    package = sub.add_parser("validate-package")
    package.add_argument("--lock", default=str(PACKAGE_LOCK))
    package.add_argument("--project-root", required=True)
    package.add_argument("--vllm-root")
    package.add_argument(
        "--phase",
        choices=["project_artifacts", "upstream_base", "post_legacy", "post_true_partial", "post_score_diagnostic"],
        default="project_artifacts",
    )

    schedule = sub.add_parser("validate-schedule")
    schedule.add_argument("--schedule", default=str(SCHEDULE))

    doctor = sub.add_parser("doctor")
    doctor.add_argument("--lock", default=str(PACKAGE_LOCK))
    doctor.add_argument("--project-root", required=True)
    doctor.add_argument("--vllm-root", required=True)
    doctor.add_argument("--model-root", required=True)
    doctor.add_argument("--expected-project-commit", required=True)
    doctor.add_argument("--output", required=True)

    prepare = sub.add_parser("prepare-probe")
    prepare.add_argument("--lock", default=str(PACKAGE_LOCK))
    prepare.add_argument("--model-root", required=True)
    prepare.add_argument("--harness-root", required=True)
    prepare.add_argument("--output", required=True)

    capture = sub.add_parser("capture")
    capture.add_argument("--lock", default=str(PACKAGE_LOCK))
    capture.add_argument("--doctor-report", required=True)
    capture.add_argument("--probe", required=True)
    capture.add_argument("--model-root", required=True)
    capture.add_argument("--output-root", required=True)
    capture.add_argument("--dry-run", action="store_true")

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--lock", default=str(PACKAGE_LOCK))
    analyze.add_argument("--doctor-report", required=True)
    analyze.add_argument("--capture-root", required=True)
    analyze.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate-package":
            lock = load_package_lock(args.lock)
            if args.phase == "project_artifacts":
                result = validate_project_artifacts(args.project_root, lock)
            else:
                if not args.vllm_root:
                    raise ConfigError("--vllm-root is required for a vLLM phase")
                result = validate_vllm_tree(args.vllm_root, lock, args.phase)
        elif args.command == "validate-schedule":
            result = validate_schedule(args.schedule)
        elif args.command == "doctor":
            result = run_doctor(
                project_root=args.project_root,
                vllm_root=args.vllm_root,
                model_root=args.model_root,
                expected_project_commit=args.expected_project_commit,
                output=args.output,
                lock_path=args.lock,
            )
            if result["payload"]["status"] != "passed":
                print(json.dumps(result, indent=2, sort_keys=True))
                return 2
        elif args.command == "prepare-probe":
            result = prepare_probe(
                model_root=args.model_root,
                harness_root=args.harness_root,
                output=args.output,
                lock_path=args.lock,
            )
        elif args.command == "capture":
            result = capture_probe(
                doctor_report=args.doctor_report,
                probe_path=args.probe,
                model_root=args.model_root,
                output_root=args.output_root,
                lock_path=args.lock,
                dry_run=args.dry_run,
            )
        else:
            result = analyze_probe(
                doctor_report=args.doctor_report,
                capture_root=args.capture_root,
                output_root=args.output_root,
                lock_path=args.lock,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ConfigError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
