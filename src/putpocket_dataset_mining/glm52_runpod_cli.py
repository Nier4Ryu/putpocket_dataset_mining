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
    analyze_query_sum_probe,
    capture_matrix_episode,
    capture_probe,
    capture_query_sum_probe,
    load_package_lock,
    prepare_probe,
    prepare_final_two_query_probe,
    run_doctor,
    score_matrix_capture,
    validate_project_artifacts,
    validate_schedule,
    validate_vllm_tree,
)


def _range(value: str) -> list[int]:
    try:
        left, right = value.split(":", 1)
        result = [int(left), int(right)]
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("range must be START:END") from exc
    if not 0 <= result[0] < result[1]:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= START < END")
    return result


def _layers(value: str) -> list[int]:
    try:
        result = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not result or result != sorted(set(result)) or min(result) < 0:
        raise argparse.ArgumentTypeError("layers must be nonempty, sorted, and unique")
    return result


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

    final_prepare = sub.add_parser("prepare-final-probe")
    final_prepare.add_argument("--lock", default=str(PACKAGE_LOCK))
    final_prepare.add_argument("--model-root", required=True)
    final_prepare.add_argument("--harness-root", required=True)
    final_prepare.add_argument("--output", required=True)
    final_prepare.add_argument("--matrix-output", required=True)

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

    query_capture = sub.add_parser("capture-query-sum")
    query_capture.add_argument("--lock", default=str(PACKAGE_LOCK))
    query_capture.add_argument("--doctor-report", required=True)
    query_capture.add_argument("--probe", required=True)
    query_capture.add_argument("--model-root", required=True)
    query_capture.add_argument("--output-root", required=True)
    query_capture.add_argument("--dry-run", action="store_true")

    query_analyze = sub.add_parser("analyze-query-sum")
    query_analyze.add_argument("--lock", default=str(PACKAGE_LOCK))
    query_analyze.add_argument("--doctor-report", required=True)
    query_analyze.add_argument("--probe", required=True)
    query_analyze.add_argument("--capture-root", required=True)
    query_analyze.add_argument("--output-root", required=True)

    matrix_capture = sub.add_parser("capture-matrix")
    matrix_capture.add_argument("--lock", default=str(PACKAGE_LOCK))
    matrix_capture.add_argument("--doctor-report", required=True)
    matrix_capture.add_argument("--episode-manifest", required=True)
    matrix_capture.add_argument("--model-root", required=True)
    matrix_capture.add_argument("--output-root", required=True)
    matrix_capture.add_argument("--dry-run", action="store_true")

    matrix_score = sub.add_parser("score-matrix")
    matrix_score.add_argument("--episode-manifest", required=True)
    matrix_score.add_argument("--capture-root", required=True)
    matrix_score.add_argument("--output-root", required=True)
    matrix_score.add_argument("--q1-range", type=_range, action="append", required=True)
    matrix_score.add_argument("--q2-range", type=_range, action="append", required=True)
    matrix_score.add_argument("--window", type=_range, required=True)
    matrix_score.add_argument("--layers", type=_layers, required=True)
    matrix_score.add_argument("--max-level", type=int, required=True)
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
        elif args.command == "prepare-final-probe":
            result = prepare_final_two_query_probe(
                model_root=args.model_root,
                harness_root=args.harness_root,
                output=args.output,
                matrix_output=args.matrix_output,
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
        elif args.command == "analyze":
            result = analyze_probe(
                doctor_report=args.doctor_report,
                capture_root=args.capture_root,
                output_root=args.output_root,
                lock_path=args.lock,
            )
        elif args.command == "capture-query-sum":
            result = capture_query_sum_probe(
                doctor_report=args.doctor_report,
                probe_path=args.probe,
                model_root=args.model_root,
                output_root=args.output_root,
                lock_path=args.lock,
                dry_run=args.dry_run,
            )
        elif args.command == "analyze-query-sum":
            result = analyze_query_sum_probe(
                doctor_report=args.doctor_report,
                probe_path=args.probe,
                capture_root=args.capture_root,
                output_root=args.output_root,
                lock_path=args.lock,
            )
        elif args.command == "capture-matrix":
            result = capture_matrix_episode(
                doctor_report=args.doctor_report,
                episode_path=args.episode_manifest,
                model_root=args.model_root,
                output_root=args.output_root,
                lock_path=args.lock,
                dry_run=args.dry_run,
            )
        else:
            result = score_matrix_capture(
                episode_path=args.episode_manifest,
                capture_root=args.capture_root,
                output_root=args.output_root,
                q1_ranges=args.q1_range,
                q2_ranges=args.q2_range,
                layers=args.layers,
                window=args.window,
                max_level=args.max_level,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ConfigError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
