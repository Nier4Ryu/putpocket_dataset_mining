from __future__ import annotations

import argparse
import json
import sys

from .runner import cleanup, job_status, preflight, promote_incoming, protocol_version, result_status, start_worker, verify, worker_status


def _print(data: dict) -> int:
    print(json.dumps(data, indent=2, sort_keys=True))
    ok_statuses = {
        None,
        "passed",
        "failed",
        "timeout",
        "uncertain",
        "missing",
        "noop",
        "ready",
        "running",
        "completed",
        "incoming",
        "worker_started",
        "already_completed",
        "exited",
    }
    return 0 if data.get("status") in ok_statuses else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="putpocket-remote-verifier")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol-version")
    pre = sub.add_parser("preflight")
    pre.add_argument("--docker-image", default=None)
    pre.add_argument("--dockerfile", default="docker/classeval_python/Dockerfile")
    ensure = sub.add_parser("ensure-image")
    ensure.add_argument("--docker-image", required=True)
    ensure.add_argument("--dockerfile", default="docker/classeval_python/Dockerfile")
    promote = sub.add_parser("promote")
    promote.add_argument("--job-id", required=True)
    ver = sub.add_parser("verify")
    ver.add_argument("--job-id", required=True)
    start = sub.add_parser("start-worker")
    start.add_argument("--job-id", required=True)
    jstatus = sub.add_parser("job-status")
    jstatus.add_argument("--job-id", required=True)
    status = sub.add_parser("result-status")
    status.add_argument("--job-id", required=True)
    wstatus = sub.add_parser("worker-status")
    wstatus.add_argument("--job-id", required=True)
    clean = sub.add_parser("cleanup")
    clean.add_argument("--job-id", action="append", default=[])
    clean.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.command == "protocol-version":
            return _print(protocol_version())
        if args.command == "preflight":
            return _print(preflight(args.docker_image, dockerfile=args.dockerfile))
        if args.command == "ensure-image":
            from .image import ensure_image
            from putpocket_dataset_mining.constants import REPO_ROOT

            result = ensure_image(args.docker_image, REPO_ROOT / args.dockerfile)
            return _print({"schema_version": 1, "status": "passed", **result.__dict__})
        if args.command == "promote":
            path = promote_incoming(args.job_id)
            return _print({"schema_version": 1, "status": "passed", "ready_path": str(path)})
        if args.command == "verify":
            return _print(verify(args.job_id))
        if args.command == "start-worker":
            return _print(start_worker(args.job_id))
        if args.command == "job-status":
            return _print(job_status(args.job_id))
        if args.command == "result-status":
            return _print(result_status(args.job_id))
        if args.command == "worker-status":
            return _print(worker_status(args.job_id))
        if args.command == "cleanup":
            return _print(cleanup(args.job_id, dry_run=not args.execute))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"schema_version": 1, "status": "infra_failed", "error_class": exc.__class__.__name__, "error_message": str(exc)}), file=sys.stdout)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
