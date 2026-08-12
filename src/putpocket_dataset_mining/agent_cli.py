from __future__ import annotations

import argparse

from . import agent_control


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="putpocket-agent")
    sub = parser.add_subparsers(dest="command", required=True)

    task = sub.add_parser("task")
    task_sub = task.add_subparsers(dest="task_command", required=True)
    start = task_sub.add_parser("start")
    start.add_argument("--topic", required=True)
    start.add_argument("--task-id", default=None)
    start.add_argument("--runtime-mode", choices=["shared-python-overlay", "isolated-native", "audit-only"], default="shared-python-overlay")
    start.add_argument("--wait-lock", type=float, default=None, help="Seconds to wait for Agent advisory locks before recording pending status.")
    start.set_defaults(func=agent_control.task_start)
    status = task_sub.add_parser("status")
    status.set_defaults(func=agent_control.task_status)
    close = task_sub.add_parser("close")
    close.set_defaults(func=agent_control.task_close)
    integrate = task_sub.add_parser("integrate")
    integrate.add_argument("--branch", required=True)
    integrate.add_argument("--skip-bootstrap", action="store_true")
    integrate.add_argument("--wait-lock", type=float, default=None, help="Seconds to wait for Agent advisory locks before recording pending status.")
    integrate.set_defaults(func=agent_control.task_integrate)

    runtime = sub.add_parser("runtime")
    runtime_sub = runtime.add_subparsers(dest="runtime_command", required=True)
    sync = runtime_sub.add_parser("sync")
    sync.add_argument("--skip-bootstrap", action="store_true")
    sync.add_argument("--wait-lock", type=float, default=None, help="Seconds to wait for Agent advisory locks before recording pending status.")
    sync.set_defaults(func=agent_control.runtime_sync)

    locks = sub.add_parser("locks")
    locks_sub = locks.add_subparsers(dest="locks_command", required=True)
    locks_status = locks_sub.add_parser("status")
    locks_status.set_defaults(func=agent_control.locks_status)

    doctor = sub.add_parser("doctor")
    doctor.set_defaults(func=agent_control.doctor)

    worktrees = sub.add_parser("worktrees")
    worktrees_sub = worktrees.add_subparsers(dest="worktrees_command", required=True)
    audit = worktrees_sub.add_parser("audit")
    audit.add_argument("--markdown", action="store_true")
    audit.set_defaults(func=agent_control.worktrees_audit)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
