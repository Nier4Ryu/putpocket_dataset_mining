from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from .constants import REPO_ROOT
from .dataset import SourceTask
from .execution_config import DEFAULT_VERIFIER_TIMEOUT_SEC, ExecutionConfig
from .ssh_transport import SshRsyncTransport
from .verifier import SshRsyncVerifierTransport


FIXTURE_TESTS = {
    "pass": "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    "fail": "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 6\n",
    "timeout": "import time\n\n\ndef test_timeout():\n    time.sleep(30)\n",
}


def run_remote_verifier_fixtures(
    *,
    execution_config: ExecutionConfig,
    fixtures: list[str],
    timeout_fixture_sec: int = 2,
    output_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir or REPO_ROOT / "data" / "remote_verifier" / "live_fixtures" / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    execution_config.validate_for_evaluation_start()
    if dry_run:
        return _dry_run_summary(execution_config, fixtures, timeout_fixture_sec, output_dir)

    transport = SshRsyncVerifierTransport(execution_config)
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name in fixtures:
            if name not in FIXTURE_TESTS:
                raise ValueError(f"Unknown remote fixture: {name}")
            workspace = _fixture_workspace(root / name, FIXTURE_TESTS[name])
            task = _fixture_task(name)
            fixture_timeout = timeout_fixture_sec if name == "timeout" else execution_config.verifier_timeout_sec
            attempt_dir = output_dir / name / "attempt"
            result = transport.run(
                stage="history1",
                verifier_workspace=workspace,
                task=task,
                docker_image=execution_config.remote.docker_image or "putpocket-classeval-python:ubuntu22.04-py313-v1",
                test_command="pytest -q tests/test_solution.py",
                cpus=1,
                memory="512m",
                timeout_sec=fixture_timeout,
                attempt_dir=attempt_dir,
            )
            rows.append(
                {
                    "fixture": name,
                    "timeout_sec": fixture_timeout,
                    "status": result.final_status,
                    "passed": result.passed,
                    "returncode": result.returncode,
                    "timed_out": result.timeout,
                    "failure_class": result.failure_class,
                    "artifact_dir": str(attempt_dir),
                }
            )
    summary = {
        "schema_version": 1,
        "dry_run": False,
        "fixtures": rows,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _dry_run_summary(execution_config: ExecutionConfig, fixtures: list[str], timeout_fixture_sec: int, output_dir: Path) -> dict[str, Any]:
    transport = SshRsyncTransport(execution_config.remote)
    rows = []
    for name in fixtures:
        if name not in FIXTURE_TESTS:
            raise ValueError(f"Unknown remote fixture: {name}")
        fixture_timeout = timeout_fixture_sec if name == "timeout" else execution_config.verifier_timeout_sec
        rows.append(
            {
                "fixture": name,
                "timeout_sec": fixture_timeout,
                "ssh_argv_prefix": transport.ssh_base_argv(),
                "rsync_argv_prefix": transport.rsync_base_argv(),
                "wrapper": execution_config.remote.wrapper,
                "remote_target": transport.target,
                "would_connect": False,
            }
        )
    summary = {
        "schema_version": 1,
        "dry_run": True,
        "fixtures": rows,
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _fixture_workspace(path: Path, test_code: str) -> Path:
    tests = path / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    (path / "solution.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (tests / "test_solution.py").write_text(test_code, encoding="utf-8")
    return path


def _fixture_task(name: str) -> SourceTask:
    return SourceTask(
        adapter="remote_fixture",
        dataset_id="remote_fixture",
        split="remote_fixture",
        row_index=0,
        task_id=f"remote_fixture_{name}",
        prompt="remote verifier fixture",
        reference_solution="",
        tests=[],
        test_setup="",
        raw={"fixture": name},
    )
