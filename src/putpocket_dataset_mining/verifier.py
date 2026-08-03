from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import SourceTask, verifier_materializer_for_task
from .docker_workspace import run_verifier_container


@dataclass(frozen=True)
class VerificationResult:
    stage: str
    passed: bool
    final_status: str
    failure_class: str | None
    returncode: int
    stdout: str
    stderr: str
    timeout: bool
    workspace: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "checks": [
                {
                    "name": "hidden_mbpp_pytest",
                    "passed": self.passed,
                    "command": "pytest -q tests/test_solution.py",
                    "returncode": self.returncode,
                    "timeout": self.timeout,
                }
            ],
            "final_status": self.final_status,
            "failure_class": self.failure_class,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "workspace": self.workspace,
        }


class HiddenVerifier:
    def __init__(
        self,
        attempt_dir: Path,
        docker_image: str,
        cpus: int | float = 8,
        memory: str = "8g",
        test_command: str = "pytest -q tests/test_solution.py",
        timeout_sec: int = 120,
    ) -> None:
        self.attempt_dir = attempt_dir
        self.docker_image = docker_image
        self.cpus = cpus
        self.memory = memory
        self.test_command = test_command
        self.timeout_sec = timeout_sec

    def verify(self, stage: str, snapshot_dir: Path, task: SourceTask) -> VerificationResult:
        verification_dir = self.attempt_dir / "verification" / stage
        verification_dir.mkdir(parents=True, exist_ok=True)
        verifier_workspace = verification_dir / "workspace"
        if verifier_workspace.exists():
            shutil.rmtree(verifier_workspace)
        shutil.copytree(snapshot_dir, verifier_workspace, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        verifier_materializer_for_task(task).write(task, verifier_workspace)
        result = run_verifier_container(
            workspace=verifier_workspace,
            image=self.docker_image,
            command=self.test_command,
            cpus=self.cpus,
            memory=self.memory,
            timeout_sec=self.timeout_sec,
        )
        passed = result.returncode == 0
        failure_class = None if passed else f"{stage}.unit_test.timeout" if result.timeout else f"{stage}.unit_test.failed"
        verification = VerificationResult(
            stage=stage,
            passed=passed,
            final_status="passed" if passed else "failed",
            failure_class=failure_class,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timeout=result.timeout,
            workspace=str(verifier_workspace),
        )
        (verification_dir / "checklist.json").write_text(
            json.dumps(verification.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (verification_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
        (verification_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
        return verification
