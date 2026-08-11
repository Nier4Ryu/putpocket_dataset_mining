from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import InfraError


@dataclass(frozen=True)
class JudgeResult:
    decision: str
    reason: str
    backend: str
    failure_class: str | None
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "backend": self.backend,
            "failure_class": self.failure_class,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }


class CodexJudge:
    def __init__(self, attempt_dir: Path, timeout_sec: int = 300, workdir: Path | None = None) -> None:
        self.attempt_dir = attempt_dir
        self.timeout_sec = timeout_sec
        self.workdir = workdir or attempt_dir
        (attempt_dir / "judge").mkdir(parents=True, exist_ok=True)

    def write_skipped(self, reason: str) -> JudgeResult:
        result = JudgeResult(
            decision="skipped",
            reason=reason,
            backend="codex_cli",
            failure_class=None,
        )
        self._write(result)
        return result

    def run(
        self,
        cline_rules_v1: str,
        files_after_history1: dict[str, str],
        cline_rules_v2: str,
        query2: str,
        files_after_history2: dict[str, str],
        history2_unit_test_summary: dict[str, Any],
    ) -> JudgeResult:
        prompt = self._build_prompt(
            cline_rules_v1,
            files_after_history1,
            cline_rules_v2,
            query2,
            files_after_history2,
            history2_unit_test_summary,
        )
        (self.attempt_dir / "judge" / "judge_prompt.txt").write_text(prompt, encoding="utf-8")
        cmd = [
            "codex",
            "--ask-for-approval",
            "never",
            "exec",
            "--cd",
            str(self.workdir),
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-",
        ]
        try:
            result = subprocess.run(cmd, input=prompt, text=True, capture_output=True, timeout=self.timeout_sec)
        except subprocess.TimeoutExpired as exc:
            judge = JudgeResult(
                decision="uncertain",
                reason="Codex judge timed out.",
                backend="codex_cli",
                failure_class="judge.cli_error",
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
            )
            self._write(judge)
            return judge
        if result.returncode != 0:
            judge = JudgeResult(
                decision="uncertain",
                reason=f"Codex CLI returned {result.returncode}.",
                backend="codex_cli",
                failure_class="judge.cli_error",
                stdout=result.stdout,
                stderr=result.stderr,
            )
            self._write(judge)
            return judge
        parsed = self._parse_decision(result.stdout)
        judge = JudgeResult(
            decision=parsed.get("decision", "uncertain"),
            reason=str(parsed.get("reason", "")),
            backend="codex_cli",
            failure_class=None if parsed.get("decision") in {"pass", "fail"} else "judge.uncertain",
            stdout=result.stdout,
            stderr=result.stderr,
        )
        self._write(judge)
        return judge

    def _write(self, result: JudgeResult) -> None:
        (self.attempt_dir / "judge" / "judge_decision.json").write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _parse_decision(self, stdout: str) -> dict[str, Any]:
        try:
            data = json.loads(stdout)
            if data.get("decision") in {"pass", "fail", "uncertain"}:
                return data
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*?\}", stdout, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if data.get("decision") in {"pass", "fail", "uncertain"}:
                    return data
            except json.JSONDecodeError:
                pass
        return {"decision": "uncertain", "reason": "Judge did not return valid decision JSON."}

    def _build_prompt(
        self,
        cline_rules_v1: str,
        files_after_history1: dict[str, str],
        cline_rules_v2: str,
        query2: str,
        files_after_history2: dict[str, str],
        history2_unit_test_summary: dict[str, Any],
    ) -> str:
        payload = {
            "cline_rules_v1": cline_rules_v1,
            "files_after_history1": files_after_history1,
            "cline_rules_v2": cline_rules_v2,
            "query2": query2,
            "files_after_history2": files_after_history2,
            "history2_unit_test_summary": history2_unit_test_summary,
        }
        return (
            "You are the read-only dataset-mining judge. Return only JSON matching "
            '{"decision":"pass|fail|uncertain","reason":"short reason"}.\n'
            "Return pass only if files_after_history2 appear to satisfy query2 while following cline_rules_v2. "
            "Unit-test correctness is handled separately.\n\n"
            + json.dumps(payload, indent=2, sort_keys=True)
        )


def read_text_files(root: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(root))
        if rel.startswith("tests/"):
            continue
        try:
            files[rel] = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise InfraError(f"Non-text file in judge scope: {path}") from exc
    return files
