from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import load_yaml
from .constants import MODEL_EVALUATION_ROOT, REPO_ROOT
from .dataset import SourceTask
from .serving import GenerationRequest, GenerationResult
from .single import SingleSampleRunner


class ScriptedTwoTurnEngine:
    """Deterministic Cline XML generator for end-to-end transport tests."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        if self.calls == 1:
            text = (
                "<write_to_file>\n"
                "<path>solution.py</path>\n"
                "<content>def add(a, b):\n"
                "    return a + b\n"
                "</content>\n"
                "</write_to_file>"
            )
        elif self.calls == 2:
            text = "<attempt_completion><result>History-1 implementation complete.</result></attempt_completion>"
        elif self.calls == 3:
            text = (
                "<write_to_file>\n"
                "<path>solution.py</path>\n"
                "<content>def add(a: int, b: int) -> int:\n"
                "    \"\"\"Return the sum of a and b.\"\"\"\n"
                "    return a + b\n"
                "</content>\n"
                "</write_to_file>"
            )
        elif self.calls == 4:
            text = "<attempt_completion><result>History-2 update complete.</result></attempt_completion>"
        else:
            text = "<attempt_completion><result>No further changes.</result></attempt_completion>"
        return GenerationResult(
            text=text,
            metadata={
                "serving_mode": "scripted_two_turn_engine",
                "call_index": self.calls,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "seed": request.seed,
            },
        )


def scripted_task() -> SourceTask:
    return SourceTask(
        adapter="mbpp_huggingface",
        dataset_id="scripted_remote_e2e",
        split="scripted",
        row_index=0,
        task_id="two_turn_remote_e2e",
        prompt="Write a function add(a, b) that returns the sum of a and b.",
        reference_solution="def add(a, b):\n    return a + b\n",
        tests=[
            "assert add(2, 3) == 5",
            "assert add(-1, 1) == 0",
        ],
        test_setup="from solution import add",
        raw={"kind": "scripted_two_turn_remote_e2e"},
    )


def build_scripted_config(remote_config: Path, *, docker_image: str | None = None) -> dict[str, Any]:
    base = load_yaml(REPO_ROOT / "configs" / "dataset_mining" / "mbpp_stateful_single.yaml")
    base["run"]["output_root"] = "data/model_evaluation/runs"
    base["model"]["generation_model_id"] = "scripted-two-turn-engine"
    base["workspace"]["initial_files"] = {"solution.py": "# TODO: implement add(a, b).\n"}
    base["history"]["history1_max_turns"] = 4
    base["history"]["history2_max_turns"] = 4
    base["history"]["max_parse_failures_per_history"] = 0
    base["judge"]["enabled"] = False
    base["docker"]["image"] = docker_image or "putpocket-classeval-python:ubuntu22.04-py313-v1"
    base["docker"]["dockerfile"] = "docker/classeval_python/Dockerfile"
    base["docker"]["memory"] = "2g"
    base["docker"]["cpus"] = 1
    base["docker"]["timeouts"]["per_tool_command_sec"] = 30
    base["verifier"]["timeout_sec"] = 3600
    base["execution"] = {
        "workspace_backend": "local_docker",
        "verifier_backend": "remote_ssh_docker",
        "remote_config": str(remote_config),
        "verifier_timeout_sec": 3600,
    }
    return base


def run_scripted_two_turn_e2e(
    *,
    remote_config: Path,
    run_id: str | None = None,
    output_root: Path | None = None,
) -> dict[str, Any]:
    run_id = run_id or f"e2e_server2_inference_server1_verifier_scripted_{time.strftime('%Y%m%d_%H%M%S')}"
    config = build_scripted_config(remote_config)
    if output_root is not None:
        rel_or_abs = output_root if output_root.is_absolute() else REPO_ROOT / output_root
        config["run"]["output_root"] = str(rel_or_abs)
    runner = SingleSampleRunner(config)
    summary = runner.run_task(
        task=scripted_task(),
        run_id=run_id,
        attempt_id="scripted_attempt",
        write_index=False,
        engine=ScriptedTwoTurnEngine(),
        dataset_version="scripted_two_turn_remote_e2e",
    )
    result_path = Path(summary["artifact_path"]) / "scripted_e2e_summary.json"
    result_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
