from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .cline_tools import execute_cline_tool, parse_cline_tool_calls
from .docker_workspace import DockerWorkspace
from .errors import InfraError, ToolParseError
from .jsonl import append_jsonl
from .prompts import ChatTemplateRenderer, Message
from .serving import GenerationEngine, GenerationRequest


@dataclass(frozen=True)
class RolloutResult:
    history_name: str
    completed: bool
    failure_class: str | None
    turns: int
    parse_failures: int
    messages: list[Message]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EpisodeTimeline:
    def __init__(self, attempt_dir: Path) -> None:
        self.markdown_path = attempt_dir / "episode_timeline.md"
        self.jsonl_path = attempt_dir / "episode_timeline.jsonl"
        self.markdown_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.markdown_path.exists():
            self.markdown_path.write_text("# Episode Timeline\n\n", encoding="utf-8")
        if not self.jsonl_path.exists():
            self.jsonl_path.write_text("", encoding="utf-8")

    def append(self, event_type: str, summary: str, payload: dict[str, Any] | None = None) -> None:
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "summary": summary,
            "payload": payload or {},
        }
        append_jsonl(self.jsonl_path, event)
        with self.markdown_path.open("a", encoding="utf-8") as handle:
            handle.write(f"## {event_type}\n\n{summary}\n\n")


class HeadlessClineRuntime:
    def __init__(
        self,
        attempt_dir: Path,
        renderer: ChatTemplateRenderer,
        engine: GenerationEngine,
        workspace: DockerWorkspace,
        timeline: EpisodeTimeline,
        max_parse_failures: int = 3,
        per_generation_timeout_sec: int = 300,
        per_tool_timeout_sec: int = 120,
        max_tokens: int = 2048,
        generation_seed: int | None = None,
    ) -> None:
        self.attempt_dir = attempt_dir
        self.renderer = renderer
        self.engine = engine
        self.workspace = workspace
        self.timeline = timeline
        self.max_parse_failures = max_parse_failures
        self.per_generation_timeout_sec = per_generation_timeout_sec
        self.per_tool_timeout_sec = per_tool_timeout_sec
        self.max_tokens = max_tokens
        self.generation_seed = generation_seed
        (attempt_dir / "trajectories").mkdir(parents=True, exist_ok=True)
        (attempt_dir / "serving").mkdir(parents=True, exist_ok=True)

    def run_history(self, history_name: str, initial_messages: list[Message], max_turns: int) -> RolloutResult:
        messages = [dict(message) for message in initial_messages]
        parse_failures = 0
        trajectory_path = self.attempt_dir / "trajectories" / f"{history_name}_trajectory.jsonl"
        trajectory_path.write_text("", encoding="utf-8")
        self.timeline.append(f"{history_name}.start", f"Starting {history_name} rollout.")

        for turn in range(1, max_turns + 1):
            try:
                rendered = self.renderer.render(messages)
                prompt_path = self.attempt_dir / "serving" / f"{history_name}_turn_{turn}_rendered_prompt.txt"
                token_path = self.attempt_dir / "serving" / f"{history_name}_turn_{turn}_tokenization.json"
                prompt_path.write_text(rendered.rendered_prompt, encoding="utf-8")
                token_path.write_text(json.dumps(rendered.metadata, indent=2, sort_keys=True), encoding="utf-8")
                generation = self.engine.generate(
                    GenerationRequest(
                        rendered_prompt=rendered.rendered_prompt,
                        max_tokens=self.max_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        n=1,
                        seed=self.generation_seed,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                failure = f"{history_name}.infra.vllm_generation_failed"
                self.timeline.append(f"{history_name}.failed", str(exc), {"failure_class": failure})
                raise InfraError(str(exc)) from exc

            assistant_message: Message = {"role": "assistant", "content": generation.text}
            messages.append(assistant_message)
            append_jsonl(
                trajectory_path,
                {
                    "event": "model_response",
                    "history": history_name,
                    "turn": turn,
                    "text": generation.text,
                    "generation_metadata": generation.metadata,
                    "prompt_metadata": rendered.metadata,
                },
            )
            self.timeline.append(
                f"{history_name}.model_response",
                f"Turn {turn} generated {len(generation.text)} characters.",
                {"turn": turn, "prompt_sha256": rendered.metadata["prompt_sha256"]},
            )

            try:
                calls = parse_cline_tool_calls(generation.text)
            except ToolParseError as exc:
                parse_failures += 1
                observation = (
                    "FORMAT ERROR: Use exactly one original Cline XML tool call such as "
                    "<read_file><path>solution.py</path></read_file> or "
                    "<attempt_completion><result>done</result></attempt_completion>."
                )
                messages.append({"role": "user", "content": observation})
                append_jsonl(
                    trajectory_path,
                    {
                        "event": "parse_failure",
                        "history": history_name,
                        "turn": turn,
                        "error": str(exc),
                        "parse_failures": parse_failures,
                    },
                )
                self.timeline.append(f"{history_name}.parse_failure", str(exc), {"turn": turn})
                if parse_failures > self.max_parse_failures:
                    failure = f"{history_name}.rollout.agent_parse_failed"
                    return RolloutResult(history_name, False, failure, turn, parse_failures, messages)
                continue

            for call in calls:
                observation = execute_cline_tool(self.workspace, call, timeout_sec=self.per_tool_timeout_sec)
                messages.append({"role": "user", "content": observation.as_message_content()})
                append_jsonl(
                    trajectory_path,
                    {
                        "event": "tool_observation",
                        "history": history_name,
                        "turn": turn,
                        "tool": call.name,
                        "params": call.params,
                        "ok": observation.ok,
                        "observation": observation.content,
                        "data": observation.data,
                    },
                )
                self.timeline.append(
                    f"{history_name}.tool.{call.name}",
                    observation.content[:1000],
                    {"turn": turn, "ok": observation.ok},
                )
                if call.name == "attempt_completion":
                    self.timeline.append(f"{history_name}.complete", f"{history_name} ended with attempt_completion.")
                    return RolloutResult(history_name, True, None, turn, parse_failures, messages)

        failure = f"{history_name}.rollout.max_turns_exceeded"
        self.timeline.append(f"{history_name}.failed", f"{history_name} exceeded {max_turns} turns.", {"failure_class": failure})
        return RolloutResult(history_name, False, failure, max_turns, parse_failures, messages)
