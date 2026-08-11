from __future__ import annotations

import json
import hashlib
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
from .timing import TimingRecorder, utc_now_iso


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
        timing_recorder: TimingRecorder | None = None,
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
        self.timing_recorder = timing_recorder
        (attempt_dir / "trajectories").mkdir(parents=True, exist_ok=True)
        (attempt_dir / "serving").mkdir(parents=True, exist_ok=True)

    def run_history(self, history_name: str, initial_messages: list[Message], max_turns: int) -> RolloutResult:
        messages = [dict(message) for message in initial_messages]
        parse_failures = 0
        trajectory_path = self.attempt_dir / "trajectories" / f"{history_name}_trajectory.jsonl"
        trajectory_path.write_text("", encoding="utf-8")
        self.timeline.append(f"{history_name}.start", f"Starting {history_name} rollout.")
        if self.timing_recorder:
            self.timing_recorder.mark(f"{history_name}.rollout.start")

        for turn in range(1, max_turns + 1):
            try:
                prompt_start = time.perf_counter_ns()
                if self.timing_recorder:
                    self.timing_recorder.mark(f"{history_name}.prompt_prepare.start", turn=turn)
                rendered = self.renderer.render(messages)
                prompt_end = time.perf_counter_ns()
                if self.timing_recorder:
                    self.timing_recorder.mark(f"{history_name}.prompt_prepare.end", turn=turn)
                prompt_path = self.attempt_dir / "serving" / f"{history_name}_turn_{turn}_rendered_prompt.txt"
                token_path = self.attempt_dir / "serving" / f"{history_name}_turn_{turn}_tokenization.json"
                prompt_path.write_text(rendered.rendered_prompt, encoding="utf-8")
                token_path.write_text(json.dumps(rendered.metadata, indent=2, sort_keys=True), encoding="utf-8")
                request_id = f"{history_name}-{turn}-{hashlib.sha256(rendered.rendered_prompt.encode('utf-8')).hexdigest()[:12]}"
                vllm_start = time.perf_counter_ns()
                if self.timing_recorder:
                    self.timing_recorder.mark(f"{history_name}.vllm_request.{turn}.start", request_id=request_id)
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
                vllm_end = time.perf_counter_ns()
                if self.timing_recorder:
                    self.timing_recorder.mark(f"{history_name}.vllm_request.{turn}.end", request_id=request_id)
                    output_sha = hashlib.sha256(generation.text.encode("utf-8")).hexdigest()
                    self.timing_recorder.add_vllm_request(
                        {
                            "request_ordinal": turn,
                            "request_id": request_id,
                            "stage": history_name,
                            "start_utc": utc_now_iso(),
                            "elapsed_sec": (vllm_end - vllm_start) / 1_000_000_000,
                            "prompt_preparation_sec": (prompt_end - prompt_start) / 1_000_000_000,
                            "prompt_token_count": rendered.metadata.get("token_count"),
                            "completion_token_count": generation.metadata.get("completion_token_count"),
                            "total_token_count": (
                                (rendered.metadata.get("token_count") or 0)
                                + (generation.metadata.get("completion_token_count") or 0)
                            ),
                            "time_to_first_token_sec": generation.metadata.get("time_to_first_token_sec"),
                            "prefill_time_sec": generation.metadata.get("prefill_time_sec"),
                            "decode_time_sec": generation.metadata.get("decode_time_sec"),
                            "output_tokens_per_second": generation.metadata.get("output_tokens_per_second"),
                            "prompt_sha256": rendered.metadata["prompt_sha256"],
                            "output_sha256": output_sha,
                            "stop_reason": generation.metadata.get("finish_reason"),
                            "cache_read_disabled": generation.metadata.get("skip_reading_prefix_cache"),
                            "prefix_cache_hit_tokens": generation.metadata.get("prefix_cache_hit_tokens"),
                        }
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
                tool_start = time.perf_counter_ns()
                if self.timing_recorder:
                    self.timing_recorder.mark(f"{history_name}.tool_execution.{turn}.start", tool=call.name)
                observation = execute_cline_tool(self.workspace, call, timeout_sec=self.per_tool_timeout_sec)
                tool_end = time.perf_counter_ns()
                if self.timing_recorder:
                    self.timing_recorder.mark(f"{history_name}.tool_execution.{turn}.end", tool=call.name)
                    self.timing_recorder.add_tool_call(
                        {
                            "stage": history_name,
                            "turn": turn,
                            "tool": call.name,
                            "elapsed_sec": (tool_end - tool_start) / 1_000_000_000,
                            "ok": observation.ok,
                        }
                    )
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
                    if self.timing_recorder:
                        self.timing_recorder.mark(f"{history_name}.rollout.end", completed=True)
                    return RolloutResult(history_name, True, None, turn, parse_failures, messages)

        failure = f"{history_name}.rollout.max_turns_exceeded"
        self.timeline.append(f"{history_name}.failed", f"{history_name} exceeded {max_turns} turns.", {"failure_class": failure})
        if self.timing_recorder:
            self.timing_recorder.mark(f"{history_name}.rollout.end", completed=False, failure_class=failure)
        return RolloutResult(history_name, False, failure, max_turns, parse_failures, messages)
