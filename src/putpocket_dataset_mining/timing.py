from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def kst_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime())


@dataclass(frozen=True)
class TimingEvent:
    name: str
    monotonic_ns: int
    utc: str
    kst: str
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "monotonic_ns": self.monotonic_ns,
            "utc": self.utc,
            "kst": self.kst,
            "payload": self.payload,
        }


class TimingRecorder:
    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root
        self.timing_dir = run_root / "timing"
        self.timing_dir.mkdir(parents=True, exist_ok=True)
        self.timeline_path = self.timing_dir / "timeline.jsonl"
        self.timeline_path.write_text("", encoding="utf-8")
        self.events: list[TimingEvent] = []
        self.durations: dict[str, float] = {}
        self.vllm_requests: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []

    def mark(self, name: str, **payload: Any) -> TimingEvent:
        event = TimingEvent(
            name=name,
            monotonic_ns=time.perf_counter_ns(),
            utc=utc_now_iso(),
            kst=kst_now_iso(),
            payload=payload,
        )
        self.events.append(event)
        with self.timeline_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
        return event

    @contextmanager
    def span(self, name: str, **payload: Any) -> Iterator[None]:
        start = self.mark(f"{name}.start", **payload)
        try:
            yield
        finally:
            end = self.mark(f"{name}.end", **payload)
            self.durations[name] = (end.monotonic_ns - start.monotonic_ns) / 1_000_000_000

    def duration_between(self, start_name: str, end_name: str) -> float | None:
        start = next((event for event in self.events if event.name == start_name), None)
        end = next((event for event in reversed(self.events) if event.name == end_name), None)
        if start is None or end is None:
            return None
        return (end.monotonic_ns - start.monotonic_ns) / 1_000_000_000

    def add_vllm_request(self, row: dict[str, Any]) -> None:
        self.vllm_requests.append(row)
        with (self.timing_dir / "vllm_requests.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def add_tool_call(self, row: dict[str, Any]) -> None:
        self.tool_calls.append(row)

    def write_json_atomic(self, relative: str, payload: dict[str, Any]) -> Path:
        path = self.timing_dir / relative
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return path
