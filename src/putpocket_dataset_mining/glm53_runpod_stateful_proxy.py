"""Loopback proxy for the frozen GLM-5.3 stateful-edit-v3 episode.

The proxy captures one old-system turn, replays the frozen A1 exactly in every
ratio arm, and applies the one-token system edit before forwarding Q2 and later
turns.  All model traffic is pinned to DP rank zero so the donor snapshots and
target cache remain owned by one vLLM engine process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .glm53_runpod_stateful_edit import EDIT_SENTENCE_NEW, EDIT_SENTENCE_OLD


DP_RANK_HEADER = "X-data-parallel-rank"
DP_RANK = "0"


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any, lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n"
    with lock, path.open("a", encoding="utf-8") as stream:
        stream.write(line)
        stream.flush()


def usage_fields(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices", [])
    first = choices[0] if isinstance(choices, list) and choices else {}
    message = first.get("message", {}) if isinstance(first, dict) else {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    return {
        "usage": response.get("usage", {}),
        "finish_reason": first.get("finish_reason") if isinstance(first, dict) else None,
        "completion_content_sha256": sha256(str(content).encode()),
    }


def message_core(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    return [
        {"role": item.get("role"), "content": item.get("content")}
        for item in messages
    ]


def make_server(
    *,
    listen_port: int,
    backend_port: int,
    phase_file: Path,
    log_path: Path,
    capture_path: Path,
    episode_path: Path | None,
) -> ThreadingHTTPServer:
    """Build a server; split out for deterministic loopback unit tests."""

    log_lock = threading.Lock()
    state_lock = threading.Lock()
    phase_ordinals: dict[str, int] = {}

    def forward(
        path: str,
        method: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> tuple[int, bytes, dict[str, str]]:
        request_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in {"host", "content-length", "connection"}
        }
        request_headers[DP_RANK_HEADER] = DP_RANK
        request = urllib.request.Request(
            f"http://127.0.0.1:{backend_port}{path}",
            data=body,
            headers=request_headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=3600) as response:
                return response.status, response.read(), dict(response.headers.items())
        except urllib.error.HTTPError as error:
            return error.code, error.read(), dict(error.headers.items())

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *values: object) -> None:
            return

        def _send(
            self,
            status: int,
            body: bytes,
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header(
                "Content-Type", (headers or {}).get("Content-Type", "application/json")
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else None
            clean_path = self.path.split("?", 1)[0]
            if clean_path != "/v1/chat/completions":
                status, body, headers = forward(
                    self.path, self.command, dict(self.headers.items()), raw
                )
                self._send(status, body, headers)
                return

            phase = json.loads(phase_file.read_text(encoding="utf-8"))
            phase_id = phase["phase_id"]
            mode = phase["mode"]
            with state_lock:
                ordinal = phase_ordinals.get(phase_id, 0)
                phase_ordinals[phase_id] = ordinal + 1
            try:
                request_json = json.loads(raw or b"{}")
            except Exception:
                self._send(400, b'{"error":"REQUEST_JSON_INVALID"}')
                return

            request_kind = ""
            system_edit_applied = False
            frozen_q2_attested = False
            forwarded_json = request_json
            if mode == "capture_old_turn1":
                messages = request_json.get("messages")
                if ordinal != 0 or capture_path.exists():
                    self._send(409, b'{"error":"CAPTURE_REQUIRES_EXACTLY_ONE_REQUEST"}')
                    return
                if (
                    not isinstance(messages, list)
                    or len(messages) != 2
                    or [item.get("role") for item in messages] != ["system", "user"]
                ):
                    self._send(409, b'{"error":"CAPTURE_NOT_OLD_TURN1"}')
                    return
                system = messages[0].get("content", "")
                if system.count(EDIT_SENTENCE_OLD) != 1 or EDIT_SENTENCE_NEW in system:
                    self._send(409, b'{"error":"CAPTURE_SYSTEM_NOT_OLD"}')
                    return
                status, response_body, response_headers = forward(
                    self.path, self.command, dict(self.headers.items()), raw
                )
                try:
                    response_json = json.loads(response_body)
                except Exception:
                    response_json = {}
                atomic_json(
                    capture_path,
                    {
                        "schema_version": 3,
                        "mode": mode,
                        "http_status": status,
                        "data_parallel_rank": 0,
                        "request": request_json,
                        "response": response_json,
                    },
                )
                request_kind = "captured_old_turn1"
            elif mode == "replay_then_edit":
                if episode_path is None or not episode_path.is_file():
                    self._send(409, b'{"error":"FROZEN_EPISODE_MISSING"}')
                    return
                episode = json.loads(episode_path.read_text(encoding="utf-8"))
                messages = request_json.get("messages")
                expected_turn1 = episode["donor_turn1_messages"]
                if ordinal == 0:
                    if message_core(messages) != message_core(expected_turn1):
                        self._send(
                            409,
                            b'{"error":"TURN1_REQUEST_DIVERGED_FROM_FROZEN_EPISODE"}',
                        )
                        return
                    response_json = episode["frozen_backend_response"]
                    response_body = canonical(response_json)
                    response_headers = {"Content-Type": "application/json"}
                    status = 200
                    request_kind = "replayed_frozen_old_turn1"
                else:
                    expected_prefix = expected_turn1 + [episode["frozen_a1_message"]]
                    if (
                        not isinstance(messages, list)
                        or len(messages) < 4
                        or message_core(messages[:3]) != message_core(expected_prefix)
                    ):
                        self._send(409, b'{"error":"POST_EDIT_HISTORY_DIVERGED_BEFORE_Q2"}')
                        return
                    if message_core([messages[3]]) != message_core(
                        [episode["frozen_q2_message"]]
                    ):
                        self._send(
                            409,
                            b'{"error":"REPLAYED_A1_OBSERVATION_OR_REPO_STATE_DIVERGED"}',
                        )
                        return
                    frozen_q2_attested = True
                    old_system = messages[0].get("content", "")
                    if (
                        old_system.count(EDIT_SENTENCE_OLD) != 1
                        or EDIT_SENTENCE_NEW in old_system
                    ):
                        self._send(409, b'{"error":"PRE_EDIT_SYSTEM_NOT_OLD"}')
                        return
                    forwarded_json = json.loads(json.dumps(request_json))
                    forwarded_json["messages"][0]["content"] = old_system.replace(
                        EDIT_SENTENCE_OLD, EDIT_SENTENCE_NEW
                    )
                    forwarded_raw = canonical(forwarded_json)
                    status, response_body, response_headers = forward(
                        self.path,
                        self.command,
                        dict(self.headers.items()),
                        forwarded_raw,
                    )
                    try:
                        response_json = json.loads(response_body)
                    except Exception:
                        response_json = {}
                    request_kind = "forwarded_after_system_edit"
                    system_edit_applied = True
            else:
                self._send(409, b'{"error":"PROXY_PHASE_MODE_INVALID"}')
                return

            self._send(status, response_body, response_headers)
            append_jsonl(
                log_path,
                {
                    "schema_version": 3,
                    "phase_id": phase_id,
                    "ratio_percent": phase.get("ratio_percent"),
                    "phase_request_ordinal": ordinal,
                    "path": clean_path,
                    "request_kind": request_kind,
                    "http_status": status,
                    "system_edit_applied": system_edit_applied,
                    "frozen_q2_attested": frozen_q2_attested,
                    "data_parallel_rank": 0,
                    "original_request_sha256": sha256(canonical(request_json)),
                    "forwarded_request_sha256": sha256(canonical(forwarded_json)),
                    "response_body_sha256": sha256(response_body),
                    **usage_fields(response_json),
                },
                log_lock,
            )

        do_GET = _handle
        do_POST = _handle

    return ThreadingHTTPServer(("127.0.0.1", listen_port), Handler)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--backend-port", type=int, required=True)
    parser.add_argument("--phase-file", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--episode", type=Path)
    args = parser.parse_args()
    make_server(
        listen_port=args.listen_port,
        backend_port=args.backend_port,
        phase_file=args.phase_file.resolve(),
        log_path=args.log.resolve(),
        capture_path=args.capture.resolve(),
        episode_path=args.episode.resolve() if args.episode else None,
    ).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
