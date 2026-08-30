#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from putpocket_dataset_mining.serving import (
    GenerationRequest,
    OpenAICompatibleHTTPGenerationEngine,
)


def _json_request(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def _wait_ready(base_url: str, timeout_sec: float) -> tuple[float, dict[str, Any]]:
    started = time.monotonic()
    last_error = "not attempted"
    while time.monotonic() - started < timeout_sec:
        try:
            models = _json_request(f"{base_url}/v1/models")
            return time.monotonic() - started, models
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(10)
    raise RuntimeError(f"server did not become ready within {timeout_sec}s: {last_error}")


def _response_text(payload: dict[str, Any]) -> str:
    message = payload["choices"][0]["message"]
    parts = [message.get("reasoning") or "", message.get("content") or ""]
    return "".join(parts)


def run(args: argparse.Namespace) -> dict[str, Any]:
    client_started_at = dt.datetime.now(dt.timezone.utc)
    base_url = args.base_url.rstrip("/")
    ready_after_sec, models = _wait_ready(base_url, args.startup_timeout_sec)
    ready_at = dt.datetime.now(dt.timezone.utc)
    model_ids = [item.get("id") for item in models.get("data", [])]
    if args.model not in model_ids:
        raise RuntimeError(f"/v1/models did not expose {args.model!r}: {model_ids!r}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    messages = [
        {
            "role": "user",
            "content": "Reply with exactly GLM53_SMOKE_OK and no other text.",
        }
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        clear_thinking=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("tokenizer produced an empty rendered prompt")
    rendered_ids = tokenizer.encode(rendered, add_special_tokens=False)

    chat_payload = {
        "model": args.model,
        "messages": messages,
        "temperature": 0,
        "top_p": 1,
        "seed": 530053,
        "max_tokens": 48,
        "reasoning_effort": "low",
        "chat_template_kwargs": {"clear_thinking": True},
        "stream": False,
    }
    chat_first = _json_request(f"{base_url}/v1/chat/completions", chat_payload)
    chat_second = _json_request(f"{base_url}/v1/chat/completions", chat_payload)
    first_text = _response_text(chat_first)
    second_text = _response_text(chat_second)
    if not first_text:
        raise RuntimeError("deterministic chat smoke returned empty text")
    if first_text != second_text:
        raise RuntimeError("temperature-zero seeded chat responses were not deterministic")

    putpocket_engine = OpenAICompatibleHTTPGenerationEngine(
        base_url=base_url,
        model_id=args.model,
        timeout_sec=300,
    )
    putpocket_result = putpocket_engine.generate(
        GenerationRequest(
            rendered_prompt=rendered,
            max_tokens=48,
            temperature=0.0,
            top_p=1.0,
            seed=530053,
        )
    )
    if not putpocket_result.text:
        raise RuntimeError("PutPocket HTTP GenerationEngine returned empty text")

    prompt_digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return {
        "schema_version": 1,
        "status": "ok",
        "base_url": base_url,
        "served_model": args.model,
        "client_started_at_utc": client_started_at.isoformat(),
        "ready_at_utc": ready_at.isoformat(),
        "ready_after_client_start_sec": ready_after_sec,
        "models_response": models,
        "prompt": {
            "sha256": prompt_digest,
            "token_count": len(rendered_ids),
            "token_ids": rendered_ids,
            "rendered_text": rendered,
            "messages": messages,
            "clear_thinking": True,
        },
        "deterministic_chat": {
            "request": chat_payload,
            "first": chat_first,
            "second": chat_second,
            "text_sha256": hashlib.sha256(first_text.encode("utf-8")).hexdigest(),
            "exact_match": True,
        },
        "putpocket_generation_engine": {
            "text": putpocket_result.text,
            "text_sha256": hashlib.sha256(
                putpocket_result.text.encode("utf-8")
            ).hexdigest(),
            "metadata": putpocket_result.metadata,
            "input_was_same_rendered_prompt": True,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "name_or_path": str(args.model_dir),
            "vocab_size": len(tokenizer),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8137")
    parser.add_argument("--model", default="glm-5.3-flash-nvfp4")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--startup-timeout-sec", type=float, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run(args)
    except Exception as exc:  # noqa: BLE001 - evidence must retain exact integration failure.
        report = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
