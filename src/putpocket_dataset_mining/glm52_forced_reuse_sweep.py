"""Drive the pinned GLM donor/transplant ratio sweep against a local server."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from .glm52_forced_reuse import (
    DOWNSTREAM_END,
    DOWNSTREAM_START,
    EDIT_POSITION,
    FULL_LAYERS,
    NEW_TOKEN_ID,
    OLD_TOKEN_ID,
    PROMPT_TOKENS,
)


MAIN_LAYERS = tuple(range(78))
INDEXER_LAYERS = FULL_LAYERS
UNSAFE_ACK = "I_UNDERSTAND_ZERO_SAFE_PAGES"


class SweepError(RuntimeError):
    pass


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SweepError(reason)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_prompt(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("prompt"), list):
        values = payload["prompt"]
    elif isinstance(payload, dict):
        values = payload.get("prompt_token_ids", payload.get("token_ids"))
    else:
        values = payload
    _require(isinstance(values, list) and len(values) == PROMPT_TOKENS, "PROMPT_TOKEN_COUNT_INVALID")
    _require(all(isinstance(value, int) for value in values), "PROMPT_TOKEN_TYPE_INVALID")
    return values


def _token_digest(values: list[int]) -> str:
    return _sha256_bytes(json.dumps(values, separators=(",", ":")).encode())


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _control(
    *,
    mode: str,
    experiment_id: str,
    phase_id: str,
    prompt_side: str,
    ratio: int,
    selector_path: Path,
    selector_sha256: str,
    donor_digest: str,
    edited_digest: str,
    evidence_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": mode,
        "experiment_id": experiment_id,
        "phase_id": phase_id,
        "prompt_side": prompt_side,
        "unsafe_forced_reuse_ack": UNSAFE_ACK,
        "production_default_enabled": False,
        "prompt_token_count": PROMPT_TOKENS,
        "edit_position": EDIT_POSITION,
        "real_block_size": 64,
        "main_layers": list(MAIN_LAYERS),
        "indexer_layers": list(INDEXER_LAYERS),
        "donor_prompt_token_ids_sha256": donor_digest,
        "edited_prompt_token_ids_sha256": edited_digest,
        "selector_source_baseline_digest": donor_digest,
        "selector_source_edited_digest": edited_digest,
        "selector_path": str(selector_path),
        "selector_sha256": selector_sha256,
        "requested_ratio_percent": ratio,
        "evidence_dir": str(evidence_dir),
    }


def _request(url: str, prompt: list[int], max_tokens: int, timeout: float) -> tuple[dict[str, Any], float, str]:
    body = {
        "model": os.environ.get("PUTPOCKET_SERVED_MODEL_NAME", "nvidia/GLM-5.2-NVFP4"),
        "prompt": prompt,
        "temperature": 0,
        "top_p": 1,
        "max_tokens": max_tokens,
        "n": 1,
        "seed": 0,
        "stream": False,
        "return_token_ids": True,
    }
    encoded = json.dumps(body, separators=(",", ":"), sort_keys=True).encode()
    request = urllib.request.Request(url.rstrip("/") + "/v1/completions", data=encoded, headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            _require(response.status == 200, f"COMPLETION_HTTP_{response.status}")
    except urllib.error.HTTPError as error:
        raise SweepError(f"COMPLETION_HTTP_{error.code}:{error.read(4096)!r}") from error
    latency = time.monotonic() - started
    return json.loads(raw), latency, _sha256_bytes(raw)


def _wait_health(endpoint: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    error = "not_attempted"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(endpoint.rstrip("/") + "/health", timeout=5) as response:
                if response.status == 200:
                    return
                error = f"HTTP_{response.status}"
        except (OSError, urllib.error.URLError) as exc:
            error = type(exc).__name__
        time.sleep(5)
    raise SweepError(f"SERVER_HEALTH_TIMEOUT:{error}")


def _runtime_records(evidence_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(evidence_dir.glob("runtime.rank-*.jsonl")):
        with path.open("r", encoding="utf-8") as stream:
            records.extend(json.loads(line) for line in stream)
    return records


def _verify_runtime(evidence_dir: Path, action: str, ratio: int, expected_ranks: int) -> dict[str, Any]:
    records = _runtime_records(evidence_dir)
    ranks = sorted({int(record["rank"]) for record in records})
    _require(ranks == list(range(expected_ranks)), "RUNTIME_TP_RANK_COVERAGE_INVALID")
    selected = [record for record in records if record.get("action") == action]
    expected_products = len(MAIN_LAYERS) + len(INDEXER_LAYERS)
    _require(len(selected) == expected_ranks * expected_products, "RUNTIME_LAYER_PRODUCT_COVERAGE_INVALID")
    by_rank: dict[str, Any] = {}
    for rank in ranks:
        rank_records = [record for record in selected if int(record["rank"]) == rank]
        keys = {(record["product"], int(record["layer"])) for record in rank_records}
        expected = {("mla_kv", layer) for layer in MAIN_LAYERS} | {("indexer_k", layer) for layer in INDEXER_LAYERS}
        _require(keys == expected, "RUNTIME_LAYER_PRODUCT_SET_INVALID")
        if action == "destination_transplant":
            count = (DOWNSTREAM_END - DOWNSTREAM_START) * ratio // 100
            _require(all(record.get("actual_row_count") == count for record in rank_records), "RUNTIME_ROW_COUNT_INVALID")
            _require(all(record.get("consumed_by_native_attention") is True for record in rank_records), "RUNTIME_CONSUMPTION_ATTESTATION_MISSING")
            _require(all(record.get("edited_token_excluded") is True for record in rank_records), "EDITED_TOKEN_REUSED")
        by_rank[str(rank)] = {
            "layer_product_count": len(rank_records),
            "actual_transplanted_rows": sum(int(record.get("actual_row_count", 0)) for record in rank_records),
        }
    files = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for path in sorted(evidence_dir.glob("runtime.rank-*.jsonl"))
    ]
    return {"rank_count": len(ranks), "action": action, "by_rank": by_rank, "files": files}


def run_sweep(arguments: argparse.Namespace) -> dict[str, Any]:
    donor = _load_prompt(arguments.donor_prompt.resolve())
    edited = _load_prompt(arguments.edited_prompt.resolve())
    differences = [index for index, pair in enumerate(zip(donor, edited, strict=True)) if pair[0] != pair[1]]
    _require(differences == [EDIT_POSITION], "PROMPTS_NOT_EXACT_SINGLE_TOKEN_EDIT")
    _require(donor[EDIT_POSITION] == OLD_TOKEN_ID and edited[EDIT_POSITION] == NEW_TOKEN_ID, "EDIT_TOKEN_IDS_INVALID")
    donor_digest = _token_digest(donor)
    edited_digest = _token_digest(edited)
    selector_path = arguments.selector.resolve()
    selector_sha256 = _sha256_file(selector_path)
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    _require(selector.get("status") == "attested_before_benchmark_outcomes", "SELECTOR_NOT_PREATTESTED")
    source = selector.get("source_evidence", {})
    _require(source.get("baseline_prompt_token_ids_sha256") == donor_digest, "SELECTOR_DONOR_DIGEST_MISMATCH")
    _require(source.get("edited_prompt_token_ids_sha256") == edited_digest, "SELECTOR_EDITED_DIGEST_MISMATCH")

    ratios = tuple(int(value) for value in arguments.ratios.split(","))
    _require(bool(ratios) and ratios[0] == 0 and all(value in range(0, 101, 10) for value in ratios), "RATIO_SEQUENCE_INVALID")
    _require(len(set(ratios)) == len(ratios), "RATIO_DUPLICATE")
    root = arguments.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    control_path = arguments.control.resolve()
    _wait_health(arguments.endpoint, arguments.health_timeout)
    results = []

    off_dir = root / "runtime" / "ratio-000"
    _atomic_json(control_path, _control(mode="OFF", experiment_id=arguments.experiment_id, phase_id="ratio-000", prompt_side="edited", ratio=0, selector_path=selector_path, selector_sha256=selector_sha256, donor_digest=donor_digest, edited_digest=edited_digest, evidence_dir=off_dir))
    response, latency, response_sha = _request(arguments.endpoint, edited, arguments.max_tokens, arguments.request_timeout)
    response_path = root / "responses" / "ratio-000.json"
    _atomic_json(response_path, response)
    results.append({"ratio": 0, "latency_seconds": latency, "response_sha256": response_sha, "response_file_sha256": _sha256_file(response_path), "actual_transplanted_rows": 0})

    donor_dir = root / "runtime" / "donor"
    _atomic_json(control_path, _control(mode="SNAPSHOT", experiment_id=arguments.experiment_id, phase_id="donor", prompt_side="donor", ratio=100, selector_path=selector_path, selector_sha256=selector_sha256, donor_digest=donor_digest, edited_digest=edited_digest, evidence_dir=donor_dir))
    donor_response, donor_latency, donor_response_sha = _request(arguments.endpoint, donor, 1, arguments.request_timeout)
    _atomic_json(root / "responses" / "donor.json", donor_response)
    donor_runtime = _verify_runtime(donor_dir, "source_snapshot", 100, arguments.tp_size)

    for ratio in ratios[1:]:
        evidence = root / "runtime" / f"ratio-{ratio:03d}"
        _atomic_json(control_path, _control(mode="TRANSPLANT", experiment_id=arguments.experiment_id, phase_id=f"ratio-{ratio:03d}", prompt_side="edited", ratio=ratio, selector_path=selector_path, selector_sha256=selector_sha256, donor_digest=donor_digest, edited_digest=edited_digest, evidence_dir=evidence))
        response, latency, response_sha = _request(arguments.endpoint, edited, arguments.max_tokens, arguments.request_timeout)
        response_path = root / "responses" / f"ratio-{ratio:03d}.json"
        _atomic_json(response_path, response)
        runtime = _verify_runtime(evidence, "destination_transplant", ratio, arguments.tp_size)
        results.append({
            "ratio": ratio,
            "latency_seconds": latency,
            "response_sha256": response_sha,
            "response_file_sha256": _sha256_file(response_path),
            "actual_transplanted_rows": sum(value["actual_transplanted_rows"] for value in runtime["by_rank"].values()),
            "runtime": runtime,
        })

    report = {
        "schema_version": 1,
        "status": "completed_cross_environment_unsafe_ablation",
        "authoritative_cluster_center_pass": False,
        "unsafe_zero_safe_page_override": True,
        "experiment_id": arguments.experiment_id,
        "scenario": {
            "prompt_token_count": PROMPT_TOKENS,
            "edit_position": EDIT_POSITION,
            "old_token_id": OLD_TOKEN_ID,
            "new_token_id": NEW_TOKEN_ID,
            "downstream_range": [DOWNSTREAM_START, DOWNSTREAM_END],
            "same_length": True,
            "rope_absolute_positions_unchanged": True,
            "insertion_deletion_generalization": False,
        },
        "selector_path": str(selector_path),
        "selector_sha256": selector_sha256,
        "donor_prompt_token_ids_sha256": donor_digest,
        "edited_prompt_token_ids_sha256": edited_digest,
        "donor_latency_seconds": donor_latency,
        "donor_response_sha256": donor_response_sha,
        "donor_runtime": donor_runtime,
        "results": results,
    }
    report_path = root / "sweep-report.json"
    _atomic_json(report_path, report)
    _atomic_json(root / "SHA256SUMS.json", {
        "files": [
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.name != "SHA256SUMS.json"
        ]
    })
    return {"report": str(report_path), "sha256": _sha256_file(report_path)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--selector", type=Path, required=True)
    parser.add_argument("--donor-prompt", type=Path, required=True)
    parser.add_argument("--edited-prompt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--ratios", default="0,10,20,30,40,50,60,70,80,90,100")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--health-timeout", type=float, default=3600)
    parser.add_argument("--request-timeout", type=float, default=1800)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(json.dumps(run_sweep(arguments), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
