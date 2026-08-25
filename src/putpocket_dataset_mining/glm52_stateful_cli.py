"""Fail-closed utilities for the GLM-5.2 stateful mid-trajectory edit sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


BASE_PROMPT_TOKENS = 2071
EDIT_POSITION = 114
OLD_TOKEN_ID = 17526
NEW_TOKEN_ID = 11660
ELIGIBLE_START = 115
RATIOS = tuple(range(0, 101, 10))
MAIN_LAYERS = tuple(range(78))
INDEXER_LAYERS = (0, 1, 2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 66, 70, 74)
UNSAFE_ACK = "I_UNDERSTAND_ZERO_SAFE_PAGES"
MODEL_ID = "nvidia/GLM-5.2-NVFP4"
INSTANCE_ID = "instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
EDIT_SENTENCE_OLD = "Failure to follow these rules will cause your response to be rejected."
EDIT_SENTENCE_NEW = "Failure to follow these rules will cause your response to be accepted."


class ContractError(RuntimeError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ContractError(reason)


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_sha(tokens: list[int]) -> str:
    return sha_bytes(json.dumps(tokens, separators=(",", ":")).encode())


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tokens(path: Path) -> list[int]:
    value = load_json(path)
    if isinstance(value, dict):
        value = value.get("prompt", value.get("prompt_token_ids", value.get("token_ids")))
    require(isinstance(value, list) and value and all(isinstance(item, int) for item in value), "TOKEN_FILE_INVALID")
    return value


def configure_agent(args: argparse.Namespace) -> None:
    import yaml

    value = yaml.safe_load(args.input.resolve().read_text(encoding="utf-8"))
    system = value["agent"]["system_template"]
    require(system.count(EDIT_SENTENCE_OLD) == 1 and EDIT_SENTENCE_NEW not in system, "DONOR_SYSTEM_TEXT_INVALID")
    value["model"]["model_kwargs"]["api_base"] = args.api_base
    value["model"]["model_kwargs"]["temperature"] = 0.0
    value["agent"]["step_limit"] = int(args.step_limit)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    temporary.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    temporary.replace(output)


def _chat_ids(tokenizer: Any, messages: list[dict[str, Any]], *, generation: bool) -> list[int]:
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=generation, return_dict=False)
    require(isinstance(ids, list) and all(isinstance(item, int) for item in ids), "CHAT_TEMPLATE_IDS_INVALID")
    return ids


def _base_ids(base_selector: dict[str, Any]) -> tuple[list[int], dict[int, Any]]:
    ranking = base_selector.get("ranking")
    require(isinstance(ranking, list), "BASE_SELECTOR_RANKING_INVALID")
    expected = set(range(ELIGIBLE_START, BASE_PROMPT_TOKENS))
    require(len(ranking) == len(expected) and set(ranking) == expected, "BASE_SELECTOR_NOT_PERMUTATION")
    metrics = base_selector.get("token_metrics", {})
    require(isinstance(metrics, dict), "BASE_SELECTOR_METRICS_INVALID")
    return ranking, {int(key): value for key, value in metrics.items()}


def _hybrid_ranking(base_ranking: list[int], history_count: int) -> list[int]:
    # Preserve the native DSA order over the original prompt. Newly generated
    # A1 positions have no pre-outcome DSA capture, so assign a deterministic
    # hash quantile and interleave both groups by quantile.
    scored: list[tuple[float, int, int]] = []
    base_count = len(base_ranking)
    for ordinal, position in enumerate(base_ranking):
        scored.append(((ordinal + 0.5) / base_count, 0, position))
    for position in range(BASE_PROMPT_TOKENS, history_count):
        raw = hashlib.sha256(f"sr-cc-1-stateful-v3:{position}".encode()).digest()[:8]
        scored.append((int.from_bytes(raw, "big") / 2**64, 1, position))
    return [position for _, _, position in sorted(scored)]


def build_episode(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    capture = load_json(args.capture.resolve())
    trajectory = load_json(args.trajectory.resolve())
    request = capture.get("request")
    response = capture.get("response")
    require(capture.get("http_status") == 200 and isinstance(request, dict) and isinstance(response, dict), "EPISODE_CAPTURE_INVALID")
    request_messages = request.get("messages")
    require(isinstance(request_messages, list) and len(request_messages) == 2, "DONOR_TURN1_REQUEST_INVALID")
    require([item.get("role") for item in request_messages] == ["system", "user"], "DONOR_TURN1_ROLES_INVALID")
    messages = trajectory.get("messages")
    require(isinstance(messages, list) and len(messages) >= 4, "EPISODE_TRAJECTORY_TOO_SHORT")
    frozen = messages[:4]
    require([item.get("role") for item in frozen] == ["system", "user", "assistant", "user"], "EPISODE_MESSAGE_ROLES_INVALID")
    require(frozen[:2] == request_messages, "TRAJECTORY_REQUEST_DIVERGENCE")
    choices = response.get("choices", [])
    require(isinstance(choices, list) and len(choices) == 1 and isinstance(choices[0].get("message"), dict), "EPISODE_RESPONSE_INVALID")
    require(choices[0]["message"].get("content") == frozen[2].get("content"), "FROZEN_A1_RESPONSE_DIVERGENCE")
    require(isinstance(frozen[3].get("content"), str) and frozen[3]["content"], "A1_WAS_NOT_EXECUTED")
    actions = re.findall(r"```bash\s*\n(.*?)\n```", frozen[2].get("content", ""), re.DOTALL)
    require(len(actions) == 1, "A1_NOT_EXACTLY_ONE_BASH_ACTION")

    old_system = frozen[0]["content"]
    require(old_system.count(EDIT_SENTENCE_OLD) == 1 and EDIT_SENTENCE_NEW not in old_system, "OLD_SYSTEM_EDIT_SITE_INVALID")
    new_system = old_system.replace(EDIT_SENTENCE_OLD, EDIT_SENTENCE_NEW)
    edited_history_messages = [dict(item) for item in frozen[:3]]
    edited_history_messages[0] = {**edited_history_messages[0], "content": new_system}

    tokenizer = AutoTokenizer.from_pretrained(args.model_root.resolve(), local_files_only=True)
    donor_base = load_tokens(args.base_donor_prompt.resolve())
    edited_base = load_tokens(args.base_edited_prompt.resolve())
    require(len(donor_base) == len(edited_base) == BASE_PROMPT_TOKENS, "BASE_PROMPT_LENGTH_INVALID")
    require(_chat_ids(tokenizer, frozen[:2], generation=True) == donor_base, "DONOR_INITIAL_PROMPT_DIVERGENCE")
    edited_initial = [dict(frozen[0], content=new_system), frozen[1]]
    require(_chat_ids(tokenizer, edited_initial, generation=True) == edited_base, "EDITED_INITIAL_PROMPT_DIVERGENCE")

    donor_history = _chat_ids(tokenizer, frozen[:3], generation=False)
    edited_history = _chat_ids(tokenizer, edited_history_messages, generation=False)
    require(len(donor_history) == len(edited_history) and len(donor_history) > BASE_PROMPT_TOKENS, "STATEFUL_HISTORY_LENGTH_INVALID")
    differences = [index for index, pair in enumerate(zip(donor_history, edited_history, strict=True)) if pair[0] != pair[1]]
    require(differences == [EDIT_POSITION], "STATEFUL_HISTORY_EDIT_NOT_SINGLE_TOKEN_114")
    require(donor_history[EDIT_POSITION] == OLD_TOKEN_ID and edited_history[EDIT_POSITION] == NEW_TOKEN_ID, "STATEFUL_HISTORY_EDIT_TOKEN_INVALID")
    require(donor_history[:BASE_PROMPT_TOKENS] == donor_base, "DONOR_HISTORY_BASE_PREFIX_INVALID")
    require(edited_history[:BASE_PROMPT_TOKENS] == edited_base, "EDITED_HISTORY_BASE_PREFIX_INVALID")
    target_messages = edited_history_messages + [frozen[3]]
    target_ids = _chat_ids(tokenizer, target_messages, generation=True)
    require(target_ids[: len(edited_history)] == edited_history and len(target_ids) > len(edited_history), "Q2_NOT_STRICT_POST_EDIT_EXTENSION")

    base_selector = load_json(args.base_selector.resolve())
    base_ranking, base_metrics = _base_ids(base_selector)
    ranking = _hybrid_ranking(base_ranking, len(donor_history))
    expected = set(range(ELIGIBLE_START, len(donor_history)))
    require(len(ranking) == len(expected) and set(ranking) == expected, "STATEFUL_SELECTOR_NOT_PERMUTATION")
    by_ratio = {str(ratio): sorted(ranking[: len(ranking) * ratio // 100]) for ratio in RATIOS}
    output_root = args.output_root.resolve()
    selector = {
        "schema_version": 2,
        "status": "attested_before_benchmark_outcomes",
        "unsafe_research_ablation": True,
        "scenario": {
            "name": "sr_cc_1_stateful_mid_trajectory_equal_token_edit_v3",
            "base_prompt_token_count": BASE_PROMPT_TOKENS,
            "history_token_count": len(donor_history),
            "edit_position": EDIT_POSITION,
            "old_token_id": OLD_TOKEN_ID,
            "new_token_id": NEW_TOKEN_ID,
            "eligible_history_range": [ELIGIBLE_START, len(donor_history)],
            "same_length": True,
            "rope_positions_unchanged": True,
            "q2_and_post_edit_extensions_recomputed": True,
        },
        "selection_definition": {
            "requested_ratio_denominator": len(ranking),
            "algorithm": "base_dsa_rank_interleaved_with_frozen_a1_hash_quantiles_v1",
            "base_selector_sha256": file_sha(args.base_selector.resolve()),
            "a1_ranking_is_outcome_independent": True,
        },
        "source_evidence": {
            "donor_history_token_ids_sha256": token_sha(donor_history),
            "edited_history_token_ids_sha256": token_sha(edited_history),
            "base_selector_sha256": file_sha(args.base_selector.resolve()),
        },
        "ranking": ranking,
        "positions_by_ratio": by_ratio,
        "token_metrics": {
            **{str(position): base_metrics.get(position, {"source": "base_dsa"}) for position in range(ELIGIBLE_START, BASE_PROMPT_TOKENS)},
            **{str(position): {"source": "frozen_a1_deterministic_hash"} for position in range(BASE_PROMPT_TOKENS, len(donor_history))},
        },
    }
    episode = {
        "schema_version": 2,
        "status": "frozen_before_ratio_outcomes",
        "instance_id": INSTANCE_ID,
        "model_id": MODEL_ID,
        "scenario": "SYS_old+Q1 -> frozen A1/action/Q2; edit SYS_old->SYS_new; SYS_new+Q1+A1+Q2 -> post-edit continuation",
        "donor_turn1_messages": frozen[:2],
        "frozen_a1_message": frozen[2],
        "frozen_q2_message": frozen[3],
        "frozen_backend_response": response,
        "frozen_action_sha256": sha_bytes(actions[0].encode()),
        "frozen_q2_sha256": sha_bytes(frozen[3]["content"].encode()),
        "history_token_count": len(donor_history),
        "eligible_start": ELIGIBLE_START,
        "eligible_end": len(donor_history),
        "donor_history_token_ids_sha256": token_sha(donor_history),
        "edited_history_token_ids_sha256": token_sha(edited_history),
        "first_post_edit_prompt_token_count": len(target_ids),
        "first_post_edit_prompt_prefix_attested": True,
        "source_capture_sha256": file_sha(args.capture.resolve()),
        "source_trajectory_sha256": file_sha(args.trajectory.resolve()),
    }
    atomic_json(output_root / "episode.json", episode)
    atomic_json(output_root / "donor-history-token-ids.json", {"prompt": donor_history})
    atomic_json(output_root / "edited-history-token-ids.json", {"prompt": edited_history})
    atomic_json(output_root / "selector" / "selector.json", selector)


def write_control(args: argparse.Namespace) -> None:
    donor = load_tokens(args.donor_history.resolve())
    edited = load_tokens(args.edited_history.resolve())
    require(len(donor) == len(edited) and len(donor) > BASE_PROMPT_TOKENS, "CONTROL_HISTORY_LENGTH_INVALID")
    require([i for i, pair in enumerate(zip(donor, edited, strict=True)) if pair[0] != pair[1]] == [EDIT_POSITION], "CONTROL_HISTORY_DIFF_INVALID")
    ratio = int(args.ratio)
    require(ratio in RATIOS and args.mode in {"OFF", "SNAPSHOT", "TRANSPLANT"}, "CONTROL_MODE_OR_RATIO_INVALID")
    if args.mode == "SNAPSHOT":
        require(args.prompt_side == "donor" and ratio == 100, "SNAPSHOT_CONTROL_INVALID")
    if args.mode == "TRANSPLANT":
        require(args.prompt_side == "edited" and ratio > 0, "TRANSPLANT_CONTROL_INVALID")
    selector_path = args.selector.resolve()
    selector = load_json(selector_path)
    scenario = selector.get("scenario", {})
    require(scenario.get("history_token_count") == len(donor), "CONTROL_SELECTOR_HISTORY_MISMATCH")
    atomic_json(args.output.resolve(), {
        "schema_version": 2,
        "mode": args.mode,
        "experiment_id": args.experiment_id,
        "phase_id": args.phase_id,
        "prompt_side": args.prompt_side,
        "unsafe_forced_reuse_ack": UNSAFE_ACK,
        "production_default_enabled": False,
        "base_prompt_token_count": BASE_PROMPT_TOKENS,
        "history_token_count": len(donor),
        "eligible_start": ELIGIBLE_START,
        "eligible_end": len(donor),
        "edit_position": EDIT_POSITION,
        "real_block_size": 64,
        "main_layers": list(MAIN_LAYERS),
        "indexer_layers": list(INDEXER_LAYERS),
        "donor_history_token_ids_sha256": token_sha(donor),
        "edited_history_token_ids_sha256": token_sha(edited),
        "selector_path": str(selector_path),
        "selector_sha256": file_sha(selector_path),
        "requested_ratio_percent": ratio,
        "evidence_dir": str(args.evidence_dir.resolve()),
    })


def runtime_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("runtime.rank-*.jsonl")):
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return records


def finalize_ratio(args: argparse.Namespace) -> None:
    ratio = int(args.ratio)
    episode = load_json(args.episode.resolve())
    history_count = int(episode["history_token_count"])
    denominator = history_count - ELIGIBLE_START
    expected_rows = denominator * ratio // 100
    records = [json.loads(line) for line in args.proxy_log.resolve().read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [item for item in records if item.get("phase_id") == args.phase_id and item.get("path") == "/v1/chat/completions"]
    replay = [item for item in records if item.get("request_kind") == "replayed_frozen_old_turn1"]
    post_edit = [item for item in records if item.get("request_kind") == "forwarded_after_system_edit"]
    require(len(replay) == 1, "FROZEN_A1_REPLAY_COUNT_INVALID")
    require(len(post_edit) >= 1, "NO_POST_EDIT_MODEL_TURN")
    require(all(item.get("http_status") == 200 and item.get("system_edit_applied") is True for item in post_edit), "POST_EDIT_HTTP_OR_REWRITE_FAILURE")
    require(post_edit[0].get("frozen_q2_attested") is True, "FROZEN_Q2_REPLAY_DIVERGED")
    prompt_tokens = [item.get("usage", {}).get("prompt_tokens") for item in post_edit]
    completion_tokens = [item.get("usage", {}).get("completion_tokens") for item in post_edit]
    require(all(isinstance(value, int) and value > history_count for value in prompt_tokens), "POST_EDIT_PROMPT_NOT_EXTENSION")
    require(all(isinstance(value, int) and value > 0 for value in completion_tokens), "POST_EDIT_COMPLETION_USAGE_INVALID")
    require(all(left <= right for left, right in zip(prompt_tokens, prompt_tokens[1:])), "POST_EDIT_PROMPT_LENGTH_NOT_MONOTONIC")
    evaluation = load_json(args.eval_results.resolve())
    require(set(evaluation) == {INSTANCE_ID} and isinstance(evaluation[INSTANCE_ID], bool), "OFFICIAL_RESULT_INVALID")

    transplant: dict[str, Any] = {"runtime_evidence_expected": ratio > 0}
    if ratio == 0:
        transplant.update({"actual_transplanted_rows_per_layer_product_per_turn": 0, "actual_transplanted_rows_all_ranks_all_turns": 0})
    else:
        runtime = runtime_records(args.evidence_dir.resolve())
        ranks = sorted({int(item["rank"]) for item in runtime})
        require(ranks == [0, 1, 2, 3], "RUNTIME_TP_RANK_COVERAGE_INVALID")
        products = {("mla_kv", layer) for layer in MAIN_LAYERS} | {("indexer_k", layer) for layer in INDEXER_LAYERS}
        total = 0
        for rank in ranks:
            rank_records = [item for item in runtime if int(item["rank"]) == rank]
            attested = [item for item in rank_records if item.get("action") == "prompt_attested"]
            mutations = [item for item in rank_records if item.get("action") == "destination_transplant"]
            ordinals = sorted({int(item["prefill_ordinal"]) for item in attested})
            require(ordinals == list(range(len(post_edit))), "RUNTIME_POST_EDIT_TURN_COVERAGE_INVALID")
            require([next(item for item in attested if int(item["prefill_ordinal"]) == ordinal)["full_prefill_token_count"] for ordinal in ordinals] == prompt_tokens, "RUNTIME_PROXY_PROMPT_MISMATCH")
            require(len(mutations) == len(post_edit) * len(products), "RUNTIME_LAYER_PRODUCT_COVERAGE_INVALID")
            for ordinal in ordinals:
                turn = [item for item in mutations if int(item["prefill_ordinal"]) == ordinal]
                require({(item["product"], int(item["layer"])) for item in turn} == products, "RUNTIME_PRODUCT_SET_INVALID")
                require(all(item.get("actual_row_count") == expected_rows for item in turn), "RUNTIME_ROW_COUNT_INVALID")
                require(all(item.get("extension_rows_recomputed") == prompt_tokens[ordinal] - history_count for item in turn), "POST_EDIT_EXTENSION_REUSE_DETECTED")
                require(all(item.get("edited_token_excluded") is True and item.get("consumed_by_native_attention") is True for item in turn), "RUNTIME_MUTATION_INVARIANT_FAILED")
            total += sum(int(item["actual_row_count"]) for item in mutations)
        transplant.update({"actual_transplanted_rows_per_layer_product_per_turn": expected_rows, "actual_transplanted_rows_all_ranks_all_turns": total})

    atomic_json(args.output.resolve(), {
        "schema_version": 2,
        "status": "complete",
        "claim": "single_pinned_swebench_pro_instance_stateful_mid_trajectory_edit_ablation",
        "scenario_attested": "old system generated and executed frozen A1 exactly once; new system applied before Q2 model turn",
        "ratio_percent_requested": ratio,
        "ratio_rows_denominator": denominator,
        "ratio_rows_effective": expected_rows / denominator,
        "history_token_count": history_count,
        "frozen_a1_replay_count": 1,
        "post_edit_agent_turn_count": len(post_edit),
        "official_evaluator_resolved": evaluation[INSTANCE_ID],
        "single_instance_accuracy": int(evaluation[INSTANCE_ID]),
        "total_post_edit_decode_tokens": sum(completion_tokens),
        "per_post_edit_turn_prompt_tokens": prompt_tokens,
        "transplant": transplant,
        "latency_or_skipped_flop_claim": False,
        "patches_sha256": file_sha(args.patches.resolve()),
        "eval_results_sha256": file_sha(args.eval_results.resolve()),
    })


def finalize_sweep(args: argparse.Namespace) -> None:
    paths = [Path(value).resolve() for value in args.ratio_results]
    results = sorted((load_json(path) for path in paths), key=lambda item: item["ratio_percent_requested"])
    require([item["ratio_percent_requested"] for item in results] == [int(item) for item in args.expected_ratios.split(",")], "SWEEP_RATIO_SET_INVALID")
    require(results[0]["ratio_percent_requested"] == 0, "RATIO_ZERO_MISSING")
    baseline = results[0]
    for result in results:
        result["post_edit_decode_delta_vs_ratio_0"] = result["total_post_edit_decode_tokens"] - baseline["total_post_edit_decode_tokens"]
    atomic_json(args.output.resolve(), {
        "schema_version": 2,
        "status": "complete",
        "primary_outcome": "official single-instance resolved accuracy after a true mid-trajectory system edit",
        "scenario": "SYS_old+Q1 -> one frozen A1/action/Q2; edit; SYS_new+Q1+A1+Q2 -> continuation",
        "results": results,
    })


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    config = commands.add_parser("configure-agent")
    config.add_argument("--input", type=Path, required=True)
    config.add_argument("--output", type=Path, required=True)
    config.add_argument("--api-base", required=True)
    config.add_argument("--step-limit", type=int, default=0)
    episode = commands.add_parser("build-episode")
    episode.add_argument("--capture", type=Path, required=True)
    episode.add_argument("--trajectory", type=Path, required=True)
    episode.add_argument("--base-donor-prompt", type=Path, required=True)
    episode.add_argument("--base-edited-prompt", type=Path, required=True)
    episode.add_argument("--base-selector", type=Path, required=True)
    episode.add_argument("--model-root", type=Path, required=True)
    episode.add_argument("--output-root", type=Path, required=True)
    control = commands.add_parser("control")
    control.add_argument("--mode", required=True)
    control.add_argument("--experiment-id", required=True)
    control.add_argument("--phase-id", required=True)
    control.add_argument("--prompt-side", required=True)
    control.add_argument("--ratio", type=int, required=True)
    control.add_argument("--selector", type=Path, required=True)
    control.add_argument("--donor-history", type=Path, required=True)
    control.add_argument("--edited-history", type=Path, required=True)
    control.add_argument("--evidence-dir", type=Path, required=True)
    control.add_argument("--output", type=Path, required=True)
    ratio = commands.add_parser("finalize-ratio")
    ratio.add_argument("--ratio", type=int, required=True)
    ratio.add_argument("--phase-id", required=True)
    ratio.add_argument("--episode", type=Path, required=True)
    ratio.add_argument("--proxy-log", type=Path, required=True)
    ratio.add_argument("--evidence-dir", type=Path, required=True)
    ratio.add_argument("--patches", type=Path, required=True)
    ratio.add_argument("--eval-results", type=Path, required=True)
    ratio.add_argument("--output", type=Path, required=True)
    sweep = commands.add_parser("finalize-sweep")
    sweep.add_argument("--ratio-results", nargs="+", required=True)
    sweep.add_argument("--expected-ratios", required=True)
    sweep.add_argument("--output", type=Path, required=True)
    return root


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "configure-agent":
        configure_agent(args)
    elif args.command == "build-episode":
        build_episode(args)
    elif args.command == "control":
        write_control(args)
    elif args.command == "finalize-ratio":
        finalize_ratio(args)
    else:
        finalize_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
