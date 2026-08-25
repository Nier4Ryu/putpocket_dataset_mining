from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from putpocket_dataset_mining import glm52_stateful_cli as MODULE


ROOT = Path(__file__).resolve().parents[1]
HOOK_PATH = ROOT / "instrumentation/vllm/glm52_forced_edit_reuse.py"
RUNNER_PATH = ROOT / "scripts/cluster/run_glm52_forced_reuse_ablation.sh"
SUBMIT_PATH = ROOT / "scripts/cluster/submit_glm52_forced_reuse_ablation.sh"
PACKAGE_PATH = ROOT / "scripts/cluster/package_glm52_forced_reuse_ablation.sh"
LOCK_PATH = ROOT / "configs/cluster/glm52_forced_reuse_ablation.lock.json"
PROXY_PATH = ROOT / "src/putpocket_dataset_mining/glm52_stateful_proxy.py"
CLI_PATH = ROOT / "src/putpocket_dataset_mining/glm52_stateful_cli.py"


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _history() -> tuple[list[int], list[int]]:
    donor = list(range(MODULE.BASE_PROMPT_TOKENS + 23))
    donor[MODULE.EDIT_POSITION] = MODULE.OLD_TOKEN_ID
    edited = donor.copy()
    edited[MODULE.EDIT_POSITION] = MODULE.NEW_TOKEN_ID
    return donor, edited


class StatefulEpisodeTests(unittest.TestCase):
    def test_hybrid_ranking_is_full_deterministic_permutation(self) -> None:
        base = list(range(MODULE.ELIGIBLE_START, MODULE.BASE_PROMPT_TOKENS))
        history_count = MODULE.BASE_PROMPT_TOKENS + 173
        first = MODULE._hybrid_ranking(base, history_count)
        second = MODULE._hybrid_ranking(base, history_count)
        self.assertEqual(first, second)
        self.assertEqual(len(first), history_count - MODULE.ELIGIBLE_START)
        self.assertEqual(set(first), set(range(MODULE.ELIGIBLE_START, history_count)))
        self.assertTrue(any(position >= MODULE.BASE_PROMPT_TOKENS for position in first[: len(first) // 2]))

    def test_ratio_prefix_counts_cover_dynamic_endpoints_and_exclude_edit(self) -> None:
        history_count = MODULE.BASE_PROMPT_TOKENS + 41
        ranking = MODULE._hybrid_ranking(
            list(range(MODULE.ELIGIBLE_START, MODULE.BASE_PROMPT_TOKENS)),
            history_count,
        )
        by_ratio = {ratio: sorted(ranking[: len(ranking) * ratio // 100]) for ratio in MODULE.RATIOS}
        self.assertEqual(by_ratio[0], [])
        self.assertEqual(set(by_ratio[100]), set(range(MODULE.ELIGIBLE_START, history_count)))
        self.assertNotIn(MODULE.EDIT_POSITION, by_ratio[100])

    def test_build_episode_freezes_a1_q2_and_dynamic_selector(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_system = "prefix " + MODULE.EDIT_SENTENCE_OLD + " suffix"
            new_system = "prefix " + MODULE.EDIT_SENTENCE_NEW + " suffix"
            q1 = {"role": "user", "content": "repair the repository"}
            a1 = {"role": "assistant", "content": "```bash\nprintf ok\n```"}
            q2 = {"role": "user", "content": "Chunk ID: abc\nProcess exited with code 0"}
            turn1 = [{"role": "system", "content": old_system}, q1]
            frozen = turn1 + [a1, q2]
            response = {
                "choices": [{"message": {"role": "assistant", "content": a1["content"]}}],
                "usage": {"prompt_tokens": MODULE.BASE_PROMPT_TOKENS, "completion_tokens": 9},
            }
            capture = root / "capture.json"
            trajectory = root / "trajectory.json"
            _json(capture, {"http_status": 200, "request": {"messages": turn1}, "response": response})
            _json(trajectory, {"messages": frozen})

            donor_base = list(range(MODULE.BASE_PROMPT_TOKENS))
            donor_base[MODULE.EDIT_POSITION] = MODULE.OLD_TOKEN_ID
            edited_base = donor_base.copy()
            edited_base[MODULE.EDIT_POSITION] = MODULE.NEW_TOKEN_ID
            donor_history = donor_base + list(range(30000, 30023))
            edited_history = edited_base + list(range(30000, 30023))
            target = edited_history + [40001, 40002]
            donor_path = root / "donor.json"
            edited_path = root / "edited.json"
            selector_path = root / "base-selector.json"
            _json(donor_path, {"prompt": donor_base})
            _json(edited_path, {"prompt": edited_base})
            _json(selector_path, {
                "ranking": list(reversed(range(MODULE.ELIGIBLE_START, MODULE.BASE_PROMPT_TOKENS))),
                "token_metrics": {str(MODULE.ELIGIBLE_START): {"source": "base_dsa"}},
            })

            class FakeTokenizer:
                def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict):  # type: ignore[no-untyped-def]
                    self.assert_contract(tokenize, return_dict)
                    edited = MODULE.EDIT_SENTENCE_NEW in messages[0]["content"]
                    if len(messages) == 2 and add_generation_prompt:
                        return edited_base if edited else donor_base
                    if len(messages) == 3 and not add_generation_prompt:
                        return edited_history if edited else donor_history
                    if len(messages) == 4 and add_generation_prompt and edited:
                        return target
                    raise AssertionError((len(messages), add_generation_prompt, edited))

                @staticmethod
                def assert_contract(tokenize: bool, return_dict: bool) -> None:
                    if tokenize is not True or return_dict is not False:
                        raise AssertionError("chat template flags diverged")

            class FakeAutoTokenizer:
                @classmethod
                def from_pretrained(cls, path, *, local_files_only):  # type: ignore[no-untyped-def]
                    if not local_files_only:
                        raise AssertionError("tokenizer must be local")
                    return FakeTokenizer()

            fake_transformers = types.ModuleType("transformers")
            fake_transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
            output = root / "episode"
            args = argparse.Namespace(
                capture=capture,
                trajectory=trajectory,
                base_donor_prompt=donor_path,
                base_edited_prompt=edited_path,
                base_selector=selector_path,
                model_root=root / "model",
                output_root=output,
            )
            with patch.dict(sys.modules, {"transformers": fake_transformers}):
                MODULE.build_episode(args)

            episode = MODULE.load_json(output / "episode.json")
            selector = MODULE.load_json(output / "selector/selector.json")
            self.assertEqual(episode["history_token_count"], len(donor_history))
            self.assertEqual(episode["eligible_end"], len(donor_history))
            self.assertEqual(episode["first_post_edit_prompt_token_count"], len(target))
            self.assertEqual(selector["selection_definition"]["requested_ratio_denominator"], len(donor_history) - MODULE.ELIGIBLE_START)
            self.assertEqual(set(selector["positions_by_ratio"]["100"]), set(range(MODULE.ELIGIBLE_START, len(donor_history))))
            self.assertNotIn(MODULE.EDIT_POSITION, selector["positions_by_ratio"]["100"])
            self.assertTrue(all(position < len(donor_history) for position in selector["positions_by_ratio"]["100"]))
            self.assertEqual(episode["frozen_q2_sha256"], MODULE.sha_bytes(q2["content"].encode()))
            self.assertEqual(frozen[0]["content"].replace(MODULE.EDIT_SENTENCE_OLD, MODULE.EDIT_SENTENCE_NEW), new_system)

    def test_finalize_ratio_requires_all_products_and_q2_extension_recompute(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            donor, _ = _history()
            history_count = len(donor)
            denominator = history_count - MODULE.ELIGIBLE_START
            ratio = 10
            rows = denominator * ratio // 100
            phase = "ratio-010"
            prompt_tokens = history_count + 7
            episode_path = root / "episode.json"
            proxy_path = root / "proxy.jsonl"
            eval_path = root / "eval.json"
            patches_path = root / "patches.json"
            evidence = root / "runtime"
            output = root / "ratio.json"
            _json(episode_path, {"history_token_count": history_count})
            proxy_records = [
                {"phase_id": phase, "path": "/v1/chat/completions", "request_kind": "replayed_frozen_old_turn1", "http_status": 200, "usage": {"prompt_tokens": MODULE.BASE_PROMPT_TOKENS, "completion_tokens": 9}},
                {"phase_id": phase, "path": "/v1/chat/completions", "request_kind": "forwarded_after_system_edit", "http_status": 200, "system_edit_applied": True, "frozen_q2_attested": True, "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 11}},
            ]
            proxy_path.write_text("".join(json.dumps(record) + "\n" for record in proxy_records), encoding="utf-8")
            _json(eval_path, {MODULE.INSTANCE_ID: True})
            _json(patches_path, {"patch": "diff"})
            products = {("mla_kv", layer) for layer in MODULE.MAIN_LAYERS} | {("indexer_k", layer) for layer in MODULE.INDEXER_LAYERS}
            self.assertEqual(len(products), 99)
            for rank in range(4):
                records = [{"rank": rank, "action": "prompt_attested", "prefill_ordinal": 0, "full_prefill_token_count": prompt_tokens}]
                records.extend({
                    "rank": rank,
                    "action": "destination_transplant",
                    "prefill_ordinal": 0,
                    "product": product,
                    "layer": layer,
                    "actual_row_count": rows,
                    "extension_rows_recomputed": prompt_tokens - history_count,
                    "edited_token_excluded": True,
                    "consumed_by_native_attention": True,
                } for product, layer in sorted(products))
                path = evidence / f"runtime.rank-{rank}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            args = argparse.Namespace(
                ratio=ratio,
                phase_id=phase,
                episode=episode_path,
                proxy_log=proxy_path,
                evidence_dir=evidence,
                patches=patches_path,
                eval_results=eval_path,
                output=output,
            )
            MODULE.finalize_ratio(args)
            result = MODULE.load_json(output)
            self.assertEqual(result["ratio_rows_denominator"], denominator)
            self.assertEqual(result["transplant"]["actual_transplanted_rows_per_layer_product_per_turn"], rows)
            self.assertFalse(result["latency_or_skipped_flop_claim"])

            rank_zero = evidence / "runtime.rank-0.jsonl"
            records = [json.loads(line) for line in rank_zero.read_text(encoding="utf-8").splitlines()]
            records[1]["extension_rows_recomputed"] = 0
            rank_zero.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            with self.assertRaisesRegex(MODULE.ContractError, "POST_EDIT_EXTENSION_REUSE_DETECTED"):
                MODULE.finalize_ratio(args)


class ProxyIntegrationTests(unittest.TestCase):
    def _free_port(self) -> int:
        with socket.socket() as stream:
            stream.bind(("127.0.0.1", 0))
            return int(stream.getsockname()[1])

    def _post(self, port: int, payload: object) -> tuple[int, dict[str, object]]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def _wait(self, port: int) -> None:
        for _ in range(100):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.2):
                    return
            except Exception:
                time.sleep(0.02)
        self.fail("proxy did not become ready")

    def _wait_for_log_records(self, path: Path, minimum: int) -> list[dict[str, object]]:
        for _ in range(100):
            records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            if len(records) >= minimum:
                return records
            time.sleep(0.02)
        self.fail(f"proxy log did not reach {minimum} records")

    def test_proxy_captures_replays_and_rewrites_only_at_turn_two(self) -> None:
        backend_posts: list[dict[str, object]] = []
        a1_content = "```bash\nprintf ok\n```"
        backend_response = {"choices": [{"message": {"role": "assistant", "content": a1_content}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 2071, "completion_tokens": 9}}

        class Backend(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_GET(self) -> None:
                body = b"{}"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                backend_posts.append(json.loads(raw))
                body = json.dumps(backend_response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        backend = ThreadingHTTPServer(("127.0.0.1", 0), Backend)
        thread = threading.Thread(target=backend.serve_forever, daemon=True)
        thread.start()
        old_system = "prefix " + MODULE.EDIT_SENTENCE_OLD + " suffix"
        turn1 = [{"role": "system", "content": old_system}, {"role": "user", "content": "q1"}]
        a1 = {"role": "assistant", "content": a1_content}
        q2 = {"role": "user", "content": "q2-observation"}
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        processes: list[subprocess.Popen[bytes]] = []
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                phase = root / "phase.json"
                log = root / "proxy.jsonl"
                capture = root / "capture.json"
                _json(phase, {"phase_id": "capture", "mode": "capture_old_turn1", "ratio_percent": None})
                port = self._free_port()
                command = [sys.executable, "-m", "putpocket_dataset_mining.glm52_stateful_proxy", "--listen-port", str(port), "--backend-port", str(backend.server_port), "--phase-file", str(phase), "--log", str(log), "--capture", str(capture)]
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                processes.append(process)
                self._wait(port)
                status, response = self._post(port, {"model": MODULE.MODEL_ID, "messages": turn1})
                self.assertEqual(status, 200)
                self.assertEqual(response, backend_response)
                self.assertEqual(MODULE.load_json(capture)["request"]["messages"], turn1)
                process.terminate()
                process.wait(timeout=5)
                processes.remove(process)

                episode = root / "episode.json"
                _json(episode, {"donor_turn1_messages": turn1, "frozen_a1_message": a1, "frozen_q2_message": q2, "frozen_backend_response": backend_response})
                _json(phase, {"phase_id": "ratio-010", "mode": "replay_then_edit", "ratio_percent": 10})
                port = self._free_port()
                command.extend(["--episode", str(episode)])
                command[command.index("--listen-port") + 1] = str(port)
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                processes.append(process)
                self._wait(port)
                post_count = len(backend_posts)
                status, response = self._post(port, {"model": MODULE.MODEL_ID, "messages": turn1})
                self.assertEqual(status, 200)
                self.assertEqual(response, backend_response)
                self.assertEqual(len(backend_posts), post_count)
                status, _ = self._post(port, {"model": MODULE.MODEL_ID, "messages": turn1 + [a1, q2]})
                self.assertEqual(status, 200)
                forwarded = backend_posts[-1]["messages"]  # type: ignore[index]
                self.assertIn(MODULE.EDIT_SENTENCE_NEW, forwarded[0]["content"])  # type: ignore[index]
                self.assertNotIn(MODULE.EDIT_SENTENCE_OLD, forwarded[0]["content"])  # type: ignore[index]
                records = self._wait_for_log_records(log, 3)
                self.assertEqual([record["request_kind"] for record in records[-2:]], ["replayed_frozen_old_turn1", "forwarded_after_system_edit"])
                self.assertTrue(records[-1]["frozen_q2_attested"])
        finally:
            for process in processes:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            backend.shutdown()
            backend.server_close()
            thread.join(timeout=5)


class StatefulPackageContractTests(unittest.TestCase):
    def test_login1_source_postimages_and_scope_are_bound(self) -> None:
        expected = {
            CLI_PATH: "8fa25ec1b9454616805a5e83da3a3bec3279beb654066e71372fa27cf02b3d6f",
            PROXY_PATH: "512ff6026e6cb61b4f636d0addb2e4a47b08a4e2d9330336186b7be237aee73b",
            HOOK_PATH: "d3bb32c706e902931c2a19948ed56ba9aac03a8fc86c591a7ebaba2fcdf65c1b",
            RUNNER_PATH: "60d45c68ae2b668316d8a0771b9e2ae1d52aeac8feffa906ac3cd408b7cff548",
        }
        for path, digest in expected.items():
            with self.subTest(path=path.name):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        self.assertFalse(lock["production_default_enabled"])
        self.assertTrue(lock["predeclared_analysis"]["no_true_partial_prefill_claim"])
        self.assertTrue(lock["predeclared_analysis"]["no_speedup_claim"])
        self.assertIn("full target KV is computed", lock["predeclared_analysis"]["hybrid_state_semantics"])

    def test_runner_packager_and_submitter_preserve_immutable_chain(self) -> None:
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        submit = SUBMIT_PATH.read_text(encoding="utf-8")
        package = PACKAGE_PATH.read_text(encoding="utf-8")
        self.assertIn("--no-enable-chunked-prefill", runner)
        self.assertIn("PUTPOCKET_SWEEP_PROFILE == episode", runner)
        self.assertIn("swe_bench_pro_eval.py", runner)
        self.assertIn("FROZEN_EPISODE_DIGEST_MISMATCH", runner)
        self.assertIn("POST_EDIT_EXTENSION_REUSE_DETECTED", CLI_PATH.read_text(encoding="utf-8"))
        self.assertIn("PUTPOCKET_SWEEP_PROFILE=episode", submit)
        self.assertIn('afterok:$EPISODE_JOB_ID', submit)
        self.assertIn('afterok:$SMOKE_JOB_ID', submit)
        self.assertNotIn("--array", submit)
        self.assertIn("STATEFUL-EDIT-V3-MANIFEST.json", package)
        self.assertIn("compute_then_overwrite; no true partial prefill and no speedup claim", package)
        self.assertIn("find . -type f -not -path './SHA256SUMS'", package)
        self.assertIn('chmod -R a-w "$STAGING"', package)


if __name__ == "__main__":
    unittest.main()
