from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from putpocket_dataset_mining import glm52_shift_fixture as fixture
from putpocket_dataset_mining.glm52_forced_reuse_sweep import (
    SweepError,
    _validate_offset_fixture_selection,
)


class _FakeTokenizer:
    @staticmethod
    def _text(messages: list[dict[str, str]]) -> str:
        return json.dumps(messages, ensure_ascii=True, separators=(",", ":")) + "<assistant>"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
    ) -> object:
        assert add_generation_prompt
        text = self._text(messages)
        if not tokenize:
            return text
        values = list(text.encode("utf-8"))
        return {"input_ids": values} if return_dict else values

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(text.encode("utf-8"))


def _messages() -> list[dict[str, str]]:
    system = (
        "safe system"
        "\n# Google Docstring Required\n\n"
        "The primary function must include a concise Google-style docstring with Args and Returns sections."
    )
    duplicate = {"role": "user", "content": "do the task"}
    return [
        {"role": "system", "content": system},
        duplicate,
        dict(duplicate),
        {"role": "assistant", "content": "write"},
        {"role": "user", "content": "write result"},
        {"role": "assistant", "content": "execute"},
        {"role": "user", "content": "execute result"},
        {"role": "assistant", "content": "bad format"},
        {"role": "user", "content": "format error"},
        {"role": "assistant", "content": "complete"},
        {"role": "user", "content": "complete result"},
        {"role": "user", "content": "refactor"},
    ]


class AlignmentTests(unittest.TestCase):
    def test_exact_deletion_alignment(self) -> None:
        old = [1, 2, 7, 8, 3, 4]
        new = [1, 2, 3, 4]
        self.assertEqual(fixture._edit_span(old, new), (2, 4, 2, 2))

    def test_candidate_records_page_crossing_and_identity(self) -> None:
        tokenizer = _FakeTokenizer()
        baseline = _messages()
        edited = [value for index, value in enumerate(baseline) if index != 2]
        record, _, _ = fixture._candidate(
            tokenizer,
            "test",
            baseline,
            edited,
            provenance="unit",
            edit_type="deletion",
            criticality="non-control",
            rationale="unit",
        )
        self.assertLess(record["downstream_offset_delta"], 0)
        self.assertTrue(record["unchanged_suffix_token_identity_exact"])
        self.assertGreater(record["unchanged_suffix_token_count"], 0)


class BuilderTests(unittest.TestCase):
    def test_builder_materializes_all_contract_artifacts_without_gold(self) -> None:
        tokenizer = _FakeTokenizer()
        messages = _messages()
        edited = [value for index, value in enumerate(messages) if index != 2]
        old_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        new_ids = tokenizer.apply_chat_template(edited, tokenize=True, add_generation_prompt=True)
        assert isinstance(old_ids, list) and isinstance(new_ids, list)
        span = fixture._edit_span(old_ids, new_ids)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / fixture.SOURCE_ATTEMPT_ID
            source.joinpath("prepared").mkdir(parents=True)
            message_path = source / "prepared/messages_history2.json"
            message_path.write_text(json.dumps(messages), encoding="utf-8")
            tokenizer_root = root / "tokenizer"
            tokenizer_root.mkdir()
            for name in fixture.TOKENIZER_FILES:
                (tokenizer_root / name).write_text(name, encoding="utf-8")
            output = root / "output"
            patches = {
                "SOURCE_MESSAGES_SHA256": hashlib.sha256(message_path.read_bytes()).hexdigest(),
                "TOKENIZER_FILES": {name: hashlib.sha256(name.encode()).hexdigest() for name in fixture.TOKENIZER_FILES},
                "EXPECTED_OLD_TOKENS": len(old_ids),
                "EXPECTED_NEW_TOKENS": len(new_ids),
                "EXPECTED_OLD_SPAN": span[:2],
                "EXPECTED_NEW_SPAN": span[2:],
                "EXPECTED_POSITION_DELTA": len(new_ids) - len(old_ids),
                "EXPECTED_SUFFIX_TOKENS": len(old_ids) - span[1],
                "EXPECTED_OLD_TOKEN_SHA256": fixture._token_digest(old_ids),
                "EXPECTED_NEW_TOKEN_SHA256": fixture._token_digest(new_ids),
            }
            with mock.patch.multiple(fixture, **patches), mock.patch.object(fixture, "_load_tokenizer", return_value=tokenizer):
                summary = fixture.build_fixture(source, tokenizer_root, output)
            self.assertEqual(summary["scenario_id"], fixture.SCENARIO_ID)
            self.assertTrue((output / "fixture-contract.json").is_file())
            self.assertTrue((output / "old-to-new-token-alignment.json").is_file())
            self.assertTrue((output / "physical-page-map.json").is_file())
            self.assertTrue((output / "MANIFEST.json").is_file())
            self.assertTrue((output / "SHA256SUMS").is_file())
            combined = (output / "baseline_request.json").read_text() + (output / "edited_request.json").read_text()
            self.assertNotIn("reference_solution", combined)
            self.assertNotIn("test_list", combined)


class RunnerFailClosedTests(unittest.TestCase):
    def _contract(self, root: Path) -> Path:
        path = root / "fixture-contract.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "scenario_id": fixture.SCENARIO_ID,
            "production_default_enabled": False,
            "edit": {
                "type": "deletion",
                "same_length": False,
                "downstream_position_delta": -48,
                "unchanged_suffix_token_identity_exact": True,
            },
            "runtime_support": {
                "raw_shifted_reuse_safe": False,
                "rope_correction_implemented": False,
            },
            "request_attestation": {
                "baseline_prompt_token_ids_sha256": "1" * 64,
                "edited_prompt_token_ids_sha256": "2" * 64,
            },
        }), encoding="utf-8")
        return path.resolve()

    def test_raw_shifted_reuse_is_rejected_before_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SweepError, "SHIFTED_RAW_KV_REUSE_UNSAFE_AND_UNSUPPORTED"):
                _validate_offset_fixture_selection(self._contract(Path(directory)), "raw_donor")

    def test_missing_rope_correction_and_position_mapping_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            contract = self._contract(Path(directory))
            with self.assertRaisesRegex(SweepError, "ROPE_CORRECTION_RUNTIME_NOT_IMPLEMENTED"):
                _validate_offset_fixture_selection(contract, "rope_corrected")
            with self.assertRaisesRegex(SweepError, "SHIFTED_RUNTIME_POSITION_MAP_NOT_IMPLEMENTED"):
                _validate_offset_fixture_selection(contract, "forced_unsafe_raw")


if __name__ == "__main__":
    unittest.main()
