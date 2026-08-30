from __future__ import annotations

import sys
import types
import unittest
from io import BytesIO
from unittest.mock import patch

from putpocket_dataset_mining.errors import InfraError
from putpocket_dataset_mining.serving import (
    GenerationRequest,
    LocalVLLMEngine,
    OpenAICompatibleHTTPGenerationEngine,
)


class _Output:
    text = "done"
    token_ids = [1, 2, 3]
    finish_reason = "stop"


class _RequestOutput:
    outputs = [_Output()]


class _LLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def generate(self, prompts, sampling):
        return [_RequestOutput()]


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _HTTPResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class ServingTimingTests(unittest.TestCase):
    def test_generation_records_monotonic_and_wall_timestamps(self) -> None:
        module = types.SimpleNamespace(LLM=_LLM, SamplingParams=_SamplingParams)
        with patch.dict(sys.modules, {"vllm": module}):
            engine = LocalVLLMEngine(model_id="/immutable/model", gpu_devices=[])
            result = engine.generate(GenerationRequest("prompt"))
        metadata = result.metadata
        self.assertLess(metadata["request_start_monotonic_ns"], metadata["request_end_monotonic_ns"])
        self.assertTrue(metadata["request_start_utc"].endswith("Z"))
        self.assertIsNone(metadata["time_to_first_token_sec"])
        self.assertGreater(metadata["output_tokens_per_second"], 0)

    def test_openai_http_engine_preserves_rendered_prompt_contract(self) -> None:
        response = _HTTPResponse(
            b'{"id":"cmpl-test","choices":[{"text":"done","finish_reason":"stop"}],'
            b'"usage":{"completion_tokens":3}}'
        )
        with patch(
            "putpocket_dataset_mining.serving.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            engine = OpenAICompatibleHTTPGenerationEngine(
                base_url="http://127.0.0.1:8137/", model_id="glm-5.3-flash"
            )
            result = engine.generate(
                GenerationRequest(
                    rendered_prompt="exact rendered prompt",
                    max_tokens=7,
                    temperature=0.0,
                    seed=123,
                )
            )
        sent = urlopen.call_args.args[0]
        self.assertEqual(sent.full_url, "http://127.0.0.1:8137/v1/completions")
        self.assertEqual(
            __import__("json").loads(sent.data),
            {
                "model": "glm-5.3-flash",
                "prompt": "exact rendered prompt",
                "max_tokens": 7,
                "temperature": 0.0,
                "top_p": 1.0,
                "n": 1,
                "stream": False,
                "seed": 123,
            },
        )
        self.assertEqual(result.text, "done")
        self.assertEqual(result.metadata["completion_token_count"], 3)
        self.assertFalse(result.metadata["vllm_internal_chat_template_applied"])

    def test_openai_http_engine_rejects_credentials_in_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "embedded credentials"):
            OpenAICompatibleHTTPGenerationEngine(
                base_url="http://secret@example.test", model_id="model"
            )

    def test_openai_http_engine_fails_closed_when_named_secret_is_absent(self) -> None:
        engine = OpenAICompatibleHTTPGenerationEngine(
            base_url="http://127.0.0.1:8137",
            model_id="model",
            api_key_env="PUTPOCKET_TEST_MISSING_API_KEY",
        )
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(InfraError, "is unset"):
                engine.generate(GenerationRequest("prompt"))


if __name__ == "__main__":
    unittest.main()
