from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from putpocket_dataset_mining.serving import GenerationRequest, LocalVLLMEngine


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


if __name__ == "__main__":
    unittest.main()
