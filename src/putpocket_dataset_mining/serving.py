from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .constants import ALLOWED_CUDA_DEVICES, DEFAULT_MODEL_ID, SHARED_HF_HUB_CACHE_DIR
from .errors import DependencyError, InfraError


@dataclass(frozen=True)
class GenerationRequest:
    rendered_prompt: str
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    n: int = 1
    seed: int | None = None


@dataclass(frozen=True)
class GenerationResult:
    text: str
    metadata: dict[str, Any]


class GenerationEngine(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...


class LocalVLLMEngine:
    """Local vLLM Python engine wrapper.

    The wrapper accepts rendered prompt strings only. Chat templating is handled
    before this layer by PromptPreparer/ChatTemplateRenderer.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        gpu_devices: list[int] | None = None,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        cache_dir: Path = SHARED_HF_HUB_CACHE_DIR,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
        max_num_seqs: int = 1,
        enforce_eager: bool = True,
        enable_prefix_caching: bool | None = None,
    ) -> None:
        self.model_id = model_id
        self.gpu_devices = list(gpu_devices) if gpu_devices is not None else [ALLOWED_CUDA_DEVICES[0]]
        self.tensor_parallel_size = tensor_parallel_size
        self.pipeline_parallel_size = pipeline_parallel_size
        self.cache_dir = cache_dir
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_seqs = max_num_seqs
        self.enforce_eager = enforce_eager
        self.enable_prefix_caching = enable_prefix_caching
        self._llm: Any | None = None
        self._sampling_params_cls: Any | None = None
        self.initialized_at_monotonic_ns: int | None = None
        self.ready_at_monotonic_ns: int | None = None
        self.controller_pid = os.getpid()
        self.engine_pid: int | None = None
        self.worker_pids: list[int] = []

    @property
    def llm(self) -> Any:
        if self._llm is None:
            if self.gpu_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in self.gpu_devices)
            try:
                from vllm import LLM, SamplingParams
            except ImportError as exc:
                raise DependencyError("vLLM is required for dataset mining generation.") from exc
            try:
                self._sampling_params_cls = SamplingParams
                self.initialized_at_monotonic_ns = time.perf_counter_ns()
                llm_kwargs: dict[str, Any] = {
                    "model": self.model_id,
                    "download_dir": str(self.cache_dir),
                    "tensor_parallel_size": self.tensor_parallel_size,
                    "pipeline_parallel_size": self.pipeline_parallel_size,
                    "max_model_len": self.max_model_len,
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "max_num_seqs": self.max_num_seqs,
                    "enforce_eager": self.enforce_eager,
                    "trust_remote_code": True,
                }
                if self.enable_prefix_caching is not None:
                    llm_kwargs["enable_prefix_caching"] = self.enable_prefix_caching
                self._llm = LLM(
                    **llm_kwargs,
                )
                self.ready_at_monotonic_ns = time.perf_counter_ns()
                self.worker_pids = _child_pids(os.getpid())
                self.engine_pid = _select_engine_pid(self.worker_pids) or os.getpid()
            except Exception as exc:  # noqa: BLE001 - preserve engine load failure.
                raise InfraError(f"Failed to initialize local vLLM engine for {self.model_id}: {exc}") from exc
        return self._llm

    def generate(self, request: GenerationRequest) -> GenerationResult:
        sampling_cls = self._sampling_params_cls
        if sampling_cls is None:
            _ = self.llm
            sampling_cls = self._sampling_params_cls
        assert sampling_cls is not None
        sampling_kwargs: dict[str, Any] = {
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
            "max_tokens": request.max_tokens,
        }
        if request.seed is not None:
            sampling_kwargs["seed"] = request.seed
        sampling = sampling_cls(**sampling_kwargs)
        started = time.time()
        try:
            outputs = self.llm.generate([request.rendered_prompt], sampling)
        except Exception as exc:  # noqa: BLE001
            raise InfraError(f"vLLM generation failed: {exc}") from exc
        elapsed = time.time() - started
        output = outputs[0].outputs[0] if outputs and outputs[0].outputs else None
        text = output.text if output is not None else ""
        token_ids = getattr(output, "token_ids", None) if output is not None else None
        completion_token_count = len(token_ids) if token_ids is not None else None
        finish_reason = getattr(output, "finish_reason", None) if output is not None else None
        return GenerationResult(
            text=text,
            metadata={
                "model_id": self.model_id,
                "serving_mode": "local_vllm_python_engine",
                "input_kind": "rendered_prompt_string",
                "vllm_internal_chat_template_applied": False,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "n": request.n,
                "seed": request.seed,
                "max_tokens": request.max_tokens,
                "completion_token_count": completion_token_count,
                "finish_reason": finish_reason,
                "max_model_len": self.max_model_len,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_num_seqs": self.max_num_seqs,
                "enforce_eager": self.enforce_eager,
                "enable_prefix_caching": self.enable_prefix_caching,
                "skip_reading_prefix_cache": self.enable_prefix_caching is False,
                "prefix_cache_hit_tokens": 0 if self.enable_prefix_caching is False else None,
                "elapsed_sec": elapsed,
                "gpu_devices": self.gpu_devices,
            },
        )


def _child_pids(pid: int) -> list[int]:
    try:
        output = subprocess.check_output(["pgrep", "-P", str(pid)], text=True, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        return []
    pids: list[int] = []
    for line in output.splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            continue
    return pids


def _select_engine_pid(pids: list[int]) -> int | None:
    for pid in pids:
        try:
            cmd = subprocess.check_output(["ps", "-p", str(pid), "-o", "args="], text=True, stderr=subprocess.DEVNULL)
        except Exception:  # noqa: BLE001
            continue
        if "EngineCore" in cmd or "SpawnProcess" in cmd:
            return pid
    return pids[0] if pids else None
