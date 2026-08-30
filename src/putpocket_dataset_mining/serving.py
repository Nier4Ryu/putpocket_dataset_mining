from __future__ import annotations

import os
import atexit
import gc
import json
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .constants import ALLOWED_CUDA_DEVICES, DEFAULT_MODEL_ID, SHARED_HF_HUB_CACHE_DIR
from .errors import DependencyError, InfraError
from .timing import kst_now_iso, utc_now_iso


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


class OpenAICompatibleHTTPGenerationEngine:
    """PutPocket ``GenerationEngine`` adapter for an OpenAI-compatible server.

    PutPocket owns prompt rendering, so this adapter deliberately calls the
    legacy-compatible ``/v1/completions`` endpoint with the exact rendered
    prompt.  It does not apply a second chat template.  Credentials, when
    needed, are read from a named environment variable and are never retained
    in result metadata.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model_id: str,
        timeout_sec: float = 300.0,
        api_key_env: str | None = None,
    ) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain embedded credentials")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.timeout_sec = float(timeout_sec)
        self.api_key_env = api_key_env

    def generate(self, request: GenerationRequest) -> GenerationResult:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "prompt": request.rendered_prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
            "stream": False,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        headers = {"Content-Type": "application/json"}
        if self.api_key_env is not None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise InfraError(
                    f"OpenAI-compatible API credential environment variable "
                    f"{self.api_key_env!r} is unset"
                )
            headers["Authorization"] = f"Bearer {api_key}"

        started_ns = time.perf_counter_ns()
        started_utc = utc_now_iso()
        started_kst = kst_now_iso()
        http_request = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_sec) as response:
                response_payload = json.load(response)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            raise InfraError(f"OpenAI-compatible generation request failed: {exc}") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise InfraError("OpenAI-compatible server returned invalid JSON") from exc

        ended_ns = time.perf_counter_ns()
        try:
            choice = response_payload["choices"][0]
            text = choice["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise InfraError("OpenAI-compatible response has no completion text") from exc
        if not isinstance(text, str):
            raise InfraError("OpenAI-compatible completion text is not a string")
        usage = response_payload.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        elapsed = (ended_ns - started_ns) / 1_000_000_000
        return GenerationResult(
            text=text,
            metadata={
                "model_id": self.model_id,
                "serving_mode": "openai_compatible_http",
                "input_kind": "rendered_prompt_string",
                "vllm_internal_chat_template_applied": False,
                "endpoint": f"{self.base_url}/v1/completions",
                "temperature": request.temperature,
                "top_p": request.top_p,
                "n": request.n,
                "seed": request.seed,
                "max_tokens": request.max_tokens,
                "completion_token_count": completion_tokens,
                "finish_reason": choice.get("finish_reason"),
                "request_id": response_payload.get("id"),
                "elapsed_sec": elapsed,
                "request_start_monotonic_ns": started_ns,
                "request_end_monotonic_ns": ended_ns,
                "request_start_utc": started_utc,
                "request_start_kst": started_kst,
                "request_end_utc": utc_now_iso(),
                "request_end_kst": kst_now_iso(),
                "time_to_first_token_sec": None,
                "output_tokens_per_second": (
                    completion_tokens / elapsed
                    if isinstance(completion_tokens, int) and elapsed > 0
                    else None
                ),
            },
        )


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
        self.initialized_at_utc: str | None = None
        self.initialized_at_kst: str | None = None
        self.ready_at_utc: str | None = None
        self.ready_at_kst: str | None = None
        self.controller_pid = os.getpid()
        self.engine_pid: int | None = None
        self.worker_pids: list[int] = []
        self._shutdown_done = False
        atexit.register(self.shutdown)

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
                self.initialized_at_utc = utc_now_iso()
                self.initialized_at_kst = kst_now_iso()
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
                self.ready_at_utc = utc_now_iso()
                self.ready_at_kst = kst_now_iso()
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
        started_ns = time.perf_counter_ns()
        started_utc = utc_now_iso()
        started_kst = kst_now_iso()
        try:
            outputs = self.llm.generate([request.rendered_prompt], sampling)
        except Exception as exc:  # noqa: BLE001
            raise InfraError(f"vLLM generation failed: {exc}") from exc
        ended_ns = time.perf_counter_ns()
        elapsed = (ended_ns - started_ns) / 1_000_000_000
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
                "request_start_monotonic_ns": started_ns,
                "request_end_monotonic_ns": ended_ns,
                "request_start_utc": started_utc,
                "request_start_kst": started_kst,
                "request_end_utc": utc_now_iso(),
                "request_end_kst": kst_now_iso(),
                "time_to_first_token_sec": None,
                "output_tokens_per_second": (
                    completion_token_count / elapsed if completion_token_count is not None and elapsed > 0 else None
                ),
                "gpu_devices": self.gpu_devices,
            },
        )

    def shutdown(self) -> None:
        """Best-effort teardown for experiment-owned vLLM resources."""
        if self._shutdown_done:
            return
        self._shutdown_done = True
        llm = self._llm
        self._llm = None
        for attr in ("shutdown", "close"):
            method = getattr(llm, attr, None) if llm is not None else None
            if callable(method):
                try:
                    method()
                except Exception:
                    pass
                break
        engine = getattr(llm, "llm_engine", None) if llm is not None else None
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:
                pass
        gc.collect()
        for pid in list(self.worker_pids):
            _terminate_child_pid(pid, self.controller_pid)
        self.worker_pids = []

    def __del__(self) -> None:
        self.shutdown()


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


def _terminate_child_pid(pid: int, expected_parent: int) -> None:
    try:
        parent = int(subprocess.check_output(["ps", "-o", "ppid=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL).strip())
    except Exception:
        return
    if parent != expected_parent:
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        return
    deadline = time.time() + 5
    while time.time() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass
