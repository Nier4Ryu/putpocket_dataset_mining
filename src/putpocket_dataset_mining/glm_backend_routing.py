from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .errors import ConfigError


@dataclass(frozen=True)
class GlmBackendTrace:
    model_identifier: str | None
    compute_capability: tuple[int, int]
    kv_lora_rank: int | None
    qk_rope_head_dim: int | None
    derived_head_size: int | None
    v_head_dim: int | None
    index_n_heads: int | None
    index_head_dim: int | None
    selected_backend: str
    selected_custom_kernel: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def select_glm_sparse_mla_backend(config: Any, compute_capability: tuple[int, int], *, model_identifier: str | None = None, custom_kernel_available: bool = True) -> GlmBackendTrace:
    kv = _get_int(config, "kv_lora_rank")
    rope = _get_int(config, "qk_rope_head_dim")
    v_head = _get_int(config, "v_head_dim")
    index_n_heads = _get_int(config, "index_n_heads")
    index_head_dim = _get_int(config, "index_head_dim")
    head = kv + rope if kv is not None and rope is not None else None
    tiny_match = (
        compute_capability in {(12, 0), (12, 1)}
        and kv == 128
        and rope == 64
        and head == 192
        and v_head == 128
        and index_n_heads == 8
        and index_head_dim == 64
        and _is_glm_moe_dsa(config)
        and custom_kernel_available
    )
    if tiny_match:
        _assert_allowed_tiny_path()
        return GlmBackendTrace(model_identifier, compute_capability, kv, rope, head, v_head, index_n_heads, index_head_dim, "putpocket_triton_h192_sm120", True, "exact_tiny_glm_h192_sm120_match")
    if os.environ.get("SR_ASSERT_NO_TINY_GLM_KERNEL") == "1" and head == 192:
        raise ConfigError("SR_ASSERT_NO_TINY_GLM_KERNEL=1: Tiny GLM h192 path attempted during a run that forbids it.")
    if compute_capability[0] == 9 and head == 576:
        backend = "upstream_hopper_sparse_mla_h576"
        reason = "hopper_h576_full_glm"
    elif compute_capability in {(12, 0), (12, 1)} and head == 576:
        backend = "upstream_sm120_sparse_mla_h576"
        reason = "sm120_h576_full_glm"
    else:
        backend = "unsupported_or_upstream_selector"
        reason = "custom_tiny_kernel_conditions_not_met"
    return GlmBackendTrace(model_identifier, compute_capability, kv, rope, head, v_head, index_n_heads, index_head_dim, backend, False, reason)


def assert_no_tiny_glm_kernel_invocation(trace: GlmBackendTrace) -> None:
    if os.environ.get("SR_ASSERT_NO_TINY_GLM_KERNEL") == "1" and trace.selected_custom_kernel:
        raise ConfigError("SR_ASSERT_NO_TINY_GLM_KERNEL=1: custom Tiny GLM h192 backend selected.")


def _get_int(config: Any, name: str) -> int | None:
    value = getattr(config, name, None) if not isinstance(config, dict) else config.get(name)
    return None if value is None else int(value)


def _get(config: Any, name: str) -> Any:
    return getattr(config, name, None) if not isinstance(config, dict) else config.get(name)


def _is_glm_moe_dsa(config: Any) -> bool:
    model_type = _get(config, "model_type")
    architectures = _get(config, "architectures") or []
    return model_type == "glm_moe_dsa" or "GlmMoeDsaForCausalLM" in architectures


def _assert_allowed_tiny_path() -> None:
    if os.environ.get("SR_ASSERT_NO_TINY_GLM_KERNEL") == "1":
        raise ConfigError("SR_ASSERT_NO_TINY_GLM_KERNEL=1: custom Tiny GLM h192 backend selected.")
