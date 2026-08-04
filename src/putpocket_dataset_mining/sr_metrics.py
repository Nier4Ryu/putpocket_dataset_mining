from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PrefixCacheMetrics:
    prefix_cache_queries: int | None = None
    prefix_cache_hits: int | None = None
    external_prefix_cache_queries: int | None = None
    external_prefix_cache_hits: int | None = None
    exact_prefix_cache_bypass_state: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class SrReuseMetrics:
    reused_token_count: int | None = None
    recomputed_token_count: int | None = None
    reused_token_layer_count: int | None = None
    recomputed_token_layer_count: int | None = None
    overwritten_or_corrected_token_layer_count: int | None = None
    reuse_map_path: str | None = None
    reuse_map_sha256: str | None = None
    sr_cache_path_selected: str | None = None
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def default_unavailable_sr_metrics(reason: str = "sr_runtime_hooks_not_yet_exposed") -> SrReuseMetrics:
    return SrReuseMetrics(unavailable_reason=reason)
