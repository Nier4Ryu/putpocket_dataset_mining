"""A6000 package binding for the frozen GLM-5.3 stateful-edit-v3 episode.

Episode semantics are retained for reproducibility, but control generation is
blocked because the current official sparse runtime has no SM86 backend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EDIT_SENTENCE_OLD = "Always write code without any comments."
EDIT_SENTENCE_NEW = "Always write code with extensive comments."
MODEL_ID = "zai-org/GLM-5.3-Flash-BF16"
MODEL_REVISION = "a5b45eb41df6402735dedc900be14a42e8d5e538"
VLLM_COMMIT = "9cd956c7e6cf54efa366b803cafa15ec6c2df827"
ATTENTION_BACKEND_POLICY = "official_sparse_mla_required_but_unavailable_on_sm86"
KV_CACHE_DTYPE = "bfloat16"
MLA_LAYERS = tuple(range(3, 45, 4))
KDA_LAYERS = tuple(layer for layer in range(45) if layer not in MLA_LAYERS)
INDEX_KPOOL = 4


def model_binding() -> dict[str, Any]:
    return {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "architecture": "Glm5NextForConditionalGeneration",
        "vllm_commit": VLLM_COMMIT,
        "attention_backend_policy": ATTENTION_BACKEND_POLICY,
        "kv_cache_dtype": KV_CACHE_DTYPE,
        "index_kpool": INDEX_KPOOL,
    }


def load_frozen_episode(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 4:
        raise ValueError("A6000_EPISODE_SCHEMA_INVALID")
    if payload.get("model") != model_binding():
        raise ValueError("A6000_EPISODE_MODEL_BINDING_INVALID")
    if payload.get("exact_a1_q2_replay") is not True:
        raise ValueError("A6000_EPISODE_REPLAY_ATTESTATION_MISSING")
    if payload.get("benchmark_outcomes_read") is not False:
        raise ValueError("A6000_EPISODE_OUTCOME_LEAKAGE")
    return payload


def refuse_runtime_control() -> None:
    raise RuntimeError(
        "GLM53_SM86_RUNTIME_UNSUPPORTED: no official sparse MLA/indexer backend"
    )
