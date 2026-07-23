"""Correctness-first Triton prototype for tiny GLM sparse MLA decode.

This module intentionally does not patch vLLM.  It validates the reduced
GLM-5.2-0.8B decode contract:

    q:      [T, H, 128 + 64]
    cache:  [S, 128] latent KV plus [S, 64] RoPE K
    output: [T, H, 128]

The prototype models the MLA data-movement decode path described in vLLM's
``mla_attention.py``:

    softmax((q_latent @ kv_c.T) + (q_rope @ k_rope.T)) @ kv_c

restricted to the sparse top-k token indices for each query token.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, sqrt

import torch
import triton
import triton.language as tl


GLM_TINY_LATENT_DIM = 128
GLM_TINY_ROPE_DIM = 64
GLM_TINY_HEAD_SIZE = GLM_TINY_LATENT_DIM + GLM_TINY_ROPE_DIM


@dataclass(frozen=True)
class SparseMLAPrototypeResult:
    output: torch.Tensor
    max_abs_error: float
    mean_abs_error: float


@triton.jit
def _glm_h192_sparse_mla_decode_kernel(
    q_latent_ptr,
    q_rope_ptr,
    kv_c_ptr,
    k_rope_ptr,
    topk_ptr,
    out_ptr,
    scale: tl.constexpr,
    num_heads: tl.constexpr,
    topk: tl.constexpr,
    block_topk: tl.constexpr,
    block_v: tl.constexpr,
    latent_dim: tl.constexpr,
    rope_dim: tl.constexpr,
) -> None:
    token_head = tl.program_id(0)
    value_block = tl.program_id(1)
    token = token_head // num_heads
    head = token_head - token * num_heads

    k_offsets = tl.arange(0, block_topk)
    v_offsets = value_block * block_v + tl.arange(0, block_v)
    latent_offsets = tl.arange(0, latent_dim)
    rope_offsets = tl.arange(0, rope_dim)

    q_latent = tl.load(
        q_latent_ptr
        + (token * num_heads + head) * latent_dim
        + latent_offsets
    ).to(tl.float32)
    q_rope = tl.load(
        q_rope_ptr + (token * num_heads + head) * rope_dim + rope_offsets
    ).to(tl.float32)

    max_score = tl.full((), -float("inf"), tl.float32)
    denom = tl.full((), 0.0, tl.float32)
    for start in range(0, topk, block_topk):
        topk_offsets = start + k_offsets
        topk_ids = tl.load(
            topk_ptr + token * topk + topk_offsets,
            mask=topk_offsets < topk,
            other=-1,
        )
        valid_k = (topk_offsets < topk) & (topk_ids >= 0)

        kv_latent = tl.load(
            kv_c_ptr + topk_ids[:, None] * latent_dim + latent_offsets[None, :],
            mask=valid_k[:, None],
            other=0.0,
        ).to(tl.float32)
        k_rope = tl.load(
            k_rope_ptr + topk_ids[:, None] * rope_dim + rope_offsets[None, :],
            mask=valid_k[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(kv_latent * q_latent[None, :], axis=1)
        scores += tl.sum(k_rope * q_rope[None, :], axis=1)
        scores *= scale
        scores = tl.where(valid_k, scores, -float("inf"))

        tile_max = tl.max(scores, axis=0)
        new_max = tl.maximum(max_score, tile_max)
        denom = denom * tl.exp(max_score - new_max) + tl.sum(
            tl.exp(scores - new_max), axis=0
        )
        max_score = new_max

    out = tl.zeros((block_v,), tl.float32)
    for start in range(0, topk, block_topk):
        topk_offsets = start + k_offsets
        topk_ids = tl.load(
            topk_ptr + token * topk + topk_offsets,
            mask=topk_offsets < topk,
            other=-1,
        )
        valid_k = (topk_offsets < topk) & (topk_ids >= 0)

        kv_latent = tl.load(
            kv_c_ptr + topk_ids[:, None] * latent_dim + latent_offsets[None, :],
            mask=valid_k[:, None],
            other=0.0,
        ).to(tl.float32)
        k_rope = tl.load(
            k_rope_ptr + topk_ids[:, None] * rope_dim + rope_offsets[None, :],
            mask=valid_k[:, None],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(kv_latent * q_latent[None, :], axis=1)
        scores += tl.sum(k_rope * q_rope[None, :], axis=1)
        scores *= scale
        scores = tl.where(valid_k, scores, -float("inf"))
        weights = tl.exp(scores - max_score) / denom

        values = tl.load(
            kv_c_ptr + topk_ids[:, None] * latent_dim + v_offsets[None, :],
            mask=valid_k[:, None] & (v_offsets[None, :] < latent_dim),
            other=0.0,
        ).to(tl.float32)
        out += tl.sum(weights[:, None] * values, axis=0)
    tl.store(
        out_ptr
        + (token * num_heads + head) * latent_dim
        + v_offsets,
        out,
        mask=v_offsets < latent_dim,
    )


def torch_glm_h192_sparse_mla_decode(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_c: torch.Tensor,
    k_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference sparse MLA decode in PyTorch float32 math."""

    _validate_inputs(q_latent, q_rope, kv_c, k_rope, topk_indices)
    if scale is None:
        scale = 1.0 / sqrt(GLM_TINY_HEAD_SIZE)

    selected_c = kv_c[topk_indices].float()
    selected_rope = k_rope[topk_indices].float()
    q_latent_f = q_latent.float()
    q_rope_f = q_rope.float()

    scores = torch.einsum("thd,tkd->thk", q_latent_f, selected_c)
    scores = scores + torch.einsum("thd,tkd->thk", q_rope_f, selected_rope)
    weights = torch.softmax(scores * scale, dim=-1)
    return torch.einsum("thk,tkd->thd", weights, selected_c)


def triton_glm_h192_sparse_mla_decode(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_c: torch.Tensor,
    k_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    scale: float | None = None,
    block_topk: int | None = None,
    block_v: int = 32,
) -> torch.Tensor:
    """Run the Triton prototype for top-k decode.

    This prototype streams over top-k blocks and can validate the full GLM
    ``index_topk=2048`` synthetic shape.  It is not yet a vLLM paged-cache
    backend because it consumes dense ``kv_c`` and ``k_rope`` tensors.
    """

    _validate_inputs(q_latent, q_rope, kv_c, k_rope, topk_indices)
    if not q_latent.is_cuda:
        raise ValueError("Triton prototype requires CUDA tensors")
    if scale is None:
        scale = 1.0 / sqrt(GLM_TINY_HEAD_SIZE)
    topk = topk_indices.shape[1]
    if block_topk is None:
        block_topk = min(triton.next_power_of_2(topk), 256)
    if block_topk < topk:
        if topk % block_topk:
            raise ValueError("streaming prototype requires topk % block_topk == 0")
    if block_topk > 256:
        raise ValueError("prototype supports block_topk <= 256")

    q_latent = q_latent.contiguous()
    q_rope = q_rope.contiguous()
    kv_c = kv_c.contiguous()
    k_rope = k_rope.contiguous()
    topk_indices = topk_indices.to(torch.int32).contiguous()

    num_tokens, num_heads, _ = q_latent.shape
    out = torch.empty(
        (num_tokens, num_heads, GLM_TINY_LATENT_DIM),
        dtype=torch.float32,
        device=q_latent.device,
    )
    grid = (num_tokens * num_heads, ceil(GLM_TINY_LATENT_DIM / block_v))
    _glm_h192_sparse_mla_decode_kernel[grid](
        q_latent,
        q_rope,
        kv_c,
        k_rope,
        topk_indices,
        out,
        float(scale),
        num_heads,
        topk,
        block_topk,
        block_v,
        GLM_TINY_LATENT_DIM,
        GLM_TINY_ROPE_DIM,
        num_warps=8,
    )
    return out


def compare_triton_to_torch(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_c: torch.Tensor,
    k_rope: torch.Tensor,
    topk_indices: torch.Tensor,
    *,
    scale: float | None = None,
    block_topk: int | None = None,
) -> SparseMLAPrototypeResult:
    """Run both implementations and return numeric error metrics."""

    expected = torch_glm_h192_sparse_mla_decode(
        q_latent, q_rope, kv_c, k_rope, topk_indices, scale=scale
    )
    actual = triton_glm_h192_sparse_mla_decode(
        q_latent,
        q_rope,
        kv_c,
        k_rope,
        topk_indices,
        scale=scale,
        block_topk=block_topk,
    )
    torch.cuda.synchronize(actual.device)
    diff = (actual.float() - expected.float()).abs()
    return SparseMLAPrototypeResult(
        output=actual,
        max_abs_error=float(diff.max().item()),
        mean_abs_error=float(diff.mean().item()),
    )


def _validate_inputs(
    q_latent: torch.Tensor,
    q_rope: torch.Tensor,
    kv_c: torch.Tensor,
    k_rope: torch.Tensor,
    topk_indices: torch.Tensor,
) -> None:
    if q_latent.ndim != 3 or q_latent.shape[-1] != GLM_TINY_LATENT_DIM:
        raise ValueError("q_latent must have shape [T, H, 128]")
    if q_rope.ndim != 3 or q_rope.shape[-1] != GLM_TINY_ROPE_DIM:
        raise ValueError("q_rope must have shape [T, H, 64]")
    if q_rope.shape[:2] != q_latent.shape[:2]:
        raise ValueError("q_latent and q_rope must share [T, H]")
    if kv_c.ndim != 2 or kv_c.shape[-1] != GLM_TINY_LATENT_DIM:
        raise ValueError("kv_c must have shape [S, 128]")
    if k_rope.ndim != 2 or k_rope.shape[-1] != GLM_TINY_ROPE_DIM:
        raise ValueError("k_rope must have shape [S, 64]")
    if k_rope.shape[0] != kv_c.shape[0]:
        raise ValueError("kv_c and k_rope must share sequence length")
    if topk_indices.ndim != 2 or topk_indices.shape[0] != q_latent.shape[0]:
        raise ValueError("topk_indices must have shape [T, K]")
    if topk_indices.numel() == 0:
        raise ValueError("topk_indices must be non-empty")
    if int(topk_indices.min().item()) < 0:
        raise ValueError("topk_indices must be non-negative")
    if int(topk_indices.max().item()) >= kv_c.shape[0]:
        raise ValueError("topk_indices contains an out-of-range token index")
