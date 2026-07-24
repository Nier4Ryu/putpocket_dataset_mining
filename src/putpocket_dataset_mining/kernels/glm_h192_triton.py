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
VLLM_FP8_DS_MLA_PADDED_BYTES = 656
VLLM_FP8_DS_MLA_H192_SCALE_OFFSET = 512
VLLM_FP8_DS_MLA_H192_ROPE_OFFSET = 528


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

    valid = topk_indices >= 0
    safe_indices = topk_indices.clamp_min(0)
    selected_c = kv_c[safe_indices].float()
    selected_rope = k_rope[safe_indices].float()
    q_latent_f = q_latent.float()
    q_rope_f = q_rope.float()

    scores = torch.einsum("thd,tkd->thk", q_latent_f, selected_c)
    scores = scores + torch.einsum("thd,tkd->thk", q_rope_f, selected_rope)
    scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
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
    validate: bool = True,
) -> torch.Tensor:
    """Run the Triton prototype for top-k decode.

    This prototype streams over top-k blocks and can validate the full GLM
    ``index_topk=2048`` synthetic shape.  It is not yet a vLLM paged-cache
    backend because it consumes dense ``kv_c`` and ``k_rope`` tensors.
    """

    if validate:
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


def dequantize_vllm_fp8_ds_mla_h192_cache(
    kv_cache: torch.Tensor,
    physical_indices: torch.Tensor,
    *,
    scale_offset: int = VLLM_FP8_DS_MLA_H192_SCALE_OFFSET,
    rope_offset: int = VLLM_FP8_DS_MLA_H192_ROPE_OFFSET,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather and dequantize tiny GLM entries from vLLM ``fp8_ds_mla`` cache.

    The current vLLM GLM experiment stores h128/h64 entries inside the 656-byte
    full-layout cache allocation.  This is storage padding only: the computation
    below reads the first 128 FP8 latent bytes, the first FP32 scale, and the
    64 BF16 RoPE values at ``rope_offset``.  The h576 full-compute padding path is
    not used.

    Returns ``(selected_kv_c, selected_k_rope, flat_indices)`` where the first
    two tensors are dense selected caches and ``flat_indices`` addresses them in
    the shape expected by :func:`triton_glm_h192_sparse_mla_decode`.
    """

    if kv_cache.dtype != torch.uint8:
        kv_cache = kv_cache.view(torch.uint8)
    if kv_cache.ndim == 4 and kv_cache.shape[2] == 1:
        kv_cache = kv_cache.squeeze(2)
    if kv_cache.ndim != 3:
        raise ValueError("kv_cache must have shape [blocks, block_size, bytes]")
    if kv_cache.shape[-1] < rope_offset + GLM_TINY_ROPE_DIM * 2:
        raise ValueError(
            "kv_cache byte stride is too small for h192 fp8_ds_mla layout: "
            f"got {kv_cache.shape[-1]}"
        )
    if physical_indices.ndim != 2:
        raise ValueError("physical_indices must have shape [T, K]")

    num_tokens, topk = physical_indices.shape
    valid = physical_indices >= 0
    flat_cache = kv_cache.reshape(-1, kv_cache.shape[-1])
    safe_indices = physical_indices.clamp_min(0).to(torch.long)
    entries = flat_cache[safe_indices].contiguous()

    nope_bytes = entries[..., :GLM_TINY_LATENT_DIM].contiguous()
    nope_fp8 = nope_bytes.view(torch.float8_e4m3fn).to(torch.float32)
    scales = (
        entries[..., scale_offset : scale_offset + 4]
        .contiguous()
        .view(torch.float32)
        .reshape(num_tokens, topk, 1)
    )
    selected_kv_c = nope_fp8 * scales
    selected_kv_c = torch.where(valid[..., None], selected_kv_c, 0.0)

    selected_k_rope = (
        entries[..., rope_offset : rope_offset + GLM_TINY_ROPE_DIM * 2]
        .contiguous()
        .view(torch.bfloat16)
        .to(torch.float32)
        .reshape(num_tokens, topk, GLM_TINY_ROPE_DIM)
    )
    selected_k_rope = torch.where(valid[..., None], selected_k_rope, 0.0)

    flat_indices = torch.arange(
        num_tokens * topk, device=physical_indices.device, dtype=torch.int32
    ).reshape(num_tokens, topk)
    flat_indices = torch.where(valid, flat_indices, -torch.ones_like(flat_indices))
    return selected_kv_c, selected_k_rope, flat_indices


def pack_vllm_fp8_ds_mla_h192_cache_for_test(
    kv_c: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    block_size: int = 64,
    cache_bytes: int = VLLM_FP8_DS_MLA_PADDED_BYTES,
    scale_offset: int = VLLM_FP8_DS_MLA_H192_SCALE_OFFSET,
    rope_offset: int = VLLM_FP8_DS_MLA_H192_ROPE_OFFSET,
) -> torch.Tensor:
    """Pack dense h192 test tensors into the vLLM padded byte layout."""

    if kv_c.ndim != 2 or kv_c.shape[-1] != GLM_TINY_LATENT_DIM:
        raise ValueError("kv_c must have shape [S, 128]")
    if k_rope.ndim != 2 or k_rope.shape[-1] != GLM_TINY_ROPE_DIM:
        raise ValueError("k_rope must have shape [S, 64]")
    if k_rope.shape[0] != kv_c.shape[0]:
        raise ValueError("kv_c and k_rope must share sequence length")
    if cache_bytes < rope_offset + GLM_TINY_ROPE_DIM * 2:
        raise ValueError("cache_bytes is too small")

    seq_len = kv_c.shape[0]
    num_blocks = ceil(seq_len / block_size)
    cache = torch.zeros(
        (num_blocks, block_size, cache_bytes),
        dtype=torch.uint8,
        device=kv_c.device,
    )
    flat = cache.reshape(-1, cache_bytes)
    kv_c_f = kv_c.float()
    scales = kv_c_f.abs().amax(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    scales = scales / 448.0
    quantized = (kv_c_f / scales[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    flat[:seq_len, :GLM_TINY_LATENT_DIM] = quantized
    flat[:seq_len, scale_offset : scale_offset + 4] = (
        scales.contiguous().view(torch.uint8).reshape(seq_len, 4)
    )
    flat[:seq_len, rope_offset : rope_offset + GLM_TINY_ROPE_DIM * 2] = (
        k_rope.to(torch.bfloat16)
        .contiguous()
        .view(torch.uint8)
        .reshape(seq_len, GLM_TINY_ROPE_DIM * 2)
    )
    return cache


def triton_glm_h192_sparse_mla_decode_vllm_fp8_paged(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    physical_indices: torch.Tensor,
    *,
    scale: float | None = None,
    scale_offset: int = VLLM_FP8_DS_MLA_H192_SCALE_OFFSET,
    rope_offset: int = VLLM_FP8_DS_MLA_H192_ROPE_OFFSET,
    block_topk: int | None = None,
) -> torch.Tensor:
    """Run tiny GLM sparse MLA decode from vLLM's padded fp8 paged cache."""

    if q.ndim != 3 or q.shape[-1] != GLM_TINY_HEAD_SIZE:
        raise ValueError("q must have shape [T, H, 192]")
    selected_kv_c, selected_k_rope, flat_indices = dequantize_vllm_fp8_ds_mla_h192_cache(
        kv_cache,
        physical_indices,
        scale_offset=scale_offset,
        rope_offset=rope_offset,
    )
    q_latent, q_rope = q.float().split([GLM_TINY_LATENT_DIM, GLM_TINY_ROPE_DIM], dim=-1)
    selected_kv_c_flat = selected_kv_c.reshape(-1, GLM_TINY_LATENT_DIM)
    selected_k_rope_flat = selected_k_rope.reshape(-1, GLM_TINY_ROPE_DIM)
    return triton_glm_h192_sparse_mla_decode(
        q_latent,
        q_rope,
        selected_kv_c_flat,
        selected_k_rope_flat,
        flat_indices,
        scale=scale,
        block_topk=block_topk,
        validate=False,
    )


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
    if int(topk_indices.min().item()) < -1:
        raise ValueError("topk_indices may only use -1 as the invalid sentinel")
    if int(topk_indices.max().item()) >= kv_c.shape[0]:
        raise ValueError("topk_indices contains an out-of-range token index")
