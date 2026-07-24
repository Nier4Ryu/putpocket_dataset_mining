from __future__ import annotations

import pytest
import torch

from putpocket_dataset_mining.kernels.glm_h192_triton import (
    GLM_TINY_HEAD_SIZE,
    GLM_TINY_LATENT_DIM,
    GLM_TINY_ROPE_DIM,
    compare_triton_to_torch,
    dequantize_vllm_fp8_ds_mla_h192_cache,
    pack_vllm_fp8_ds_mla_h192_cache_for_test,
    triton_glm_h192_sparse_mla_decode_vllm_fp8_paged,
    torch_glm_h192_sparse_mla_decode,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _inputs(topk: int = 64):
    torch.manual_seed(20260724)
    device = torch.device("cuda")
    num_tokens = 2
    num_heads = 16
    seq_len = max(192, topk)
    q_latent = torch.randn(
        num_tokens, num_heads, GLM_TINY_LATENT_DIM, device=device, dtype=torch.float32
    )
    q_rope = torch.randn(
        num_tokens, num_heads, GLM_TINY_ROPE_DIM, device=device, dtype=torch.float32
    )
    kv_c = torch.randn(seq_len, GLM_TINY_LATENT_DIM, device=device, dtype=torch.float32)
    k_rope = torch.randn(seq_len, GLM_TINY_ROPE_DIM, device=device, dtype=torch.float32)
    topk_indices = torch.stack(
        [
            torch.arange(row * 3, row * 3 + topk, device=device, dtype=torch.int32)
            for row in range(num_tokens)
        ]
    )
    return q_latent, q_rope, kv_c, k_rope, topk_indices


def test_dense_triton_matches_torch_reference():
    q_latent, q_rope, kv_c, k_rope, topk_indices = _inputs(topk=64)
    result = compare_triton_to_torch(q_latent, q_rope, kv_c, k_rope, topk_indices)
    assert not torch.isnan(result.output).any()
    assert result.max_abs_error < 5e-3
    assert result.mean_abs_error < 5e-4


def test_vllm_fp8_paged_adapter_matches_dequantized_reference():
    q_latent, q_rope, kv_c, k_rope, topk_indices = _inputs(topk=64)
    q = torch.cat([q_latent, q_rope], dim=-1)
    assert q.shape[-1] == GLM_TINY_HEAD_SIZE
    cache = pack_vllm_fp8_ds_mla_h192_cache_for_test(kv_c, k_rope)
    selected_kv_c, selected_k_rope, flat_indices = dequantize_vllm_fp8_ds_mla_h192_cache(
        cache, topk_indices
    )
    expected = torch_glm_h192_sparse_mla_decode(
        q_latent,
        q_rope,
        selected_kv_c.reshape(-1, GLM_TINY_LATENT_DIM),
        selected_k_rope.reshape(-1, GLM_TINY_ROPE_DIM),
        flat_indices,
    )
    actual = triton_glm_h192_sparse_mla_decode_vllm_fp8_paged(q, cache, topk_indices)
    torch.cuda.synchronize()
    diff = (actual - expected).abs()
    assert not torch.isnan(actual).any()
    assert float(diff.max().item()) < 5e-3
    assert float(diff.mean().item()) < 5e-4
