#!/usr/bin/env python
"""Run synthetic correctness checks for the tiny GLM h192/v128 Triton path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from putpocket_dataset_mining.kernels.glm_h192_triton import (
    GLM_TINY_LATENT_DIM,
    GLM_TINY_HEAD_SIZE,
    GLM_TINY_ROPE_DIM,
    compare_triton_to_torch,
    dequantize_vllm_fp8_ds_mla_h192_cache,
    pack_vllm_fp8_ds_mla_h192_cache_for_test,
    triton_glm_h192_sparse_mla_decode_vllm_fp8_paged,
    torch_glm_h192_sparse_mla_decode,
)


def _topk_indices(num_tokens: int, seq_len: int, topk: int, device: torch.device):
    rows = []
    for token in range(num_tokens):
        start = (token * 17) % max(seq_len - topk + 1, 1)
        rows.append(torch.arange(start, start + topk, device=device, dtype=torch.int32))
    return torch.stack(rows, dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-tokens", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--paged-cache", action="store_true")
    parser.add_argument("--max-abs-tol", type=float, default=5e-3)
    parser.add_argument("--mean-abs-tol", type=float, default=5e-4)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.topk > args.seq_len:
        raise SystemExit("--topk must be <= --seq-len")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float32
    q_latent = torch.randn(
        args.num_tokens,
        args.num_heads,
        GLM_TINY_LATENT_DIM,
        device=device,
        dtype=dtype,
    )
    q_rope = torch.randn(
        args.num_tokens,
        args.num_heads,
        GLM_TINY_ROPE_DIM,
        device=device,
        dtype=dtype,
    )
    kv_c = torch.randn(args.seq_len, GLM_TINY_LATENT_DIM, device=device, dtype=dtype)
    k_rope = torch.randn(args.seq_len, GLM_TINY_ROPE_DIM, device=device, dtype=dtype)
    topk = _topk_indices(args.num_tokens, args.seq_len, args.topk, device)

    if args.paged_cache:
        q = torch.cat([q_latent, q_rope], dim=-1)
        assert q.shape[-1] == GLM_TINY_HEAD_SIZE
        cache = pack_vllm_fp8_ds_mla_h192_cache_for_test(kv_c, k_rope)
        selected_kv_c, selected_k_rope, flat_indices = (
            dequantize_vllm_fp8_ds_mla_h192_cache(cache, topk)
        )
        expected = torch_glm_h192_sparse_mla_decode(
            q_latent,
            q_rope,
            selected_kv_c.reshape(-1, GLM_TINY_LATENT_DIM),
            selected_k_rope.reshape(-1, GLM_TINY_ROPE_DIM),
            flat_indices,
        )
        actual = triton_glm_h192_sparse_mla_decode_vllm_fp8_paged(q, cache, topk)
        torch.cuda.synchronize()
        diff = (actual.float() - expected.float()).abs()
        output = actual
        max_abs_error = float(diff.max().item())
        mean_abs_error = float(diff.mean().item())
        mode = "vllm_fp8_paged"
    else:
        result = compare_triton_to_torch(q_latent, q_rope, kv_c, k_rope, topk)
        output = result.output
        max_abs_error = result.max_abs_error
        mean_abs_error = result.mean_abs_error
        mode = "dense"

    has_nan = torch.isnan(output).any().item()
    summary = {
        "mode": mode,
        "num_tokens": args.num_tokens,
        "num_heads": args.num_heads,
        "seq_len": args.seq_len,
        "topk": args.topk,
        "latent_dim": GLM_TINY_LATENT_DIM,
        "rope_dim": GLM_TINY_ROPE_DIM,
        "output_shape": list(output.shape),
        "has_nan": bool(has_nan),
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "max_abs_tol": args.max_abs_tol,
        "mean_abs_tol": args.mean_abs_tol,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if has_nan:
        raise SystemExit("prototype produced NaNs")
    if max_abs_error > args.max_abs_tol:
        raise SystemExit("max_abs_error exceeded tolerance")
    if mean_abs_error > args.mean_abs_tol:
        raise SystemExit("mean_abs_error exceeded tolerance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
