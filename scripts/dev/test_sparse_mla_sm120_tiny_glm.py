#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys

import torch


def _make_tiny_glm_inputs(
    *,
    num_tokens: int,
    num_heads: int,
    d_qk: int,
    d_v: int,
    topk: int,
    block_size: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    torch.manual_seed(20260723)
    device = torch.device("cuda")
    q = (torch.randn(num_tokens, num_heads, d_qk, device=device, dtype=torch.bfloat16) / 10).contiguous()

    d_rope = d_qk - d_v
    if d_rope != 64:
        raise ValueError(f"expected tiny GLM d_rope=64, got {d_rope}")

    # Tiny GLM h192 would naturally pack 128 FP8 NoPE bytes, one FP32 scale,
    # and 64 bf16 RoPE elements: 128 + 4 + 128 = 260 bytes/token.
    # The current external kernel dispatches by q.shape[-1], so the exact
    # byte count is only used to create a plausible contiguous cache operand.
    bytes_per_token = d_v + 4 + d_rope * 2
    kv_cache = torch.empty(
        (num_blocks, block_size, 1, bytes_per_token),
        device=device,
        dtype=torch.uint8,
    ).random_(0, 128).contiguous()

    s_kv = num_blocks * block_size
    indices = torch.randint(0, s_kv, (num_tokens, topk), device=device, dtype=torch.int32)
    indices[:, -min(10, topk) :] = -1
    sm_scale = 1.0 / math.sqrt(d_qk)
    return q, kv_cache, indices.contiguous(), sm_scale


def _run_case(kind: str, *, topk: int) -> bool:
    import flash_mla_sm120

    num_tokens = 1 if kind == "decode" else 65
    q, kv_cache, indices, sm_scale = _make_tiny_glm_inputs(
        num_tokens=num_tokens,
        num_heads=16,
        d_qk=192,
        d_v=128,
        topk=topk,
        block_size=64,
        num_blocks=max(32, (topk + 63) // 64),
    )

    if kind == "decode":
        output, lse = flash_mla_sm120.sparse_mla_decode_fwd(
            q,
            kv_cache,
            indices,
            sm_scale,
            d_v=128,
            bf16_qk=True,
        )
    else:
        output, _max_logits, lse = flash_mla_sm120.sparse_mla_prefill_fwd(
            q,
            kv_cache,
            indices,
            sm_scale,
            d_v=128,
            bf16_qk=True,
        )

    expected = (num_tokens, 16, 128)
    print(f"{kind} output shape: {tuple(output.shape)}")
    print(f"{kind} lse shape: {tuple(lse.shape)}")
    print(f"{kind} output finite: {bool(torch.isfinite(output.float()).all().item())}")
    print(f"{kind} lse finite-or-neg-inf: {bool((torch.isfinite(lse) | torch.isneginf(lse)).all().item())}")
    if tuple(output.shape) != expected:
        raise RuntimeError(f"{kind} output shape mismatch: expected {expected}, got {tuple(output.shape)}")
    if not torch.isfinite(output.float()).all():
        raise RuntimeError(f"{kind} output contains NaN or Inf")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe leavelet sparse_mla_sm120 with tiny GLM h192 shapes.")
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--case", choices=["decode", "prefill", "both"], default="both")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))

    cases = ["decode", "prefill"] if args.case == "both" else [args.case]
    failed = False
    for case in cases:
        print(f"== {case} tiny GLM h192/topk={args.topk} ==")
        try:
            _run_case(case, topk=args.topk)
        except Exception as exc:
            failed = True
            print(f"{case} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            print(f"{case} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
