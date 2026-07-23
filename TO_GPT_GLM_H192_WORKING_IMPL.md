# GLM h192/v128 Working Implementation Report

## Executive Summary

Blocked before a valid vLLM-native GLM generation path.

What was completed:
- Confirmed GLM-5.2-0.8B tiny dimensions from HF config: `kv_lora_rank=128`, `qk_rope_head_dim=64`, derived vLLM MLA `head_size=192`, `v_head_dim=128`.
- Tested the installed official FlashInfer SM120 MLA API directly. It rejects the tiny dimensions with `Unsupported MLA dimensions` and only advertises `kv_lora_rank=512` or `256`.
- Implemented a correctness-first Triton sparse MLA decode prototype for the actual h192/v128 math.
- Validated that prototype against a PyTorch reference at `topk=64`, `topk=256`, and full GLM `index_topk=2048`.

What was not completed:
- No valid vLLM paged-cache backend was implemented yet.
- GLM native vLLM generation was not re-run as a success path because the only existing local vLLM h192 experiment is a padding shim that previously generated incoherent text.
- Dataset forwarding was not attempted because there is no coherent vLLM-native generation path to use.

## Branch / Git

- Branch: `blackwell`
- Local HEAD: `f8065c6efafccfd3b4fccd722c6ab7e8d35415df`
- Status while writing: branch ahead of `origin/blackwell` by 2 commits.
- Push status: previous push failed because the remote requires GitHub authentication for `https://github.com/Nier4Ryu/putpocket_dataset_mining.git/`.

## Starting Point

The previous vLLM-native GLM path failed on Blackwell SM120 because the model selects sparse MLA:

- `head_size=192`
- `use_mla=True`
- `use_sparse=True`
- compute capability SM120

Base vLLM 0.25 source supports the SM120 FlashInfer sparse MLA backend only for `head_size=576`; its packed `fp8_ds_mla` cache layout is hardcoded around the full-size `512 NoPE + 16 scale bytes + 128 RoPE = 656 bytes/token` format.

## B Option Status

No ready-made h192/v128 custom kernel package was found.

Evidence:
- Installed `flashinfer==0.6.13` exposes the official SM120 MLA API but rejects `kv_lora_rank=128`.
- `leavelet/sparse_mla_sm120` provides a working SM120 sparse MLA package for full layouts such as V32 h576/v512 and MODEL1 h512/v512.
- Its direct h192 probe previously failed with `Unsupported d_qk=192; expected 576 (V32) or 512 (MODEL1)`.

## A Path Attempt

Attempted the adapter/wrapper path first by calling the installed official FlashInfer API directly with tiny GLM dimensions.

Command log:
- `logs/glm_h192_impl/20260723T130501Z/flashinfer_tiny_api_probe.log`

Result:

```text
Unsupported MLA dimensions, got kv_lora_rank=128 and qk_rope_head_dim=64,
supported dimensions are:
MLAHeadDimensions(qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, kv_lora_rank=512)
MLAHeadDimensions(qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=128, kv_lora_rank=256)
```

This failed for `backend=auto`, `xqa`, `trtllm-gen`, and `cute-dsl`, with both natural 192-byte cache width and 656-byte full-layout cache width.

Conclusion: a Python-only vLLM routing/validation patch to call official FlashInfer is not sufficient. The underlying FlashInfer API does not currently accept tiny GLM h192/v128.

Files touched for A: none.

## C Triton Prototype

Implemented:
- `src/putpocket_dataset_mining/kernels/glm_h192_triton.py`
- `scripts/dev/test_glm_h192_triton.py`

Supported computation:

```text
scores = (q_latent @ kv_c.T) + (q_rope @ k_rope.T)
weights = softmax(scores / sqrt(192)) over sparse top-k tokens
output = weights @ kv_c
```

Shapes:
- `q_latent`: `[T, H, 128]`
- `q_rope`: `[T, H, 64]`
- `kv_c`: `[S, 128]`
- `k_rope`: `[S, 64]`
- `topk_indices`: `[T, K]`
- output: `[T, H, 128]`

Validation results:
- `topk=64`: pass, max abs error `2.38e-07`, no NaNs.
- `topk=256`: pass, max abs error `1.19e-07`, no NaNs.
- `topk=2048`: pass, max abs error `7.45e-08`, no NaNs.

Logs:
- `logs/glm_h192_impl/20260723T130736Z/triton_h192_topk64.log`
- `logs/glm_h192_impl/20260723T130751Z/triton_h192_topk256.log`
- `logs/glm_h192_impl/20260723T130936Z/triton_h192_topk2048_streaming.log`

Limitations:
- The prototype consumes dense synthetic `kv_c` and `k_rope` tensors.
- It does not yet read vLLM paged KV cache layout.
- It does not yet dequantize vLLM `fp8_ds_mla` cache bytes.
- It is not yet wired into vLLM metadata, physical top-k indexing, or CUDA graph execution.

## D Native Kernel Path

D is still required for a working vLLM-native path.

The missing layer is a real vLLM backend/kernel integration that can:
- accept `kv_lora_rank=128`, `qk_rope_head_dim=64`, `v_head_dim=128`,
- read vLLM sparse top-k physical token indices,
- read/dequantize the paged KV cache for tiny h192/v128,
- compute the validated h192 sparse MLA decode path,
- return latent `[tokens, heads, 128]` output to the existing GLM projection path.

This can be implemented as either:
- a vLLM Triton backend based on the prototype, extended to paged KV cache and `fp8_ds_mla`, or
- a native CUDA/FlashInfer h192/v128 kernel.

## Selected h192 Path

Current selected path: blocked at vLLM integration.

The standalone h192/v128 Triton math is correct for synthetic dense tensors. It is not yet selected by vLLM, and the old padding shim is not accepted as a valid working implementation because its prior generation output was incoherent.

## Correctness Validation

Commands run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/dev/test_glm_h192_triton.py --topk 64 --seq-len 192
CUDA_VISIBLE_DEVICES=0 python scripts/dev/test_glm_h192_triton.py --topk 256 --seq-len 512
CUDA_VISIBLE_DEVICES=0 python scripts/dev/test_glm_h192_triton.py --topk 2048 --seq-len 2048
```

All three passed against the PyTorch reference with no NaNs.

## vLLM Build Summary

No vLLM rebuild was performed in this step.

Reason: the validated implementation is a standalone Triton prototype under repo source, not a vLLM C++/CUDA extension change. Rebuilding vLLM would not make vLLM select this prototype until a real backend integration is written.

## GLM Native Generation Result

Not rerun as a valid success path.

Reason: the only h192-capable local vLLM tree currently contains the prior padding-shim experiment, and that experiment produced incoherent text. The current task explicitly disallows declaring success from an unproven padding shim.

Prior invalid-output log for reference:
- `logs/sparse_mla_sm120_build/20260723T030132Z/glm_native_smoke.log`

## Dataset Forwarding Result

Not attempted in this step.

Reason: dataset forwarding requires a coherent vLLM-native GLM generation path. The valid h192/v128 computation exists only as a standalone dense-tensor prototype, not as a vLLM generation backend.

Dataset presence was confirmed:
- dataset: `data/dataset_mining/datasets/mbpp_stateful_working_v0`
- accepted rows: `20`
- accepted file: `data/dataset_mining/datasets/mbpp_stateful_working_v0/accepted.jsonl`

Expected future command after vLLM backend integration:

```bash
CUDA_VISIBLE_DEVICES=0 python -m putpocket_dataset_mining.model_evaluation.glm52_vllm025_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_h192_sm120_working_on_mbpp_stateful_working_v0 \
  --serving-stack glm52_vllm025_h192_sm120 \
  --profile smoke \
  --limit 1 \
  --workers 1 \
  --gpu-slots 0
```

Secondary infrastructure note: the previous dataset smoke attempt hit Docker socket permission denial. After model generation is fixed, the user likely still needs the active shell/session to have `docker` group access.

## Full/Subsample Evaluation Result

Not run. No valid vLLM-native generation backend exists yet.

## What Is Still Not Proven

- vLLM paged-cache h192/v128 decode correctness.
- `fp8_ds_mla` tiny cache dequantization correctness.
- Coherent GLM generation from vLLM.
- Cline XML tool-call behavior from GLM.
- History-1 verifier pass/fail on mined samples.
- Full 20-sample evaluation counts.

## Known Blockers

Primary blocker:
- Failing command: direct official FlashInfer API probe in `logs/glm_h192_impl/20260723T130501Z/flashinfer_tiny_api_probe.log`.
- Error: `Unsupported MLA dimensions, got kv_lora_rank=128 and qk_rope_head_dim=64`.
- Smallest next action: implement a vLLM h192/v128 sparse MLA backend that uses the Triton prototype’s math but reads vLLM paged KV cache and top-k metadata.

Secondary blocker after model backend:
- Docker workspace execution may still fail until the current user/session can access `/var/run/docker.sock`.

## Next Recommended Action

Implement `FLASHINFER_MLA_SPARSE_SM120_H192` or equivalent as a separate vLLM backend in `externals/vllm_glm52_v025`, using the validated Triton h192/v128 math as the reference. Start with decode-only paged-cache support for one-token generation, then validate coherent text before adding prefill/full evaluation support.

