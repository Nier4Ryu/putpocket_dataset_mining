# GLM h192/v128 Kernel Working Report

## Executive Summary
I did not reach a validated working GLM path. The A path was infeasible because installed FlashInfer rejects the tiny GLM dimensions `kv_lora_rank=128` and `qk_rope_head_dim=64`. I then implemented the C path: a correctness-first Triton h192/v128 sparse MLA decode prototype plus a vLLM fp8 paged-cache adapter.

The standalone Triton path passes PyTorch reference tests, including a top-k 2048 paged-cache case. vLLM now loads GLM and selects the Putpocket Triton h192/v128 path, but generation output is incoherent garbage in both compiled and eager modes. Dataset forwarding was not run because the native generation result is not meaningful enough to feed into the headless Cline loop.

## Branch / Git
- Branch: `blackwell`
- Starting commit for this run: `d4b865de5340f8b1b80d327609da602951bf1544`
- Push target: `origin/blackwell`
- Old Qwen stack preservation: `Putpocket_env` and `externals/vllm` were not modified.
- GLM stack used: `Putpocket_env_glm52_v025` and `externals/vllm_glm52_v025`.

## Starting Failure
Previous vLLM native GLM attempts failed because the selector could not find a valid backend for:
- `head_size=192`
- `use_mla=True`
- `use_sparse=True`
- Blackwell `SM120`

That failure was a backend availability/shape support issue for tiny GLM, not a Qwen inherited config.

## Model Config
Config-only inspection of `inference-optimization/GLM-5.2-0.8B-A0.8B` showed:
- `model_type`: `glm_moe_dsa`
- `architectures`: `['GlmMoeDsaForCausalLM']`
- `num_hidden_layers`: 6
- `hidden_size`: 2048
- `num_attention_heads`: 16
- `num_key_value_heads`: 16
- `kv_lora_rank`: 128
- `q_lora_rank`: 512
- `qk_nope_head_dim`: 192
- `qk_rope_head_dim`: 64
- `v_head_dim`: 128
- `index_n_heads`: 8
- `index_head_dim`: 64
- `index_topk`: 2048
- vLLM MLA decode head size: `kv_lora_rank + qk_rope_head_dim = 192`

## A Path: Existing FlashInfer / Python-level vLLM Adapter
Installed FlashInfer state:
- `flashinfer`: `0.6.13`
- Top-level `flashinfer.sparse_mla_sm120_paged_attention`: missing
- Top-level `flashinfer.BatchSparseMLAPagedAttentionWrapper`: missing
- vLLM utility `has_flashinfer()`: true
- vLLM utility `has_flashinfer_sparse_mla_sm120()`: true

Direct FlashInfer calls rejected the tiny shape for both cache dimensions 192 and 656:

```text
Unsupported MLA dimensions, got kv_lora_rank=128 and qk_rope_head_dim=64
supported dimensions are:
  kv_lora_rank=512, qk_rope_head_dim=64
  kv_lora_rank=256, qk_rope_head_dim=64
```

Conclusion: A cannot use existing FlashInfer directly for h192/v128 without a native FlashInfer/kernel change. No vLLM rebuild was done for A.

Log: `logs/glm_h192_working_impl/20260724T122649Z/flashinfer_a_path_probe.log`

## C Path: Triton h192/v128 Prototype
Files added or changed:
- `src/putpocket_dataset_mining/kernels/glm_h192_triton.py`
- `scripts/dev/test_glm_h192_triton.py`
- `tests/test_glm_h192_triton.py`
- `patches/vllm_glm52_h192_triton/vllm_glm52_h192_triton.patch`

Implemented:
- Dense h192/v128 sparse MLA decode against explicit top-k indices.
- PyTorch reference implementation.
- vLLM `fp8_ds_mla` padded cache adapter for tiny GLM.
- Test cache packing for h128 latent values, one FP32 scale, and h64 RoPE values in the 656-byte vLLM storage layout.
- Python-level vLLM route that selects the Putpocket Triton h192/v128 path only for the tiny GLM h192/v128 case.

Validation results:
- Dense Triton top-k64: max abs error `2.38e-07`, no NaNs.
- vLLM fp8 paged top-k64: max abs error `2.98e-07`, no NaNs.
- vLLM fp8 paged top-k2048: max abs error `8.94e-08`, no NaNs.
- `python -m pytest tests/test_glm_h192_triton.py -q`: passed, 2 tests.

Logs:
- `logs/glm_h192_working_impl/20260724T123001Z/triton_dense_topk64.json`
- `logs/glm_h192_working_impl/20260724T123001Z/triton_paged_topk64.json`
- `logs/glm_h192_working_impl/20260724T123001Z/triton_paged_topk2048.json`
- `logs/glm_h192_working_impl/20260724T123001Z/pytest_glm_h192_triton.log`

Limitations:
- The Triton test validates the sparse MLA decode computation and vLLM cache byte layout adapter in isolation.
- It does not prove the whole GLM/vLLM path, including DSA indexer top-k selection, prefill path, layer fusion, and all model runner cache/update contracts.

## D Path: Native CUDA/FlashInfer Requirement
D was not implemented. A native h192/v128 FlashInfer or CUDA kernel may still be required, but the current blocker is more specific than performance: whole-model generation is numerically wrong even after the Triton decode math validates in isolation.

The next native work should not start until a layer-level correctness harness identifies whether the corruption comes from:
- DSA indexer h8/top-k path,
- h128 fp8 cache packing or scale contract,
- prefill path,
- the modified vLLM `deep_gemm` indexer-head padding,
- model runner integration,
- or the Triton decode adapter itself under real vLLM tensors.

## Selected Execution Path
Current experimental selected path:
- vLLM backend: `FLASHINFER_MLA_SPARSE_SM120`
- Tiny GLM route: Putpocket Triton h192/v128 sparse MLA path
- Log message observed: `Using Putpocket Triton h192/v128 sparse MLA path for GLM tiny on SM120.`

This path is selected but not accepted as working because model output is incoherent.

## Correctness Validation
Passed:

```bash
source scripts/env/env_activate_glm52_v025.sh
python -m compileall src/putpocket_dataset_mining/kernels scripts/dev/test_glm_h192_triton.py tests/test_glm_h192_triton.py
python -m pytest tests/test_glm_h192_triton.py -q
CUDA_VISIBLE_DEVICES=0 python scripts/dev/test_glm_h192_triton.py --paged-cache --topk 2048 --seq-len 2048
```

The standalone checks prove the local h192/v128 sparse MLA computation is shape-correct and close to the PyTorch reference on synthetic tensors.

## vLLM Build Summary
No vLLM rebuild was performed in this run. The new routing/adapter work is Python-level. The current external vLLM checkout already contains earlier experimental native changes for `fp8_ds_mla` h128 cache packing; the patch artifact records the full current diff:

- `patches/vllm_glm52_h192_triton/vllm_glm52_h192_triton.patch`

A clean reproduction from upstream vLLM 0.25 would require applying that patch and rebuilding vLLM with the configured cap:

```bash
PUTPOCKET_BUILD_THREADS=8 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_BUILD_JOBS=8 NVCC_THREADS=1 \
python -m pip install --no-build-isolation -e externals/vllm_glm52_v025
```

That rebuild was not run here.

## GLM Native Generation Result
Command used:

```bash
source scripts/env/env_activate_glm52_v025.sh
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from vllm import LLM, SamplingParams

model = "inference-optimization/GLM-5.2-0.8B-A0.8B"
llm = LLM(
    model=model,
    trust_remote_code=True,
    tensor_parallel_size=1,
    max_model_len=8192,
    max_num_seqs=1,
    dtype="auto",
)
outs = llm.generate(
    ["Write one short sentence explaining what a Python function is."],
    SamplingParams(temperature=0.0, max_tokens=80),
)
print(outs[0].outputs[0].text)
PY
```

Compiled/default mode:
- Exit status: 0
- vLLM loaded model and selected the h192 path.
- Output excerpt: `midpoint asistencia Newest Pyongyang ...`
- Result: failed correctness; output is incoherent.
- Log: `logs/glm_h192_working_impl/20260724T123337Z/glm_native_triton_h192_smoke.log`

Eager diagnostic:
- Exit status: 0
- vLLM loaded model and selected the h192 path.
- Output excerpt: `midpoint asistencia Newest Pyongyang ...`
- Result: failed correctness; output is still incoherent, so this is not a CUDA graph capture-only issue.
- Log: `logs/glm_h192_working_impl/20260724T123616Z/glm_native_triton_h192_eager_smoke.log`

Earlier compiled attempt:
- Failed during CUDA graph capture because validation called `.item()` inside capture.
- Fixed by disabling validation in the vLLM adapter call.
- Log: `logs/glm_h192_working_impl/20260724T123220Z/glm_native_triton_h192_smoke.log`

## Dataset Forwarding Result
Dataset exists and was not modified:
- `data/dataset_mining/datasets/mbpp_stateful_working_v0/accepted.jsonl`
- Accepted samples: 20
- Example artifact root: `data/dataset_mining/runs/full_server_validation_20260707T175646Z/samples/train_730/attempt_5a8d1db9b812`

Dataset forwarding was not run. The native GLM smoke generated incoherent text, so feeding it into the headless Cline tool loop would not answer whether a correct vLLM-native GLM path can use the mined samples.

Pending command once native generation is coherent:

```bash
CUDA_VISIBLE_DEVICES=0 python -m putpocket_dataset_mining.model_evaluation.glm52_vllm025_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_h192_triton_or_flashinfer_on_mbpp_stateful_working_v0 \
  --serving-stack glm52_vllm025_h192_sm120 \
  --profile smoke \
  --limit 1 \
  --workers 1 \
  --gpu-slots 0
```

## Full/Subsample Evaluation Result
Not run. Full/subsample evaluation should wait until the simple generation smoke produces coherent text.

## What Is Still Not Proven
- The real vLLM DSA indexer path is not proven correct for `index_n_heads=8`.
- The whole-layer h192/h128/h64 GLM path is not proven equivalent to Transformers.
- The current h128 fp8 cache packing layout is only probed synthetically and through vLLM startup, not layer-by-layer.
- The model runner prefill/decode handoff is not proven correct.
- The current external vLLM diff still contains earlier indexer and cache-packing experiments that need a clean layer-level audit.

## Known Blockers
Primary blocker:
- Failing condition: vLLM native GLM generation returns incoherent text despite selecting the h192 path.
- Command: the GLM native generation command in the section above.
- Logs:
  - `logs/glm_h192_working_impl/20260724T123337Z/glm_native_triton_h192_smoke.log`
  - `logs/glm_h192_working_impl/20260724T123616Z/glm_native_triton_h192_eager_smoke.log`
- Smallest next action: build a layer-level correctness harness comparing Transformers GLM against vLLM GLM for one short prompt, starting with DSA indexer logits/top-k indices, then the h192 sparse MLA decode output, then post-attention hidden states.

Secondary blocker:
- Existing FlashInfer does not support h192/v128 directly.
- Command/log: `logs/glm_h192_working_impl/20260724T122649Z/flashinfer_a_path_probe.log`
- Smallest next action: do not retry existing FlashInfer for h192; use the Triton harness to identify the exact incorrect vLLM tensor contract before writing native CUDA.

## Next Recommended Action
Implement a focused correctness harness that runs the same prompt through Transformers and the experimental vLLM GLM path and captures per-layer intermediate tensors. The first comparison targets should be:

1. DSA indexer logits and selected top-k physical indices.
2. h192 query split into latent and RoPE components.
3. h128 fp8 cache dequantization versus the source latent cache.
4. Putpocket Triton decode output versus a PyTorch reference using the real vLLM tensors.
5. Post-attention hidden state before MoE.

Only after that harness identifies the corrupted layer should native CUDA/FlashInfer work proceed.
