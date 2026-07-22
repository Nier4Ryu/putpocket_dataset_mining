# GLM vLLM Serve Smoke Report

## Executive Summary

Local `vllm serve` does not currently run `inference-optimization/GLM-5.2-0.8B-A0.8B` on this Blackwell server through the repo-local GLM vLLM stack.

The generic official-style CLI command, `--dtype auto`, and `--dtype float32` all fail before `/v1/models` readiness with the same native vLLM sparse-MLA backend blocker already seen in the Python `LLM(...)` path:

`No valid attention backend found ... head_size=192 ... use_mla=True ... use_sparse=True`

The CLI and Python path use the same vLLM package: `vllm 0.23.0` from `externals/vllm_glm52`. This means the generic `vllm serve` path does not route around the local native backend limitation.

The `--model-impl transformers` CLI variant changes the failure mode, but still does not serve. It fails during Torch/Dynamo compilation with a fake-tensor shape error:

`shape '[-1, 16, 192]' is invalid for input of size 4096*s27`

## Environment

- Branch: `blackwell`
- Commit at start: `5f73d4d1a2927fa511bff25abcdee211201666c6`
- Env path: `Putpocket_env_glm52`
- Activation: `source scripts/env/env_activate_glm52.sh`
- Python: `/home/dyryu/putpocket_dataset_mining/Putpocket_env_glm52/bin/python`
- Python version: `3.13.14`
- vLLM CLI: `/home/dyryu/putpocket_dataset_mining/Putpocket_env_glm52/bin/vllm`
- vLLM version: `0.23.0`
- vLLM file: `/home/dyryu/putpocket_dataset_mining/externals/vllm_glm52/vllm/__init__.py`
- Torch: `2.11.0+cu129`
- Torch CUDA: `12.9`
- Transformers: `5.14.1`
- CUDA_HOME: `/usr/local/cuda-12.9`
- GPU used: `CUDA_VISIBLE_DEVICES=0`
- GPU detected: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- vLLM platform capability: `DeviceCapability(major=12, minor=0)`, `12.0`
- vLLM `support_deep_gemm`: `False`

## Model Config

Config-only load for `inference-optimization/GLM-5.2-0.8B-A0.8B`:

- `model_type`: `glm_moe_dsa`
- `architectures`: `['GlmMoeDsaForCausalLM']`
- `torch_dtype`: `torch.float32`
- `hidden_size`: `2048`
- `num_attention_heads`: `16`
- `num_key_value_heads`: `16`
- `kv_lora_rank`: `128`
- `q_lora_rank`: `512`
- `qk_nope_head_dim`: `192`
- `qk_rope_head_dim`: `64`
- `v_head_dim`: `128`
- `index_n_heads`: `8`
- `index_head_dim`: `64`
- `index_topk`: `2048`
- `use_sparse`: `None`
- `use_mla`: `None`
- vLLM-derived MLA `head_size`: `kv_lora_rank + qk_rope_head_dim = 192`

The sparse/DSA behavior is model-derived via `glm_moe_dsa` and `index_topk`, not from a repo launch flag.

## Commands Tried

Generic official-style serve command:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve inference-optimization/GLM-5.2-0.8B-A0.8B --served-model-name glm52-08b --host 127.0.0.1 --port 18080 --max-model-len 8192 --max-num-seqs 1 --trust-remote-code
```

Variant A, explicit dtype auto:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve inference-optimization/GLM-5.2-0.8B-A0.8B --served-model-name glm52-08b --host 127.0.0.1 --port 18081 --max-model-len 8192 --max-num-seqs 1 --trust-remote-code --dtype auto
```

Variant B, dtype float32:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve inference-optimization/GLM-5.2-0.8B-A0.8B --served-model-name glm52-08b --host 127.0.0.1 --port 18082 --max-model-len 8192 --max-num-seqs 1 --trust-remote-code --dtype float32
```

Variant C, vLLM Transformers model implementation:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve inference-optimization/GLM-5.2-0.8B-A0.8B --served-model-name glm52-08b --host 127.0.0.1 --port 18083 --max-model-len 8192 --max-num-seqs 1 --trust-remote-code --model-impl transformers
```

## Results

Generic:

- Server ready: no
- `/v1/models`: no, zero-byte readiness capture
- Chat/completion request: not sent because server never became ready
- Status: server exited early after 6 readiness checks
- Log: `data/model_evaluation/serve_smoke/glm_vllm_serve_smoke_20260723_053145/generic.log`
- Err: `data/model_evaluation/serve_smoke/glm_vllm_serve_smoke_20260723_053145/generic.err`
- Key log lines:
  - `Resolved architecture: GlmMoeDsaForCausalLM`
  - `Downcasting torch.float32 to torch.bfloat16`
  - `ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(head_size=192, dtype=torch.bfloat16, kv_cache_dtype=auto, ... use_mla=True, ... use_sparse=True, ...)`

Variant A, `--dtype auto`:

- Server ready: no
- `/v1/models`: no, zero-byte readiness capture
- Chat/completion request: not sent
- Status: server exited early after 6 readiness checks
- Log: `data/model_evaluation/serve_smoke/glm_vllm_serve_smoke_20260723_053145/dtype_auto.log`
- Err: `data/model_evaluation/serve_smoke/glm_vllm_serve_smoke_20260723_053145/dtype_auto.err`
- Error summary: same as generic; BF16 sparse-MLA backend selection fails.

Variant B, `--dtype float32`:

- Server ready: no
- `/v1/models`: no, zero-byte readiness capture
- Chat/completion request: not sent
- Status: server exited early after 6 readiness checks
- Log: `data/model_evaluation/serve_smoke/glm_vllm_serve_smoke_20260723_053145/dtype_float32.log`
- Err: `data/model_evaluation/serve_smoke/glm_vllm_serve_smoke_20260723_053145/dtype_float32.err`
- Error summary: same sparse-MLA backend selection fails, now with `dtype=torch.float32`; backend reasons add `dtype not supported`.

Variant C, `--model-impl transformers`:

- Server ready: no
- `/v1/models`: no, zero-byte readiness capture
- Chat/completion request: not sent
- Status: server exited early after 16 readiness checks
- Log: `data/model_evaluation/serve_smoke/glm_vllm_serve_smoke_20260723_053145/model_impl_transformers.log`
- Err: `data/model_evaluation/serve_smoke/glm_vllm_serve_smoke_20260723_053145/model_impl_transformers.err`
- Error summary: no native sparse-MLA selector error in the captured key lines; it fails later in Torch/Dynamo:
  - `torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call`
  - `view(... size=(s27, 4096), dtype=torch.bfloat16), -1, 16, 192): got RuntimeError("shape '[-1, 16, 192]' is invalid for input of size 4096*s27")`

## Does CLI Serve Differ From Python LLM?

The CLI and Python import the same vLLM package:

```text
0.23.0
/home/dyryu/putpocket_dataset_mining/externals/vllm_glm52/vllm/__init__.py
/home/dyryu/putpocket_dataset_mining/Putpocket_env_glm52/bin/vllm
```

For the native path, CLI serve does not differ materially from Python `LLM(...)`: it resolves the same `GlmMoeDsaForCausalLM` architecture, downcasts to BF16 by default, reaches `AttentionSelectorConfig(head_size=192, use_mla=True, use_sparse=True)`, and fails with no valid sparse-MLA backend.

The only tested CLI config that differs materially is `--model-impl transformers`. That avoids the native sparse-MLA backend error, but still fails before serving due to Torch/Dynamo compilation/shape handling. It is not currently usable for dataset evaluation.

## Recommended Next Config

For this repo-local GLM vLLM stack, there is no working `vllm serve` command yet.

The best native-vLLM next action is to replace or update only the separate `externals/vllm_glm52` stack to a GLM-5.2/SM120-compatible vLLM build that supports sparse MLA for:

- `head_size=192`
- `use_mla=True`
- `use_sparse=True`
- RTX PRO Blackwell capability `12.0`

If the human wants an official-style server command to try after obtaining a compatible image/build, use this shape:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve inference-optimization/GLM-5.2-0.8B-A0.8B \
  --served-model-name glm52-08b \
  --host 127.0.0.1 \
  --port 18080 \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --trust-remote-code
```

For official GLM image testing, use the GLM-specific CUDA 12.9 image path as a separate container smoke, not as proof that the repo-local source build works:

```bash
docker run --gpus '"device=0"' --ipc=host \
  -p 18080:8000 \
  vllm/vllm-openai:glm52-cu129 \
  inference-optimization/GLM-5.2-0.8B-A0.8B \
  --served-model-name glm52-08b \
  --max-model-len 8192 \
  --max-num-seqs 1 \
  --trust-remote-code
```

`--model-impl transformers` is not ready as tested. A possible later debugging experiment is `--model-impl transformers --enforce-eager`, because the failure hint points at Torch/Dynamo compilation, but that was not part of this bounded serve smoke.

SGLang remains useful only as a model capability sanity check, not as a vLLM serve fix.

## Can This Be Used For Dataset Evaluation?

No. None of the tested `vllm serve` variants reached `/v1/models`, so there is no serving endpoint to use for evaluation over the Qwen-mined dataset.

The generic CLI failure matching Python `LLM(...)` means the blocker is not the repo's Python wrapper configuration. The native repo-local vLLM stack itself lacks a valid backend for this tiny GLM DSA/sparse-MLA configuration on this SM120 host.

## Known Blockers

Primary blocker for native `vllm serve`:

- Failing command: generic command in `## Commands Tried`
- Log path: `data/model_evaluation/serve_smoke/glm_vllm_serve_smoke_20260723_053145/generic.err`
- Exact error:

```text
ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(head_size=192, dtype=torch.bfloat16, kv_cache_dtype=auto, block_size=None, use_mla=True, has_sink=False, use_sparse=True, use_mm_prefix=False, use_per_head_quant_scales=False, attn_type=AttentionType.DECODER, use_non_causal=False, use_batch_invariant=False, use_kv_connector=False).
```

- Backend reasons include `head_size not supported`, `sparse not supported`, and `compute capability not supported`.
- Smallest next action: update or replace only `externals/vllm_glm52` with a GLM-5.2/SM120-compatible vLLM source/image, then rerun the generic serve command above.

Secondary blocker for `--model-impl transformers`:

- Failing command: `model_impl_transformers` command in `## Commands Tried`
- Log path: `data/model_evaluation/serve_smoke/glm_vllm_serve_smoke_20260723_053145/model_impl_transformers.log`
- Exact key error:

```text
torch._dynamo.exc.TorchRuntimeError: RuntimeError when making fake tensor call
Explanation: Dynamo failed to run FX node with fake tensors: call_method view(*(FakeTensor(..., device='cuda:0', size=(s27, 4096), dtype=torch.bfloat16), -1, 16, 192), **{}): got RuntimeError("shape '[-1, 16, 192]' is invalid for input of size 4096*s27")
```

- Smallest next action if pursuing Transformers implementation: test `--model-impl transformers --enforce-eager` as a separate, explicitly approved follow-up.

## Artifacts

Run directory:

`data/model_evaluation/serve_smoke/glm_vllm_serve_smoke_20260723_053145/`

Important files:

- `metadata.txt`
- `generic_command.txt`
- `generic.log`
- `generic.err`
- `generic_status.txt`
- `dtype_auto_command.txt`
- `dtype_auto.log`
- `dtype_auto.err`
- `dtype_auto_status.txt`
- `dtype_float32_command.txt`
- `dtype_float32.log`
- `dtype_float32.err`
- `dtype_float32_status.txt`
- `model_impl_transformers_command.txt`
- `model_impl_transformers.log`
- `model_impl_transformers.err`
- `model_impl_transformers_status.txt`
