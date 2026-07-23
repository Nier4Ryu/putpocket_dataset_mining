# GLM h192 SM120 Sparse MLA Kernel Report

## Executive Summary

Integrated the `leavelet/sparse_mla_sm120` package into the separate vLLM 0.25 GLM stack as an experiment. The original external kernel does not natively accept tiny GLM dimensions: standalone h192 decode/prefill calls fail with `Unsupported d_qk=192; expected 576 (V32) or 512 (MODEL1)`.

I patched only `externals/vllm_glm52_v025` to route tiny GLM h192 through a padded V32 adapter, padded the tiny sparse indexer path, rebuilt vLLM, and confirmed native vLLM generation runs on GPU 0. The generated text was nonsensical mixed-language/token fragments, so the path is infrastructure-successful but not yet quality/correctness-proven.

Dataset forwarding smoke was attempted but blocked before model/tool execution because this user cannot access `/var/run/docker.sock`.

## External Kernel Source

- Repo: `https://github.com/leavelet/sparse_mla_sm120`
- Path: `externals/sparse_mla_sm120`
- Branch: `master`
- Commit: `6999a13fc1a51b0eb4a38360a063944201c14112`
- Installed package import: `flash_mla_sm120`
- Build log: `logs/sparse_mla_sm120_build/20260723T020227Z/build_wheel.log`
- Install log: `logs/sparse_mla_sm120_build/20260723T020227Z/pip_install.log`
- Known-good upstream test: `CUDA_VISIBLE_DEVICES=0 pytest externals/sparse_mla_sm120/tests/test_decode.py::TestV32Decode::test_correctness -v -s --tb=short`
- Test log: `logs/sparse_mla_sm120_build/20260723T020227Z/pytest_v32_decode.log`
- Result: `8 passed`

## Kernel API / Shape Contract

The installed package exports `sparse_mla_decode_fwd`, `sparse_mla_prefill_fwd`, `flash_mla_with_kvcache`, `flash_mla_sparse_fwd`, `FlashMLASchedMeta`, and `get_mla_metadata`.

Inspection showed the current source only accepts V32/MODEL1-style QK dimensions:

- `d_qk=576` for V32
- `d_qk=512` for MODEL1
- shared model traits assume `D_V=512`
- V32 cache ABI uses a 656-byte entry layout

This is not a native tiny GLM h192/v128 implementation.

## Tiny GLM Shape Mapping

HF config for `inference-optimization/GLM-5.2-0.8B-A0.8B`:

- `model_type=glm_moe_dsa`
- `architectures=['GlmMoeDsaForCausalLM']`
- `kv_lora_rank=128`
- `qk_nope_head_dim=192`
- `qk_rope_head_dim=64`
- `v_head_dim=128`
- `index_n_heads=8`
- `index_head_dim=64`
- `index_topk=2048`

vLLM attention selector head size is `kv_lora_rank + qk_rope_head_dim = 192`.

## Standalone Kernel Test

Added `scripts/dev/test_sparse_mla_sm120_tiny_glm.py`.

Command:

```bash
source scripts/env/env_activate_glm52_v025.sh
CUDA_VISIBLE_DEVICES=0 python scripts/dev/test_sparse_mla_sm120_tiny_glm.py --topk 2048 --case both
```

Log: `logs/sparse_mla_sm120_build/20260723T020227Z/tiny_glm_shape_test.log`

Result:

- decode failed: `RuntimeError: Unsupported d_qk=192; expected 576 (V32) or 512 (MODEL1)`
- prefill failed: `RuntimeError: Unsupported d_qk=192; expected 576 (V32) or 512 (MODEL1)`

## vLLM Patch

Patch artifact: `patches/vllm_glm52_h192_sm120/vllm_glm52_h192_sm120.patch`

Files patched inside ignored external checkout `externals/vllm_glm52_v025`:

- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`
- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`
- `csrc/libtorch_stable/cache_kernels.cu`
- `vllm/compilation/passes/fusion/mla_rope_kvcache_cat_fusion.py`
- `vllm/model_executor/models/deepseek_v2.py`
- `vllm/utils/deep_gemm.py`

What changed:

- admitted `head_size=192` only for exact GLM tiny sparse MLA dimensions;
- selected `flash_mla_sm120` only for this exact h192 GLM path;
- packed/padded vLLM's h192 query/cache layout into the V32 576/656-byte ABI expected by the external kernel;
- kept the old h576 path intact;
- allowed `kv_lora_rank=128` in fp8 DS MLA cache insertion, padding unused NoPE lanes;
- avoided the fused MLA RoPE/cache path for `kv_lora_rank=128`;
- allowed indexer quant block size 64 for tiny `index_head_dim=64`;
- padded tiny `index_n_heads=8` to 16 at the DeepGEMM wrapper boundary to satisfy the backend assertion.

## vLLM Build

Command:

```bash
source scripts/env/env_activate_glm52_v025.sh
export CUDA_HOME=/usr/local/cuda-12.9
export PUTPOCKET_BUILD_THREADS=8 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 CARGO_BUILD_JOBS=8 NVCC_THREADS=1
python -m pip install --no-build-isolation --index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://pypi.org/simple -e externals/vllm_glm52_v025
```

Log: `logs/sparse_mla_sm120_build/20260723T021343Z/vllm_h192_rebuild.log`

Result: success. Active vLLM: `0.25.2.dev0+g752a3a504.d20260723` from `externals/vllm_glm52_v025`.

## GLM Native Generation Smoke

Command:

```bash
source scripts/env/env_activate_glm52_v025.sh
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from vllm import LLM, SamplingParams
model = "inference-optimization/GLM-5.2-0.8B-A0.8B"
llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=1, max_model_len=8192, max_num_seqs=1, dtype="auto")
outs = llm.generate(["Hello. Reply with one short sentence."], SamplingParams(temperature=0.0, max_tokens=64))
print(outs[0].outputs[0].text)
PY
```

Successful log: `logs/sparse_mla_sm120_build/20260723T030132Z/glm_native_smoke.log`

Backend evidence in log:

- `Using FLASHINFER_MLA_SPARSE_SM120 attention backend`
- `Using fp8_ds_mla KV cache format`
- `Using leavelet flash_mla_sm120 padded V32 adapter for GLM tiny sparse MLA h192 on SM120`

Result: model loaded and generated text. Output excerpt:

```text
价值onen kül'clock碱性ремя ... Quelleague_Surface,str着陆食品药品...
```

This is not meaningful output for the prompt. It proves native generation can run through the patched path, but it does not prove correctness of the padded h192 adapter.

## Dataset Forwarding Result

Attempted command:

```bash
source scripts/env/env_activate_glm52_v025.sh
CUDA_VISIBLE_DEVICES=0 python -m putpocket_dataset_mining.model_evaluation.glm52_vllm025_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_h192_sm120_on_mbpp_stateful_working_v0 \
  --serving-stack glm52_vllm025_blackwell \
  --profile smoke \
  --max-samples 1 \
  --workers 1 \
  --gpu-slots 0
```

Log: `logs/sparse_mla_sm120_build/20260723T030257Z/eval_smoke.log`

Run root created: `data/model_evaluation/runs/eval_glm52_08b_h192_sm120_on_mbpp_stateful_working_v0_20260723T030257Z/`

Result: blocked before agent/model rollout.

Exact error:

```text
InfraError: Docker image build failed: ERROR: permission denied while trying to connect to the docker API at unix:///var/run/docker.sock
```

Dataset audit succeeded first:

- dataset: `data/dataset_mining/datasets/mbpp_stateful_working_v0/accepted.jsonl`
- accepted count: `20`
- selected sample: `train_730`
- artifact completeness: first audited rows had no missing artifacts

## What Was Implemented

This was a vLLM wrapper/path experiment plus cache/indexer compatibility shims. It was not a native h192 kernel implementation inside `leavelet/sparse_mla_sm120`; the external kernel still only accepts V32/MODEL1 dimensions, so h192 is padded into the V32 ABI.

The old Qwen stack was preserved. No changes were made to `externals/vllm`, `Putpocket_env`, or mined dataset artifacts.

## Remaining Blockers

Primary correctness blocker: generated text is nonsensical. The most likely issue is that padding h192/v128 into a V32 h576/v512 kernel ABI is not mathematically equivalent enough for the tiny GLM model. A real h192/v128 sparse MLA kernel path is still needed for quality.

Dataset forwarding blocker:

- failing command: the dataset smoke command in the previous section
- log path: `logs/sparse_mla_sm120_build/20260723T030257Z/eval_smoke.log`
- exact error: Docker socket permission denied
- host state: user `dyryu` is in groups `dyryu,sudo`, not `docker`; `/var/run/docker.sock` is `root:docker` with mode `660`
- smallest next action: add `dyryu` to the `docker` group and start a new login/session, or run the evaluator from a user/session with Docker socket access.

Git push blocker:

- commit created locally: this report/source commit (`Add GLM h192 SM120 sparse MLA integration experiment`)
- failing command: `git push -u origin blackwell`
- exact error: `remote: No anonymous write access. fatal: Authentication failed for 'https://github.com/Nier4Ryu/putpocket_dataset_mining.git/'`
- smallest next action: restore GitHub credentials for this shell/session or push the local `blackwell` branch from a credentialed session.

## Next Recommended Action

First fix Docker socket access and rerun the one-sample smoke to collect actual Cline/tool-call behavior. In parallel, treat the current padded-kernel generation as suspect: implement or obtain a true h192/v128 SM120 sparse MLA kernel instead of relying on V32 padding if meaningful text is required.
