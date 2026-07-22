# GLM-5.2 vLLM 0.23 Stack + Evaluation Report

## Executive Summary

Separate GLM infrastructure was added without overwriting the existing Qwen / Putpocket-v0.19.1 stack. The repo-local GLM environment `Putpocket_env_glm52` was created successfully, vLLM `0.23.0+cu129` was built from source at the required 8-thread cap, and import validation passed.

GLM model load smoke did not pass. Local vLLM 0.23.0 resolves `inference-optimization/GLM-5.2-0.8B-A0.8B` as `GlmMoeDsaForCausalLM`, but engine initialization fails before weights are fully loaded because no valid sparse MLA attention backend is available for CUDA / SM120 with `head_size=192` and `use_sparse=True`. Per instruction, GLM evaluation smoke and full evaluation were not run after this backend failure.

## Branch / Git

- Branch: `blackwell`
- Remote: `origin https://github.com/Nier4Ryu/putpocket_dataset_mining.git`
- Starting HEAD before these changes: `0c242a8197ddfe0136d32c8e3d062a25979ba8b7`
- Final commit: created after this report is written; verify with `git rev-parse HEAD`
- Push target: `origin/blackwell`

## Existing Stack Preservation

- Existing env `Putpocket_env` was not overwritten.
- Existing vLLM checkout `externals/vllm` was not overwritten.
- Existing vLLM checkout status:
  - Branch: `Putpocket-v0.19.1`
  - Commit: `b65d39ddbab966bb72110056a481d17e4726892b`
  - Remote: `https://github.com/Nier4Ryu/vllm_mod.git`
- Existing Qwen dataset mining configs and copied mined dataset artifacts were not modified.

## New GLM Stack

- Env path: `Putpocket_env_glm52`
- Activation script: `scripts/env/env_activate_glm52.sh`
- Bootstrap script: `scripts/env/bootstrap_glm52_env.sh`
- Serving config: `configs/serving/glm52_vllm023_blackwell.yaml`
- Evaluation config: `configs/model_evaluation/glm52_08b_blackwell.yaml`
- vLLM source path: `externals/vllm_glm52`
- vLLM remote: `https://github.com/vllm-project/vllm.git`
- vLLM ref/describe: `v0.23.0`
- vLLM commit: `0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665`
- vLLM import path: `externals/vllm_glm52/vllm/__init__.py`
- CUDA path: `/usr/local/cuda-12.9`
- Python: `3.13.14`
- Torch: `2.11.0+cu129`
- Torch CUDA: `12.9`
- Transformers: `5.14.1`
- DeepGEMM: `2.5.0`

## vLLM Build

- Build mode: editable source build from `externals/vllm_glm52`
- Build threads: `8`
- Build environment:
  - `PUTPOCKET_BUILD_THREADS=8`
  - `MAX_JOBS=8`
  - `CMAKE_BUILD_PARALLEL_LEVEL=8`
  - `CARGO_BUILD_JOBS=8`
  - `NVCC_THREADS=1`
  - `TORCH_CUDA_ARCH_LIST=12.0`
- Build result: pass
- Main vLLM build log: `logs/env_setup_glm52/20260722T185256Z/build-vllm-glm52.log`
- DeepGEMM build log: `logs/env_setup_glm52/20260722T185256Z/install-deepgemm-glm52.log`
- Setup manifest: `logs/env_setup_glm52/20260722T185256Z/setup_summary.json`

## GLM Load Smoke

Command:

```bash
source scripts/env/env_activate_glm52.sh >/tmp/putpocket_glm52_activate_check.log
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from vllm import LLM, SamplingParams

model = "inference-optimization/GLM-5.2-0.8B-A0.8B"
llm = LLM(
    model=model,
    trust_remote_code=True,
    tensor_parallel_size=1,
)
params = SamplingParams(
    temperature=0.0,
    max_tokens=64,
)
outs = llm.generate(["Hello. Reply with one short sentence."], params)
print(outs[0].outputs[0].text)
PY
```

- GPU used: `CUDA_VISIBLE_DEVICES=0`
- Result: failed
- Log: `data/model_evaluation/logs/glm52_vllm023_load_smoke_20260722T194254Z.log`
- Root failure:

```text
ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(head_size=192, dtype=torch.bfloat16, kv_cache_dtype=auto, block_size=None, use_mla=True, has_sink=False, use_sparse=True, use_mm_prefix=False, use_per_head_quant_scales=False, attn_type=AttentionType.DECODER, use_non_causal=False, use_batch_invariant=False, use_kv_connector=False).
```

Backend rejection summary from the log:

```text
FLASH_ATTN_MLA: head_size not supported, sparse not supported, compute capability not supported
FLASHMLA: head_size not supported, sparse not supported, compute capability not supported, vllm._flashmla_C is not available
FLASHINFER_MLA: head_size not supported, sparse not supported, compute capability not supported
TRITON_MLA: sparse not supported
FLASHMLA_SPARSE: head_size not supported, compute capability not supported
```

## Existing Code Compatibility Audit

- Existing local vLLM Python engine is in `src/putpocket_dataset_mining/serving.py`.
- The engine uses public vLLM imports: `LLM`, `SamplingParams`, and `llm.generate(...)`.
- Existing GLM evaluation CLI is `src/putpocket_dataset_mining/model_evaluation/glm_eval.py`.
- The evaluation code did not hardcode `externals/vllm`; the active environment determines which vLLM import is used.
- Compatibility changes added now:
  - Added a `--serving-stack` option to the GLM eval CLI.
  - Added `glm52_vllm023_blackwell` as a separate serving stack.
  - Added GLM-specific 8-thread build/runtime env overrides.
  - Preserved the old Putpocket-v0.19.1 defaults for existing Qwen/mining paths.
- Current evaluation code could be invoked under `Putpocket_env_glm52`, but actual evaluation is blocked earlier by vLLM model-load backend selection.

## Dataset Used

- Dataset version: `mbpp_stateful_working_v0`
- Accepted file: `data/dataset_mining/datasets/mbpp_stateful_working_v0/accepted.jsonl`
- Accepted count: `20`
- First accepted sample inspected:
  - `sample_id`: `train_730`
  - `task_id`: `730`
  - `artifact_path`: `data/dataset_mining/runs/full_server_validation_20260707T175646Z/samples/train_730/attempt_5a8d1db9b812`
- Dataset artifacts were not modified.

## GLM Evaluation Smoke

- Result: not run
- Reason: GLM vLLM load smoke failed with `No valid attention backend found`.

Intended command once the backend issue is fixed:

```bash
CUDA_VISIBLE_DEVICES=0 python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --serving-stack glm52_vllm023_blackwell \
  --profile smoke \
  --workers 1 \
  --gpu-slots 0
```

## GLM Full Evaluation

- Result: not run
- Reason: GLM vLLM load smoke failed before evaluation.

Intended command once the backend issue is fixed:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --serving-stack glm52_vllm023_blackwell \
  --profile full \
  --workers 3 \
  --gpu-slots 0,1,2
```

## Logs

- Env setup log dir: `logs/env_setup_glm52/20260722T185256Z/`
- vLLM build log: `logs/env_setup_glm52/20260722T185256Z/build-vllm-glm52.log`
- DeepGEMM build log: `logs/env_setup_glm52/20260722T185256Z/install-deepgemm-glm52.log`
- Doctor smoke log: `logs/env_setup_glm52/20260722T185256Z/doctor-smoke.log`
- Setup summary: `logs/env_setup_glm52/20260722T185256Z/setup_summary.json`
- GLM load smoke log: `data/model_evaluation/logs/glm52_vllm023_load_smoke_20260722T194254Z.log`
- Eval run path: none created for this GLM run because model load smoke failed first.

## Known Blockers

Blocker: local vLLM `0.23.0+cu129` built from `v0.23.0` cannot initialize `inference-optimization/GLM-5.2-0.8B-A0.8B` on this SM120 host because no valid sparse MLA attention backend is selected for `head_size=192`, `use_mla=True`, `use_sparse=True`.

Exact failing command:

```bash
source scripts/env/env_activate_glm52.sh >/tmp/putpocket_glm52_activate_check.log
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from vllm import LLM, SamplingParams
model = "inference-optimization/GLM-5.2-0.8B-A0.8B"
llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=1)
params = SamplingParams(temperature=0.0, max_tokens=64)
outs = llm.generate(["Hello. Reply with one short sentence."], params)
print(outs[0].outputs[0].text)
PY
```

Failing log path: `data/model_evaluation/logs/glm52_vllm023_load_smoke_20260722T194254Z.log`

Smallest next action: move `externals/vllm_glm52` to a GLM-5.2/SM120-compatible vLLM 0.23.x commit or GLM vendor branch/image that provides a valid sparse MLA backend for `head_size=192` on Blackwell SM120, then rebuild with `./scripts/env/bootstrap_glm52_env.sh --force-vllm-build` and rerun the load smoke.

## Next Recommended Action

Use the newly added separate GLM env/scripts/config as the preserved integration point, but replace or patch only the separate `externals/vllm_glm52` stack so that sparse MLA backend selection supports this GLM-5.2 config on SM120. After model load succeeds, run the smoke and full evaluation commands above against the fixed dataset.
