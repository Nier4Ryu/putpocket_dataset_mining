# Blackwell Env + GLM Evaluation Report

## Executive Summary
- Environment setup for the Blackwell server is implemented on branch `blackwell`.
- The repo-local `Putpocket_env` is built and usable with Python `3.13.14`, CUDA `/usr/local/cuda-12.9`, PyTorch `2.10.0+cu129`, Ray `2.55.1`, editable vLLM, editable LMCache, and DeepGEMM `2.3.0`.
- Docker is installed and usable through a refreshed group shell via `sg docker`; the repo Docker image exists.
- GLM smoke was attempted on GPU `0` before and after DeepGEMM install. After DeepGEMM, model loading still fails in vLLM sparse MLA attention backend selection for this SM 12.0 RTX Blackwell device and GLM config.
- Full GLM evaluation was not run because the one-sample smoke did not pass.

## Branch / Git
- Branch name: `blackwell`
- Upstream: `origin/blackwell`
- Base commit before Blackwell changes: `09906fc63f71bfc950a33b506b0c9441e25ae6df`
- Source/config/report commit pushed earlier: `e043f774fdf6385baacd4c1badfdfb93c6067a2b`
- Source/report continuation commits:
  - `297cf1c3f08c4f26c41ca1deb794793095d0b5d1`
  - `74f05513bfd8b8e70d523760763ee4688bcbc02b`
  - `b28c4cdd96dded67a17ec05c9f58d2b3e90a5601`
  - `adee93cb71309352847749ed3c37f8e9d8318b28`
- Pushed status: `git push -u origin blackwell` succeeded through `adee93cb71309352847749ed3c37f8e9d8318b28`.
- This final status edit is report-only; the final assistant handoff reports the final branch HEAD after it is committed and pushed.

## Hardware Detected
- GPUs:
  - `0`: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability `12.0`, `97887 MiB`, driver `580.159.03`
  - `1`: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability `12.0`, `97887 MiB`, driver `580.159.03`
  - `2`: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability `12.0`, `97887 MiB`, driver `580.159.03`
- CUDA path: `/usr/local/cuda-12.9`
- CUDA version: `12.9`, `nvcc` release `12.9`, `V12.9.41`
- CPU cores: `64`
- CPU RAM: `62Gi`
- Python 3.13 was provided by uv-managed Python under the repo-local environment.

## Env Setup Changes
- Modified environment/bootstrap files:
  - `scripts/env/bootstrap_env.sh`
  - `scripts/env/env_activate.sh`
  - `scripts/env/env_activate_ref.sh`
  - `scripts/env/README.md`
- Modified runtime/config/source defaults:
  - `configs/dataset_mining/mbpp_stateful_multi.yaml`
  - `configs/model_evaluation/glm52_08b_blackwell.yaml`
  - `src/putpocket_dataset_mining/constants.py`
  - `src/putpocket_dataset_mining/multi.py`
  - `src/putpocket_dataset_mining/model_evaluation/glm_eval.py`
  - `src/putpocket_dataset_mining/externals.py`
  - `src/putpocket_dataset_mining/docker_workspace.py`
  - `src/putpocket_dataset_mining/cli.py`
- Python version: `Python 3.13.14`
- CUDA path: `/usr/local/cuda-12.9`
- Torch: `2.10.0+cu129`
- `torch.version.cuda`: `12.9`
- `torch.cuda.is_available()`: `True`
- Ray: `2.55.1`
- vLLM: `0.1.dev15375+gb65d39ddb`
- LMCache: `0.1.dev1451`, imported with `lmcache.c_ops`
- DeepGEMM: `2.3.0`, installed from the vLLM-pinned commit `477618cd51baffca09c4b0b87e97c03fe827ef03`
- Build cap defaults confirmed:
  - `PUTPOCKET_BUILD_THREADS=16`
  - `MAX_JOBS=16`
  - `CMAKE_BUILD_PARALLEL_LEVEL=16`
  - `CARGO_BUILD_JOBS=16`
  - `NVCC_THREADS=1`
- PyTorch/CUDA decision: installed official PyTorch CUDA 12.9 wheel `torch==2.10.0+cu129` from `https://download.pytorch.org/whl/cu129`; CPU torch was not accepted.

## vLLM Build Retry Summary
- Retry log root: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z`
- Attempted thread counts:
  - `16`: fail, OOM-like
    - Log: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/vllm_build_threads_16.log`
  - `12`: fail, OOM-like
    - Log: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/vllm_build_threads_12.log`
  - `8`: pass
    - Log: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/vllm_build_threads_8.log`
- Final successful thread count: `8`
- Summary file: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/vllm_build_retry_summary.tsv`
- No remaining vLLM build blocker after the 8-thread retry passed.

## Bootstrap Usage
- First-time command:

```bash
./scripts/env/bootstrap_env.sh
```

- The original real run failed at Docker before Docker was installed:
  - Failed stage: `ensure-docker-image`
  - Failing command: `ensure_docker_image`
  - Log path: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/ensure-docker-image.log`
- Successful idempotent validation before Docker:

```bash
./scripts/env/bootstrap_env.sh --skip-docker --skip-vllm-build --skip-lmcache-build
```

- Successful skip-Docker log root: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T122119Z`
- DeepGEMM install is now integrated into bootstrap as `install-deepgemm`.
- Successful DeepGEMM-stage validation command:

```bash
./scripts/env/bootstrap_env.sh --skip-externals --skip-vllm-build --skip-lmcache-build --skip-docker
```

- Successful DeepGEMM-stage validation log root: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T180715Z`
- Activation command:

```bash
source scripts/env/env_activate.sh
```

## Validation Commands Run
- `bash -n scripts/env/bootstrap_env.sh scripts/env/env_activate.sh scripts/env/env_activate_ref.sh`: pass
- `./scripts/env/bootstrap_env.sh`: env/vLLM/LMCache build passed; failed only at Docker before Docker was installed
- `./scripts/env/bootstrap_env.sh --skip-docker --skip-vllm-build --skip-lmcache-build`: pass
- `./scripts/env/bootstrap_env.sh --doctor-only --skip-docker`: pass
  - Log root: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T180645Z`
- `./scripts/env/bootstrap_env.sh --skip-externals --skip-vllm-build --skip-lmcache-build --skip-docker`: pass
  - Log root: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T180715Z`
- torch/Ray/datasets/transformers import check: pass
- repo package import check: pass
- vLLM/LMCache import check: pass
- DeepGEMM import check: pass
- `putpocket-dataset-mining doctor --json`: pass
- `python -m compileall src tests`: pass
- `python -m unittest discover -s tests -v`: pass, `13` tests
- Docker image build through refreshed group shell:

```bash
sg docker -c 'cd /home/dyryu/putpocket_dataset_mining && bash -lc "source scripts/env/env_activate.sh >/tmp/putpocket_activate_check.log && putpocket-dataset-mining docker ensure-image"'
```

  - Result: pass
  - Image: `putpocket-default-python:ubuntu22.04-py313-v1`, image id `1ea08521b3c6`, disk usage `1.19GB`

## Dataset Found
- Dataset version: `mbpp_stateful_working_v0`
- Accepted count: `20`
- Accepted path: `/home/dyryu/putpocket_dataset_mining/data/dataset_mining/datasets/mbpp_stateful_working_v0/accepted.jsonl`
- Artifact paths resolve: yes, `0` missing paths across accepted rows.
- Original copied mined dataset was not modified.

## GLM Evaluation Code Status
- GLM evaluation code exists and was updated for Blackwell defaults.
- Main CLI module: `src/putpocket_dataset_mining/model_evaluation/glm_eval.py`
- Config: `configs/model_evaluation/glm52_08b_blackwell.yaml`
- CLI command form:

```bash
CUDA_VISIBLE_DEVICES=0 python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --profile smoke \
  --workers 1 \
  --gpu-slots 0
```

## GLM Smoke Result
- Pre-DeepGEMM smoke status: blocked at vLLM model initialization because `deep_gemm` was missing.
  - Log: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/logs/glm_smoke_blackwell_20260722T175830Z.log`
  - Run path: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/runs/eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_blackwell_20260722T175830Z`
- DeepGEMM install command:

```bash
source scripts/env/env_activate.sh
PUTPOCKET_BUILD_THREADS=16 MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16 CARGO_BUILD_JOBS=16 NVCC_THREADS=1 \
  CUDA_VISIBLE_DEVICES=0 bash externals/vllm/tools/install_deepgemm.sh --cuda-version 12.9
```

  - Result: pass
  - Log: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/logs/install_deepgemm_blackwell_20260722T180144Z.log`
- Post-DeepGEMM smoke command:

```bash
sg docker -c 'cd /home/dyryu/putpocket_dataset_mining && bash -lc '"'"'source scripts/env/env_activate.sh >/tmp/putpocket_activate_check.log
CUDA_VISIBLE_DEVICES=0 python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --profile smoke \
  --workers 1 \
  --gpu-slots 0 \
  --run-id eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_after_deepgemm_blackwell_20260722T180327Z'"'"''
```

- Post-DeepGEMM smoke status: blocked
- Output path: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/runs/eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_after_deepgemm_blackwell_20260722T180327Z`
- Log path: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/logs/glm_smoke_blackwell_after_deepgemm_20260722T180327Z.log`
- Selected samples: `1`
- Counts:
  - `failed_infra=1`
  - `history1_status.failed_infra=1`
  - `history2_status.skipped=1`
  - `judge_decision.not_run=1`
- Root failure:

```text
ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(head_size=192, dtype=torch.bfloat16, kv_cache_dtype=auto, block_size=None, use_mla=True, has_sink=False, use_sparse=True, use_mm_prefix=False, use_per_head_quant_scales=False, attn_type=AttentionType.DECODER).
```

- vLLM backend reasons included `head_size not supported`, `sparse not supported`, `compute capability not supported`, and `FLASHMLA_SPARSE: [head_size not supported, compute capability not supported]`.
- Local vLLM inspection:
  - Device capability: `DeviceCapability(major=12, minor=0)`
  - `has_deep_gemm`: `True`
  - `current_platform.support_deep_gemm()`: `False`
  - `is_deep_gemm_supported()`: `False`
  - `FlashInferMLASparseBackend.get_supported_head_sizes()`: `[576]`
  - `FlashMLASparseBackend.get_supported_head_sizes()`: `[576]`
  - The GLM model path passes `head_size = kv_lora_rank + qk_rope_head_dim = 192`.

## GLM Full Evaluation Result
- Status: not run because smoke remained blocked after DeepGEMM installation.
- Planned command:

```bash
sg docker -c 'cd /home/dyryu/putpocket_dataset_mining && bash -lc '"'"'source scripts/env/env_activate.sh >/tmp/putpocket_activate_check.log
CUDA_VISIBLE_DEVICES=0,1,2 python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --profile full \
  --workers 3 \
  --gpu-slots 0,1,2'"'"''
```

- Number of samples planned: `20`
- accepted/rejected/failed_infra/uncertain: unavailable because full evaluation did not start.
- Failure stage histogram: unavailable for full evaluation because smoke failed first.

## Logs
- Preflight log dir: `/home/dyryu/putpocket_dataset_mining/logs/blackwell_preflight`
- Real bootstrap log dir: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z`
- Successful skip-Docker bootstrap log dir: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T122119Z`
- Doctor-only validation log dir: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T180645Z`
- DeepGEMM-stage validation log dir: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T180715Z`
- vLLM retry summary: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/vllm_build_retry_summary.tsv`
- DeepGEMM install log: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/logs/install_deepgemm_blackwell_20260722T180144Z.log`
- GLM post-DeepGEMM smoke log: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/logs/glm_smoke_blackwell_after_deepgemm_20260722T180327Z.log`
- GLM post-DeepGEMM smoke run path: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/runs/eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_after_deepgemm_blackwell_20260722T180327Z`

## Known Blockers
- GLM-5.2-0.8B local vLLM engine cannot initialize on this current vLLM checkout and SM 12.0 RTX Blackwell device.
  - Failing operation: vLLM local engine initialization for `inference-optimization/GLM-5.2-0.8B-A0.8B`
  - Failing command: the post-DeepGEMM smoke command in `## GLM Smoke Result`
  - Log path: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/logs/glm_smoke_blackwell_after_deepgemm_20260722T180327Z.log`
  - Root error: no valid CUDA attention backend for sparse MLA with `head_size=192`, `use_sparse=True`, and compute capability `12.0`
  - Smallest next action: update the local vLLM branch to a version that explicitly supports GLM sparse MLA for RTX Blackwell SM 12.0 and this `head_size=192` GLM layout, or obtain an upstream-supported backend/config for this model. Do not bypass the backend validation gates without a kernel-level compatibility check.
- Docker is installed but the current non-login shell does not have the refreshed `docker` supplemental group.
  - Direct `docker info` in this shell still fails with permission denied.
  - `sg docker -c 'docker info'` works and was used for Docker image setup and GLM smoke.
  - Smallest next action for direct Docker access: start a new login session for `dyryu`, then validate `id` and `docker info`.
- Git push blocker is cleared for the latest source/report commit:
  - Command: `git push -u origin blackwell`
  - Result: pass through `adee93cb71309352847749ed3c37f8e9d8318b28`

## Next Recommended Action
Update or replace the local vLLM branch with explicit support for GLM sparse MLA on RTX Blackwell SM 12.0, then rerun the post-DeepGEMM smoke command on GPU `0`. If that smoke passes, run the planned full evaluation on GPUs `0,1,2`.
