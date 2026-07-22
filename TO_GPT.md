# GLM Evaluation Report To GPT

status: blocked

## Executive Summary
- GLM-5.2-0.8B did not produce any evaluable coding trajectories in this environment.
- Full evaluation was attempted over all 20 accepted samples in `mbpp_stateful_working_v0`.
- Succeeded: 0 / 20.
- Main failure mode: local vLLM Python engine could not initialize `inference-optimization/GLM-5.2-0.8B-A0.8B` on the available RTX 3090 GPUs. After installing DeepGEMM, vLLM still reported no valid sparse MLA CUDA attention backend because compute capability 8.6 is unsupported for this model/backend combination.

## Commands Run
- Implementation validation:
  - `source scripts/env/env_activate.sh && python -m compileall src tests`
  - `source scripts/env/env_activate.sh && python -m unittest discover -s tests -v`
- Dataset audit:
  - `source scripts/env/env_activate.sh && python - <<'PY' ... load_accepted_samples('mbpp_stateful_working_v0') ... PY`
- Initial smoke evaluation:
  - `source scripts/env/env_activate.sh && python -m putpocket_dataset_mining.model_evaluation.glm_eval --dataset-version mbpp_stateful_working_v0 --model-id inference-optimization/GLM-5.2-0.8B-A0.8B --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 --profile smoke --gpu-slots 4 --workers 1 --run-id eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_20260721T000000Z 2>&1 | tee data/model_evaluation/logs/eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_20260721T000000Z.log`
- Dependency remediation attempt:
  - `source scripts/env/env_activate.sh && PUTPOCKET_BUILD_THREADS=16 MAX_JOBS=16 CMAKE_BUILD_PARALLEL_LEVEL=16 CARGO_BUILD_JOBS=16 NVCC_THREADS=1 CUDA_VISIBLE_DEVICES=0 bash externals/vllm/tools/install_deepgemm.sh 2>&1 | tee data/model_evaluation/logs/install_deepgemm_20260721T000000Z.log`
- Smoke evaluation after DeepGEMM install:
  - `source scripts/env/env_activate.sh && python -m putpocket_dataset_mining.model_evaluation.glm_eval --dataset-version mbpp_stateful_working_v0 --model-id inference-optimization/GLM-5.2-0.8B-A0.8B --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 --profile smoke --gpu-slots 4 --workers 1 --run-id eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_after_deepgemm_20260721T000000Z 2>&1 | tee data/model_evaluation/logs/eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_after_deepgemm_20260721T000000Z.log`
- Full evaluation attempt:
  - `source scripts/env/env_activate.sh && python -m putpocket_dataset_mining.model_evaluation.glm_eval --dataset-version mbpp_stateful_working_v0 --model-id inference-optimization/GLM-5.2-0.8B-A0.8B --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 --profile full --gpu-slots 0,1,2 --workers 3 --run-id eval_glm52_08b_on_mbpp_stateful_working_v0_full_20260721T000000Z 2>&1 | tee data/model_evaluation/logs/eval_glm52_08b_on_mbpp_stateful_working_v0_full_20260721T000000Z.log`
- Summary regeneration after status-classification patch:
  - `source scripts/env/env_activate.sh && python - <<'PY' ... write_summary(...) ... PY`

## Dataset Used
- Dataset version: `mbpp_stateful_working_v0`
- Accepted count: 20
- Selected subset: all 20 accepted rows
- Accepted path: `data/dataset_mining/datasets/mbpp_stateful_working_v0/accepted.jsonl`
- Artifact completeness: 20 / 20 accepted samples had all checked source artifacts present.
- Accepted sample IDs: `train_730`, `test_347`, `train_848`, `train_822`, `test_353`, `test_173`, `test_352`, `test_267`, `test_68`, `train_970`, `train_625`, `train_852`, `train_711`, `test_162`, `test_197`, `test_58`, `train_913`, `test_366`, `train_693`, `train_800`

## Target Model
- Model id: `inference-optimization/GLM-5.2-0.8B-A0.8B`
- Backend: local vLLM Python engine only
- GPU config:
  - Smoke: GPU 4
  - Full: GPUs 0,1,2 with 3 workers, `tp=1`, `pp=1`
  - Hardware observed: NVIDIA GeForce RTX 3090, compute capability 8.6
- Decoding config: deterministic greedy, `temperature=0.0`, `top_p=1.0`, `n=1`, `evaluation_seed=20260721`
- Chat-template rendering policy: semantic messages were rendered with the GLM tokenizer via `AutoTokenizer.apply_chat_template`; Qwen-rendered prompts were not used.

## Implementation Summary
- Added `src/putpocket_dataset_mining/model_evaluation/`.
- Added `dataset_loader.py` for accepted-row loading and source-artifact completeness checks.
- Added `glm_eval.py` CLI: `python -m putpocket_dataset_mining.model_evaluation.glm_eval`.
- Added `GLM52_08B_MODEL_ID`, model-evaluation run roots, and model-evaluation directory helper in `constants.py`.
- Extended `GenerationRequest`, `LocalVLLMEngine`, and `HeadlessClineRuntime` to record/use `evaluation_seed` and completion token metadata.
- Added tests in `tests/test_glm_eval.py`.

## Experiment Progress Log
- Static validation passed: compileall succeeded, 13 unit tests passed.
- Dataset audit passed: 20 accepted rows, 0 samples with missing checked artifacts.
- Initial smoke run failed before generation:
  - Failure: `Sparse Attention Indexer CUDA op requires DeepGEMM to be installed.`
  - Log: `data/model_evaluation/logs/eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_20260721T000000Z.log`
- DeepGEMM install completed successfully from the vLLM-pinned commit.
  - Log: `data/model_evaluation/logs/install_deepgemm_20260721T000000Z.log`
- Smoke after DeepGEMM still failed before generation:
  - Failure: no valid CUDA sparse MLA attention backend on compute capability 8.6.
  - Log: `data/model_evaluation/logs/eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_after_deepgemm_20260721T000000Z.log`
- Full run was attempted across all 20 accepted samples with 3 workers.
  - Full run status: blocked by backend initialization.
  - Run root: `data/model_evaluation/runs/eval_glm52_08b_on_mbpp_stateful_working_v0_full_20260721T000000Z`
  - Log: `data/model_evaluation/logs/eval_glm52_08b_on_mbpp_stateful_working_v0_full_20260721T000000Z.log`

## Results Summary
- Final status counts: `failed_infra=20`
- History 1 status counts: `failed_infra=20`
- History 1 pass/fail counts: `passed=0`, `failed=20`
- History 2 status counts: `skipped=20`
- History 2 pass/fail counts: `passed=0`, `failed=0`, `skipped=20`
- Judge counts: `not_run=20`, `pass=0`, `fail=0`, `uncertain=0`
- Failure-stage histogram: `infra=20`
- Failure-class histogram: `infra.vllm_generation_failed=20`, `history2.skipped_after_infra_failure=20`
- Representative accepted/successful sample: none
- Representative failed sample:
  - sample: `train_730`
  - artifact: `data/model_evaluation/runs/eval_glm52_08b_on_mbpp_stateful_working_v0_full_20260721T000000Z/per_sample/train_730/attempt_f023e3be40bd`

## Interpretation
- GLM-5.2-0.8B is not usable on this server with the current local vLLM Python backend and the allowed GPUs.
- This result does not measure GLM coding capability, prompt quality, tool parsing, verifier behavior, or Codex judge quality, because no sample reached the first model response.
- The blocker is infrastructure/hardware-backend compatibility: vLLM resolves the model as `GlmMoeDsaForCausalLM`, which requires sparse MLA attention support. On RTX 3090 compute capability 8.6, vLLM reports no valid attention backend. The vLLM platform code also reports DeepGEMM support only for Hopper/Blackwell-class GPUs.
- The verifier and judge did not run because History 1 generation never started.

## Visualization-Ready Data
- Results JSONL: `data/model_evaluation/runs/eval_glm52_08b_on_mbpp_stateful_working_v0_full_20260721T000000Z/results.jsonl`
- Summary JSON: `data/model_evaluation/runs/eval_glm52_08b_on_mbpp_stateful_working_v0_full_20260721T000000Z/summary.json`
- Useful plotting fields:
  - `sample_id`
  - `task_id`
  - `final_status`
  - `history1_status`
  - `history1_failure_class`
  - `history2_status`
  - `judge_decision`
  - `failure_stage`
  - `latency.sample_wall_sec`
  - `artifact_path`
- Recommended plots later:
  - final-status counts
  - failure-stage histogram
  - per-sample wall-clock time
  - after rerunning on supported hardware, History 1/History 2 verification pass rates

## Remaining Issues / Next Tasks
- Exact blocker:
  - Full command completed with all samples `failed_infra`.
  - Failing backend operation: vLLM local engine initialization for `inference-optimization/GLM-5.2-0.8B-A0.8B`.
  - Root cause in log: `No valid attention backend found for cuda ... use_mla=True ... use_sparse=True ... compute capability not supported`.
- Smallest next action:
  - Rerun the same full command on Hopper or Blackwell GPUs with vLLM sparse MLA attention support, or provide a vLLM build/backend that supports GLM sparse MLA on the target hardware.
- Do not switch to Transformers fallback or remote endpoint if preserving this evaluation policy.
