# GLM Docker Agent Evaluation Report

## Executive Summary

- Official image pulled: `vllm/vllm-openai:glm52-cu129`, digest `sha256:894456ff199741e4e6a06292e360e6582b79a8f450f34d48b4bd8a4f35124b7d`.
- The standard `docker run --gpus "device=0"` path failed because this Docker install has no NVIDIA Container Toolkit/CDI runtime configured.
- A manual GPU exposure workaround did make CUDA visible inside the official image (`torch 2.11.0+cu129`, `cuda available True`, one RTX PRO 6000 Blackwell device).
- Native official `vllm serve` still failed before `/v1/models` with the same GLM sparse MLA backend error seen in the repo-local path.
- The documented `--model-impl transformers --enforce-eager` vLLM-server variant loaded weights, but failed during vLLM profiling with a tensor shape error before readiness.
- No OpenAI server became ready, so no headless Cline agentic smoke or full dataset evaluation could be run. There is no evidence yet about GLM output quality, parseable tool calls, or verifier pass rate.

## Why This Test Was Run

The repo-local vLLM Python path previously failed native sparse MLA initialization for `inference-optimization/GLM-5.2-0.8B-A0.8B` on SM120. This test used the official GLM vLLM Docker image to check whether the packaged server path could run the model and, if so, feed the existing headless Cline tool loop over the already-mined MBPP-stateful dataset.

## Environment

- Branch: `blackwell`
- HEAD at test start: `2225e6307d7522978f4113d32ca86298d165f0a8`
- GPU used: GPU `0`
- GPU detected: `NVIDIA RTX PRO 6000 Blackwell Server Edition`, `97887 MiB`
- Driver: `580.159.03`
- Docker: `29.6.2`
- Docker command access in this Codex shell: direct `docker info` fails due stale group membership; `sg docker -c 'docker info'` works.
- Image: `vllm/vllm-openai:glm52-cu129`
- Image entrypoint: `["vllm","serve"]`
- Model id: `inference-optimization/GLM-5.2-0.8B-A0.8B`
- Served model name attempted: `glm52-08b`
- Server base URLs attempted: `http://127.0.0.1:18080/v1`, `http://127.0.0.1:18081/v1`, `http://127.0.0.1:18082/v1`

## Server Smoke

Standard Docker GPU command failed:

```bash
sg docker -c 'docker run -d --rm --gpus "device=0" --ipc=host -p 18080:8000 -v /data/shared/hf_cache:/data/shared/hf_cache -e HF_HOME=/data/shared/hf_cache -e HUGGING_FACE_HUB_CACHE=/data/shared/hf_cache/hub --name putpocket-glm52-08b-serve-smoke vllm/vllm-openai:glm52-cu129 --model inference-optimization/GLM-5.2-0.8B-A0.8B --served-model-name glm52-08b --max-model-len 8192 --max-num-seqs 1 --trust-remote-code'
```

Result:

- Status: failed before container start
- Error: `failed to discover GPU vendor from CDI: no known GPU vendor found`
- Log: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_20260723_055251/docker_run_stderr.txt`

Manual GPU/library probe worked:

```bash
sg docker -c 'docker run --rm --entrypoint python3 --device=/dev/nvidia0 --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools -v /lib/x86_64-linux-gnu:/host_lib/x86_64-linux-gnu:ro -e CUDA_VISIBLE_DEVICES=0 -e LD_LIBRARY_PATH=/host_lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib vllm/vllm-openai:glm52-cu129 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"'
```

Result:

- `torch 2.11.0+cu129`
- `cuda available True`
- `device count 1`
- `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Log: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_20260723_055251/manual_hostlibs_torch_probe.log`

Native official Docker serve with manual GPU exposure failed:

```bash
sg docker -c 'docker run -d --device=/dev/nvidia0 --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools --ipc=host -p 18080:8000 -v /data/shared/hf_cache:/data/shared/hf_cache -v /lib/x86_64-linux-gnu:/host_lib/x86_64-linux-gnu:ro -e LD_LIBRARY_PATH=/host_lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib -e CUDA_VISIBLE_DEVICES=0 -e HF_HOME=/data/shared/hf_cache -e HUGGING_FACE_HUB_CACHE=/data/shared/hf_cache/hub --name putpocket-glm52-08b-serve-smoke vllm/vllm-openai:glm52-cu129 --model inference-optimization/GLM-5.2-0.8B-A0.8B --served-model-name glm52-08b --max-model-len 8192 --max-num-seqs 1 --trust-remote-code'
```

Result:

- `/v1/models`: not ready
- Basic chat request: not run
- Exit code: `1`
- Key log line: `ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(head_size=192, dtype=torch.bfloat16, kv_cache_dtype=auto, use_mla=True, use_sparse=True, ...)`
- Log: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_manual_logs_20260723_055937/vllm_docker_server.log`

Positional model-card style command also failed the same way:

```bash
sg docker -c 'docker run -d --device=/dev/nvidia0 --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools --ipc=host -p 18081:8000 -v /data/shared/hf_cache:/data/shared/hf_cache -v /lib/x86_64-linux-gnu:/host_lib/x86_64-linux-gnu:ro -e LD_LIBRARY_PATH=/host_lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib -e CUDA_VISIBLE_DEVICES=0 -e HF_HOME=/data/shared/hf_cache -e HUGGING_FACE_HUB_CACHE=/data/shared/hf_cache/hub --name putpocket-glm52-08b-serve-positional vllm/vllm-openai:glm52-cu129 inference-optimization/GLM-5.2-0.8B-A0.8B --served-model-name glm52-08b --max-model-len 8192 --max-num-seqs 1 --trust-remote-code'
```

Result:

- `/v1/models`: not ready
- Basic chat request: not run
- Exit code: `1`
- Same sparse MLA backend selector failure
- Log: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_manual_logs_20260723_055937/positional_vllm_docker_server.log`

Documented vLLM server `model_impl=transformers` variant also failed:

```bash
sg docker -c 'docker run -d --device=/dev/nvidia0 --device=/dev/nvidiactl --device=/dev/nvidia-uvm --device=/dev/nvidia-uvm-tools --ipc=host -p 18082:8000 -v /data/shared/hf_cache:/data/shared/hf_cache -v /lib/x86_64-linux-gnu:/host_lib/x86_64-linux-gnu:ro -e LD_LIBRARY_PATH=/host_lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib -e CUDA_VISIBLE_DEVICES=0 -e HF_HOME=/data/shared/hf_cache -e HUGGING_FACE_HUB_CACHE=/data/shared/hf_cache/hub --name putpocket-glm52-08b-serve-transformers vllm/vllm-openai:glm52-cu129 inference-optimization/GLM-5.2-0.8B-A0.8B --served-model-name glm52-08b --max-model-len 8192 --max-num-seqs 1 --trust-remote-code --model-impl transformers --enforce-eager'
```

Result:

- `/v1/models`: not ready
- Basic chat request: not run
- Exit code: `1`
- It resolved `TransformersMoEForCausalLM`, used the Transformers modeling backend, selected `FLASH_ATTN`, downloaded/loaded the 3.17 GiB checkpoint, then failed in vLLM profiling.
- Key error: `RuntimeError: shape '[-1, 16, 192]' is invalid for input of size 33554432`
- Log: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_manual_logs_20260723_055937/transformers_vllm_docker_server.log`

## Dataset Used

- Dataset version: `mbpp_stateful_working_v0`
- Accepted file: `data/dataset_mining/datasets/mbpp_stateful_working_v0/accepted.jsonl`
- Accepted rows: `20`
- First sample: `train_730`, task `730`
- Example artifact path: `data/dataset_mining/runs/full_server_validation_20260707T175646Z/samples/train_730/attempt_5a8d1db9b812`
- The copied dataset was inspected read-only and not modified.

The accepted rows contain `sample_id`, `task_id`, `artifact_path`, `query1`, `query2`, and `policy_delta`. Existing source artifacts include `source_task.json`, prepared prompts, trajectories, workspace snapshots, verification outputs, and judge outputs. The older `prepared/verifier_specs/...` paths are not present in the checked examples, but the repo verifier regenerates MBPP hidden pytest files from `source_task.json`, so that was not the blocking issue.

## Evaluation Implementation

- Existing headless Cline runtime: `src/putpocket_dataset_mining/runtime.py`
- Existing GLM sample evaluator: `src/putpocket_dataset_mining/model_evaluation/glm_eval.py`
- Existing runtime accepts a `GenerationEngine`, so an OpenAI-compatible HTTP engine can be added without replacing the Qwen/local vLLM path.
- No OpenAI-compatible serving client was added in this run because no GLM Docker server reached `/v1/models`; adding the client would not enable evaluation until serving works.
- No source files or dataset artifacts were modified.

## Smoke Evaluation Result

Agentic smoke evaluation was not run.

Reason: every official-image vLLM server attempt exited before readiness, so there was no OpenAI-compatible endpoint for the headless Cline loop to call. This is infrastructure/model-serving failure, not evidence about Cline tool-call quality.

Intended smoke command after a working server and HTTP engine exist:

```bash
python -m putpocket_dataset_mining.model_evaluation.glm_docker_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --dataset-root data/dataset_mining/datasets/mbpp_stateful_working_v0 \
  --eval-name eval_glm52_08b_docker_on_mbpp_stateful_working_v0 \
  --server-base-url http://127.0.0.1:18080/v1 \
  --served-model-name glm52-08b \
  --profile smoke \
  --limit 1 \
  --temperature 0.0 \
  --output-root data/model_evaluation/runs
```

## Full/Subsample Evaluation Result

Full/subsample evaluation was not run for the same reason: no GLM server became ready.

Status counts are unavailable because no model response was generated and no per-sample result row was produced.

## Qualitative Output Assessment

Cannot assess yet.

- Meaningful text: unknown
- Parseable Cline tool calls: unknown
- History 1 completion: unknown
- History 1 verifier pass: unknown
- History 2 reach/pass: unknown
- Usability for this workflow: blocked before inference

The only qualitative conclusion supported by the run is that the official vLLM Docker image, on this RTX PRO 6000 Blackwell SM120 host, does not currently serve this tiny GLM checkpoint through the tested native or Transformers-backed vLLM server paths.

## Representative Examples

No model output examples exist because the server never became ready and no completion request was accepted.

Representative failure artifacts:

- Native sparse MLA failure: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_manual_logs_20260723_055937/vllm_docker_server.log`
- Positional native sparse MLA failure: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_manual_logs_20260723_055937/positional_vllm_docker_server.log`
- Transformers backend shape failure: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_manual_logs_20260723_055937/transformers_vllm_docker_server.log`
- Docker GPU runtime/CDI failure: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_20260723_055251/docker_run_stderr.txt`

## Known Blockers

Primary blocker:

- Failing command: native official Docker `vllm/vllm-openai:glm52-cu129` serve command with manual GPU exposure, shown in the Server Smoke section.
- Log path: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_manual_logs_20260723_055937/vllm_docker_server.log`
- Error: `No valid attention backend found for cuda with AttentionSelectorConfig(head_size=192, dtype=torch.bfloat16, kv_cache_dtype=auto, use_mla=True, use_sparse=True, ...)`
- Smallest next action: use a vLLM/GLM image or commit whose sparse MLA backend supports this GLM DSA tiny-model shape on SM120, specifically `head_size=192` with `use_sparse=True`, or run an SGLang sanity check to separate model capability from vLLM support.

Secondary Docker host blocker:

- Failing command: standard `docker run --gpus "device=0" ...`
- Log path: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_20260723_055251/docker_run_stderr.txt`
- Error: `failed to discover GPU vendor from CDI: no known GPU vendor found`
- Smallest next action: install/configure NVIDIA Container Toolkit or generate/register NVIDIA CDI specs so the standard `--gpus` path works. This is not the final model-serving blocker because manual device/library exposure proved CUDA visibility inside the official image.

Transformers-backed vLLM server fallback blocker:

- Failing command: official Docker serve with `--model-impl transformers --enforce-eager`
- Log path: `data/model_evaluation/server_smoke/glm52_docker_server_smoke_manual_logs_20260723_055937/transformers_vllm_docker_server.log`
- Error: `RuntimeError: shape '[-1, 16, 192]' is invalid for input of size 33554432`
- Smallest next action: do not use this as the next evaluation backend without a known vLLM/HF fix for GLM MoE DSA attention shape handling.

## Next Recommended Action

1. Install/configure NVIDIA Container Toolkit/CDI for clean Docker GPU support.
2. Try a GLM/SM120-compatible serving backend rather than the tested `vllm/vllm-openai:glm52-cu129` native path for this tiny 0.8B checkpoint. The strongest next sanity check is SGLang for the same model on GPU 0.
3. Once an OpenAI-compatible server reaches `/v1/models`, add the small `GenerationEngine` HTTP adapter and run the one-sample Cline smoke over `mbpp_stateful_working_v0`.
