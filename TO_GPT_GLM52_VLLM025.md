# GLM-5.2 vLLM 0.25 Evaluation Report

## Executive Summary

The separate GLM vLLM 0.25 stack was created and built successfully from vLLM `v0.25.1` without touching the existing Qwen / Putpocket-v0.19.1 stack.

Native GLM model loading still failed before generation. vLLM 0.25.1 finds the SM120 sparse MLA backend, but rejects this GLM-5.2-0.8B configuration because the model reaches `AttentionSelectorConfig(head_size=192, use_mla=True, use_sparse=True)` and `FLASHINFER_MLA_SPARSE_SM120` reports `head_size not supported`.

No dataset forwarding/evaluation smoke or full evaluation was run, by design, because the native vLLM 0.25 engine could not load `inference-optimization/GLM-5.2-0.8B-A0.8B`. The mined dataset was validated as present and unchanged.

## Branch / Git

- Branch: `blackwell`
- Base commit before this work: `e034f49eb0b94525ddb4c641ada0698da9883d71`
- Remote: `origin`
- Report/write status: source, config, script, and report changes prepared for commit to `origin/blackwell`
- Unrelated pre-existing untracked file left untouched: `TO_GPT_USE_SPARSE_AUDIT.md`

## Stack Separation

The old Qwen / vLLM 0.19 stack was preserved:

- `Putpocket_env` was not overwritten.
- `Putpocket_env_glm52` was not overwritten.
- `externals/vllm` was not modified.
- `externals/vllm_glm52` was not modified.
- Existing mined dataset artifacts under `data/dataset_mining/` were not modified.

New vLLM 0.25 stack paths:

- Env: `Putpocket_env_glm52_v025`
- Activation script: `scripts/env/env_activate_glm52_v025.sh`
- Bootstrap script: `scripts/env/bootstrap_glm52_v025_env.sh`
- vLLM source: `externals/vllm_glm52_v025`
- Serving config: `configs/serving/glm52_vllm025_blackwell.yaml`
- Evaluation config: `configs/model_evaluation/glm52_08b_vllm025_blackwell.yaml`
- Evaluation wrapper: `src/putpocket_dataset_mining/model_evaluation/glm52_vllm025_eval.py`

## vLLM 0.25 Source

- Source path: `externals/vllm_glm52_v025`
- Remote: `https://github.com/vllm-project/vllm.git`
- Selected ref: `v0.25.1`
- Commit: `752a3a504485790a2e8491cacbb35c137339ad34`
- Installed vLLM version: `0.25.1`

Evidence of SM120 sparse MLA support in v0.25.1:

- `externals/vllm_glm52_v025/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`
  - `FlashInferMLASparseSM120Backend`
  - backend name `FLASHINFER_MLA_SPARSE_SM120`
  - supports compute capability major `12`
  - requires FlashInfer sparse MLA API
  - requires `index_topk=2048`
- `externals/vllm_glm52_v025/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py`
  - requires packed `fp8_ds_mla` KV cache layout after canonicalization
  - uses GLM-specific `arbitrary_fp32` KV scale format for model types starting with `glm`

Important limitation found:

- `FlashInferMLASparseSM120Backend` inherits supported head sizes from `_FlashInferMLASparseBackendBase`, which returns `[576]`.
- The GLM-5.2-0.8B model reaches `head_size=192`, so backend validation rejects it before generation.

## GLM Env

- Env path: `Putpocket_env_glm52_v025`
- Activation: `source scripts/env/env_activate_glm52_v025.sh`
- Python: `3.13.14`
- Python executable: `/home/dyryu/putpocket_dataset_mining/Putpocket_env_glm52_v025/bin/python`
- CUDA_HOME: `/usr/local/cuda-12.9`
- Torch: `2.11.0+cu129`
- Torch CUDA: `12.9`
- CUDA available: `True`
- GPU 0: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- Compute capability: `(12, 0)`
- Transformers: `5.14.1`
- vLLM: `0.25.1`
- vLLM import path: `/home/dyryu/putpocket_dataset_mining/externals/vllm_glm52_v025/vllm/__init__.py`
- DeepGEMM: `2.5.0`

## Build Summary

Bootstrap command:

```bash
./scripts/env/bootstrap_glm52_v025_env.sh
```

Build status: succeeded.

Build parallelism was fixed at 8:

- `PUTPOCKET_BUILD_THREADS=8`
- `MAX_JOBS=8`
- `CMAKE_BUILD_PARALLEL_LEVEL=8`
- `CARGO_BUILD_JOBS=8`
- `NVCC_THREADS=1`
- `TORCH_CUDA_ARCH_LIST=12.0`

Build logs:

- Log directory: `logs/env_setup_glm52_v025/20260722T213401Z/`
- vLLM build log: `logs/env_setup_glm52_v025/20260722T213401Z/build-vllm-glm52-v025.log`
- Setup manifest: `logs/env_setup_glm52_v025/20260722T213401Z/setup_summary.json`

The vLLM source build used CUDA 12.9 and compiled SM120 targets. No OOM occurred at 8 build threads.

## GLM Config

Model config inspection command loaded only HuggingFace config, not full weights.

Relevant fields for `inference-optimization/GLM-5.2-0.8B-A0.8B`:

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

vLLM derives the failing attention shape as `head_size = kv_lora_rank + qk_rope_head_dim = 128 + 64 = 192`.

vLLM derives sparse MLA behavior from the GLM DSA architecture path and the presence of `index_topk`, not from this repo's launch config.

## GLM Native Smoke

Primary smoke command:

```bash
source scripts/env/env_activate_glm52_v025.sh
CUDA_VISIBLE_DEVICES=0 python - <<'PY' > data/model_evaluation/logs/glm52_vllm025_native_smoke_20260723_071659.log 2>&1
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
    ["Hello. Reply with one short sentence."],
    SamplingParams(temperature=0.0, max_tokens=64),
)
print(outs[0].outputs[0].text)
PY
```

Result: failed before generation.

Log path:

- `data/model_evaluation/logs/glm52_vllm025_native_smoke_20260723_071659.log`

Key error:

```text
ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(head_size=192, dtype=torch.bfloat16, kv_cache_dtype=auto, block_size=None, use_mla=True, has_sink=False, use_sparse=True, use_mm_prefix=False, use_per_head_quant_scales=False, attn_type=AttentionType.DECODER, use_non_causal=False, use_batch_invariant=False, use_kv_connector=False). Reasons: {TRITON_MLA: [sparse not supported], FLASHINFER_MLA_SPARSE_SM120: [head_size not supported]}.
RuntimeError: Engine core initialization failed. See root cause above.
```

Diagnostic float32 command:

```bash
source scripts/env/env_activate_glm52_v025.sh
CUDA_VISIBLE_DEVICES=0 python - <<'PY' > data/model_evaluation/logs/glm52_vllm025_native_smoke_float32_20260723_071751.log 2>&1
from vllm import LLM, SamplingParams

model = "inference-optimization/GLM-5.2-0.8B-A0.8B"

llm = LLM(
    model=model,
    trust_remote_code=True,
    tensor_parallel_size=1,
    max_model_len=8192,
    max_num_seqs=1,
    dtype="float32",
)

outs = llm.generate(
    ["Hello. Reply with one short sentence."],
    SamplingParams(temperature=0.0, max_tokens=64),
)
print(outs[0].outputs[0].text)
PY
```

Result: failed before generation.

Log path:

- `data/model_evaluation/logs/glm52_vllm025_native_smoke_float32_20260723_071751.log`

Key error:

```text
ValueError: No valid attention backend found for cuda with AttentionSelectorConfig(head_size=192, dtype=torch.float32, kv_cache_dtype=auto, block_size=None, use_mla=True, has_sink=False, use_sparse=True, use_mm_prefix=False, use_per_head_quant_scales=False, attn_type=AttentionType.DECODER, use_non_causal=False, use_batch_invariant=False, use_kv_connector=False). Reasons: {TRITON_MLA: [dtype not supported, sparse not supported], FLASHINFER_MLA_SPARSE_SM120: [head_size not supported, dtype not supported, dtype not supported]}.
RuntimeError: Engine core initialization failed. See root cause above.
```

Conclusion: dtype override does not fix the v0.25.1 native backend blocker.

## Dataset Found

- Dataset version: `mbpp_stateful_working_v0`
- accepted.jsonl: `data/dataset_mining/datasets/mbpp_stateful_working_v0/accepted.jsonl`
- Accepted row count: `20`
- Original dataset artifacts were not modified.

First checked artifact paths:

- `train_730`, task `730`: artifact exists; `prepared/system_prompt_1.md`, `prepared/query1.txt`, `prepared/system_prompt_2.md`, `prepared/query2.txt`, and `workspace_snapshots/initial` all exist.
- `test_347`, task `347`: artifact exists; the same prepared files and initial workspace snapshot exist.
- `train_822`, task `822`: artifact exists; the same prepared files and initial workspace snapshot exist.

## Evaluation Code Compatibility

Existing evaluation code already contains the headless Cline-style two-turn GLM evaluator in:

- `src/putpocket_dataset_mining/model_evaluation/glm_eval.py`

Changes made for a separate v0.25 path:

- Added `GLM52_VLLM025_SERVING_STACK` and 8-thread build env overrides in `src/putpocket_dataset_mining/constants.py`.
- Added `glm52_vllm025_blackwell` as a valid serving stack in `glm_eval.py`.
- Added a dedicated wrapper CLI in `src/putpocket_dataset_mining/model_evaluation/glm52_vllm025_eval.py`.
- Added `--limit` as an alias for `--max-samples`.
- Added parse/malformed tool-call counters to result rows.

The evaluator was not exercised under vLLM 0.25 because native GLM model instantiation failed first.

## Evaluation Smoke Result

Not run.

Reason: native vLLM 0.25.1 GLM load failed before any generation. Per the task requirement, dataset forwarding/evaluation must not be claimed when the native vLLM model path cannot load the target model.

Intended smoke command after backend support is fixed:

```bash
source scripts/env/env_activate_glm52_v025.sh
CUDA_VISIBLE_DEVICES=0 python -m putpocket_dataset_mining.model_evaluation.glm52_vllm025_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_vllm025_on_mbpp_stateful_working_v0 \
  --serving-stack glm52_vllm025_blackwell \
  --profile smoke \
  --limit 1 \
  --workers 1 \
  --gpu-slots 0
```

## Full/Subsample Evaluation Result

Not run.

Reason: native model-load smoke failed. No generated text, Cline-style tool calls, verifier results, or qualitative model behavior could be collected.

Intended full command after backend support is fixed:

```bash
source scripts/env/env_activate_glm52_v025.sh
CUDA_VISIBLE_DEVICES=0,1,2 python -m putpocket_dataset_mining.model_evaluation.glm52_vllm025_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_vllm025_on_mbpp_stateful_working_v0 \
  --serving-stack glm52_vllm025_blackwell \
  --profile full \
  --workers 3 \
  --gpu-slots 0,1,2
```

## Qualitative Usability

GLM-5.2-0.8B-A0.8B usability for the mined MBPP-stateful samples could not be assessed through native vLLM 0.25.1.

The model did not generate any text in the native smoke. Therefore this run provides no evidence about:

- whether GLM produces meaningful task text,
- whether it emits parseable Cline-style tool calls,
- whether it can complete History 1,
- whether it can pass History 1 verification,
- whether it can reach or pass History 2.

The only validated result is that the separate vLLM 0.25.1 build is usable as a Python/CUDA environment but is not sufficient for this GLM-0.8B sparse MLA backend layout on SM120.

## Logs

- Env setup log dir: `logs/env_setup_glm52_v025/20260722T213401Z/`
- vLLM source build log: `logs/env_setup_glm52_v025/20260722T213401Z/build-vllm-glm52-v025.log`
- Setup summary JSON: `logs/env_setup_glm52_v025/20260722T213401Z/setup_summary.json`
- Native smoke log: `data/model_evaluation/logs/glm52_vllm025_native_smoke_20260723_071659.log`
- Native float32 diagnostic log: `data/model_evaluation/logs/glm52_vllm025_native_smoke_float32_20260723_071751.log`

## Known Blockers

Blocker: vLLM 0.25.1 native local engine cannot instantiate attention for `inference-optimization/GLM-5.2-0.8B-A0.8B` on SM120.

Exact failing command:

```bash
source scripts/env/env_activate_glm52_v025.sh
CUDA_VISIBLE_DEVICES=0 python - <<'PY' > data/model_evaluation/logs/glm52_vllm025_native_smoke_20260723_071659.log 2>&1
from vllm import LLM, SamplingParams
model = "inference-optimization/GLM-5.2-0.8B-A0.8B"
llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=1, max_model_len=8192, max_num_seqs=1, dtype="auto")
outs = llm.generate(["Hello. Reply with one short sentence."], SamplingParams(temperature=0.0, max_tokens=64))
print(outs[0].outputs[0].text)
PY
```

Failing log path:

- `data/model_evaluation/logs/glm52_vllm025_native_smoke_20260723_071659.log`

Smallest next action:

Obtain or identify a vLLM/FlashInfer GLM DSA sparse MLA backend that supports SM120 with this tiny model's `head_size=192` layout, then rerun the native smoke command above. If support exists only in a GLM-specific patched vLLM branch or image newer than upstream `v0.25.1`, create a separate `externals/vllm_glm52_<patch>` path rather than modifying `externals/vllm` or this v0.25 stack in place.

## Next Recommended Action

Do not run dataset evaluation through this native vLLM 0.25.1 path yet. The next engineering step is to confirm whether official GLM/vLLM support for `glm_moe_dsa` 0.8B on SM120 includes a backend accepting `head_size=192`, or whether this model requires a GLM-specific branch/image not represented by upstream vLLM `v0.25.1`.
