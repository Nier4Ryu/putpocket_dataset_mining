# /goal — Implement and Run GLM-5.2-0.8B Evaluation Over Mined Dataset

You are operating inside the `~/putpocket_dataset_mining` repository.

This is a durable goal, not a one-shot patch request.

Your job is to **implement the GLM evaluation runner, run the actual evaluation experiment, analyze the results, and write a final handoff report to `TO_GPT.md`**.

Do not stop after static code changes.
Do not stop after smoke tests.
Do not stop after writing a plan.
Do not claim completion unless the full experiment has been attempted and the results are summarized in `TO_GPT.md`.

---

## Goal

Evaluate this target model:

```text
inference-optimization/GLM-5.2-0.8B-A0.8B
```

against the existing mined dataset generated earlier with Qwen 3.5.

Evaluation name:

```text
eval_glm52_08b_on_mbpp_stateful_working_v0
```

The goal is to determine whether GLM-5.2-0.8B is usable on the current valid dataset samples.

---

## Source dataset

Inspect the repository to find the existing mined dataset.

The expected dataset is likely:

```text
data/dataset_mining/datasets/mbpp_stateful_working_v0
```

but do not assume blindly. Verify:

```text
- dataset versions under data/dataset_mining/datasets/
- accepted.jsonl existence
- accepted sample count
- accepted row schema
- artifact_path / artifact_root fields
- per-sample artifact completeness
```

The remembered expected state is about 20 accepted samples.

The source mined dataset must be treated as **read-only**.

Do not modify:

```text
data/dataset_mining/datasets/mbpp_stateful_working_v0
data/dataset_mining/runs/*
```

unless the change is only a new evaluation output under a separate evaluation directory.

---

## Evaluation mode

Use full two-turn re-run with GLM.

For each accepted mined sample:

```text
1. Reconstruct System Prompt 1 + Query 1 for the target GLM model.
2. GLM generates History 1 through the headless Cline agent loop.
3. Apply GLM History 1 to a Docker workspace through tool execution.
4. Verify History 1 with the hidden verifier tests.
5. If History 1 verification fails:
   - do not run History 2,
   - record final status,
   - record failure stage/class,
   - continue to the next sample.
6. If History 1 verification passes:
   reconstruct System Prompt 2 + Query 1 + GLM-generated History 1 + Query 2.
7. GLM generates History 2 through the headless Cline agent loop.
8. Apply GLM History 2 to the same workspace lineage.
9. Verify History 2 with hidden verifier tests.
10. If History 2 verification passes, run Codex judge.
11. Record final status and artifacts.
```

Do **not** use the original Qwen-generated History 1 as fixed context.

This evaluation must answer:

```text
Can GLM-5.2-0.8B perform the current two-turn headless Cline coding tasks?
```

---

## Prompt rendering policy

Use stored semantic components/messages and re-render for the GLM tokenizer/chat template.

Do not feed Qwen-rendered prompts directly to GLM.

Preferred order:

```text
stored semantic messages / system prompt / query / generated GLM history
→ GLM tokenizer chat template
→ rendered prompt for GLM
→ local vLLM Python engine
```

You must save:

```text
- semantic messages used for GLM
- rendered prompt text actually sent to GLM
- tokenization metadata
- prompt SHA256
- prompt token count
```

If the current artifacts do not contain enough semantic information to re-render for GLM, do not silently use Qwen-rendered prompts. Record the blocker in `TO_GPT.md` with exact missing files.

---

## Serving backend

Use local vLLM Python engine only.

Do not use:

```text
- remote endpoint
- OpenAI-compatible HTTP server mode
- Transformers fallback
```

unless vLLM support is impossible and the failure is recorded as a blocker.

Target model:

```text
inference-optimization/GLM-5.2-0.8B-A0.8B
```

Use the repository’s model registry/constants approach if present.

If the model is missing from the shared Hugging Face cache, download it through the existing repo policy.

---

## Hardware and build constraints

This server is shared.

Build CPU cap:

```text
32 CPU build jobs max
```

Use only these GPUs:

```text
4,5,6,7
```

Do not use GPUs:

```text
0,1,2,3
```

For vLLM/external build steps, enforce:

```bash
PUTPOCKET_BUILD_THREADS=32
MAX_JOBS=32
CMAKE_BUILD_PARALLEL_LEVEL=32
CARGO_BUILD_JOBS=32
NVCC_THREADS=1
```

Evaluation worker policy:

```text
smoke/debug:
  1 worker on GPU 4

full evaluation:
  up to 4 workers on GPUs 4,5,6,7
  tp=1
  pp=1
```

Before launching multi-worker evaluation, validate GPU slot assignments.

---

## Generation policy

Use deterministic decoding.

Default generation policy:

```text
temperature = 0.0
greedy decoding
no sampling randomness
```

Record evaluation seed separately from the mining seed.

Do not conflate:

```text
mining_seed
RANDOM_SEED
evaluation_seed
```

---

## Judge policy

Use Codex CLI judge if the existing repo already has a judge pattern.

Judge policy:

```text
backend: codex_cli
sandbox: read-only
approval: never
skip_if_unit_test_failed: true
```

Judge should run only after the relevant verification checks pass.

Judge output may be simple:

```json
{"decision": "pass|fail|uncertain"}
```

---

## Required outputs

Create evaluation outputs under a separate evaluation root, recommended:

```text
data/model_evaluation/runs/<eval_run_id>/
```

Use an eval run id similar to:

```text
eval_glm52_08b_on_mbpp_stateful_working_v0_<timestamp>
```

Do not pollute `data/dataset_mining/datasets/`.

For each evaluated sample, save enough artifacts to inspect:

```text
per_sample/<sample_id>/<attempt_id>/
├── episode_timeline.md
├── episode_timeline.jsonl
├── eval_config.yaml
├── source_dataset_row.json
├── source_artifact_reference.json
├── prepared_glm/
│   ├── messages_history1.json
│   ├── messages_history2.json
│   ├── rendered_prompt_history1_turn_*.txt
│   ├── rendered_prompt_history2_turn_*.txt
│   └── tokenization_*.json
├── trajectories/
│   ├── history1_trajectory.jsonl
│   └── history2_trajectory.jsonl
├── workspace_snapshots/
│   ├── initial/
│   ├── after_history1/
│   └── after_history2/
├── verification/
│   ├── history1/
│   └── history2/
├── judge/
└── eval_sample_summary.json
```

At run level, save:

```text
eval_config.yaml
results.jsonl
summary.json
summary.md
TO_GPT.md
```

`results.jsonl` must include at least:

```text
sample_id
task_id
source_dataset_version
source_artifact_path
target_model
evaluation_mode
history1_status
history1_failure_class
history2_status
history2_failure_class
judge_decision
final_status
history1_turns
history2_turns
prompt_token_counts
completion_token_counts
latency
artifact_path
```

---

## Final report: TO_GPT.md

At the end, write:

```text
TO_GPT.md
```

at the repository root.

This file must be human-readable and must contain:

```text
# GLM Evaluation Report To GPT

## Executive Summary
- Did GLM-5.2-0.8B look usable or not?
- How many samples were evaluated?
- How many succeeded?
- Main failure modes.

## Commands Run
- exact commands used for implementation validation
- exact commands used for smoke evaluation
- exact commands used for full evaluation
- log paths

## Dataset Used
- dataset version
- accepted count
- selected subset
- accepted.jsonl path

## Target Model
- model id
- backend
- GPU config
- decoding config
- chat-template rendering policy

## Implementation Summary
- files added/modified
- new CLI commands
- reusable modules

## Experiment Progress Log
- smoke run status
- full run status
- per-stage progress
- any stop/retry/blocker events

## Results Summary
- final status counts
- history1 pass/fail counts
- history2 pass/fail counts
- judge pass/fail/uncertain counts
- failure-stage histogram
- representative accepted sample if any
- representative rejected/failed samples

## Interpretation
- Is GLM-5.2-0.8B usable for this dataset?
- What evidence supports this?
- What caveats apply?
- Is the failure mainly model capability, prompt/tool parsing, verifier, runtime, or infra?

## Visualization-Ready Data
- where results.jsonl is
- fields useful for plotting
- recommended plots later

## Remaining Issues / Next Tasks
- exact blockers if any
- follow-up recommendations
```

If the goal cannot be completed, `TO_GPT.md` must still be written and must clearly say:

```text
status: blocked
exact failing command
log path
reason
what was completed before the block
smallest next action
```

---

## Do not finish early

This goal is **not complete** when:

```text
- code compiles only
- unit tests pass only
- smoke on one sample passes only
- a plan is written only
- a CLI is added but full evaluation is not attempted
- state/config is edited without running the experiment
```

This goal is complete only when one of the following is true:

```text
A. Full evaluation over the selected working dataset completes and TO_GPT.md summarizes the result.

B. A real blocker prevents full evaluation, and TO_GPT.md records the exact failing command, logs, reason, completed partial work, and next action.
```

---

## Implementation tasks

Implement whatever is necessary, but keep the source mined dataset read-only.

Likely implementation tasks:

```text
1. Inspect current dataset artifact schema.
2. Add GLM model registry/config entry if missing.
3. Add model evaluation package/module if missing.
4. Add evaluation sample loader from accepted.jsonl.
5. Add prompt reconstruction for GLM:
   - semantic messages preferred
   - GLM chat template rendering
   - rendered prompt artifacts
6. Reuse headless Cline runtime/tool parser/Docker backend/verifier where possible.
7. Implement full two-turn evaluation runner.
8. Integrate Codex judge.
9. Add run-level result writer and summary generator.
10. Add CLI commands.
11. Add tests for loader/config/result schema where feasible.
12. Run smoke evaluation on one sample.
13. Run full evaluation on working dataset or record blocker.
14. Write TO_GPT.md.
```

---

## Suggested CLI names

Prefer commands like:

```bash
python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --profile smoke

python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --profile full \
  --gpu-slots 4,5,6,7 \
  --workers 4
```

If the repo uses a different CLI style, adapt to the existing style and document the actual commands in `TO_GPT.md`.

---

## Acceptance checks

Run as much as applicable:

```bash
source scripts/env/env_activate.sh
python -m compileall src tests
python -m unittest discover -s tests -v
```

Then run evaluation commands:

```bash
# smoke: one sample
<actual smoke evaluation command>

# full: working_v0
<actual full evaluation command>
```

Finally verify:

```bash
test -f TO_GPT.md
test -f data/model_evaluation/runs/<eval_run_id>/results.jsonl
test -f data/model_evaluation/runs/<eval_run_id>/summary.json
test -f data/model_evaluation/runs/<eval_run_id>/summary.md
```

---

## Important interpretation rule

If GLM fails many samples, that is a valid result.

Do not try to make GLM look better by weakening the verifier, changing dataset samples, using Qwen histories, or switching to a larger model.

The purpose is to know whether GLM-5.2-0.8B is usable on the current dataset.
