# GLM-5.2-0.8B Evaluation Objective Audit — Analysis Only

You are not implementing a patch in this run.

You are an architecture/evaluation auditor for this dataset-mining repo.

The repo already contains mined dataset samples generated with Qwen 3.5. The human now wants to evaluate another model:

```text
Target model:
  inference-optimization/GLM-5.2-0.8B-A0.8B

Planned evaluation name:
  eval_glm52_08b_on_mbpp_stateful_working_v0
```

Your job is to inspect the repo and determine how to evaluate this model on the existing mined dataset.

Do not modify files.
Do not patch code.
Do not edit configs.
Do not edit state.yaml.
Do not run long mining jobs.
Do not start multi-sample mining.
Do not download large models unless explicitly required by an already-existing lightweight command.
Do not build vLLM or Docker images in this audit.

You may run read-only inspection commands:

git status
find
rg
sed
cat
wc
python scripts that only inspect existing files
sqlite3 read-only queries if the database exists

If a runtime check would require model download, vLLM engine start, GPU allocation, or Docker build, do not run it. Instead, report the exact command that should be run later.

Human decisions already made

Use these as fixed decisions. Do not ask these again.

Evaluation mode

Use full two-turn re-run with GLM.

For each accepted mined sample:

1. Use System Prompt 1 + Query 1.
2. GLM generates History 1 through the headless Cline agent loop.
3. Apply GLM History 1 to a Docker workspace through tool execution.
4. Verify History 1 with the hidden verifier tests.
5. Only if History 1 verification passes:
   use System Prompt 2 + Query 1 + GLM-generated History 1 + Query 2.
6. GLM generates History 2 through the headless Cline agent loop.
7. Apply GLM History 2 to the same workspace lineage.
8. Verify History 2.
9. Run Codex judge if verification policy requires it.
10. Record final status.

Do not use the original Qwen-generated History 1 as fixed context for this first evaluation objective.

Prompt rendering policy

Use stored semantic prompt components / messages and re-render for the target GLM model tokenizer/chat template.

Do not feed Qwen-rendered prompts directly into GLM unless the report explains that no semantic-message reconstruction is available.

The preferred policy is:

semantic messages / system prompt / query / history
→ GLM tokenizer chat template
→ rendered prompt for GLM
→ local vLLM Python engine
Serving backend

Use local vLLM Python engine only.

Do not use a remote endpoint.
Do not use OpenAI-compatible HTTP server mode.
Do not use Transformers fallback unless the report explicitly says vLLM support is impossible and asks the human for approval.

Dataset/subset

Do not assume the exact dataset version. Inspect the repo and report what mined datasets exist.

The human remembers that about 20 accepted samples were mined, likely under something like:

data/dataset_mining/datasets/mbpp_stateful_working_v0

But you must verify:

dataset versions present,
accepted counts,
accepted.jsonl schema,
artifact paths,
whether each accepted row has enough data for evaluation.
GPU policy

Use only GPUs 4,5,6,7.

Do not use GPUs 0,1,2,3.

For audit, do not start GPU-heavy jobs. Just report the recommended worker/GPU config.

Generation policy

Use deterministic decoding.

Default:

temperature = 0.0
greedy decoding
no sampling randomness

Record evaluation seed separately from mining seed if evaluation code is later implemented.

Judge policy

Use judge = yes.

Use the existing Codex CLI judge pattern if present:

sandbox: read-only
approval: never

Do not run judge in this audit unless there is already a lightweight read-only test command.

History-2 condition

Only run History 2 if History 1 verification succeeds.

If History 1 fails:

final_status = rejected or failed_infra depending on failure class
failure_stage = history1.verification or earlier
history2 = skipped
Future visualization / comparison

The evaluation output should preserve enough information to later visualize whether GLM-5.2-0.8B is usable or not.

You do not need to implement visualization now.

But your report must propose what result fields should be saved to enable later visual comparison with the original Qwen-mined results.

Required investigation
1. Mined dataset inventory

Find all local mined dataset versions.

Inspect paths such as:

data/dataset_mining/datasets/
data/dataset_mining/runs/
data/dataset_mining/mining_index.sqlite

Report:

dataset_version
dataset_path
accepted.jsonl exists?
accepted_count
rejected_count if available
uncertain_count if available
artifact_index exists?
dataset_manifest exists?

Also identify which dataset is the likely default for this GLM evaluation.

Expected candidate:

mbpp_stateful_working_v0

But do not assume; verify.

2. accepted.jsonl schema

For the likely dataset version, inspect accepted.jsonl.

Report:

row fields
field that points to attempt artifact path
sample_id field
task_id field
policy_delta field
query2 class field if any
final_status field

Answer:

Can an evaluator locate each sample's artifact directory from accepted.jsonl alone?
3. Per-sample artifact completeness

Inspect several accepted sample artifact directories.

For at least 3 accepted rows, check whether these exist:

episode_summary.json
episode_timeline.md
episode_timeline.jsonl

prepared/system_prompt_1.md
prepared/system_prompt_2.md
prepared/query1.*
prepared/query2.*
prepared/messages_history1.json
prepared/messages_history2.json
prepared/rendered_prompt_history1.txt
prepared/rendered_prompt_history2.txt
prepared/tokenization_history1.json
prepared/tokenization_history2.json

trajectories/history1_trajectory.jsonl
trajectories/history2_trajectory.jsonl

workspace_snapshots/initial/
workspace_snapshots/after_history1/
workspace_snapshots/after_history2/

verification/history1/checklist.json
verification/history2/checklist.json

judge/judge_decision.json

Report missing files and whether missing files block GLM evaluation.

4. Input reconstruction modes

Determine which evaluation input reconstruction is feasible.

Check whether the repo can support each mode:

Mode A:
  Reconstruct semantic messages from stored messages_history1/messages_history2
  and re-render with GLM tokenizer.

Mode B:
  Compose prompts again from stored system_prompt_1/2, query1, query2, and generated GLM history.

Mode C:
  Reuse stored Qwen rendered prompt text directly.

Report which mode is safest and why.

Preferred answer should be Mode A or B, not C, unless required artifacts are missing.

5. Existing runtime reuse

Inspect the current code and identify reusable components for GLM evaluation:

- local vLLM Python engine wrapper
- prompt renderer / chat-template renderer
- headless Cline runtime
- original Cline tool parser
- Docker workspace backend
- hidden verifier test materializer/injector
- independent verifier runner
- Codex judge runner
- artifact writer / episode timeline writer
- SQLite mining index or result DB code

For each component, report:

file path
class/function name
whether reusable as-is
whether needs extension
6. New evaluator design

Based on the existing repo structure, propose where a new evaluator should live.

Do not implement it.

Possible names:

src/putpocket_dataset_mining/model_evaluation/

or:

src/putpocket_dataset_mining/evaluation/

Report the recommended module layout.

The evaluator should be separate from dataset mining generation, but should reuse runtime components where possible.

7. GLM model support check

Inspect model registry/constants/config code.

Report:

where model IDs / paths are configured
how Qwen/Qwen3.5-9B is configured
how to add inference-optimization/GLM-5.2-0.8B-A0.8B
where shared HF cache is configured
whether local vLLM Python engine can accept arbitrary model IDs

Also inspect current vLLM installation/configuration enough to answer:

Can this repo plausibly load GLM-5.2-0.8B-A0.8B with local vLLM Python engine?

Do not start a heavy model load during this audit.

If uncertain, provide the exact smoke command to run later.

8. Evaluation output schema

Propose output structure.

The human wants:

eval_glm52_08b_on_mbpp_stateful_working_v0

Recommended output root should be something like:

data/model_evaluation/runs/eval_glm52_08b_on_mbpp_stateful_working_v0_<timestamp>/

or another repo-consistent path.

Output should include:

eval_config.yaml
results.jsonl
summary.json
summary.md
per_sample/<sample_id>/<attempt_id>/...

Each sample result should record at least:

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

For future visualization, include fields that allow:

per-sample GLM success/failure
failure stage histogram
comparison with original Qwen accepted baseline
token/turn/latency charts
9. Required human decisions remaining

After inspection, list only decisions still required from the human.

Do not repeat already fixed decisions unless the repo state contradicts them.

Likely remaining decisions may include:

- exact dataset version if multiple valid candidates exist
- whether to run debug subset first or working_v0 directly
- whether to implement vLLM-only or add fallback if GLM is unsupported
- whether to store evaluation output under data/model_evaluation or data/dataset_mining/evaluations
10. Implementation plan proposal

Propose a staged implementation plan, but do not implement.

The plan should include:

Stage 0: dataset/evaluation audit confirmation
Stage 1: GLM model registry/config
Stage 2: evaluation sample loader from accepted.jsonl
Stage 3: prompt reconstruction/re-rendering for GLM
Stage 4: full two-turn GLM evaluation runner
Stage 5: verifier/judge integration
Stage 6: result aggregation and summary
Stage 7: smoke run on 1 sample
Stage 8: full run on working dataset

For each stage, list:

files likely touched
tests to add
acceptance command
risks
Final report format

Return a report in this exact structure:

GLM Evaluation Objective Audit Report
Executive Summary
Mined Dataset Inventory
Accepted JSONL Schema
Per-Sample Artifact Completeness
Feasible Input Reconstruction Modes
Runtime Components Reuse Map
GLM Model Support / vLLM Feasibility
Proposed Evaluation Output Schema
Remaining Human Decisions
Proposed Implementation Plan
Acceptance Criteria For Future Patch
Risks / Unknowns
Appendix: Commands Run

Remember:
Do not patch.
Do not edit files.
Do not implement.
This is analysis only.
