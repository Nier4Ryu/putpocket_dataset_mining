# START_HERE — Dataset Mining Objective v0.5

You are the implementation agent for a standalone dataset-mining repo.
This repo owns its own env, external dependencies, Docker specs, dataset-mining code, run artifacts, and local materialized datasets.
It must not depend on a separate fixing-code repo or main Putpocket repo.

## Source of truth

Read these files in order:

1. `tasks/objectives/dataset_mining/objective.yaml`
2. `tasks/objectives/dataset_mining/state.yaml`
3. `configs/dataset_mining/mbpp_stateful_single.yaml`
4. `configs/dataset_mining/mbpp_stateful_multi.yaml`
5. `docs/specs/*.md`

## Implementation lock

If `state.yaml` says:

```yaml
implementation_lock:
  status: human_review_required
```

then do not implement. Only review the specs and report issues.

If it says:

```yaml
implementation_lock:
  status: ready_for_implementation
```

then implement the objective.

## Objective

Build config-driven Dataset Mining code that can mine MBPP-stateful data through real headless Cline-style two-turn agentic episodes.

A single-sample mining attempt must:

1. Load one HuggingFace MBPP sample.
2. Convert it into a headless Cline coding task.
3. Create a Docker-backed simulated user workspace.
4. Run history-1 as a multi-step Cline tool loop using local vLLM Python engine.
5. Run hidden independent unit-test verification from a workspace snapshot.
6. Generate static Cline rules v2 and query2.
7. Run history-2 as another multi-step Cline tool loop over the same workspace lineage.
8. Run hidden independent unit-test verification again.
9. Use Codex CLI as read-only LLM judge if verification passed.
10. Store accepted/rejected/failed_infra/uncertain status and all required artifacts.

Multi-sample mining must be a parallel master-worker controller:

- one master process,
- multiple worker processes,
- full_server uses exactly 4 workers on GPU slots 4,5,6,7,
- each worker owns a GPU slot and a long-lived local vLLM Python engine,
- each worker mines one single-sample attempt at a time,
- master is the only SQLite DB writer,
- master lazily allocates jobs until target accepted count, max attempts, or stop signal.

## Non-negotiable rules

- History-1 and history-2 are not single model responses; each is a multi-step model/tool/observation trajectory.
- Use original Cline tool format, ported from `externals/cline`.
- Use compact Cline prompt by default for `Qwen/Qwen3.5-9B`.
- Do not use custom JSON tool actions unless a future explicit compatibility mode is added.
- Do not create `.clinerules` inside Docker workspaces.
- Cline rules are static prepared prompt artifacts and are rendered into system prompts.
- Do not expose MBPP unit tests to the agent workspace.
- MBPP tests are materialized only for hidden verifier containers.
- `solution.py` starts as a stub in the agent workspace.
- MBPP `code` field is reference/gold solution, not initial code.
- Putpocket must apply tokenizer chat template explicitly before vLLM generation.
- vLLM must receive rendered prompt strings, not chat messages requiring internal template application.
- Save exact rendered prompt text and tokenization metadata.
- Generation defaults to greedy deterministic decoding with `temperature=0.0`.
- `mining_seed` is separate from `RANDOM_SEED`; both default to 42.
- Docker base is Ubuntu 22.04 with Python 3.13 installed into the image.
- vLLM build must be capped to 32 CPU build threads on this shared server.
- Dataset mining workers may use only GPUs 4,5,6,7; GPUs 0,1,2,3 are unavailable.
- Docker workspace is `/workspace` and host-mounted with host UID/GID to avoid permission problems.
- Cross-repo dataset packaging/import is out of scope.

## Completion boundary

The implementation is complete only when the repo can:

- set up its own env,
- build/reuse its default Docker image,
- run a single-sample mining attempt end-to-end,
- run the debug multi-sample profile,
- run the first_parallel profile,
- define the full_server profile with 4 workers on GPUs 4,5,6,7,
- write `data/dataset_mining/mining_index.sqlite`,
- materialize local datasets under `data/dataset_mining/datasets/`,
- and document the generated dataset structure.
