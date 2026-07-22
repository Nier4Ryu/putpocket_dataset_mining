# Codex Implementation Prompt — Dataset Mining Objective v0.5

Read and follow:

```text
tasks/objectives/dataset_mining/START_HERE.md
```

Then read:

```text
tasks/objectives/dataset_mining/objective.yaml
tasks/objectives/dataset_mining/state.yaml
configs/dataset_mining/mbpp_stateful_single.yaml
configs/dataset_mining/mbpp_stateful_multi.yaml
docs/specs/*.md
```

You are implementing the standalone Dataset Mining repo.
You are not implementing Mode B, Prefix State Repair, fixing-code, or downstream dataset consumption.

Before writing implementation code, check `state.yaml`.
If it says:

```yaml
implementation_lock:
  status: human_review_required
```

stop and write a review note only.

If it says:

```yaml
implementation_lock:
  status: ready_for_implementation
```

implement the objective in phases, keeping the spec as source of truth.

Required operating behavior:

- Do not stop after a plan.
- Do not complete with smoke-only code.
- Do not replace original Cline tool format with JSON actions.
- Do not create `.clinerules` files inside Docker workspaces.
- Do not expose MBPP unit tests to the agent workspace; tests are hidden verifier-only.
- Do not let vLLM apply chat templates internally.
- Save semantic messages, rendered prompts, tokenization metadata, timelines, trajectories, verification, and judge artifacts.
- Generation default must be deterministic greedy decoding with temperature 0.
- Use local vLLM Python engine only.
- Use Codex CLI only for judge, in read-only/approval-never mode.
- Implement local dataset materialization under `data/dataset_mining/datasets/`; cross-repo import/export is out of scope.
- Cap vLLM/CUDA extension builds to 16 CPU build threads by default.
- Restrict runtime GPU workers to CUDA devices 0,1,2.

Update `tasks/objectives/dataset_mining/state.yaml` as implementation progresses.
If blocked, record the exact blocker in `state.yaml` and stop.
