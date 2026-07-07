# Codex Review Prompt — Dataset Mining Objective v0.5

Read the following files only as a reviewer:

1. `tasks/objectives/dataset_mining/START_HERE.md`
2. `tasks/objectives/dataset_mining/objective.yaml`
3. `tasks/objectives/dataset_mining/state.yaml`
4. `configs/dataset_mining/mbpp_stateful_single.yaml`
5. `configs/dataset_mining/mbpp_stateful_multi.yaml`
6. `docs/specs/*.md`

Do not create or modify implementation files.
Do not create `src/`, `docker/`, `scripts/`, or tests.
Do not modify repo state.

Your output should be a review report with:

- any contradictions you found,
- any ambiguous implementation requirements,
- any missing safety checks,
- any places where the spec might accidentally allow smoke-only completion,
- any suggestions for clearer implementation phases.

If `tasks/objectives/dataset_mining/state.yaml` contains:

```yaml
implementation_lock:
  status: human_review_required
```

then implementation is locked and you must not implement.


Additional review focus: verify that vLLM builds are capped at 32 CPU threads and full_server runtime workers use only GPUs 4,5,6,7.
