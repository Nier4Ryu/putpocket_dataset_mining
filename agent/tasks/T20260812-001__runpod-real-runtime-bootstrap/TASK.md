# T20260812-001__runpod-real-runtime-bootstrap

task identity: T20260812-001__runpod-real-runtime-bootstrap
objective: runpod-real-runtime-bootstrap
status: in_progress
base tip: 325f0c05ca68af5ce2f51bae099f9b82c1d68e7a
branch: agent/T20260812-001__runpod-real-runtime-bootstrap
worktree: /workspace/putpocket_dataset_mining_worktrees/T20260812-001__runpod-real-runtime-bootstrap
runtime mode: isolated-native
write scope:
  - source/docs/tests required for this task
forbidden paths:
  - Putpocket_env/
  - Putpocket_env_glm52/
  - Putpocket_env_glm52_v025/
  - data/
  - logs/
  - models/
  - .ssh/
fixed decisions:
  - canonical runtime checkout is /home/${USER}/putpocket_dataset_mining or /workspace/putpocket_dataset_mining
  - task worktrees live under /home/${USER}/putpocket_dataset_mining_worktrees or /workspace/putpocket_dataset_mining_worktrees
plan:
  - establish a RunPod-specific reproducible Python/torch/external-source lock
  - implement real uv environment and editable external provisioning
  - build vLLM and LMCache from clean pinned source with shared parallelism
  - validate dry-runs and focused regressions
  - integrate before canonical native execution
completion criteria:
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - focused Python 3.13 pytest: PASS, 38 tests and 11 subtests
  - blackwell-rtx explicit 32-job dry-run: PASS
  - portable-nvidia explicit 32-job dry-run: PASS, not built
  - git diff --check, shell syntax, compileall: PASS
artifacts:
  - agent/tasks/T20260812-001__runpod-real-runtime-bootstrap/
commits:
  - pending
final handoff link: agent/tasks/T20260812-001__runpod-real-runtime-bootstrap/handoffs/TO_GPT_20260812-160000.md
