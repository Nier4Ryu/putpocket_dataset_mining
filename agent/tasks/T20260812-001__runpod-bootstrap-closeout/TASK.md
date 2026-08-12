# T20260812-001__runpod-bootstrap-closeout

task identity: T20260812-001__runpod-bootstrap-closeout
objective: runpod-bootstrap-closeout
status: in_progress
base tip: 905549a7d9ac39c0148fa474d61dab167b140d78
branch: agent/T20260812-001__runpod-bootstrap-closeout
worktree: /workspace/putpocket_dataset_mining_worktrees/T20260812-001__runpod-bootstrap-closeout
runtime mode: shared-python-overlay
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
  - ignore the canonical generated native-build directory
  - record final real-bootstrap evidence
  - publish and synchronize final master
completion criteria:
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - runtime validation and focused tests: PASS
artifacts:
  - agent/tasks/T20260812-001__runpod-bootstrap-closeout/
commits:
  - pending
final handoff link: agent/tasks/T20260812-001__runpod-bootstrap-closeout/handoffs/TO_GPT_20260812-164500.md
