# T20260812-001__runpod-agent-dev-image

task identity: T20260812-001__runpod-agent-dev-image
objective: runpod-agent-dev-image
status: in_progress
base tip: 487bc28e4e799daca4c6c73134110662ce3dd335
branch: agent/T20260812-001__runpod-agent-dev-image
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260812-001__runpod-agent-dev-image
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
  - implement
  - validate
  - handoff
completion criteria:
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - pending
artifacts:
  - agent/tasks/T20260812-001__runpod-agent-dev-image/
commits:
  - pending
final handoff link: pending
