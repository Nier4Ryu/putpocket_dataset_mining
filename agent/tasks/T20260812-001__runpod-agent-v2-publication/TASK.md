# T20260812-001__runpod-agent-v2-publication

task identity: T20260812-001__runpod-agent-v2-publication
objective: runpod-agent-v2-publication
status: in_progress
base tip: 3bf167a899d2bf195e71b23aa5f4e928ffcf6f44
branch: agent/T20260812-001__runpod-agent-v2-publication
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260812-001__runpod-agent-v2-publication
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
  - agent/tasks/T20260812-001__runpod-agent-v2-publication/
commits:
  - pending
final handoff link: pending
