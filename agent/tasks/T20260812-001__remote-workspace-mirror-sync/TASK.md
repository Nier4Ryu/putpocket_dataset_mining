# T20260812-001__remote-workspace-mirror-sync

task identity: T20260812-001__remote-workspace-mirror-sync
objective: remote-workspace-mirror-sync
status: in_progress
base tip: 2f8bcf0e13e762495747b88214cb76bdee1edd0a
branch: agent/T20260812-001__remote-workspace-mirror-sync
worktree: /workspace/putpocket_dataset_mining_worktrees/T20260812-001__remote-workspace-mirror-sync
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
  - agent/tasks/T20260812-001__remote-workspace-mirror-sync/
commits:
  - pending
final handoff link: pending
