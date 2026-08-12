# T20260812-001__remote-workspace-session-path

task identity: T20260812-001__remote-workspace-session-path
objective: remote-workspace-session-path
status: in_progress
base tip: 1ede2a4ca174a4f4012c872f39c57574a443107c
branch: agent/T20260812-001__remote-workspace-session-path
worktree: /workspace/putpocket_dataset_mining_worktrees/T20260812-001__remote-workspace-session-path
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
  - agent/tasks/T20260812-001__remote-workspace-session-path/
commits:
  - pending
final handoff link: pending
