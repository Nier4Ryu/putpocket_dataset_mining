# T20260812-001__remote-server-b-workspace

task identity: T20260812-001__remote-server-b-workspace
objective: remote-server-b-workspace
status: in_progress
base tip: ec2b22642a1976501f06e560968153da2af090ca
branch: agent/T20260812-001__remote-server-b-workspace
worktree: /workspace/putpocket_dataset_mining_worktrees/T20260812-001__remote-server-b-workspace
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
  - agent/tasks/T20260812-001__remote-server-b-workspace/
commits:
  - pending
final handoff link: pending
