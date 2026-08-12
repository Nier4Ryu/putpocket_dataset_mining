# T20260812-001__server-b-judge-preflight

task identity: T20260812-001__server-b-judge-preflight
objective: server-b-judge-preflight
status: in_progress
base tip: 4e517a23e5941436a522ff92efdff7f79086a440
branch: agent/T20260812-001__server-b-judge-preflight
worktree: /workspace/putpocket_dataset_mining_worktrees/T20260812-001__server-b-judge-preflight
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
  - agent/tasks/T20260812-001__server-b-judge-preflight/
commits:
  - pending
final handoff link: pending
