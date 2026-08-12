# T20260813-001__runpod-inference-server1-verification

task identity: T20260813-001__runpod-inference-server1-verification
objective: runpod-inference-server1-verification
status: in_progress
base tip: a918d17b326e54927c251108200f79f0b629af3e
branch: agent/T20260813-001__runpod-inference-server1-verification
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260813-001__runpod-inference-server1-verification
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
  - agent/tasks/T20260813-001__runpod-inference-server1-verification/
commits:
  - pending
final handoff link: pending
