# T20260812-001__runpod-persistent-native-scratch

task identity: T20260812-001__runpod-persistent-native-scratch
objective: runpod-persistent-native-scratch
status: in_progress
base tip: 3225e44c262375604a0b820be8cfe63f8b4a9e7d
branch: agent/T20260812-001__runpod-persistent-native-scratch
worktree: /workspace/putpocket_dataset_mining_worktrees/T20260812-001__runpod-persistent-native-scratch
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
  - redirect canonical RunPod native scratch to persistent workspace storage
  - validate manifest/environment propagation
  - integrate before 32-job recovery build
completion criteria:
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - focused pytest: PASS (27 tests and 2 subtests)
  - static checks: PASS (`git diff --check`, shell syntax, and Python compileall)
artifacts:
  - agent/tasks/T20260812-001__runpod-persistent-native-scratch/
commits:
  - pending
final handoff link: agent/tasks/T20260812-001__runpod-persistent-native-scratch/handoffs/TO_GPT_20260812-163000.md
