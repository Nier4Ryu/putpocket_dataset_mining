# T20260814-001__fix-server2-torch-cu129-runtime

task identity: T20260814-001__fix-server2-torch-cu129-runtime
objective: fix-server2-torch-cu129-runtime
status: in_progress
base tip: 29ed706d805c63dd8e1359109ec609221e5e7fba
branch: agent/T20260814-001__fix-server2-torch-cu129-runtime
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260814-001__fix-server2-torch-cu129-runtime
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
  - implement
  - validate
  - handoff
completion criteria:
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - git diff --check: PASS
  - bash -n scripts/env/*.sh: PASS
  - compileall src tests: PASS
  - unittest discover: PASS (182 tests)
  - focused pytest: PASS (40 tests)
artifacts:
  - agent/tasks/T20260814-001__fix-server2-torch-cu129-runtime/
commits:
  - 7f7498a fix(env): pin server2 torch cu129 runtime
final handoff link: agent/tasks/T20260814-001__fix-server2-torch-cu129-runtime/handoffs/TO_GPT_20260814-164420.md
