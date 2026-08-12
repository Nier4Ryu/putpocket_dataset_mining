# T20260812-001__runpod-agent-v2-publication

task identity: T20260812-001__runpod-agent-v2-publication
objective: runpod-agent-v2-publication
status: ready_for_close
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
  - implemented startup config-only reconciliation
  - built and pushed v2 Docker image
  - pinned v2 remote digest in RunPod metadata
  - validated
  - wrote handoff
completion criteria:
  - tests pass: done
  - task-local TO_GPT handoff exists: done
validation:
  - git diff --check: PASS
  - bash -n scripts/env/*.sh: PASS
  - bash -n cloud/runpod/*.sh: PASS
  - compileall: PASS
  - unittest discover: PASS, 148 tests, 6 skipped
  - focused pytest: PASS, 41 passed, 14 subtests passed
  - local image smoke: PASS
  - secret audit: PASS
artifacts:
  - agent/tasks/T20260812-001__runpod-agent-v2-publication/
  - nier4ryu/putpocket-runpod-dev:cuda12.9.1-ubuntu22.04-agent-v2@sha256:2185c3aa20246557347d8bdbba776bbf7b2f3438c1bba82a71b8e2404780653c
commits:
  - 4f3e310abc9d94d208f3319ecd5a12641e3b4d2e fix(runpod): reconcile Codex config in config-only startup
  - pending metadata commit
final handoff link: agent/tasks/T20260812-001__runpod-agent-v2-publication/handoffs/TO_GPT_20260812-235240.md
