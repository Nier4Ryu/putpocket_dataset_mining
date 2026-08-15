# T20260815-001__repository-hygiene-post-vllm026

task identity: T20260815-001__repository-hygiene-post-vllm026
objective: repository-hygiene-post-vllm026
status: in_progress
base tip: 442e51b6f301c0a7052f633dcd3c8706d7dd1a84
branch: agent/T20260815-001__repository-hygiene-post-vllm026
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260815-001__repository-hygiene-post-vllm026
runtime mode: shared-python-overlay
write scope:
  - source/docs/tests required for this task
forbidden paths:
  - Putpocket_env/
  - data/
  - logs/
  - models/
  - .ssh/
fixed decisions:
  - canonical runtime checkout is /home/${USER}/putpocket_dataset_mining or /workspace/putpocket_dataset_mining
  - task worktrees live under /home/${USER}/putpocket_dataset_mining_worktrees or /workspace/putpocket_dataset_mining_worktrees
plan:
  - inventory root environments, root generated Markdown, and scripts/env
  - remove obsolete tracked root reports, legacy env scripts, and stale legacy configs
  - update active source, tests, and docs to the canonical env interface
  - validate frozen dataset, runtime imports, shell syntax, compileall, unittest, and focused pytest
  - integrate tracked cleanup, then delete explicit untracked legacy env directories
completion criteria:
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - git diff --check: PASS
  - git diff --cached --check: PASS
  - bash -n scripts/env/*.sh: PASS
  - compileall src tests: PASS
  - unittest discover: PASS, 181 tests
  - focused pytest: PASS, 72 tests and 14 subtests
  - canonical runtime regression: PASS
artifacts:
  - agent/tasks/T20260815-001__repository-hygiene-post-vllm026/
commits:
  - c10b12d01e1f2853604aaac5fce16e449afcc1de chore(repo): remove legacy environment artifacts
final handoff link: agent/tasks/T20260815-001__repository-hygiene-post-vllm026/handoffs/TO_GPT_20260815-162151.md
