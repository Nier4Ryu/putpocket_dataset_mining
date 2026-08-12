# T20260813-001__publish-frozen-classeval-dataset

task identity: T20260813-001__publish-frozen-classeval-dataset
objective: publish-frozen-classeval-dataset
status: integrated
base tip: 6d5a10496bda4b03b37e71b72fcae36276651f4c
branch: agent/T20260813-001__publish-frozen-classeval-dataset
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260813-001__publish-frozen-classeval-dataset
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
  - CODE_CONFIRMED: frozen accepted.jsonl SHA-256 is 6031d368ee8359c9dfc3c7b785d5c30e4db9ae5b2969bfba3a7e09512a46b30d
  - CODE_CONFIRMED: frozen accepted row count is 18 with 18 unique sample IDs
  - CODE_CONFIRMED: staged package is limited to accepted.jsonl, dataset_manifest.yaml, and the 18 referenced source_task.json files
  - TEST_CONFIRMED: loader smoke loaded 18 accepted samples
  - TEST_CONFIRMED: compileall passed
  - TEST_CONFIRMED: unittest discovery for ClassEval support and GLM eval passed
  - TEST_CONFIRMED: focused pytest passed, 24 tests
  - HANDOFF_REPORTED: agent/tasks/T20260813-001__publish-frozen-classeval-dataset/handoffs/TO_GPT_20260813-070217.md
artifacts:
  - agent/tasks/T20260813-001__publish-frozen-classeval-dataset/
commits:
  - c7273c2d33443613f2bec476931baf006ce18ea9 data: publish frozen 18-sample ClassEval dataset
final handoff link: agent/tasks/T20260813-001__publish-frozen-classeval-dataset/handoffs/TO_GPT_20260813-070217.md
