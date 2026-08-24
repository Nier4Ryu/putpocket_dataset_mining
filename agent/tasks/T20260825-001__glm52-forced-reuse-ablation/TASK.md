# T20260825-001__glm52-forced-reuse-ablation

task identity: T20260825-001__glm52-forced-reuse-ablation
objective: >
  Implement and execute a default-OFF, unsafe forced-ratio GLM-5.2 edit-aware
  MLA/indexer cache transplant ablation for the pinned SR-CC-1 prompt.
status: in_progress
base tip: 6e8f8920ff5074ffeb7073223fa88fc3ea65ee0d
branch: agent/T20260825-001__glm52-forced-reuse-ablation
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260825-001__glm52-forced-reuse-ablation
runtime mode: isolated-native
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
  - preserve the RunPod container-not-found evidence and prior Track A/B artifacts
  - implement an exact-vLLM-4a3447d default-OFF transplant hook in isolated source
  - freeze a Track-B-score-derived downstream-token selector before outcomes
  - test prompt attestation, layer/product completeness, slot/page maps, and 0/100 endpoints
  - package a login-node-safe Slurm smoke gate and dependent full ratio sweep
  - submit only if Login-1 H200x4 assets, account, and allocation policy pass
completion criteria:
  - CPU/static/focused tests pass
  - actual cache consumption is evidenced or an exact fail-closed blocker is recorded
  - task-local TO_GPT handoff exists
validation:
  - pending
artifacts:
  - agent/tasks/T20260825-001__glm52-forced-reuse-ablation/
commits:
  - pending
final handoff link: pending
