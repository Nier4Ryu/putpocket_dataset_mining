# T20260826-001__glm52-stateful-edit-v3-port

task identity: T20260826-001__glm52-stateful-edit-v3-port
objective: glm52-stateful-edit-v3-port
status: complete
base tip: b9d44af05a464993001ecaf19acc07f188f072a7
branch: agent/T20260826-001__glm52-stateful-edit-v3-port
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260826-001__glm52-stateful-edit-v3-port
runtime mode: shared-python-overlay
write scope:
  - GLM-5.2 stateful edit source, instrumentation, runner, packaging, lock, docs, and CPU tests
forbidden paths:
  - Putpocket_env/
  - data/
  - logs/
  - models/
  - .ssh/
  - /home/dyryu/putpocket_dataset_mining
  - /home/dyryu/putpocket_dataset_mining_worktrees/T20260819-001__dsv4-edit-aware-compressed-kv
fixed decisions:
  - canonical runtime checkout is /home/${USER}/putpocket_dataset_mining or /workspace/putpocket_dataset_mining
  - task worktrees live under /home/${USER}/putpocket_dataset_mining_worktrees or /workspace/putpocket_dataset_mining_worktrees
plan:
  - reconstruct Login-1 stateful v3 sources and dynamic-history hook
  - port runner, lock, immutable packaging, and episode/smoke/full submission chain
  - validate with static and CPU-only repository tests
  - record isolation and handoff evidence
follow-up objective:
  - save a reusable model-neutral stateful mid-trajectory edit scenario contract
  - make top-ranked importance positions recompute under SYS_new while the unselected complement retains old KV
  - distinguish true sparse prefill from compute-then-overwrite accuracy emulation
completion criteria:
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - focused stateful/forced-reuse CPU suite: 23 passed, 5 subtests passed
  - broader GLM/cluster CPU suite: 106 passed, 34 subtests passed
  - full repository CPU suite: 338 passed, 90 subtests passed
  - git diff check, JSON parse, shell syntax, package heredoc compile, and Python compileall passed
  - authoritative bundle SHA-256 and Login-1 source postimage hashes verified
  - follow-up schema and generic YAML example validate under Draft 2020-12
  - follow-up focused docs/schema/GLM CPU suite: 37 passed, 28 subtests passed
  - follow-up full repository CPU suite: 341 passed, 89 subtests passed
  - follow-up JSON/YAML parse and git diff check passed
artifacts:
  - agent/tasks/T20260826-001__glm52-stateful-edit-v3-port/
  - /home/dyryu/.cache/putpocket-handoffs/T20260826-001__glm52-stateful-edit-v3-port/montblanc-extract-3fdaa44d7fbd/
commits:
  - 9ced53fd605c0dd6c3c87f0e303e1fe6a256b3f2 (stateful v3 implementation)
  - final documentation/contract follow-up commit containing this record
final handoff link: agent/tasks/T20260826-001__glm52-stateful-edit-v3-port/handoffs/TO_GPT_20260826-084555.md
