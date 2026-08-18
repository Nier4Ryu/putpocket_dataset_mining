# T20260818-001__glm52-cluster-package-foundation

task identity: T20260818-001__glm52-cluster-package-foundation
objective: >
  Deliver the phase-1 Git foundation for provider-neutral Cluster Center
  execution of full GLM-5.2 under Slurm: source/profile/environment contracts,
  dry rendering, allocation guards, provenance, and staged readiness only.
status: in_progress
base tip: 6e8f8920ff5074ffeb7073223fa88fc3ea65ee0d
branch: agent/T20260818-001__glm52-cluster-package-foundation
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260818-001__glm52-cluster-package-foundation
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
  - no Vikunja or AFFiNE access; the orchestrator alone owns external task state
  - no network, dependency/model/checkpoint download, GPU query, or remote Cluster access on Montblanc
  - current clean vLLM v0.26 source is pinned at 568afb3a13806beb53bb2e6bd518269357b237c0
  - phase 2 SWE-bench Pro >=40 adapter and phase 3 SM90 tracing are explicit later tasks
plan:
  - define Cluster package schemas, profiles, environment lock, and Git artifact exclusions
  - implement render-only Slurm jobs and allocation-guarded heavy actions
  - implement staged static/allocation/GPU/import/checkpoint/backend/model-load/generation-handoff readiness
  - validate with CPU/static/synthetic tests only
  - commit and hand off without integration or push
completion criteria:
  - three required GLM-5.2 profiles validate
  - Slurm rendering never submits and embeds no provider site values
  - all heavy action classes refuse outside an explicit allocation
  - provenance is secret-safe and never hashes checkpoint tensors
  - focused and broad safe CPU-only tests pass
  - task-local TO_GPT handoff exists
validation:
  - focused: 23 passed, 22 subtests passed
  - broad CPU-only (GPU-querying module excluded): 201 passed, 50 subtests passed
artifacts:
  - agent/tasks/T20260818-001__glm52-cluster-package-foundation/
  - configs/cluster/
  - configs/env/cluster_h200_sm90_vllm026.lock.yaml
commits:
  - pending
final handoff link: pending
