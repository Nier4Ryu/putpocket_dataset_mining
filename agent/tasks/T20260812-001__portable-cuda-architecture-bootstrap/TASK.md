# T20260812-001__portable-cuda-architecture-bootstrap

task identity: T20260812-001__portable-cuda-architecture-bootstrap
objective: portable-cuda-architecture-bootstrap
status: closed
base tip: 817d422be4a7d4343f634f5b1729374bf2e9c965
branch: agent/T20260812-001__portable-cuda-architecture-bootstrap
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260812-001__portable-cuda-architecture-bootstrap
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
  - implement canonical CUDA architecture profile resolution
  - propagate resolved architecture list into vLLM, Docker, and manifests
  - validate dry-run profile regressions
  - update RunPod/bootstrap documentation
  - create task-local final handoff
completion criteria:
  - canonical portable profile resolves to 8.6 9.0 10.0 12.0
  - server1/server2/runpod preset defaults resolve to required lists
  - editable vLLM build plan exports TORCH_CUDA_ARCH_LIST
  - Docker build helpers emit torch_cuda_arch_list build arg
  - heavy multi-architecture build is not executed
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - git diff --check: PASS
  - bash -n scripts/env/*.sh and scripts/env/legacy/*.sh: PASS
  - compileall src tests: PASS
  - unittest discover: PASS, 133 tests, 6 skipped
  - focused pytest: PASS, 37 tests, 9 subtests
  - RTX PRO 6000 blackwell-rtx dry-run: PASS
  - portable-nvidia dry-run: PASS
  - Server-1 RTX3090 regression: PASS
  - Server-2 Blackwell regression: PASS
  - RunPod Hopper regression: PASS
artifacts:
  - agent/tasks/T20260812-001__portable-cuda-architecture-bootstrap/
  - /tmp/putpocket_arch_dryruns/
commits:
  - pending integration commit
final handoff link: agent/tasks/T20260812-001__portable-cuda-architecture-bootstrap/handoffs/TO_GPT_20260812-201926.md
