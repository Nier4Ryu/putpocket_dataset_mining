# T20260812-001__runpod-native-build-parallelism

task identity: T20260812-001__runpod-native-build-parallelism
objective: finalize RunPod native build parallelism and Codex runtime policy
status: implementation_validated
base tip: a6ae12868e09eb03ca96d81f2cfefa640f1e3d1d
branch: agent/T20260812-001__runpod-native-build-parallelism
worktree: /workspace/putpocket_dataset_mining_worktrees/T20260812-001__runpod-native-build-parallelism
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
  - resolve runpod-dev build jobs with CLI > environment > nproc precedence
  - propagate one effective count to vLLM/CMake/Ninja/native extension environments
  - record CPU, parallelism, memory, timing, architecture, and fallback fields
  - validate focused regressions without a heavy build
  - publish and synchronize source before real build
  - install bubblewrap for compatibility while recording that RunPod blocks its required user namespace
  - reconcile persistent non-secret Codex config to danger-full-access/on-request without disturbing authentication or unrelated settings
completion criteria:
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - git diff --check: PASS
  - bash syntax: PASS
  - Python 3.13 compileall: PASS
  - focused pytest: PASS, 41 tests and 14 subtests
  - full unittest discovery: BLOCKED by absent project dependencies and unrelated image environment assumptions
  - explicit blackwell-rtx/32-job dry-run: PASS
  - current Pod nproc verification: observed 256; explicit PUTPOCKET_BUILD_JOBS=32 resolves all native job variables to 32
  - RunPod Codex config reconciliation: PASS, preserves unrelated TOML, creates one mode-0600 backup, is idempotent, and creates no credential file
  - secret-safety and .dockerignore review: PASS
  - real build: BLOCKED before execution; canonical Putpocket_env and externals/vllm are absent
artifacts:
  - agent/tasks/T20260812-001__runpod-native-build-parallelism/
commits:
  - a0b9e47ed8cde2d43110f683d24e91afd08417bb
  - final policy/closeout commit pending
final handoff link: agent/tasks/T20260812-001__runpod-native-build-parallelism/handoffs/TO_GPT_20260812-143556.md
