# T20260818-002__glm52-swebench-pro-baseline-job

task identity: T20260818-002__glm52-swebench-pro-baseline-job
objective: >
  Deliver a Git-pinned, restartable, unmodified GLM-5.2 NVFP4 baseline package
  for the public SWE-bench Pro test split and an exact four-H200 Slurm job.
status: in_progress
stacked base tip: 819854e28ae170ef43722118dfc3d2a53f43c7ce
branch: agent/T20260818-001__glm52-cluster-package-foundation
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260818-001__glm52-cluster-package-foundation
stacking note: >
  The repository task CLI has no same-worktree resume/start operation. This
  phase-2 task is therefore recorded separately while intentionally sharing
  the clean phase-1 branch and worktree, without rewriting phase-1 history.
runtime mode: isolated-native
write scope:
  - phase-2 source, configs, docs, tests, and task-local handoff
forbidden paths:
  - Putpocket_env/
  - data/
  - logs/
  - models/
  - .ssh/
fixed decisions:
  - no Vikunja or AFFiNE access; the orchestrator alone owns external task state
  - unmodified nvidia/GLM-5.2-NVFP4 baseline; no cache/model behavior patches or custom quantization
  - one Slurm node with exactly four H200 GPUs
  - TP1+PCP4+EP primary; TP2+PCP2+EP only for classified startup incompatibility
  - ScaleAI/SWE-bench_Pro test split and unchanged official swe_bench_pro_eval.py scoring
  - official Docker images are jefzda/sweap-images with each row's dockerhub_tag
  - external downloads, installs, model load, inference, and evaluation require a compute allocation
  - Login-1 work is limited to lightweight inventory and sbatch submission
plan:
  - pin the official harness, submodules, dataset revision, image namespace, and licenses
  - add deterministic smoke/full selection and a local-vLLM mini-swe-agent scaffold overlay
  - add restartable inference, official patch-gathering, evaluation, and full-only score gates
  - add exact-four-H200 Slurm rendering with fail-closed container preflight
  - validate on Montblanc with CPU/static/synthetic tests only
  - commit, push only the stacked task branch, inspect Login-1, and submit without integration
completion criteria:
  - official sources and all mutable benchmark identities are commit addressed
  - heavy stages refuse outside explicit Slurm allocation
  - smoke can never produce a >=40 claim and full results require complete official coverage
  - rendered job requests exactly one node/four H200 GPUs and never submits itself
  - focused and broad safe CPU-only tests pass
  - branch push and Login submission outcomes are recorded exactly
