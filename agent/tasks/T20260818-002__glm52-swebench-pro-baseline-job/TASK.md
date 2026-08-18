# T20260818-002__glm52-swebench-pro-baseline-job

task identity: T20260818-002__glm52-swebench-pro-baseline-job
objective: >
  Deliver a Git-pinned, restartable, unmodified GLM-5.2 NVFP4 one-instance
  SWE-bench Pro smoke package and an exact four-H200 Slurm job. The smoke must
  exercise official per-row evaluation while remaining non-score-eligible.
status: smoke_only_package_ready_not_submitted
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
  - the 2026-08-19 user scope override forbids running or transitioning to the full 731-row manifest
  - the smoke result is always labeled NON_SCORE_ELIGIBLE_SMOKE_ONLY
observed Herdr site contract:
  - partition H200; account gsai-account; qos hpgpu
  - one node and exactly --gres=gpu:H200:4; 32 CPUs; 512G; 06:00:00
  - Slurm logs under /home2/jslee202403/putpocket-slurm
  - allocated-node cache/artifacts under /local-data/jslee202403/putpocket-glm52-smoke
  - Login has no uv; the pinned uv 0.11.31 bootstrap is allocation-only and follows Docker preflight
  - load the observed cuda/12.9 module in the allocation and reject any nvcc version other than 12.9
  - /usr/bin/nvidia-smi is observed; Login's default CUDA 13.1 nvcc must never be used for the build
plan:
  - pin the official harness, submodules, dataset revision, image namespace, and licenses
  - use the deterministic one-instance smoke selection and local-vLLM mini-swe-agent scaffold overlay
  - run restartable inference, official patch gathering, and unchanged official per-row evaluation only
  - add exact-four-H200 Slurm rendering with fail-closed container preflight
  - validate on Montblanc with CPU/static/synthetic tests only
  - commit, push only the stacked task branch, inspect Login-1, and submit without integration
completion criteria:
  - official sources and all mutable benchmark identities are commit addressed
  - heavy stages refuse outside explicit Slurm allocation
  - smoke can never produce a score, >=40 claim, or transition to the full manifest
  - rendered job requests exactly one node/four H200 GPUs and never submits itself
  - focused and broad safe CPU-only tests pass
  - branch push is recorded exactly; the orchestrator submits through its authenticated Herdr pane
validation:
  - focused phase-1 plus phase-2 suite: 42 passed, 27 subtests passed
  - phase-2-only recheck after final edits: 19 passed, 3 subtests passed
  - broad CPU-only suite: 220 passed, 53 subtests passed
  - exact smoke-only renderer, compact wrapper, and tracked entrypoint pass bash syntax validation
phase-2 base source commit: 9e40d710b1432c6b8a05cf40611900b802b101a3
follow-up push: pending final commit
submission boundary: >
  Codex must not connect to Login or submit this follow-up. The orchestrator
  owns the authenticated Herdr pane and will stream the compact sbatch wrapper.
final handoff link: agent/tasks/T20260818-002__glm52-swebench-pro-baseline-job/handoffs/TO_GPT_20260818-120732.md
