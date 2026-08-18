# T20260819-001__glm52-sglang-feasibility-gate

task identity: T20260819-001__glm52-sglang-feasibility-gate

stacked worktree: `T20260818-001__glm52-cluster-package-foundation`

stacked branch: `agent/T20260818-001__glm52-cluster-package-foundation`

tracker references: Vikunja #2875 / AFFiNE xHYylBm6pP (orchestrator-owned; repository workers must not access either service)

## Objective

Deliver a Git-addressed, fail-closed SGLang-first feasibility gate for the unmodified
`nvidia/GLM-5.2-NVFP4` checkpoint on one Slurm node and exactly four full H200
141GB GPUs. The gate ends after allocation inventory, weightless backend/config
validation, TP4 all-resident model load, and one deterministic correctness sentinel.

## Boundaries

- The existing benchmark adapter remains dormant and is unreachable from this task's renderer and compute entrypoint.
- No benchmark, score, tracker mutation, Login execution, canonical integration, or model-behavior patch is in scope.
- A failure never changes GPU count, model format, backend, host, quantization, or offload policy.
- The task is recorded separately on the existing stacked branch because it depends on the prior cluster-package commits; phase-1 and phase-2 history is not rewritten.

## Acceptance

- Exact one-node/four-H200/no-MIG/full-memory allocation proof precedes all model/config/weight access.
- Official SGLang source and runtime image are immutable-addressed and the installed runtime passes a weightless capability probe.
- Official model config resolves to a 40-character commit and exactly matches the architecture, 78-layer, 21/57 indexer, top-2048, and NVFP4 contract.
- Runtime uses TP4, ModelOpt FP4, Marlin, required DSA backends, 4096 context, concurrency one, and no MTP/disaggregation/offload/trace path.
- A deterministic one-shot response, backend evidence, HBM peak/headroom, and a classified final manifest are retained.
- Static/unit and broad safe CPU-only tests pass; tracked shell scripts pass syntax checks and are executable.
- Source changes are committed and pushed on the stacked task branch without integration or force-push.

final handoff link: `agent/tasks/T20260819-001__glm52-sglang-feasibility-gate/handoffs/TO_GPT_20260819-000000.md`
