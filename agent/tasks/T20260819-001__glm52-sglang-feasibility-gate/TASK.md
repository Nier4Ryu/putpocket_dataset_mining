# T20260819-001__glm52-sglang-feasibility-gate

task identity: T20260819-001__glm52-sglang-feasibility-gate

stacked worktree: `T20260818-001__glm52-cluster-package-foundation`

stacked branch: `agent/T20260818-001__glm52-cluster-package-foundation`

tracker references: Vikunja #2875 / AFFiNE xHYylBm6pP (orchestrator-owned; repository workers must not access either service)

## Objective

Deliver a Git-addressed, fail-closed SGLang-first diagnostic for the unmodified
`nvidia/GLM-5.2-NVFP4` checkpoint on one Slurm node and exactly four full H200
141GB GPUs. The authoritative diagnostic override retains allocation inventory,
weightless backend/config validation, and TP4 all-resident model load, then runs
one hard-pinned SWE-bench Pro coding prompt twice on the same server (trace OFF
and ON), captures bounded native DSA indexer evidence, gathers one patch, and
invokes the unchanged official evaluator for only that row.

## Boundaries

- The existing full benchmark adapter remains dormant and is unreachable from this task's renderer and compute entrypoint.
- The single official evaluator result is diagnostic metadata only. It is never a quality score and cannot transition to the full 731-row selection.
- The tracked SGLang patch instruments native score exposure only. It does not change model behavior, select with a second backend, or replace the fused path.
- No quality score, tracker mutation, Login execution, canonical integration, GPU fallback, or model-behavior patch is in scope.
- A failure never changes GPU count, model format, backend, host, quantization, or offload policy.
- The task is recorded separately on the existing stacked branch because it depends on the prior cluster-package commits; phase-1 and phase-2 history is not rewritten.

## Acceptance

- Exact one-node/four-H200/no-MIG/full-memory allocation proof precedes all model/config/weight access.
- Official SGLang source and runtime image are immutable-addressed and the installed runtime passes a weightless capability probe.
- Official model config resolves to a 40-character commit and exactly matches the architecture, 78-layer, 21/57 indexer, top-2048, and NVFP4 contract.
- Runtime uses TP4, ModelOpt FP4, Marlin, required DSA backends, 4096 context, concurrency one, and no MTP/disaggregation/offload/fallback.
- The exact 2,071-token pinned coding prompt is run with seed zero and greedy decoding OFF and ON on one server with radix caching disabled and cache flushes; output token IDs must match exactly.
- Native score evidence covers every full indexer layer at the last prefill query and existing decode samples 0/1/8/32, all four TP ranks, with the 57-layer sharing map preserved.
- Raw score non-exposure, coordinate ambiguity, output mismatch, coverage damage, or validation failure produces classified BLOCKED/FAIL artifacts rather than a substituted path.
- The unchanged official per-row scorer runs only for the pinned instance and its resolved result is recorded without score/threshold calculation.
- Backend evidence, OFF/ON overhead, per-run HBM peak/headroom, compressed record hashes, and a classified final manifest are retained.
- Static/unit and broad safe CPU-only tests pass; tracked shell scripts pass syntax checks and are executable.
- Source changes are committed and pushed on the stacked task branch without integration or force-push.

final handoff link: `agent/tasks/T20260819-001__glm52-sglang-feasibility-gate/handoffs/TO_GPT_20260819-120000.md`
