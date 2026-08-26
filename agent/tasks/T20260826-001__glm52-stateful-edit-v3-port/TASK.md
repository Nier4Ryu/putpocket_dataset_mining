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
server follow-up objective:
  - add a distinct default-OFF live-donor request hook independent of prefix matching
  - allocate and patch target KV from copied donor rows plus exact sparse recompute rows
  - wire arbitrary target positions through the pinned GLM sparse indexer and FLASHMLA_SPARSE backend
  - preserve the legacy compute-then-overwrite path only as accuracy emulation
catalog follow-up objective:
  - save six provenance-complete, project-authored scenario templates for the next experiment driver
  - expose a digest-pinned JSON index and Draft 2020-12 schema with safe list/load/filter/validate APIs
  - preserve the exact ScaleAI/SWE-bench_Pro revision, smoke selection, and Ansible instance provenance
  - distinguish benchmark-native repository/problem/evaluator inputs from project-authored A1/Q2/edit transformations
  - keep every catalog entry non-runnable until a distinct frozen episode and donor/target server manifests exist
RunPod packaging follow-up objective:
  - publish only this isolated branch so a fresh RunPod can reproduce the pinned GLM/vLLM source chain
  - provide an ordered, fail-closed JSON environment doctor before any GPU capture
  - capture aligned full-candidate main-attention reference logits and native pre-top-k indexer scores under a distinct default-OFF diagnostic
  - analyze raw, rank, normalized-distribution, and top-k metrics without an invented scientific pass threshold
  - preserve the exact SWE-bench Pro row provenance while labeling the SYS edit and probe as PutPocket-authored
offline raw-indexer scoring follow-up objective:
  - preserve the sampled attention/indexer comparison unchanged and add a separate bounded full-row native indexer matrix capture
  - compute per-layer c_l,1=uA_l and c_l,n+1=c_l,nA_l in float64 with cumulative s_l,L and exact unnormalized layer sums
  - bind explicit frozen Q1/Q2 ranges, complete strict-causal rows, TP consensus, input/output hashes, and benchmark/authorship provenance
  - keep the scorer offline-only, default OFF, threshold-free, and outside vLLM inference decisions
registered next slice:
  - name: glm52-rope-aware-shifted-reuse
  - owner boundary: FLASHMLA_SPARSE main K-cache representation and DeepSeek V3.2 packed indexer representation
  - implement inverse-old/forward-new RoPE transformation for reused K rows whose aligned absolute position shifts
  - define whether the packed indexer K representation requires the same transformation
  - pass old/new positions into the row-copy kernel and attest the transformation per rank/layer/row
  - test equal/replacement/insertion/deletion, shifted/unshifted numerical parity, and Q2 decode
  - do not claim shifted reuse is RoPE-correct until that slice passes GPU integration
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
  - true-partial focused stateful/legacy CPU suite: 45 passed, 4 subtests passed
  - true-partial full repository CPU suite: 355 passed, 89 subtests passed
  - exact zero-context vLLM patch applies with locked --unidiff-zero to a fresh post-legacy source tree
  - all patched vLLM files and the installed instrumentation module pass py_compile
  - patch, instrumentation, schema, and all seven vLLM pre/post source hashes verified against the lock
  - sparse token selection, exact physical slot mapping, post-patch H boundary, P > H requirement, and Q2 block-boundary extension have focused CPU/static evidence
  - scenario catalog focused contract/server/loader suite: 25 passed
  - scenario catalog full repository CPU suite: 363 passed, 89 subtests passed
  - catalog schema, index, six scenario JSON files, exact SHA-256 bindings, CLI validate/list/show, Python compile, and git diff check passed
  - RunPod focused package/true-partial/legacy/catalog CPU suite: 46 passed
  - RunPod full repository CPU suite: 371 passed, 89 subtests passed
  - fresh exact vLLM source bootstrap applied the legacy base with GNU patch, both overlays with git apply --unidiff-zero, and matched every phase/postimage hash
  - RunPod Python compile, JSON/schema validation, shell syntax, project-artifact hashes, and synthetic TP4 score/report evidence passed
  - offline raw-indexer focused CPU suite: 24 passed
  - offline raw-indexer full repository CPU suite: 387 passed, 89 subtests passed
  - offline matrix schemas, deterministic float64 recurrence/report digests, package hashes, schedule, compile, shell, and diff checks passed
  - fresh exact vLLM chain installs the updated dual-mode score instrumentation and matches every locked postimage hash
artifacts:
  - agent/tasks/T20260826-001__glm52-stateful-edit-v3-port/
  - /home/dyryu/.cache/putpocket-handoffs/T20260826-001__glm52-stateful-edit-v3-port/montblanc-extract-3fdaa44d7fbd/
commits:
  - 9ced53fd605c0dd6c3c87f0e303e1fe6a256b3f2 (stateful v3 implementation)
  - b77c9514e296d711e249a1cb91adf2f2f2618d8b (model-neutral scenario contract follow-up)
  - 2e3f4fb6899c81cfbdc43098dbaf5bdb1187aef1 (server-side true-partial-prefill implementation)
  - d84bc3c5cf988b54a80a472032da7e44b8abfeff (machine-loadable scenario catalog)
  - current RunPod doctor and score-diagnostic packaging commit containing this record
final handoff link: agent/tasks/T20260826-001__glm52-stateful-edit-v3-port/handoffs/TO_GPT_20260826-033431.md
