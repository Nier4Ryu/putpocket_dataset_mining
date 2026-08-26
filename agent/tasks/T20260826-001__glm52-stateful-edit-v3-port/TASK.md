# T20260826-001__glm52-stateful-edit-v3-port

task identity: T20260826-001__glm52-stateful-edit-v3-port
objective: glm52-stateful-edit-v3-port
status: complete
benchmark evaluation status: not run; the completed RunPod score diagnostics are not a SWE-bench Pro task result or a stateful-cache quality evaluation
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
RunPod execution follow-up objective:
  - audit and use the dedicated 4xH200 RunPod from a unique persistent task root
  - replace the non-final four-row probe with a bounded all-Q1/Q2 content-token query-range capture and like-for-like query-sum report
  - freeze a deterministic two-query benchmark-derived fallback because no executed A1 artifact exists; prohibit stateful/cache-quality claims
  - execute the query-sum comparison and strict-causal matrix recurrence through level 6, retain raw evidence, transfer it to Montblanc, and plot only from copied artifacts
  - bind the score package to the audited NVFP4 model layout: 64 global main-attention heads and 32 TP-replicated Lightning indexer heads, with native indexer head scale `32^-0.5`
completion-audit follow-up objective:
  - commit every reusable execution-time implementation that was previously present only in the Montblanc artifact cache
  - preserve accepted RunPod originals while transferring only missing identity, retry, and root-cause evidence
  - build a deterministic remote-relative Montblanc inventory with SHA-256 verification and explicit reproducible-infrastructure exclusions
  - reproduce the accepted plots from checksum-attested raw reports using the committed generalized plotter
  - audit local and GitHub branch equality without merging, force-pushing, or changing canonical/DSV4 state
offline raw-indexer scoring follow-up objective:
  - preserve the sampled attention/indexer comparison unchanged and add a separate bounded full-row native indexer matrix capture
  - compute per-layer c_l,1=uA_l and c_l,n+1=c_l,nA_l in float64 with cumulative s_l,L and exact unnormalized layer sums
  - bind explicit frozen Q1/Q2 ranges, complete strict-causal rows, TP consensus, input/output hashes, and benchmark/authorship provenance
  - keep the scorer offline-only, default OFF, threshold-free, and outside vLLM inference decisions
rank-normalized multihop scientific-correction objective:
  - preserve the signed recurrence and its artifacts as immutable legacy evidence while rejecting it as a selector-relevance metric because sign cancellation and scale explosion obscure ranks
  - transform every layer/causal row to a nonnegative normalized transition, average normalized layers, and propagate the all-Q1/Q2 seed distribution with explicit backward-causal orientation
  - make top-K DCG rank weights at K 64 primary, sweep K 16/64/256, and retain population-z-score softmax at temperature 1 as a scale-sensitive nonnegative sanity variant
  - normalize every hop and report a separate equal-hop cumulative mixture, entropy/effective support, top-K stability against level 1 and the previous level, token rows, input attestations, and checksum-bound plots
  - keep the correction CPU-offline-only and out of vLLM scheduling, inference decisions, selector integration, and stateful-cache claims
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
  - live pinned NVFP4 config and vLLM `Indexer` audit corrected indexer provenance from 64 to 32 TP-replicated heads and native head scale from `64^-0.5` to `32^-0.5`
  - 32-head correction focused suite: 26 passed; full repository CPU suite: 387 passed, 2 skipped, 89 subtests passed
  - 32-head correction package/schema/compile/shell/diff checks and fresh exact three-overlay vLLM bootstrap/postimage validation passed
  - live harness preflight bound SWE-bench Pro outer commit `ca10a60a...` and its `mini-swe-agent` gitlink `d74716a...`; the scaffold path is submodule-qualified and hash-pinned
  - exact tokenizer preflight produced 2,105 prompt tokens, Q1 `[117,2069)`, and Q2 `[2088,2103)`; the corrected full candidate/propagation window is `[0,2103)` so the SYS edit at 114 and complete causal history are retained while only Q1/Q2 content rows seed the analyses
  - the reviewed 2,103-token window requires 8,841,012 matrix edge values/rank and 138,495,040 main-reference values/rank; bounded caps were raised to 2,176 tokens, 9,465,600 matrix values/rank, and 150,994,944 main-reference values/rank without truncation
  - live wrapper audit corrected the installed default-OFF vLLM diagnostic module's independent matrix-window ceiling to the same 2,176-token fail-closed package boundary
  - live environment preflight bound the doctor's exact Python distribution check to the observed CUDA-qualified `torch==2.13.0+cu129` wheel
  - live RunPod editable-build attempt 2 failed closed at DeepGEMM's host extension because `mqa_logits.cuh` names `__nv_fp8_e4m3` without including `cuda_fp8.h`; doctor and GPU launch remained blocked
  - a scoped third-overlay build fix now forces `cuda_fp8.h` into `tools/build_deepgemm_C.py`, with exact upstream/postimage hashes and a fresh zero-context patch-chain check; runtime inference behavior is unchanged
  - clean RunPod build attempt 3 passed all native targets and editable import; doctor attempt 3 then failed closed because its matrix-capture assertion searched a function body for a literal stored in module constant `MATRIX_MODE`, while all hardware/CUDA/NCCL/model/hash checks passed and Test 2 remained blocked
  - the doctor now validates both exact `MATRIX_MODE == "strict_causal_indexer_matrix"` and that the capture function dispatches through `MATRIX_MODE`; the capture implementation and inference behavior are unchanged
  - live query-sum attempt 4 loaded the exact NVFP4 model and generated one token but failed closed with `CAPTURE_FILES_MISSING`; audit proved `GlmMoeDsaForCausalLM` uses `DeepseekV2Model` while batch attestation was wired only to `Glm4MoeLiteModel`
  - the score overlay now wires fail-closed batch attestation into the actual DeepSeek/GLM DSA model path, pins its pre/postimage, and makes the doctor inspect that exact class before GPU capture
  - the DeepSeek/GLM hook correction passed 76 focused tests, the full `389 passed, 2 skipped, 89 subtests` pytest suite, 291 unittests, project hash/schema/compile/shell/diff checks, and a genuinely fresh exact three-overlay source-chain validation
  - live query-sum attempt 5 reached the corrected model hook but vLLM's position-zero synthetic memory-profile batch failed the frozen-prompt count assertion before serving a request; the hook now ignores only nonmatching-count initialization/decode batches while exact-count position and token-digest mismatches remain fail-closed
  - the profile-batch correction passed 61 focused tests plus project hashes/schema/compile/shell/diff checks, the full `391 passed, 2 skipped, 89 subtests` pytest suite, 291 unittests, and a brand-new exact three-overlay source-chain/postimage validation
  - live query-sum attempt 6 passed engine startup and emitted every layer-0 native indexer row on all four ranks, then failed closed before main-reference output because the exact model uses a 192-dimensional non-RoPE main Q/K component while the package incorrectly declared 128; the corrected package distinguishes main Q/K 192+64 from the independent 128-dimensional indexer head and makes the doctor validate the exact main layout
  - the main-layout correction passed 61 focused tests plus package hashes/schema/compile/shell/diff checks, the full `391 passed, 2 skipped, 89 subtests` pytest suite, 291 unittests, and a brand-new exact three-overlay source-chain/postimage validation
  - `REAL_EXECUTION_CONFIRMED`: RunPod attempt 7 passed the doctor and both ordered TP4 H200 diagnostics at science commit `d85582ad8d74f0280c6833cf0be19fe853d3f52f`; the doctor payload digest is `daba05d918deb69449d4a35d7a1c8fa6425d9739195d318ad9eb71def504489f`
  - attempt 7 retained 1,967 Q1/Q2 query rows across layers 0/22/46/74 and all four ranks, plus 264 strict-causal matrix chunks containing 8,412 rows/rank and 8,841,012 native edge values/rank; TP consensus max absolute difference was zero
  - exact signed float64 multihop recurrence passed independently per layer through level 6, retained every hop and cumulative value, and remained finite without clipping, normalization, or selector integration
  - corrected offline query-sum report v2 commit `ccc5de08df139b9b5a81b87f6c6d3e995909c4a6` replaced saturated native-softmax top-k/JS interpretation with raw descending top-k and independently z-scored-softmax JS while retaining native-softmax support diagnostics
  - analyzer correction focused suite: 32 passed; full repository CPU suite: 393 passed, 2 skipped, 89 subtests passed; 291 unittests passed; schema/JSON/compile/shell/diff/package checks passed
  - authoritative v2 raw-capture replay passed with 1,967 seed rows, candidate window `[0,2103)`, report file SHA-256 `64937285cd24fe1a902879ce5abe3f176e8996d445d19451d633c7330021bf95`, and payload SHA-256 `38e1cb105e49074fa6daeb4b9ec108c4fc01d81d87005c35e0c61c549153181c`
  - the immutable 276-file raw artifact manifest passed before transfer; the exact 4,898,098,644-byte RunPod bundle and Montblanc copy both hash to `855cf68457cd4540f06e3767d03c6e997362d1118b37ca90d0539a2b01889949`
  - the corrected derived bundle transferred independently with matching SHA-256 `3cbff97a76ce530c88401dc7e50427b7cb7f3a651896ec3ae73a977c50fbe405`; all seven internal hashes, the v2 schema/payload digest, and the consolidated Montblanc manifest passed
  - five required plots were generated as PNG and PDF only from transferred evidence, visually inspected, and verified against `PLOT_SHA256SUMS`; plot summary SHA-256 is `89b04faf72c63d35dd2193c6c15b4cc127df4729b883f5f9a3a14ddce00e1ff8`
  - the frozen input is explicitly a deterministic benchmark-derived two-query score probe with project-authored bridge/Q2, not an executed A1-to-tool-observation episode; no stateful-cache, benchmark-outcome, task-quality, or native-FP4-compute claim is allowed
  - completion audit found one reusable cache-only implementation, `plot_glm52_score_evidence.py` SHA-256 `e1f5cdd967a48514750d9c3074a8ca4a33cd217060e9a7b5cbadf9c0df03363f`; commit `770ff33` saves a checksum-attested, path-independent implementation plus schema, tests, and pinned plotting dependencies
  - two host-bound transfer prototypes, SHA-256 `42b26b61940fb6ed5aec2d384890387416567f9e04c1dc2df12ce214270a5ad7` and `26387857faf6225b76bafac58d55f7b5e0ef6cfe33efcc35682318b93dbdce70`, were generalized into a no-overwrite, size/SHA-256-verified framed transfer helper without committing an address, credential, or host path
  - the completion delta retained 92 missing remote-relative files (21,494,149 bytes) covering environment/model/source identity, build/doctor setup, immutable tokenizer/config evidence, and essential failed-attempt root-cause logs; its 3,284,795-byte archive hashes to `4cb095d4ba6e21fb9ba062f875015ee4ea307fc48d105a55a193f6230fb8b411`
  - the completion delta excludes the already-local accepted 13 GB raw tree and four 62.8 MB failed attempt-6 rank partials; all selected remote source hashes and the local archive hash passed, and RunPod originals were preserved
  - repository-reproduced plots passed all 11 `PLOT_SHA256SUMS` entries from the corrected v2 query-sum and accepted level-6 multihop inputs; the new plot summary SHA-256 is `8fbfa44b96c2a0982052b8b75e208a4001dca4fd499955f26d454074070e737c`
  - the consolidated Montblanc inventory covers 453 files and 18,009,870,455 bytes with payload SHA-256 `30da1f00612d5e937da315dbf6f7b0aaad44f5378e0b8fe672a8d759b9fdd839`; its JSON file hashes to `5749dca569b59e3f4d9b913166e9887eb2fcffc20a535da1ebce0c42c5f39a31` and `COMPLETION_AUDIT_SHA256SUMS` hashes to `cd823013a9f88de113f1a9053fa665bae261daf90567adfce60783834248f3a4`
  - completion-audit focused suite: 37 passed; full repository CPU suite: 398 passed, 2 skipped, 89 subtests passed; Python compile, JSON Schema, shell syntax, package hashes, secret/large-file scope, and cumulative git diff checks passed
  - rank-normalized correction focused scientific/package suite: 43 passed; full repository CPU suite: 407 passed, 2 skipped, 89 subtests passed
  - rank-normalized source/report/token/plot schemas, Python compile, package artifact hashes, git diff checks, secret/large-file scope, and synthetic orientation/tie/normalization/regression tests passed
  - accepted-data CPU replay used the complete checksum-attested `[0,2103)` matrix, layers 0/22/46/74, and exactly 1,967 Q1/Q2 seed positions; every hop and equal-hop cumulative distribution through level 6 had mass 1 within `1e-10`
  - primary `rank_dcg_k64` level-6 top-64 retention/Jaccard was `0.328125/0.196262` versus level 1 and `0.96875/0.939394` versus level 5 for hop-only; equal-hop cumulative was `0.5/0.333333` versus level 1 and `0.953125/0.910448` versus level 5
  - K 16/64/256 and z-score-softmax sensitivity all showed the same qualitative result: progressive drift from level 1, high adjacent-level stability, and stronger level-1 retention for the cumulative mixture than the hop-only distribution
  - the new 14-file, 8,847,690-byte artifact passed both checksum manifests; report payload SHA-256 is `63c65db17ff8e8190d8791d0b1160929081316c5d72a919bbea606808d8d8780`, report SHA-256 is `cc00b6f41e79873b2258cdb17bb3bb5e86324799762c6953db373e8a3c992bfe`, and `FINAL_SHA256SUMS` SHA-256 is `1a9e367ef7a6732f7980a16ffc8d52e51cea4a937877b567c71e280987d4be8f`
artifacts:
  - agent/tasks/T20260826-001__glm52-stateful-edit-v3-port/
  - /home/dyryu/.cache/putpocket-handoffs/T20260826-001__glm52-stateful-edit-v3-port/montblanc-extract-3fdaa44d7fbd/
commits:
  - 9ced53fd605c0dd6c3c87f0e303e1fe6a256b3f2 (stateful v3 implementation)
  - b77c9514e296d711e249a1cb91adf2f2f2618d8b (model-neutral scenario contract follow-up)
  - 2e3f4fb6899c81cfbdc43098dbaf5bdb1187aef1 (server-side true-partial-prefill implementation)
  - d84bc3c5cf988b54a80a472032da7e44b8abfeff (machine-loadable scenario catalog)
  - d85582ad8d74f0280c6833cf0be19fe853d3f52f (RunPod integration fixes and exact live science package)
  - ccc5de08df139b9b5a81b87f6c6d3e995909c4a6 (offline query-sum saturation correction)
  - 770ff33 (completion-audit reproducibility implementation)
  - 401f8a4f7dd12bd7c85f4129f09c570399eded1b (rank-normalized multihop implementation, schemas, plots, docs, and tests)
  - current evidence-only RunPod execution handoff commit containing this record
final handoff link: agent/tasks/T20260826-001__glm52-stateful-edit-v3-port/handoffs/TO_GPT_20260826-215354.md
