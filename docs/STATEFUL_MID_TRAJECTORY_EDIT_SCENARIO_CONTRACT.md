# Stateful mid-trajectory edit scenario contract

Status: normative, model-neutral authoring contract for SWE-bench and
SWE-bench Pro scenarios.

Machine-readable companions:

- schema: `configs/cluster/schemas/stateful_mid_trajectory_edit_scenario.schema.json`
- authoring example: `configs/cluster/stateful_mid_trajectory_edit_scenario.example.yaml`

This contract turns a normally single-turn benchmark instance into a
deterministic edit episode. It defines scientific state and accounting; it does
not authorize model execution, container execution, or benchmark submission.

## 1. Normative state and selector orientation

The reusable baseline is the KV state already produced for the frozen old
trajectory:

```text
F_old = serialize([SYS_old, Q1, A1])
H     = token_count(F_old)
```

`Q2` is intentionally absent from `F_old`. Its content is frozen after the A1
tool action, but its tokens have no old-trajectory KV in this scenario.

Let:

- `E` be edited token positions and any positions invalidated directly by the
  edit mapping;
- `M` be the complete mandatory-recompute set, including `E` and any declared
  alignment, page, dependency, or implementation closure;
- `D` be eligible downstream frozen-history positions, normally a declared
  half-open range after the edit and before `H`, minus `M`;
- `pi` be a permutation of `D`, ordered from highest to lowest importance for
  recomputation under `SYS_new`;
- `S_b` be the first `k_b` positions of `pi` for budget `b`;
- `U_b = D - S_b` be the positions that retain or reuse old KV.

The required final frozen-history state for budget `b` is:

```text
new KV: M union S_b
old KV: U_b, plus any separately attested unchanged prefix
```

The top of an importance/indexer/selector ranking is therefore selected for
**recomputation**, never for donor reuse. A conforming manifest MUST name the
ranked quantity `importance_for_recompute`, set the direction to
`higher_score_first`, and persist both `pi` and the actual `S_b` lists in
selection order. Native scores with the opposite direction may be transformed
only by a frozen, documented, outcome-independent transform.

“Reuse ratio” and “recompute ratio” are complementary accounting labels:

```text
recompute_ratio = |S_b| / |D|
reuse_ratio     = |U_b| / |D| = 1 - recompute_ratio
```

`M` is excluded from that denominator because it is always recomputed. Ratio
names alone are not evidence. The manifest's exact ordered recompute positions,
sorted recompute set, retained-old-KV set, and complement attestation are the
authority.

## 2. Deterministic episode construction from a benchmark row

Pin the benchmark dataset revision, instance ID, repository base commit,
container image digest, official harness/evaluator commit, agent version,
model revision, tokenizer revision, chat template, decoding settings, and tool
protocol before construction.

Construct one episode as follows:

1. Create a clean repository and container from the pinned benchmark instance.
   Derive `Q1` only from the public issue/problem statement and permitted row
   metadata. Do not expose tests, gold patches, or evaluator outcomes.
2. Render `[SYS_old, Q1]` with the pinned serializer and tokenizer. Generate A1
   once with deterministic decoding. Require the declared bounded tool action
   or action sequence and freeze the exact backend response, assistant message,
   structured tool request, and their digests.
3. Execute the frozen A1 action in that clean repository/container. Freeze the
   exact resulting tool observation as Q2, together with command, exit status,
   stdout/stderr or structured result, repository tree/status digest, and Q2
   message digest.
4. Serialize `[SYS_old, Q1, A1]` as `F_old`. Apply only the declared edit to
   produce `[SYS_new, Q1, A1]` as `F_new`. Freeze both token arrays, edit/alignment
   map, `H`, mandatory set `M`, eligible set `D`, and their digests.
5. Freeze selector evidence and every budget before generating A2 or reading
   any patch, test, score, resolution, or evaluator output.
6. For each method and budget, start a fresh copy of the same pinned repository
   and container. Replay the exact A1 response/action, require the resulting Q2
   and repository digest to match, patch the old KV according to `M`, `S_b`, and
   `U_b`, then issue the first post-edit request:

   ```text
   [SYS_new, Q1, frozen A1, frozen Q2] -> A2
   ```

7. Continue the agent normally from A2 and run the unchanged official
   benchmark evaluator in fresh evaluator state.

The episode-construction run is evidence generation, not a scored budget. A1,
its action, and Q2 are generated/executed once for freezing and then replayed
identically for every compared method and budget.

## 3. Exact replay and no outcome leakage

Every compared run MUST have the same:

- benchmark row and source revision;
- initial filesystem/container state;
- `SYS_old`, `SYS_new`, Q1, frozen A1 backend response, tool request, tool
  result/Q2, and repository digest after A1;
- model/tokenizer/chat-template revisions and deterministic decoding fields;
- selector source evidence, aggregation, tie-break, selection order, and budget
  definitions;
- official evaluator revision and invocation.

Fail if A1 is regenerated, a replayed command differs, Q2 differs bytewise or
structurally, repository state differs, or a method receives a different
pre-edit trajectory. Network-derived and time-varying tool actions are not
acceptable unless their complete result is hermetically replayed and attested.

Selector construction MUST finish before any A2 generation or official outcome
is observed. The following may not influence scores, ties, budgets, or scenario
acceptance: candidate patches, hidden/public test results, resolved booleans,
judge output, decode length, latency, or any post-edit model output.

## 4. Edit, tokenization, and position invariants

Persist the exact serialized text/messages and token IDs for `F_old`, `F_new`,
and the first post-edit request. At minimum validate:

- one pinned tokenizer and chat template produced every array;
- edit message/path and text spans match the authoring declaration;
- token edit positions, old IDs, new IDs, and an old-to-new alignment are exact;
- same-position KV retention is used only for equal-length aligned positions;
- insertion/deletion scenarios provide an explicit mapping and validated RoPE
  treatment, or fail closed as unsupported;
- `M` contains every edited token and all declared closure positions;
- `D` contains only aligned positions in frozen history, is disjoint from `M`,
  and ends at `H`;
- `pi` is a unique permutation of `D`;
- every `S_b` is the exact prefix of `pi` of the declared size;
- every `U_b` is exactly `D - S_b`;
- no Q2 token or genuinely later token appears in `M`, `D`, `pi`, `S_b`, or
  `U_b`.

The unchanged prefix before the edit may retain old KV when its token identity,
position, and dependency contract are attested. It is not part of the downstream
budget denominator.

## 5. Q2 boundary versus genuinely later tokens

Q2 has two properties that must not be conflated:

1. **Content provenance:** Q2 is the frozen pre-edit tool observation obtained
   by executing A1 under `SYS_old`.
2. **KV provenance:** Q2 is first tokenized/prefilled in the first post-edit
   model request, after the frozen-history cache patch. It has no reusable
   `F_old` KV and runs normally against the patched cache.

Record `q2_token_range = [H, P)` in the first post-edit request, where `P` is
the generation boundary for A2. Tokens at positions `P` and later are genuinely
later tokens: A2 decode tokens, later tool observations, and later turns. They
also run normally and never enter the frozen-history selector.

## 6. Importance evidence and frozen manifest

The selector section MUST record:

- the semantic quantity being ranked (`importance_for_recompute`);
- native score source and exact capture stage;
- model/indexer/layer/query/TP-rank coverage and source artifact digests;
- aggregation and any score-direction normalization;
- deterministic tie-break, normally absolute position ascending after all
  scientific fields;
- the full selection-order permutation `pi`;
- per-budget `k_b`, ordered `S_b`, sorted `S_b`, sorted `U_b`, and digests;
- proof that it was frozen before A2 and outcomes.

Scores that represent stability or reuse suitability are not importance scores.
They must be labeled as such and MUST NOT be described as “top importance for
reuse.” A new conforming scenario either derives an outcome-independent
importance-for-recompute score with an explicit transform or rejects the score
source.

Recommended immutable digests include canonical JSON message arrays, token-ID
arrays, tool request/result, repository tree/status, selector scores/order,
each budget set, runtime backend/package, generated patch, and evaluator result.
Declare the canonicalization used for structured values. The companion schema
requires RFC 8785 JSON Canonicalization Scheme encoded as UTF-8; hash raw bytes
only for explicitly identified file artifacts. Each ordered, sorted, and
complement position array has its own named digest so selection order cannot be
lost through sorting.

## 7. Runtime backends and claim boundary

### True selective prefill

A compute-saving backend starts from existing `F_old` KV. At every model layer
it recomputes `M union S_b` under the edited prompt using the backend's declared
hybrid dependency semantics, writes those sparse rows into the existing cache,
and leaves `U_b` unchanged. Layer, product, rank, logical position, physical
slot, and write evidence must prove that only declared rows were computed and
patched. Only after the frozen-history patch is complete may Q2 be prefetched
normally and A2 decoded.

Any causal closure, page granularity, or kernel constraint that forces extra
work must expand `M` or an explicitly reported effective recompute set. Silent
full prefill is noncompliant with compute-saving claims.

### Accuracy-emulation/reference backend

A backend may compute the full target prefill under `SYS_new` and then overwrite
the positions in `U_b` with donor/old KV, leaving target/new rows at `M union
S_b`. This may emulate the requested final cache state for accuracy studies.
Its manifest MUST set:

```text
backend_kind: accuracy_emulation_compute_then_overwrite
compute_saving_claim: false
true_selective_prefill_claim: false
```

It must also record actual donor-overwrite positions as `U_b`; high-importance
`S_b` positions remain target/new rows. Such a run cannot report skipped FLOPs,
selective-prefill latency, or production speedup.

The current GLM-5.2 v3 hook is a legacy stability-ranked forced-reuse accuracy
ablation. Its selected donor rows describe reuse suitability, not
importance-for-recompute, and it does not by itself conform to this contract's
true selective-prefill semantics. See
`docs/CLUSTER_GLM52_STATEFUL_EDIT_V3.md` for the explicit reference boundary.

## 8. Official evaluation and retained evidence

Each method/budget starts from a fresh repository/container, replays A1, attests
Q2 and repository state, applies its cache method, runs the post-edit agent, and
is evaluated by the pinned unchanged official SWE-bench or SWE-bench Pro
evaluator. Never reuse a mutated repository across budgets.

Retain at minimum:

- scenario manifest, schema version, and checksums;
- benchmark row identity and source/harness/evaluator revisions;
- clean-state and post-A1 repository/container attestations;
- exact messages, rendered prompts, token IDs, edit/alignment maps, and digests;
- frozen A1 response/action and Q2 observation;
- selector source evidence, scores, order, and budget sets;
- backend kind, source/package digest, command/config, per-layer/product/rank
  sparse-write or donor-overwrite evidence, and effective sets;
- full agent trajectory, generated patch, evaluator logs, official result, and
  a run-level checksum manifest.

## 9. Fail-closed checklist

Reject the scenario or run if any answer is no:

- Are dataset, instance, repository, container, harness, evaluator, model,
  tokenizer, and chat template pinned by immutable revision/digest?
- Was A1 generated once, was its exact declared tool action executed, and were
  A1/Q2/repository state frozen before budgets?
- Does every run replay identical A1/Q2 from fresh state?
- Are `F_old`, `F_new`, edit positions, alignment, `H`, Q2 range, and generation
  boundary exact and digest-bound?
- Are `M` and `D` disjoint, complete, in bounds, and free of Q2/later tokens?
- Is `pi` a permutation of `D`, highest importance first, with deterministic
  outcome-independent provenance and tie-breaking?
- Does each `S_b` equal the requested prefix of `pi`, and does each `U_b` equal
  its exact complement in `D`?
- Does runtime evidence match the declared backend and actual row sets on every
  required layer, product, rank, position, and slot?
- Are accuracy emulation and true selective prefill labeled distinctly, with no
  compute-saving claim from compute-then-overwrite?
- Did the unchanged official evaluator run in fresh state and produce one
  complete result for every required budget?

## 10. Scenario acceptance criteria

A scenario is authoring-complete only when:

1. the machine-readable manifest validates against the committed schema;
2. all cross-field set, prefix, complement, boundary, alignment, and digest
   invariants above pass;
3. episode and selector artifacts are frozen before outcomes;
4. at least one declared runtime backend can enforce its own claim boundary;
5. replay and official-evaluator procedures are executable from pinned inputs;
6. the complete evidence inventory and checksum policy are declared.

The committed YAML example is illustrative, uses small synthetic token
coordinates, and intentionally contains no GLM-specific token or system-text
constant. Copy it to a task-local scenario path, replace every placeholder with
real pinned evidence, validate it, and freeze it before any scored run.
