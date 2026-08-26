# GLM-5.2 stateful mid-trajectory edit sweep (v3)

The ordered RunPod environment-doctor and main-attention/indexer score package
is documented in
[`RUNPOD_GLM52_ATTENTION_INDEXER.md`](RUNPOD_GLM52_ATTENTION_INDEXER.md).
Its final score probe is a deterministic benchmark-derived Q1 plus
project-authored bridge and Q2, tied to the recommended equal-position
scenario. It captures every Q1/Q2 content-token query row but has no executed
A1 or tool-observation Q2 and must not be cited as a stateful continuation
result. The older four-row `SYS_new + Q1` mode is retained only as a non-final
sampled diagnostic. That package also has a separate bounded full-row raw-indexer
matrix capture and strictly offline multi-hop scorer. Unlike the original
sampled comparison, its frozen input includes explicit Q1 and frozen-Q2 token
ranges. It is not part of true-partial scheduling and cannot change runtime
selection or inference.

This chain evaluates a real edit event, not independent runs that start with the
new system prompt. For authoring new model-neutral SWE-bench or SWE-bench Pro
episodes, use the normative
[`STATEFUL_MID_TRAJECTORY_EDIT_SCENARIO_CONTRACT.md`](STATEFUL_MID_TRAJECTORY_EDIT_SCENARIO_CONTRACT.md).
The project-internal templates, exact benchmark provenance, loader API, and
readiness classifications for the next experiment script are in
[`STATEFUL_EDIT_SCENARIO_CATALOG.md`](STATEFUL_EDIT_SCENARIO_CATALOG.md).
The GLM v3 experiment below predates that contract and is retained as a legacy
accuracy-emulation reference.

1. The capture job freezes the original 2,071-token equal-length edit contract
   and the base DSA selector evidence.
2. The episode job runs `SYS_old + Q1 -> A1` exactly once, executes the one A1
   shell action, and freezes A1 plus the resulting Q2 observation. It serializes
   `[SYS_old,Q1,A1]` and `[SYS_new,Q1,A1]` and requires that token 114 is their
   only difference.
3. Every ratio starts a clean, identical SWE-bench Pro container. The proxy
   returns the frozen A1 response for turn 1, verifies that executing it produces
   the frozen Q2, then replaces only the system edit sentence before forwarding
   turn 2 to GLM-5.2.
4. The legacy reuse denominator is every eligible frozen-history token in
   positions `[115, history_token_count)`, including A1. Token 114 is mandatory
   target recompute and is never a donor row. Q2 is the frozen pre-edit tool
   observation included in the first post-edit request, but it has no donor KV;
   Q2 and every genuinely later post-edit token run normally after the cache
   patch.
5. Ratio 0 and every nonzero ratio use the official single-instance SWE-bench
   Pro evaluator. The full sweep remains dependent on the ratio-0/10 smoke job.

The transplant hook computes full target KV and then overwrites its legacy
donor-selected rows from the old snapshot. Therefore the result is an
accuracy-emulation ablation only; it is not true selective prefill and makes no
latency, skipped-FLOP, compute-saving, or production-safe gating claim.

The base prompt positions preserve a pre-outcome native DSA ranking oriented to
stability/reuse suitability. A1 did not exist during that capture, so its
positions receive fixed SHA-256 quantiles and are interleaved with that legacy
donor ranking before any ratio outcome is evaluated. These selected positions
are donor rows for this experiment; they MUST NOT be described as the most
important positions to recompute.

New scenarios use the opposite, explicit scientific orientation: rank eligible
positions by `importance_for_recompute`, recompute the top-ranked prefix under
`SYS_new`, and retain old KV only at the unselected complement. If this final
state is emulated with full target prefill plus donor overwrite, the donor rows
are that complement, and the run remains a no-speedup reference backend.

## Audited server architecture

The exact stateful-v3 server source is vLLM commit
`4a3447d200e5aa428d68d1a00aa00f1a19a1a729`, GLM architecture
`GlmMoeDsaForCausalLM`, `FLASHMLA_SPARSE`, BF16 KV, block size 64, TP=4,
PP=1, 78 main MLA cache layers and 21 packed sparse-indexer cache layers.
Prefix caching and chunked prefill are disabled and the server is eager with
one sequence at a time.

The legacy overlay has three post-write hooks. A model hook attests the full
prompt, `unified_mla_kv_cache_update` snapshots or overwrites main MLA rows
after the native cache write, and the sparse indexer hook does the same for its
packed value and scale planes. A transplant request has already executed the
complete target prompt before donor rows are restored. Its private cloned
snapshots are not live donor request blocks. That architecture is the
`accuracy_emulation_compute_then_overwrite` backend and remains separate.

The maintained true-path overlay is:

- `instrumentation/vllm/putpocket_true_partial_prefill.py`
- `patches/vllm/4a3447d200e5aa428d68d1a00aa00f1a19a1a729/putpocket_true_partial_prefill.patch`
- `configs/cluster/schemas/vllm_true_partial_prefill_server_manifest.schema.json`

This generated overlay has the exact post-legacy source as its required
packaging base. Packaging must apply the verified legacy patch first, then the
true-path patch, and install the instrumentation module at
`vllm/v1/putpocket_true_partial_prefill.py`. That source dependency does not
combine the runtime modes: the legacy emulation remains inactive unless its
own control is set, and the true hook refuses to start when
`PUTPOCKET_GLM52_FORCED_REUSE_CONTROL` is set. The two modes are isolated and
mutually exclusive at runtime.

The true-path artifact is intentionally a whitespace-clean zero-context patch
so repository `git diff --check` remains strict. Apply it with its locked
argument; plain `git apply` is not the packaging contract:

```bash
git apply --check --unidiff-zero putpocket_true_partial_prefill.patch
git apply --unidiff-zero putpocket_true_partial_prefill.patch
```

## Client/server boundary and request hook

The model-neutral scenario manifest stays on the client. A client-side resolver
chooses one frozen budget and emits two server manifests: a donor registration
manifest and a target sparse manifest. Neither manifest contains SWE-bench
turn names, tool semantics, repository state, or outcomes. The server accepts
only token/cache coordinates and never derives an alignment from prefix
matching.

The feature is default OFF. A dedicated server must set all of:

```text
PUTPOCKET_VLLM_TRUE_PARTIAL_PREFILL_ENABLE=1
PUTPOCKET_VLLM_TRUE_PARTIAL_MANIFEST_ROOT=/absolute/read-only/manifest/root
PUTPOCKET_VLLM_TRUE_PARTIAL_EVIDENCE_ROOT=/absolute/write-only/evidence/root
PUTPOCKET_VLLM_TRUE_PARTIAL_TOKENIZER_IDENTITY_SHA256=<sha256>
PUTPOCKET_VLLM_TRUE_PARTIAL_SERIALIZER_IDENTITY_SHA256=<sha256>
PUTPOCKET_VLLM_SOURCE_COMMIT=4a3447d200e5aa428d68d1a00aa00f1a19a1a729
```

Each donor or target API request then supplies exactly these `vllm_xargs`:

```json
{
  "putpocket_true_partial_operation": "donor_register or target_sparse",
  "putpocket_true_partial_manifest_path": "/absolute/path/under/allowed/root.json",
  "putpocket_true_partial_manifest_sha256": "<64 lowercase hex>",
  "putpocket_true_partial_owner_token": "<unguessable value, at least 32 bytes>"
}
```

The donor request prompt is exactly `F_old` and uses `max_tokens=1`. After its
one output, the scheduler pins its block objects under the explicit handle and
then releases the ordinary request allocation. This lifetime is independent of
prefix-cache hashes. The target API prompt contains `F_new + frozen Q2`; its
server manifest identifies `H` as both `frozen_history_token_count` and
`continuation_start`. The target request consumes the donor handle exactly
once.

The target manifest is structurally equivalent to this generic template. The
real file must contain exact digests and the complete layer-name arrays; this
example intentionally has no GLM experiment token or edit-sentence constant.

```json
{
  "schema_version": 1,
  "backend_kind": "true_partial_prefill",
  "operation": "target_sparse",
  "handle": "opaque-donor-handle-0001",
  "owner_token_sha256": "<sha256 of owner token>",
  "runtime": {
    "vllm_commit": "<pinned commit>",
    "model": "<model id>",
    "model_revision": "<revision>",
    "architecture": "<architecture>",
    "tokenizer": "<tokenizer id>",
    "tokenizer_revision": "<revision>",
    "tokenizer_identity_sha256": "<sha256>",
    "serializer_identity_sha256": "<sha256>",
    "kv_cache_dtype": "bfloat16",
    "attention_backend": "FLASHMLA_SPARSE",
    "tensor_parallel_size": 4,
    "pipeline_parallel_size": 1,
    "decode_context_parallel_size": 1,
    "prefill_context_parallel_size": 1
  },
  "cache_contract": {
    "block_size": 64,
    "main_dtype": "bfloat16",
    "main_row_width": 576,
    "main_layer_names": ["<every main cache layer name>"],
    "indexer_dtype": "uint8",
    "indexer_value_bytes": 128,
    "indexer_scale_bytes": 4,
    "indexer_layer_names": ["<every indexer cache layer name>"],
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "num_attention_heads": 64
  },
  "source": {"token_count": 4, "token_ids_sha256": "<sha256>"},
  "target": {
    "request_token_count": 6,
    "request_token_ids_sha256": "<sha256>",
    "frozen_history_token_count": 5,
    "frozen_history_token_ids_sha256": "<sha256>",
    "continuation_start": 5
  },
  "alignment": [
    {"target_position": 0, "donor_position": 0},
    {"target_position": 2, "donor_position": 1},
    {"target_position": 3, "donor_position": 2},
    {"target_position": 4, "donor_position": 3}
  ],
  "edit_operations": [
    {"kind": "insert", "donor_positions": [], "target_positions": [1]}
  ],
  "selection": {
    "semantic_quantity": "importance_for_recompute",
    "score_direction": "higher_score_first",
    "eligible_recompute_positions": [2, 3, 4],
    "selection_order": [4, 2, 3],
    "mandatory_recompute_positions": [1],
    "selected_recompute_positions": [4],
    "recompute_positions": [1, 4]
  },
  "evidence_path": "/absolute/path/under/evidence/root.jsonl",
  "production_default_enabled": false,
  "lifetime": "single_use_after_donor_finish"
}
```

For deletion, deleted donor positions appear only in a `delete` operation and
are absent from the target and alignment. For insertion, every inserted target
position is mandatory. For replacement, source and target positions appear in
a `replace` operation and the target positions are mandatory. Alignment is
one-to-one and covers unchanged tokens; selected aligned positions remain in
the alignment but move from the reuse complement into exact recomputation.

## True sparse execution

The scheduler resolves the live donor record, checks owner and one-use state,
compares source/target token digests and complete runtime/cache contracts, and
requires live pinned block objects. It allocates target blocks through `H`
without a prefix-cache lookup and zeros them. The worker copies only the
unselected aligned rows into their target physical slots on every declared
main and indexer layer. The indexer copy preserves both its 128-byte packed
value row and the corresponding 4-byte page scale-plane entry.

Only `mandatory_recompute_positions + selected_recompute_positions` are loaded
as model input IDs. Their target absolute positions drive RoPE and their slot
mapping points directly at target cache rows. Both the sparse indexer builder
and FLASHMLA_SPARSE builder reinterpret the rows as independent one-token
causal decode lanes. Every lane shares the target block table and has sequence
length `absolute_position + 1`, so each sparse query sees the complete target
logical prefix: copied old rows plus all sparse K/V rows already written for
that layer, with later rows causally masked. Native cache update therefore
writes only the declared sparse target slots layer by layer.

After that forward returns, scheduler state advances directly to `H`. Q2 was
included in the target API prompt but was not in the cache patch: the next
ordinary prefill step computes `[H, P)`. A2 and later decode then proceed
normally. The server requires `P > H`; a target ending at `H` fails closed
instead of sampling from the sparse-patch step. On the first cached step the
scheduler must report `num_computed_tokens == H`, and the worker must select
the target token IDs and physical slots at the contiguous absolute Q2 positions
starting at `H`. If Q2 crosses a cache-block boundary, the emitted continuation
payload retains immutable manifest/boundary attestation fields but refreshes
the current target block table, requiring the original block table to remain
an unchanged prefix. Separate patch and continuation evidence events attest
these checks. Evidence also records the exact sparse model-input positions,
token digest and physical-slot digest, copied-row and layer-set counts, and the
explicit full-prefill prohibition.

## Fail-closed supported boundary

This slice supports only the pinned MRV1 GLM server with
`FLASHMLA_SPARSE`/`DEEPSEEK_V32_INDEXER`, BF16 main KV, the declared packed
indexer layout, eager execution, max-num-seqs 1, PP/DCP/PCP 1, no speculative
decode, no KV/encoder connector, no Mamba/hybrid cache, no LoRA or multimodal
input, no prompt embeddings, no async scheduler, and prefix caching disabled.
TP=4 is supported because every rank validates and copies its local cache
layers and emits rank-scoped evidence. An absent hook leaves ordinary prefix
caching and request behavior unchanged; an active hook never spoofs token
equality or changes global prefix-cache correctness.

The code is source-wired and CPU/static tested, but it has not been executed on
a GPU in this task. It therefore does not establish a measured compute saving,
quality result, or production runtime claim. Required first GPU tests are:

1. byte equality for every reused main and packed-indexer row on every TP rank;
2. genuinely computed KV for inserted and replacement rows;
3. deletion compaction and absence of deleted cache coordinates;
4. sparse causal attention against copied plus recomputed prefix rows;
5. intended selected-row divergence from full target prefill;
6. Q2 ordinary prefill and subsequent decode correctness;
7. evidence/model counters equal the exact recompute list and never the full
   frozen history;
8. shifted-position evidence and deliberate donor-byte preservation.

## Accepted RoPE limitation and follow-up

For an insert/delete shift, the target slot receives donor K/V bytes unchanged.
The donor K row therefore retains RoPE for its old absolute position. Every
such pair is recorded in `shifted_reuse_alignment`, with
`shifted_reused_rows_are_rope_correct=false` and policy
`preserve_donor_kv_bytes_without_rope_rerotation`. Results from this slice must
not be described as RoPE-correct shifted reuse.

The next server slice is a concrete RoPE-aware K-row remapping work item:
FLASHMLA_SPARSE backend ownership must expose or implement inverse-old/forward-
new rotary transformation for reused main K rows; the packed indexer owner must
define whether its cached key representation also needs transformation; the
row-copy kernel must receive old/new positions; scheduler evidence must record
the applied transform; and equal/replacement/insertion/deletion plus per-rank
byte/numerical tests must cover shifted and unshifted rows. No partial
re-rotation is implemented or implied here.
