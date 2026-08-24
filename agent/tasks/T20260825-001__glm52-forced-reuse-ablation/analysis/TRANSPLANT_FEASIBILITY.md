# GLM-5.2 forced edit-reuse transplant feasibility

Status: CPU/static implementation complete; GPU mutation not yet executed.
This is an explicitly unsafe research ablation because the preceding shadow
gate found zero wholly reusable 64-token pages. The feature remains default
OFF and requires both an absolute control path and the literal acknowledgement
`I_UNDERSTAND_ZERO_SAFE_PAGES`.

## Exact state and hook points

- The pinned model has 78 MLA layers. Each main cache token is one BF16 row of
  576 elements: the 512-element compressed latent plus the 64-element RoPE key.
  The hook runs immediately after `do_kv_cache_update` and before native
  attention can consume the destination cache.
- There are 21 independent indexer caches at layers
  `0,1,2,6,10,...,74`. Each token uses 128 packed value bytes plus four scale
  bytes stored in the page scale plane. Both regions are copied immediately
  after `indexer_k_quant_and_cache` and before prefill top-k. The 57 intervening
  layers share the preceding top-k buffer and own no additional indexer cache.
- Source rows are private clones retained by each TP worker. Destination
  allocator ownership, physical slots, block tables, request/sequence metadata,
  cache lifetime, and top-k buffers are never transplanted. Logical positions
  are resolved to destination slots on each request with uniqueness and bounds
  checks.
- All 78 main plus 21 indexer snapshots must exist before the first transplant.
  Any missing product, prompt digest mismatch, non-exact position vector,
  chunked/mixed prefill, invalid slot, dtype/layout mismatch, or selector drift
  raises an explicit invariant error before reuse.
- The donor and edit are both exactly 2,071 tokens. Only absolute token 114
  changes (`17526` to `11660`), so RoPE positions are identical. Eligible forced
  positions are defined only over 115..2070. This design is invalid for
  insertion, deletion, shifted positions, batching, PP>1, DCP/PCP, or a block
  size other than 64.

The main-cache transplant is guaranteed to affect subsequent decode attention;
the indexer transplant is also consumed during the full prefill because the
native prefill path gathers the just-written packed cache before score/top-k.
Runtime evidence records every copy by rank/product/layer, actual and recomputed
row counts, physical complete/touched/mixed pages, destination slot hashes, and
the native-consumer ordering attestation. Planner output alone cannot satisfy
the verifier.

## Frozen selector

`glm52_forced_reuse.py` consumes only baseline/edit native score JSONL and the
two exact token-ID artifacts. It validates record hashes, finite values, full
21-layer x five-point x TP coverage, and exact TP raw-score consensus. It never
opens model completions, patches, test outcomes, or evaluator results.

Positions are ranked lexicographically using worst-case and p95 per-query IQR
normalized score delta, membership stability, cutoff-margin retention, and rank
shift. Group Pearson, Spearman, top-k Jaccard, and IQR-normalized RMSE are
attested in the selector. A single global raw-score cutoff is forbidden. Each
0,10,...,100 ratio is a sorted prefix of the one frozen ranking; counts are
`0,195,391,586,782,978,1173,1369,1564,1760,1956`.

## Runtime order

One server process performs: edited full-recompute 0%, donor snapshot, then the
selected transplant ratios. Controls are atomically replaced only between
completed requests. Each ratio uses the identical edited prompt and deterministic
API parameters. A 0/10 smoke job must complete and prove 99 layer-products per
TP rank before an `afterok` full sweep is eligible. Official SWE-bench Pro
evaluation is a separate dependent compute/CPU allocation; it must never be
invented from completion text.

No result from this path is an authoritative Cluster Center PASS. A single
sweep cannot support a speedup claim.
