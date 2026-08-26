# RunPod GLM-5.2 environment and attention/indexer score tests

This package schedules two ordered RunPod-only tests. It contains source,
patches, validation code, schemas, and commands; it contains no model weights,
caches, credentials, container archives, or GPU results. Montblanc CPU/static
validation does not establish that either GPU test passes.

The machine-readable schedule is
`configs/runpod/glm52_attention_indexer_schedule.json`; its package lock is
`configs/runpod/glm52_attention_indexer_package.lock.json`.

## Provenance and measurement boundary

The benchmark input is exactly `ScaleAI/SWE-bench_Pro`, unnamed dataset config,
revision `7ab5114912baf22bb098818e604c02fe7ad2c11f`, split `test`, instance
`instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5`.
The repository identifier is `ansible/ansible`; the official harness is
`scaleapi/SWE-bench_Pro-os` at
`ca10a60a5fcae51e6948ffe1485d4153d421e6c5`. Its `mini-swe-agent`
gitlink is initialized at `d74716a3c8104a113f77cc9ab94cf407ecdcf1e9`;
the pinned scaffold lives inside that submodule.

SWE-bench Pro supplies the problem row, repository/container mapping, and
evaluator. It does not supply the stateful edit trajectory. PutPocket authors
the `SYS_old`/`SYS_new` replacement, A1-execute-Q2 episode contract, and this
score diagnostic.

The final score run uses the recommended equal-position edit, but no executed
A1 artifact exists. It therefore freezes a deterministic benchmark-derived
two-query diagnostic: pinned public Q1, a preregistered project-authored
assistant bridge, and a preregistered project-authored Q2. Every Q1/Q2 content
token is a score query. This is not an A1-execute-tool-observation episode and
supports no stateful cache or task-quality claim. The equal-position edit
avoids stale-RoPE confounding. The older four-row `SYS_new + Q1` sampled mode
remains available for debugging and is explicitly non-final.

## Exact source and setup contract

Use the pushed isolated branch only. Record its supplied commit at runtime;
the lock cannot embed the hash of the commit that contains itself.

```bash
git clone git@github.com:Nier4Ryu/putpocket_dataset_mining.git /workspace/putpocket_dataset_mining
cd /workspace/putpocket_dataset_mining
git switch --track -c agent/T20260826-001__glm52-stateful-edit-v3-port \
  origin/agent/T20260826-001__glm52-stateful-edit-v3-port
export PUTPOCKET_EXPECTED_PROJECT_COMMIT=$(git rev-parse HEAD)
```

The audited RunPod host uses Ubuntu 22.04, CUDA toolkit 12.9, Python 3.12, and
torch `2.13.0+cu129`. The exact vLLM commit is
`4a3447d200e5aa428d68d1a00aa00f1a19a1a729`. It intentionally does not use the
general `runpod-dev` vLLM 0.26 lock, whose vLLM and torch pins differ.

Clone vLLM and the evaluation harness at exact commits. These are explicit,
pinned network operations; the doctor itself performs no setup or download.

```bash
git clone https://github.com/vllm-project/vllm.git /workspace/vllm
git -C /workspace/vllm checkout --detach 4a3447d200e5aa428d68d1a00aa00f1a19a1a729
git clone https://github.com/scaleapi/SWE-bench_Pro-os.git /workspace/SWE-bench_Pro-os
git -C /workspace/SWE-bench_Pro-os checkout --detach ca10a60a5fcae51e6948ffe1485d4153d421e6c5
git -C /workspace/SWE-bench_Pro-os submodule update --init --depth 1 mini-swe-agent
git -C /workspace/SWE-bench_Pro-os/mini-swe-agent rev-parse HEAD | \
  grep -Fx d74716a3c8104a113f77cc9ab94cf407ecdcf1e9
```

The local model must be the explicit revision
`aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`. Downloading it, if required, is
an explicit operator step with that revision; neither test silently downloads
a model. After verifying the snapshot, create its revision marker:

```bash
printf '%s\n' aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa \
  > /workspace/models/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa/.putpocket_model_revision
```

Apply the three overlays before the editable vLLM build:

```bash
PYTHON_BIN=/workspace/Putpocket_env/bin/python \
  ./scripts/runpod/bootstrap_glm52_attention_indexer.sh \
  /workspace/vllm /workspace/artifacts/test1/bootstrap.json
uv pip install --python /workspace/Putpocket_env/bin/python \
  --no-build-isolation --no-deps -e /workspace/vllm
```

The bootstrap uses `set -euo pipefail`. It validates the clean upstream tree,
runs POSIX `patch -p1 --dry-run --forward --batch` then the matching real
`patch` command for the required legacy packaging base, and runs
`git apply --check --unidiff-zero` then
`git apply --unidiff-zero` for both zero-context overlays. It validates every
phase hash and `py_compile`s the installed hooks. A later checksum cannot mask
a failed patch command.

The third overlay also carries two pinned live-integration corrections. First,
the bundled DeepGEMM host extension's `tools/build_deepgemm_C.py` forces
`cuda_fp8.h` into its `g++` translation unit. Without that include, the exact
pinned DeepGEMM source names `__nv_fp8_e4m3` through `mqa_logits.cuh` while the
host compiler has not seen the CUDA FP8 definition, and the editable build
fails closed. This is a build-only correction; it does not enable diagnostics
or change runtime inference decisions.

Second, score-batch attestation is wired into `DeepseekV2Model`, the actual
model path behind the pinned `GlmMoeDsaForCausalLM` architecture, as well as
the pre-existing GLM-lite path. When score capture is enabled, absence of
`input_ids` now fails closed before either score hook can consume an
unattested batch. The doctor inspects the actual DeepSeek/GLM class for this
hook. Score capture remains default OFF and does not change inference choices.
vLLM's position-zero synthetic memory-profile batches are disarmed when their
flattened token count differs from the frozen prompt. Only an exact-count batch
can arm capture; it must then have contiguous absolute positions and the exact
token digest. A wrong real prompt therefore cannot produce evidence, and the
capture/analyzer still fails closed when the required files are absent.

The legacy patch is deliberately applied with the repository's existing GNU
`patch` convention. Plain `git apply` is not equivalent for this generated
zero-context legacy artifact: it can accept the hunk yet place additions at an
incorrect boundary. The bootstrap's postimage hashes make that error fail
closed.

The legacy patch is a required generated-overlay base. This packaging fact does
not combine runtime modes: legacy accuracy emulation, true partial prefill, and
score capture remain default OFF and mutually exclusive for these tests.

## Test 1: read-only environment doctor

The bootstrap is mutating setup. The doctor below is a separate read-only
inspection of source, environment, hardware, and model; its only write is the
requested evidence JSON.

```bash
./scripts/runpod/run_glm52_runpod_doctor.sh \
  /workspace/vllm \
  /workspace/models/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa \
  "$PUTPOCKET_EXPECTED_PROJECT_COMMIT" \
  /workspace/artifacts/test1/doctor.json \
  /workspace/Putpocket_env/bin/python
```

It fails closed on project branch/commit, vLLM commit/postimages, patch and
instrumentation hashes, Python/dependency imports, the CUDA 12.9 `nvcc`
toolchain, local-only tokenizer load,
model architecture/config boundaries, four visible full H200 GPUs, BF16,
memory, NCCL and peer access, FLASHMLA_SPARSE, DEEPSEEK_V32_INDEXER, required
true-partial/diagnostic symbols, or non-default experimental flags. The model
`config.json` digest is recorded as observed; repository evidence pins its
semantic fields and model revision but does not claim a previously evidenced
exact `config.json` byte digest. Tokenizer artifacts are exact-hash pinned.

Archive the whole doctor JSON. Test 2 validates its canonical
`payload_sha256`; a copied `passed` string without the matching digest is not a
dependency token.

## Test 2: aligned main-attention and raw indexer scores

```bash
./scripts/runpod/run_glm52_attention_indexer_score_test.sh \
  /workspace/models/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa \
  /workspace/SWE-bench_Pro-os \
  /workspace/artifacts/test1/doctor.json \
  /workspace/artifacts/test2 \
  /workspace/Putpocket_env/bin/python
```

`prepare-final-probe` loads only the pinned dataset revision, checks the exact
row and scaffold hashes, serializes the public Q1 plus the preregistered bridge
and Q2, resolves exact half-open Q1/Q2 content-token ranges with tokenizer
offsets, applies the equal-position target edit, estimates bounded storage,
and freezes prompt text, token IDs, ranges, provenance, and digests before any
model capture or evaluator outcome. The model runs one ordinary prefill plus one
generated token with TP4, BF16 KV, block size 64, FLASHMLA_SPARSE,
`sparse_mla_force_mqa=true`, eager mode, no prefix cache, and no chunked
prefill. The true-partial and legacy emulation controls must be OFF.

For layers 0, 22, 46, and 74 and **every** token row in the frozen Q1 and Q2
content ranges:

- Main attention is reference-recomputed from the exact post-RoPE Q and the
  model's own projected K state as
  `scale * ((q_nope dot k_nope) + (q_rope dot k_rope))` for every causal
  strictly earlier candidate in the frozen window and every main head. It is mathematically faithful full-candidate
  diagnostic scoring, not a production-kernel return and not a post-top-k
  sparse substitute. The exact NVFP4 model layout is 64 main heads with a
  192-dimensional non-RoPE component, a 64-dimensional RoPE component, and a
  256-dimensional value head.
- The Lightning/DeepSeek V3.2 indexer vector is the kernel-native
  `fp8_fp4_mqa_logits` output captured before `top_k_per_row_prefill`. It is the
  learned weighted sum across 32 TP-replicated indexer heads with the native
  quantization scales, `128^-0.5` softmax scale, and `32^-0.5` head scale.
  This 128-dimensional indexer scale is independent of the main attention's
  192-dimensional non-RoPE component.
- Candidate logical positions and token IDs must agree exactly. TP-local main
  heads are concatenated to all 64 heads; replicated indexer vectors must agree
  exactly across TP ranks.

The candidate window is the complete frozen prompt prefix `[0, Q2_end)`, not
`[Q1_start, Q2_end)`. This includes the SYS edit and every causal candidate
available to Q1/Q2. Seed/query rows remain all and only the explicit Q1 and Q2
content-token ranges; system, assistant, and tool rows are candidates but are
never silently added to the seed.

The report retains every per-query raw row, then independently for each layer
computes a signed sum over exactly the same Q1/Q2 rows for each candidate
position. Main heads are first averaged per query; the native indexer aggregate
is used unchanged. `query-summed-token-scores.jsonl` preserves both raw sums,
normalized distributions, ranks, and receiving-query counts. The report also
includes raw statistics, Pearson, Spearman,
z-score cosine, JS divergence between explicitly normalized distributions,
and top-k overlap/recall/NDCG. Query summation can make native raw logits large
enough that direct float64 softmax underflows almost every candidate to zero.
The v2 offline analyzer therefore independently population-z-scores the two
query-summed vectors before softmax for the primary JS/distribution fields.
It preserves the direct native-softmax JS and effective-support counts only as
an explicit saturation diagnostic. Top-k membership is ranked directly from
the raw pre-softmax sums, so mathematically order-preserving softmax cannot
lose the order through numerical ties; NDCG uses the main raw descending rank
as nonnegative scale-independent relevance. No per-query vector is compared
to a summed vector, and these offline transformations never affect inference.

This first run has no preregistered similarity threshold. Capture, alignment,
finite-value, schema, or digest errors fail the test; low similarity does not.
Consistently high positive rank/correlation and cosine, low JS divergence, and
strong top-k recall/NDCG would support the distillation-similarity hypothesis.
The opposite cross-layer/query pattern would refute it.

## Offline raw-indexer multi-hop score within Test 2

Neither the older sampled artifact nor the all-query artifact is a valid input
for exact multi-hop scoring.
It contains only four inclusive causal query rows. Exact propagation through a
declared window needs every query/intermediate row and uses strict causality
`k < q`; a missing row is not a zero row. The package therefore provides a
separate default-OFF `strict_causal_indexer_matrix` mode. The final wrapper
executes it after query-sum capture over the identical prompt/window/layers.
It is a postprocessor within Test 2, not a third GPU test, and it does not
change vLLM scheduling, selection, or inference.

First freeze `frozen-matrix-episode.json` against
`configs/runpod/schemas/glm52_indexer_matrix_episode.schema.json`. It must hold
the exact full prompt token IDs and digest, the exact source frozen-episode
manifest and digest, selected indexer layers, the propagation window, and
disjoint half-open Q1 and Q2 token ranges. In a true episode Q2 is the frozen
pre-edit tool observation; in this explicitly non-stateful fallback it is the
preregistered follow-up user query. The manifest records which semantic is
used. SWE-bench Pro still supplies only the problem/repository/evaluator.

The native hook records `fp8_fp4_mqa_logits` before top-k and normalization for
every query `q` in the window, but writes only keys in `[window_start, q)`.
Rows are chunked by query position into
`matrix-rank-XX-chunk-YYYY.jsonl`. Each row retains layer, rank, query/key
absolute positions and token IDs, native causal bounds, dtype, native scales,
32-head learned aggregation provenance, raw signed values, and its digest.
All TP replicas must agree under the manifest tolerance. The capture cost and
artifact size are `O(layers * window_tokens^2)`: defaults are 256 tokens, 32
rows per chunk, and layers 0/22/46/74; the reviewed hard limits are 2176 tokens,
9465600 raw edge values per rank, and propagation level 16. The exact frozen
2,103-token probe uses 8,841,012 raw edge values and 138,495,040 main-reference
logit values per rank (about 554 MB of float32 values before JSON encoding).
Exceeding a cap fails
before model execution.

After Test 1 and after an operator has frozen the matrix episode, run the
explicit optional path (replace the ranges with the exact manifest values):

```bash
./scripts/runpod/run_glm52_indexer_matrix_multihop.sh \
  /workspace/models/aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa \
  /workspace/artifacts/test1/doctor.json \
  /workspace/artifacts/test2/frozen-matrix-episode.json \
  /workspace/artifacts/test2 \
  Q1_START:Q1_END Q2_START:Q2_END WINDOW_START:WINDOW_END \
  0,22,46,74 6 /workspace/Putpocket_env/bin/python
```

The equivalent CLI is `capture-matrix` followed by `score-matrix`; use
`capture-matrix --dry-run` to validate and inspect the bounded plan without
loading the model. The scorer verifies that its explicit `--q1-range`,
`--q2-range`, `--window`, and `--layers` agree with the frozen manifest.

The following signed recurrence is retained as the original, auditable v1
definition. It is no longer the recommended selector-relevance metric: native
raw scores have both signs and unrelated layer scales, so repeated matrix
products alternate sign, explode in magnitude, and cancel under summation.
Those properties are evidence about the declared v1 formula, but they obscure
the top-k relevance ordering that a selector needs.

For each layer independently, its strict lower-triangular matrix is
`A_l[q,k]`, and `u[q]=1` exactly for frozen Q1/Q2 tokens. Level 1 is
`c_l,1 = u A_l`; level `n+1` is `c_l,n+1 = c_l,n A_l`; the reported level-L
score is cumulative, `s_l,L = sum(c_l,n, n=1..L)`. Computation uses CPython
IEEE-754 binary64 and fails on a non-finite product or sum. It does not
normalize, clip, rectify, softmax, or discard negative values. Layers are
never multiplied: every per-layer hop/final value is retained, plus the exact
unnormalized layer sum. Query tokens remain in the output and are marked as
Q1/Q2 seeds; downstream selectors may filter eligibility later.

Retain `indexer-multihop-report.json` and
`indexer-multihop-token-scores.jsonl`. The report binds every input file hash,
episode/ranges/window/layers, recurrence/version, dtype policy, TP consensus,
and token artifact hash. Each token row records token ID, segment membership,
per-layer hop contributions, per-layer cumulative score, and the hop/final
layer sums. This offline score has no threshold and must not be wired into a
vLLM request or described as a runtime inference decision.

## Rank-normalized multihop selector-relevance analysis

The v1 signed files above are immutable legacy inputs. The separate
`putpocket_rank_normalized_indexer_multihop_v1` postprocessor uses the same
complete native raw matrix, frozen token IDs, Q1/Q2 ranges, layers, causal
orientation, and input hashes without rerunning a GPU or changing inference.

For layer `l` and valid row `q`, rank all `k < q` by descending native raw
indexer score, breaking an exact tie by lower absolute key position. For
transition cutoff `K`, retain ranks `r <= min(K,q)` and define

`T_l^K[q,k] = (1/log2(r_lq(k)+1)) / sum_{j:r_lq(j)<=K}(1/log2(r_lq(j)+1))`.

Every valid layer row is nonnegative and has mass one. The analysis uses the
uniform layer mean `T^K = mean_l(T_l^K)`, so no layer can dominate because of
raw scale or sign. The seed `u` is a probability distribution uniform over all
and only the frozen Q1 and Q2 content-token rows. With the original
left-row-vector/backward-causal orientation:

- `h_1 = normalize(u T^K)`;
- `h_(n+1) = normalize(h_n T^K)`;
- `g_L = (1/L) sum_{n=1..L} h_n`.

`h_n` is the hop-only distribution. `g_L` is an explicitly labeled equal-hop
mixture, not a raw cumulative sum. Each is normalized to mass one at every
reported level. Because the transition is strictly causal, position zero has
no outgoing row; mass reaching a terminal row is reported before remaining
outgoing mass is normalized. The primary transition is DCG rank weighting at
`K=64`, with sensitivity at `K=16` and `K=256`.

The secondary `zscore_softmax_t1` transition independently population-z-scores
every full valid causal row and applies temperature-1 softmax before the same
layer averaging and propagation. It preserves within-row score-spacing
information while remaining nonnegative and normalized; it is a sanity check,
not the only answer.

For hop-only and equal-hop views, the report records mass, natural-log entropy,
effective support `exp(entropy)`, and stability against level 1 and the previous
level. Primary metrics at `k={16,64,256}` are overlap, retention, Jaccard, and
NDCG using baseline deterministic rank as graded relevance. Spearman is
secondary and is computed only on the explicit union of the two top-k sets
using their global deterministic ranks.

Run the scorer and plotter from the committed isolated branch:

```bash
export MONTBLANC_EVIDENCE_ROOT=/path/to/runpod-d85582a-attempt7
putpocket-glm52-rank-multihop \
  --evidence-root "$MONTBLANC_EVIDENCE_ROOT" \
  --checksum-manifest completion-audit-evidence-v1/COMPLETION_AUDIT_SHA256SUMS \
  --episode extracted/artifacts/test2-d85582a-attempt7/frozen-matrix-episode.json \
  --capture-root extracted/artifacts/test2-d85582a-attempt7/matrix \
  --matrix-run extracted/artifacts/test2-d85582a-attempt7/matrix/matrix-capture-run.json \
  --matrix-config extracted/artifacts/test2-d85582a-attempt7/matrix/matrix-instrumentation-config.json \
  --legacy-report extracted/artifacts/test2-d85582a-attempt7/multihop/indexer-multihop-report.json \
  --legacy-token-rows extracted/artifacts/test2-d85582a-attempt7/multihop/indexer-multihop-token-scores.jsonl \
  --transition-k 16,64,256 --evaluation-k 16,64,256 --max-level 6 \
  --softmax-temperature 1 \
  --output-root "$MONTBLANC_EVIDENCE_ROOT/completion-audit-evidence-v1/rank-normalized-multihop-v1"
PYTHONPATH=src python scripts/analysis/plot_glm52_rank_normalized_multihop.py \
  --artifact-root "$MONTBLANC_EVIDENCE_ROOT/completion-audit-evidence-v1/rank-normalized-multihop-v1"
```

The scorer refuses a pre-existing output directory. Every episode, control,
raw matrix, and legacy file must match the consolidated checksum manifest. It
reuses fail-closed TP/completeness/token validation and writes a distinct input
attestation, report, token JSONL, and checksums. The plotter accepts only that
pristine checksum-valid output and adds four PNG/PDF pairs, a schema-valid
summary, and final checksums. Neither tool overwrites or deletes signed v1
artifacts.

## Retain and do not claim

Retain bootstrap/doctor JSON, exact commits, local model revision evidence,
probe JSON, all per-rank capture JSONL, the token-level JSONL, final report,
logs/HBM inventory, and `SHA256SUMS`. Do not retain or upload weights, caches,
credentials, or unrelated workspace contents.

The final all-query and matrix modes fail closed at their declared caps. This
diagnostic does not itself validate
true-partial KV byte preservation, Q2 continuation, stateful task quality, or
latency/compute savings.

## Montblanc evidence completion and plot reproduction

GPU evidence remains outside Git. After an accepted run, preserve remote
originals and transfer only an immutable, explicitly sized archive. The
generic proxy-safe helper uses a caller-supplied SSH target and identity, reads
the remote file without modifying it, retains framed chunks, and verifies the
declared byte count and SHA-256 before publishing the destination:

```bash
scripts/runpod/transfer_framed_file.sh \
  --ssh-target "$RUNPOD_SSH_TARGET" \
  --identity "$RUNPOD_SSH_IDENTITY" \
  --remote-file /workspace/TASK/bundles/completion-evidence.tar.gz \
  --destination "$MONTBLANC_EVIDENCE_ROOT/completion-evidence.tar.gz" \
  --expected-sha256 "$EXPECTED_SHA256" \
  --expected-bytes "$EXPECTED_BYTES"
```

No address, credential, private-key material, or workstation path is stored in
the repository. The helper refuses an existing destination, candidate, or
parts directory and never deletes remote or local evidence. Its two
execution-time prototypes are retained only in the task artifact cache because
they embed one host layout; their SHA-256 values are
`42b26b61940fb6ed5aec2d384890387416567f9e04c1dc2df12ce214270a5ad7` and
`26387857faf6225b76bafac58d55f7b5e0ef6cfe33efcc35682318b93dbdce70`.

Build a deterministic consolidated inventory from a JSON spec whose groups
map Montblanc-relative paths to remote-relative source paths and whose
exclusions explain intentionally absent infrastructure:

```bash
python -m putpocket_dataset_mining.evidence_inventory build \
  --root "$MONTBLANC_EVIDENCE_ROOT" \
  --spec "$MONTBLANC_EVIDENCE_ROOT/completion-audit-spec.json" \
  --output "$MONTBLANC_EVIDENCE_ROOT/completion-audit-inventory.json" \
  --checksums "$MONTBLANC_EVIDENCE_ROOT/COMPLETION_AUDIT_SHA256SUMS"
python -m putpocket_dataset_mining.evidence_inventory verify \
  --root "$MONTBLANC_EVIDENCE_ROOT" \
  --inventory "$MONTBLANC_EVIDENCE_ROOT/completion-audit-inventory.json"
```

The validator rejects duplicate local or remote paths, symlinks, path escape,
undeclared groups, unsorted or mutated entries, size/digest/count mismatches,
and an unexplained empty exclusion set. Model weights, Hugging Face caches,
Python environments, vLLM build objects, compiler caches, failed partial raw
captures, credentials, and duplicate copies of the accepted raw tree are
reproducible-infrastructure or safety exclusions, not missing experimental
evidence.

The reusable plotter is the generalized, path-independent form of the script
used for the accepted run. The cache-origin implementation has SHA-256
`e1f5cdd967a48514750d9c3074a8ca4a33cd217060e9a7b5cbadf9c0df03363f`.
Install the exact plotting extra and render from four inputs that are already
bound by the supplied checksum manifest:

```bash
python -m pip install -e '.[plots]'
python scripts/analysis/plot_glm52_score_evidence.py \
  --evidence-root "$MONTBLANC_EVIDENCE_ROOT" \
  --query-report-root accepted/query-sum-v2 \
  --multihop-root accepted/multihop \
  --checksum-manifest COMPLETION_AUDIT_SHA256SUMS \
  --output-root "$MONTBLANC_EVIDENCE_ROOT/plots-reproduced"
```

It refuses unattested or digest-mismatched inputs and a nonempty output
directory. It produces five PNG/PDF plot pairs, a schema-validated numeric
summary, and `PLOT_SHA256SUMS`. Input paths recorded in the summary are always
evidence-root-relative. Display-only z-score and symmetric-log color mappings
do not modify the retained signed raw query-sum or multi-hop values.
