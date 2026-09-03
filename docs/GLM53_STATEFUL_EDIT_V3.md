# GLM-5.3 stateful-edit-v3 accuracy ablation

This package ports the existing single-instance GLM-5.2 experiment to the
locked Montblanc GLM-5.3 Flash NVFP4 deployment. It does not add a new
SWE-bench task, conversation meaning, edit type, or scientific claim.

## Claim boundary

The experimental hook is default OFF. Every donor and edited request performs
an ordinary complete model prefill. After a native target cache write, selected
frozen-history rows may be replaced with private donor-cache rows. This is an
accuracy emulation/ablation only. It is not selective prefill, does not skip
model FLOPs, and cannot support speedup or latency claims. The later GLM-5.2
true-partial-prefill package is deliberately not ported in this slice.

SWE-bench Pro supplies the pinned problem row, repository/container mapping,
and official evaluator. PutPocket authors the multi-turn construction, A1
execution and Q2 freeze, system edit, selector, and ratio sweep. The benchmark
does not natively contain this trajectory.

## Exact authority

- Model: `RedHatAI/GLM-5.3-Flash-NVFP4` at
  `36c184c6cda000a481711306df5adde42f63321a`.
- Architecture: `Glm5NextForConditionalGeneration`; compressed-tensors NVFP4.
- vLLM: PR 53906 commit `878631b6079d2cf9fb80830ef9cb41b43aded098`.
- Attention/cache: `FLASHINFER_MLA_SPARSE_SM120`, `fp8_ds_mla`.
- Parallelism: TP1, DP3, EP3, PP1. Every experimental request is routed with
  `X-data-parallel-rank: 0`; normal prefix matching is disabled and is never
  weakened or spoofed.
- Benchmark: `ScaleAI/SWE-bench_Pro` revision
  `7ab5114912baf22bb098818e604c02fe7ad2c11f`, test split, exactly
  `instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5`;
  official harness commit `ca10a60a5fcae51e6948ffe1485d4153d421e6c5`.

The deployment model lock remains the sole model/runtime authority. The
stateful experiment lock references it by exact SHA-256 and adds only the
experimental contract and overlay hashes.

## Why the cache contract differs from GLM-5.2

The exact model config has 45 layers: 11 `deepseek_sparse_attention` MLA
layers (`3,7,…,43`) and 34 `linear_attention` KDA layers. Each MLA main cache
row is a 656-byte `uint8` `fp8_ds_mla` payload. The Lightning/DeepSeek indexer
uses 32 heads × 128 dimensions and `index_kpool=4`; its cache stores one
132-byte compressed row only at the end of each complete four-token pool.

Therefore:

- selected MLA rows are copied by exact absolute position;
- an indexer pool row is copied only when all four constituent positions are
  selected and the whole pool lies in eligible frozen history;
- partial pools, pools crossing the edit/history boundary, and the live
  token-granular kpool tail retain native target computation;
- KDA terminal recurrent/conv state is not token-addressable and is never
  donor-transplanted.

This gives an intentional GLM-5.3 hybrid accuracy ablation, not a claim that
all layer state was transplanted. GLM-5.3 NoPE (`qk_rope_head_dim=0`) removes
the stale-RoPE issue for this model, but this task still activates only the
same equal-token-count replacement scenario.

## Episode and selector invariants

The only active episode is:

1. Clean pinned instance with `SYS_old + Q1` generates A1 once.
2. Execute A1 once and freeze the resulting Q2 observation.
3. Before reading any evaluator outcome, freeze A1, Q2, token serialization,
   source digests, and the selector.
4. For each ratio, replay the same A1 response, attest the same Q2, change only
   the one system sentence, and send `SYS_new + Q1 + A1 + Q2`.
5. Q2 and all later tokens are outside donor history and always compute
   normally. Each ratio gets fresh official evaluator state.

The old/new sentence pair is 13 tokens in isolation under the exact GLM-5.3
tokenizer, with only token `17526 → 11660` differing. The full serialized edit
position is deliberately not fixed: `build-episode` derives it after exact
GLM-5.3 chat-template serialization and requires one difference. A GLM-5.2
position such as 114 is never imported.

The experiment retains the historical ratio orientation: the frozen selector
prefix identifies donor-overwrite rows. This is only an accuracy-ablation
control. It must not be confused with the separate true-partial contract in
which high-ranked positions would be recomputed.

## Source preparation (CPU only)

```bash
./scripts/glm53/prepare_stateful_edit_v3_overlay.sh \
  /path/to/exact-vllm-878631b-git \
  /new/task-local/glm53-stateful-overlay
```

The script uses `git archive`, applies the three required GLM-5.3 deployment
patches in order (including explicit `--unidiff-zero` for the NoPE, SM120
top-k, and whitespace-clean zero-context stateful overlays), then verifies
every postimage. It
does not invoke Docker, CUDA, a model loader, or NVIDIA devices.

After separate GPU authorization, build the thin Python overlay image:

```bash
./scripts/glm53/build_stateful_edit_v3_image.sh /new/task-local/glm53-stateful-overlay
```

The derived image must receive a digest and image-doctor evidence before use;
no derived image has been GPU-validated by this CPU-only task.

## Future episode/sweep sequence

The authoring template at
`configs/experiments/glm53_stateful_edit_v3_single_instance.template.json` is
not a frozen episode. A GLM-5.3 native raw pre-top-k selector capture for both
old and edited exact prompts must first be frozen; GLM-5.2 selector files are
incompatible and prohibited.

The required frozen selector shape is machine-readable in
`configs/experiments/schemas/glm53_stateful_edit_v3_base_selector.schema.json`.
It preserves the GLM-5.2 stability-ranking rule but binds it to all 11 GLM-5.3
indexer layers, DP rank 0, the last prefill query, and native compressed-pool
scores expanded deterministically to their four constituent token positions.
The default-OFF overlay exposes a separate bounded capture control for this
purpose. It records exactly one last-query raw pool-score row per indexer layer
before top-k or normalization. The prompt must exceed 2,048 tokens so the
native sparse scoring path really executes; short prompts fail closed rather
than substituting membership.

```bash
putpocket-glm53-stateful-edit serialize-base-prompts \
  --capture exact-old-turn1-capture.json --model-root /exact/pinned/model \
  --output-root /new/base-prompt-freeze
putpocket-glm53-stateful-edit selector-capture-control \
  --prompt-side donor --tokens /new/base-prompt-freeze/donor-initial-token-ids.json \
  --evidence-dir "$GLM53_STATEFUL_SHARED_ROOT/capture/donor" \
  --output "$GLM53_STATEFUL_SHARED_ROOT/selector-capture-donor-control.json"
# Start a fresh exact donor prefill against a server launched with
# GLM53_STATEFUL_SERVER_MODE=selector-capture and
# GLM53_SELECTOR_CAPTURE_CONTROL=$GLM53_STATEFUL_SHARED_ROOT/selector-capture-donor-control.json,
# then repeat in a fresh process for the edited prompt.
putpocket-glm53-stateful-edit freeze-base-selector \
  --donor-root "$GLM53_STATEFUL_SHARED_ROOT/capture/donor" \
  --edited-root "$GLM53_STATEFUL_SHARED_ROOT/capture/edited" \
  --donor-tokens /new/base-prompt-freeze/donor-initial-token-ids.json \
  --edited-tokens /new/base-prompt-freeze/edited-initial-token-ids.json \
  --output glm53-base-selector.json
```

The serializer uses the exact captured `SYS_old+Q1` request and exact pinned
chat template; it is the bridge that prevents a hard-coded GLM-5.2 prompt or
edit position from entering the GLM-5.3 capture. After freezing the selector,
`build-episode` combines that same capture, its executed A1/Q2 trajectory, and
the selector. No evaluator result is read during these steps.

Because an incomplete final four-token pool has no native pool logit, those
one-to-three tail positions are retained, explicitly marked unscored, and
ordered last by absolute position. No score is invented. Producing the native
records and validating their 11-layer coverage remain GPU-only.

For an episode freeze, use an ordinary locked server without the stateful
control environment. For a sweep, launch the stateful image with a unique
shared control root; the missing control file is the explicit startup OFF
state, while any present malformed file fails closed.

```bash
export GLM53_STATEFUL_SHARED_ROOT=/new/task-local/shared-control
export PUTPOCKET_ALLOW_TASK_PRODUCTION=1
RUN_DIR="$(./scripts/glm53/launch_stateful_edit_v3_server.sh)"

export PUTPOCKET_GLM53_ACCURACY_ABLATION_ACK=I_UNDERSTAND_FULL_TARGET_PREFILL_THEN_DONOR_OVERWRITE
export GLM53_STATEFUL_CONTROL_ROOT="$GLM53_STATEFUL_SHARED_ROOT"
export GLM53_RUN_ROOT="$GLM53_STATEFUL_SHARED_ROOT/run-unique"
export GLM53_STATEFUL_PROFILE=smoke
export GLM53_BACKEND_PORT=8137
export GLM53_MODEL_DIR=/exact/pinned/model
export GLM53_HARNESS_ROOT=/exact/pinned/harness
export GLM53_AGENT_ENV=/exact/pinned/agent-env
export GLM53_EPISODE_ROOT="$GLM53_STATEFUL_SHARED_ROOT/frozen/glm53-episode"
./scripts/glm53/run_stateful_edit_v3_harness.sh

./scripts/glm53/stop_stateful_edit_v3_server.sh "$RUN_DIR"
```

The harness is endpoint-oriented: it never starts a model, probes a GPU,
passes devices to Docker, or submits a scheduler job. The launch/stop scripts
are retained for the one remaining separately authorized GPU validation.

## Current validation status

CPU validation proves source/lock/config/tokenizer compatibility and cache-row
mapping mechanics only. Historical GLM-5.2 captures, score reports, plots, and
benchmark outcomes remain historical/superseded provenance and are not
GLM-5.3 success evidence. The GLM-5.3 model directory is currently absent and
the filesystem has less free space than the locked 197,881,155,500-byte
checkpoint, so no safe weight download or load occurred in this task.

Remaining GPU-only work is one fresh sequence: restore/verify the exact model,
freeze a native GLM-5.3 base selector and episode, build/doctor the derived
stateful image, run ratio 0/10 smoke, then (only if it passes) the full ratio
sweep and official single-instance evaluation.
