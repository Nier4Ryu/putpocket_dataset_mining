# Final handoff — T20260903-001 GLM-5.3 stateful retarget

## Result

The active GLM-5.2 stateful-edit-v3 accuracy-ablation package has been ported
onto the exact GLM-5.3 deployment base `61da82cf29c2c0319b7f90dba211713b15a2de06`
in the dedicated worktree and branch. This was a CPU-only implementation and
static-validation slice. No GLM-5.3 stateful runtime or benchmark success is
claimed.

Worktree:
`/home/dyryu/putpocket_dataset_mining_worktrees/T20260903-001__glm53-retarget-all`

Branch: `agent/T20260903-001__glm53-retarget-all`

Implementation commit: pending at handoff authoring time; record after commit.

## Preserved scientific scope

- Exactly one pinned `ScaleAI/SWE-bench_Pro` test instance remains active:
  `instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5`.
- Dataset revision is `7ab5114912baf22bb098818e604c02fe7ad2c11f`;
  official harness commit is `ca10a60a5fcae51e6948ffe1485d4153d421e6c5`.
- SWE-bench Pro supplies the problem, repository/container mapping, and
  evaluator. PutPocket supplies A1 execution/Q2 freeze, the system edit,
  selector, and ratio sweep.
- Episode order remains `SYS_old+Q1 → frozen A1/action/Q2; edit;
  SYS_new+Q1+A1+Q2 → continuation`, with exact A1/Q2 replay and no evaluator
  outcome read before freeze.
- Only the existing one-token, equal-token-count system replacement is active.
  Insertions, deletions, multiple spans, new instances, and new multi-turn
  meanings were not activated.
- Every target request still performs full target prefill before selected donor
  cache rows overwrite native target rows. It is an accuracy ablation, not true
  partial prefill, compute saving, skipped FLOPs, or a latency experiment.
- Historical GLM-5.2 runtime/results/score captures and later true-partial
  implementation remain read-only historical/superseded provenance. None is
  labeled GLM-5.3 evidence.

## Exact GLM-5.3 binding

- `RedHatAI/GLM-5.3-Flash-NVFP4` revision
  `36c184c6cda000a481711306df5adde42f63321a`.
- `Glm5NextForConditionalGeneration`, compressed-tensors NVFP4.
- vLLM PR53906 commit `878631b6079d2cf9fb80830ef9cb41b43aded098`;
  FlashInfer PR4802 commit `c2eec117219457e45126fa4fa87e7240dd4ea620`.
- `FLASHINFER_MLA_SPARSE_SM120`, `fp8_ds_mla`, TP1/DP3/EP3/PP1, DP rank 0
  request affinity.
- Exact tokenizer SHA-256
  `0cfe2c099a7702a0921abc315ee039deb51e4a34b4818fc509bd27fa3dc4acc1`;
  chat template SHA-256
  `34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891`.
- Isolated old/new sentences each tokenize to 13 tokens and differ only by
  `17526 → 11660`. Full serialized edit position is deliberately derived from
  the future frozen GLM-5.3 request, never copied from GLM-5.2 token 114.

Static config inspection proves 45 total layers: MLA/indexer layers
`3,7,11,15,19,23,27,31,35,39,43` and 34 KDA layers. Main MLA cache rows are
656-byte `uint8` `fp8_ds_mla`; native indexer rows are 132-byte compressed
four-token pool records. Only fully selected, fully eligible pools are copied.
KDA terminal recurrent/conv state and the token-granular indexer tail remain
target-computed.

## Selector capture and freeze

GLM-5.2 selectors fail model/revision/tokenizer checks. The port includes a
separate default-OFF native GLM-5.3 capture mode:

- exact last prefill query only;
- prompt length >2,048 so the sparse score path executes;
- all 11 indexer layers on DP0/TP0;
- native `fp8_fp4_mqa_logits` pool scores sliced to exact causal bounds before
  top-k or normalization;
- record SHA-256 and exact prompt digest;
- offline stability ranking matching the historical lexicographic rule after
  expanding each pool score to four constituent positions;
- incomplete final pool tokens are retained and explicitly marked unscored,
  then sorted last by absolute position rather than assigned invented scores.

Capture and accuracy controls are mutually exclusive. A missing mounted
control is startup/default OFF; a present malformed control fails closed.

## Key files

- `configs/experiments/glm53_stateful_edit_v3_accuracy_ablation.lock.json`
- `configs/experiments/glm53_stateful_edit_v3_single_instance.template.json`
- `configs/experiments/schemas/glm53_stateful_edit_v3_*.schema.json`
- `src/putpocket_dataset_mining/glm53_stateful_edit.py`
- `src/putpocket_dataset_mining/glm53_stateful_proxy.py`
- `instrumentation/vllm/glm53_stateful_edit_accuracy_ablation.py`
- `patches/vllm/878631b6079d2cf9fb80830ef9cb41b43aded098/glm53_stateful_edit_v3_accuracy_ablation.patch`
- `docker/glm53_sm120/Dockerfile.stateful-edit-v3`
- `scripts/glm53/{prepare,build,launch,run,stop}_stateful_edit_v3*`
- `docs/GLM53_STATEFUL_EDIT_V3.md`
- `tests/test_glm53_stateful_edit.py`
- task-local 97-path inventory, CPU metadata audit, and source-target matrix.

## CPU/static evidence

- Focused: `pytest -q tests/test_glm53_stateful_edit.py` → 14 passed, 6
  subtests passed.
- Full: `pytest -q` → 221 passed, 64 subtests passed.
- Python compilation passed for CLI, proxy, hook, and tests.
- `bash -n` passed for all five new GLM-5.3 stateful scripts.
- JSON parsing/schema tests and `git diff --check` passed.
- Fresh patch-chain run `overlay-validation-008` used fail-fast application to
  exact vLLM `878631b`, including the required deployment patches and both
  explicit `--unidiff-zero` arguments for all zero-context overlays. Final postimages:
  - MLA: `ae008425588218b94628058988eadc0a39018ea92851dadb26a5ff9d2554d89a`
  - kpool indexer: `1accf858644f7ed08a9fa952d9b906f9a32e167a8c389ce0cfc4ecf0e54c9b5c`
  - Glm5Next model: `bce7d0fb2816715977b3b15d7c41adcbc3034b4aba094731996addefba9f8d0d`
  - hook: `94cbc2d49f557236813156c439bb76c25f80f6b1537e15b44758958a9f1cd7f0`
- Exact model/config check output SHA-256:
  `a90d3be7cde9db6bc8dd6cb099249fdea5bc78b1be2274817ee9e9cadf2d9a43`.

No `nvidia-smi`, CUDA, model import/load, GPU process, device passthrough,
scheduler/job, Vikunja, or AFFiNE action occurred.

## Model cache/download status

The expected exact-revision model directory is currently absent. At audit time
the filesystem had 169,755,856,896 free bytes, below the locked
197,881,155,500 checkpoint weight bytes even before download overhead. A full
download was therefore not started. Exact official config/tokenizer metadata
only was downloaded and hash-attested in the task-local external cache.

Prior deployment `model_verification.json` is retained as historical evidence
(SHA-256 `04514e470991e190d2b981c7b9bf87a38884f1827b2571929674b76f50c50d2f`)
but does not prove that the model is currently cached or that this stateful
overlay works.

## Remaining blocker and GPU-only sequence

Before any SWE-bench multi-turn result can be discussed, sufficient model
storage and explicit GPU authorization are required. Then perform one fresh
sequence:

1. restore/download and fully verify the exact locked checkpoint;
2. build and image-doctor the thin stateful image, recording its digest;
3. freeze exact old-turn A1/Q2 source, serialize GLM-5.3 donor/edited prompts;
4. in fresh donor/edited processes, capture all 11 native base-selector rows;
5. offline-freeze the selector and episode before outcomes;
6. run ratio 0/10 smoke with official evaluator and verify runtime row evidence;
7. only after smoke passes, run the 0..100 sweep.

Required GPU evidence includes exact cache-byte preservation for copied MLA and
complete indexer-pool rows, KDA/tail non-mutation, DP0 ownership, model API
health, frozen Q2 continuation, and official one-instance results. Until then,
the package is CPU-port-complete but runtime-unvalidated.
