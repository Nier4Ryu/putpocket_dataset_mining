# GLM-5.2 native DSA single-instance diagnostic

This package is a feasibility diagnostic, not a SWE-bench quality run. It uses
one immutable SWE-bench Pro row to exercise a real coding prompt, patch
collection, and the unchanged official per-row evaluator while collecting
bounded native GLM-5.2 DSA indexer evidence. It has no path to the full public
test manifest and never computes a score or the 40% acceptance threshold.

## Immutable workload

- Model: `nvidia/GLM-5.2-NVFP4`
- Required model/tokenizer revision:
  `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`
- SGLang source: `83d7d453306977dd3aad4402c921c8a6b66d9a9d`
- Runtime image:
  `lmsysorg/sglang@sha256:3be8803490a8b899a44f7ab2e22d8f6a1fb877cab52faeb400769a1555317db4`
- Dataset: `ScaleAI/SWE-bench_Pro` test split at
  `7ab5114912baf22bb098818e604c02fe7ad2c11f`
- Official harness:
  `scaleapi/SWE-bench_Pro-os@ca10a60a5fcae51e6948ffe1485d4153d421e6c5`
- Instance:
  `instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5`
- Canonical row SHA-256:
  `78ff3ac298f276dfafaa311c26b7ace35be7c52d9094b4a6f658de2e7b5e25d1`
- Serialized prompt: 2,071 tokens, 8,629 UTF-8 bytes, SHA-256
  `25d6597314bbb6c4df5afa886b064bebbdb7b57d27414d1e584e6a0127eeeab5`

The selection was resolved by rendering all 731 pinned public rows through the
pinned mini-swe system and instance templates and the exact GLM tokenizer/chat
template. A viable row must exceed top-k 2,048 so SGLang produces native
prefill logits, while its prompt plus the fixed 512-token generation cap must
fit the 4,096-token context. The smallest token count wins, with instance ID as
the stable tie-break. Runtime re-renders this row and fails unless every digest
and token count matches; selection is never dynamic.

## Native instrumentation boundary

The tracked patch targets only the pinned `Indexer._get_topk_paged` and
`Indexer._get_topk_ragged` score-production sites. It copies the requested
native MQA score row immediately before SGLang applies its forced-token mask
and calls the unchanged fused `DSAIndexerMetadata.topk_transform`.

The fused `sgl-kernel` returns transformed cache coordinates rather than scores.
The hook inverts the existing page-table/ragged offset transform, then gathers
the selected scores from the saved native row. It does not call `torch.topk`,
FlashInfer top-k, a dense attention implementation, or any duplicate selector.
If the native transformed coordinates cannot be inverted uniquely, the hook
writes a bounded BLOCKED record and raises. No backend or runtime fallback is
attempted.

The source target is locked both before and after patch application. Validation
replays SGLang's native init/local-token forced mask mathematically against the
captured raw vector and requires the fused selected set to satisfy the native
top-k boundary; it never produces a replacement selection.

Captured points are the last prefill query and decode steps 0, 1, 8, and 32
only when those decode calls exist. Every full indexer layer (0, 1, 2, then
6..74 in increments of four) and all four TP ranks are required. The manifest
also records the exact 57 shared-layer-to-source-full-layer map.

## Execution order

1. Prove one Slurm node, four visible physical H200 141GB-class GPUs, MIG off,
   typed H200 TRES, topology/NVLink, driver, CUDA 12.9, and Slurm identity.
2. Pull the immutable image, fetch exact SGLang source, validate every source
   digest, apply the zero-context tracked patch with exact pre/post digests and
   `git apply --unidiff-zero --check`, inject the hashed
   support module, and run the weightless backend/config probe.
3. Resolve `main` and require the committed 40-character model SHA, stage the
   checkpoint without tensor hashing, and load it TP4/all-resident with exact
   ModelOpt/Marlin/DSA controls.
4. Render the exact public coding prompt. With radix caching disabled and a
   successful `/flush_cache` before each request, send the same OpenAI-compatible
   completion request to the same server once OFF and once ON. Require exact
   container ID/process start identity and output token-ID/hash equality, and
   record duration/HBM for both.
5. Validate, deterministically compress, and SHA-256 the bounded native records.
6. Execute the one generated mini-swe bash action in the official row image,
   gather its patch with the pinned official helper, and run unchanged
   `swe_bench_pro_eval.py` for only this row. Resolution is recorded but is not
   a PASS condition or quality score.

## Rendering and submission boundary

From a checkout of the exact public project commit, render one line to stdout:

```bash
PYTHONPATH=src /home/dyryu/putpocket_dataset_mining/Putpocket_env/bin/python \
  -m putpocket_dataset_mining.glm52_dsa_diagnostic_cli render-wrap \
  --site configs/cluster/sites/herdr_h200_sglang_gate.json \
  --project-url https://github.com/Nier4Ryu/putpocket_dataset_mining.git \
  --project-commit <exact-public-commit>
```

The orchestrator may paste that one line into its authenticated Login pane.
Login only creates `/home2/jslee202403/putpocket-slurm` and calls `sbatch`.
The `--wrap` body validates the allocation/container control plane, clones the
exact commit below `/local-data/jslee202403`, and execs the tracked diagnostic
entrypoint. Repository workers do not log in or submit it.

## Artifacts and outcomes

- stdout: `/home2/jslee202403/putpocket-slurm/pp-glm52-dsa-diagnostic-<jobid>.out`
- stderr: `/home2/jslee202403/putpocket-slurm/pp-glm52-dsa-diagnostic-<jobid>.err`
- artifact root:
  `/local-data/jslee202403/putpocket-glm52-sglang-gate/artifacts/glm52-dsa-diagnostic/<jobid>`
- authoritative result: `<artifact-root>/diagnostic_manifest.json`
- capture manifest/checksums:
  `<artifact-root>/phase3/capture/{capture_manifest.json,SHA256SUMS}`
- official single-row result:
  `<artifact-root>/official/evaluation/eval_results.json`

`PASS` requires the exact four-GPU runtime, positive measured HBM headroom,
exact OFF/ON token equality, complete validated native evidence, and a completed
official per-row evaluator. The row may resolve or remain unresolved. Native
exposure limitations are `BLOCKED`; allocation/load/backend/equivalence/
validation failures are `FAIL`. None changes GPU count, format, backend,
quantization, model, host, or offload policy.
