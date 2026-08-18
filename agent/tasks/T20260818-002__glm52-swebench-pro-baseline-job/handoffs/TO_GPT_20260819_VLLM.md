# vLLM two-stage GLM-5.2 diagnostic handoff

This handoff records the tested implementation commit. The orchestrator must
still supply measured CPU-site values before rendering or submission.

## Authoritative runtime

- vLLM source: `4a3447d200e5aa428d68d1a00aa00f1a19a1a729`
- native target preimage SHA-256:
  `a6386326379dcf94439a0b020cd51a71ebfd0a36eaeb563417c821456b82a475`
- native target postimage SHA-256:
  `65ef4a917b35cc5298ab6e93c7351db3347cbe891f0ebc39cb115e86bb49b3dd`
- direct-from-source Docker build preimage SHA-256:
  `a50d83daedbd5992259c4257533649f42162fe3cf65565a18c3a7e5f905fccdf`
- direct-from-source Docker build postimage SHA-256:
  `f7c56f7c9100285057388cff5b7b074571853f6a3e552ee9cbdebe3221d4f71d`
- patch SHA-256:
  `fc2f3734225c077fd9cfaf08341e2eaf01955a8cfd1cf1bee3c1747accfe5a9b`
- instrumentation SHA-256:
  `1dc2872ddb58fa719290e1f954c643819f4409c2ab7b1a1f78d701c13848c516`
- compiler-audit wrapper SHA-256:
  `e060c0b09e1c2eb4da90854ee81d284f7f6acce2b7375b9ee87ecf956a78f9ee`
- build base OCI index:
  `nvidia/cuda:13.0.3-devel-ubuntu22.04@sha256:3869b846a8cc495ce11c172d87cfc0da8874b910d14a9810bec6b6182e9ee9f8`
- build base linux/amd64 manifest:
  `sha256:dfb10241e392163c6ac5b8de0e55ef486b99f4e82075e6f5619cef243ef2aafb`
- build contract: Python 3.12 / PyTorch 2.13.0 / CUDA 13.0.3 / SM90,
  with upstream's final-wheel `VLLM_USE_PRECOMPILED` shortcut patched out
- immutable bundle key:
  `vllm-4a3447d200e5-sm90-cu1303-py312-torch2130-patch-fc2f3734-image-3869b846`
- model: `nvidia/GLM-5.2-NVFP4` at
  `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`

The vLLM source exposes native raw logits and selected indices at the Python
dispatch boundary in `sparse_attn_indexer.py`: prefill immediately after
`top_k_per_row_prefill`, decode immediately after the Hopper-only
`cooperative_topk`. The patch observes these existing tensors before DCP
merge. It does not substitute dense attention, recompute scores, or change
selection. Any alternate decode selector, DCP, coordinate ambiguity, raw
non-exposure, or coverage/validation failure is classified BLOCKED/FAIL.

The same pinned official source proves that DeepGEMM CUDA-kernel AOT is a TODO
and first-use native JIT is inseparable. The H200 entrypoint therefore permits
only native DeepGEMM/DSA startup JIT after CPU-bundle validation. Before model
weights it arms exact identity/environment/audit controls and a fresh run-local
cache. It records compiler commands/timestamps/logs, checksums every cache
file, rejects non-native/general compilation, freezes the cache read-only after
warmup, and enables vLLM's post-warmup JIT monitor in error mode. Missing or
out-of-scope provenance is BLOCKED; vLLM itself is never rebuilt on H200.

## Fixed row

- instance:
  `instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5`
- row SHA-256:
  `78ff3ac298f276dfafaa311c26b7ace35be7c52d9094b4a6f658de2e7b5e25d1`
- serialized prompt: 2,071 tokens, 8,629 bytes, SHA-256
  `25d6597314bbb6c4df5afa886b064bebbdb7b57d27414d1e584e6a0127eeeab5`
- tokenizer/model revision:
  `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`

## Site fields still required

Supply measured CPU build values only: partition, account, qos, CPUs, memory,
wall time, node-local scratch root, and container executable. The committed
site file keeps them null and rendering fails closed until all are supplied.
The H200 job is fixed to H200/gsai-account/hpgpu, one node, exactly four typed
H200 GPUs, 32 CPUs, 512G, and six hours.

## Paths

- shared bundle root: `/home2/jslee202403/putpocket-builds/vllm`
- Slurm logs: `/home2/jslee202403/putpocket-slurm`
- run artifacts:
  `/local-data/jslee202403/putpocket-glm52-vllm-diagnostic/artifacts/<RUN_JOB_ID>`
- final classified result: `<run-artifacts>/diagnostic_manifest.json`

## Git/test result

- implementation/source commit:
  `bb33d316f47bbabba2ffcd6db9682b8e0b3b3fcc`
- final branch tip: the docs-only handoff commit containing this record
- pushed branch: `agent/T20260818-001__glm52-cluster-package-foundation`
- focused/regression tests: 106 passed plus 59 subtests
- broad safe CPU tests: 284 passed plus 85 subtests; only
  `tests/test_glm_h192_triton.py` was excluded because it queries CUDA
- exact upstream preimage/patch/postimage validation: passed
- shell syntax: both entrypoints, compiler wrapper, outer renderer, and both
  decoded wrappers passed `bash -n`; shellcheck was unavailable
- no Login/Slurm/tracker/GPU/model-weight access occurred
