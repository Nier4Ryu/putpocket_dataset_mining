# vLLM two-stage GLM-5.2 diagnostic handoff

This handoff records the tested implementation and measured-site correction.
The orchestrator owns rendering/submission; Codex made no Login or Slurm
connection.

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

## Measured site contract

- CPU: `cpu-max24` / `gsai-account` / `nogpu`, one node, 24 CPUs,
  192G, 06:00:00, `/usr/bin/docker`
- CPU scratch:
  `/local-data/user-data/jslee202403/putpocket-vllm-build-scratch`
- H200: `H200` / `gsai-account` / `hpgpu`, one node, exactly
  `--gres=gpu:H200:4`, 32 CPUs, 512G, 06:00:00
- measured local-storage parent: `/local-data/user-data`
- H200 work root:
  `/local-data/user-data/jslee202403/putpocket-glm52-vllm-diagnostic`

The H200 allocation wrapper validates that the measured parent exists and is
writable, creates and validates the work/artifact roots, and only then fetches
the exact project commit. The tracked entrypoint independently revalidates the
same relationship before any model metadata or weights.

## Paths

- shared bundle root: `/home2/jslee202403/putpocket-builds/vllm`
- Slurm logs: `/home2/jslee202403/putpocket-slurm`
- run artifacts:
  `/local-data/user-data/jslee202403/putpocket-glm52-vllm-diagnostic/artifacts/<RUN_JOB_ID>`
- final classified result: `<run-artifacts>/diagnostic_manifest.json`

## Exact renderer

```bash
PYTHONPATH=src python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli render-wrap --site configs/cluster/sites/herdr_vllm_diagnostic.json --project-url https://github.com/Nier4Ryu/putpocket_dataset_mining.git --project-commit 186d096a99bbe7a86c8eb6dff5302f88774c9133 --cpu-partition cpu-max24 --cpu-account gsai-account --cpu-qos nogpu --cpu-cpus-per-task 24 --cpu-memory 192G --cpu-wall-time 06:00:00 --cpu-local-scratch-root /local-data/user-data/jslee202403/putpocket-vllm-build-scratch --container-executable /usr/bin/docker
```

The raw one-line output excluding its trailing newline is 5,291 bytes and has
SHA-256 `d57e0e782c25ada66c7b3bcc9b6ef2419c64662562adb9b17b45c40b6be495c0`.

## H200 import-probe incident

CPU build job `747490` completed and published the immutable bundle. Its wheel
is 603,540,543 bytes with SHA-256
`3c408df63c56e2a711116449d4324fcef5f2043de1b5c3dee4d3bf561908af52`.
H200 job `747491` then exited 31 at the compiled-import probe before model
metadata or weights. The pinned source contains every requested import symbol,
but source and bundle metadata cannot distinguish Docker/GPU initialization,
nvcc, extension loading, Python dependency, hook import, or capability-assert
failures. The exact prior traceback is only in
`artifacts/747491/phase1/compiled_import_probe.log` on n87.

Replacement source keeps all checks unchanged and makes the probe stepwise. On
failure it replays that log, with explicit begin/end markers and the container
exit code, into the shared Slurm stderr. The existing validated bundle remains
reusable; this runtime-script-only correction does not require recompiling the
wheel or rebuilding the runtime image. The historical renderer example above
must be re-rendered with the replacement pushed commit returned in the final
handoff; do not resubmit its older project commit.

## Git/test result

The Cluster CPU build job `746239` subsequently proved compilation and wheel
creation but hit only upstream's 500 MB release-wheel policy (`575.59 MB`
reported). Replacement source passes `RUN_WHEEL_CHECK=false` solely for that
policy. The allocated build records the exact rebuilt wheel bytes and SHA-256
in `logs/wheel_artifact.json`, `build_manifest.json#wheel_release_policy`, the
immutable `files.vllm_wheel` entry, and `SHA256SUMS`; bundle validation requires
all copies to agree before publication or H200 use.

Replacement-package verification: 31 focused vLLM tests passed; the focused
Cluster/GLM group passed 113 tests plus 59 subtests; the broad safe CPU suite
passed 291 tests plus 85 subtests with only the known CUDA-querying
`tests/test_glm_h192_triton.py` excluded. Both tracked entrypoints passed
`bash -n`.

- implementation/source commit:
  `186d096a99bbe7a86c8eb6dff5302f88774c9133`
- final branch tip: the docs-only handoff commit containing this record
- pushed branch: `agent/T20260818-001__glm52-cluster-package-foundation`
- focused measured-site/vLLM tests: 27 passed
- broad safe CPU tests: 287 passed plus 85 subtests; only
  `tests/test_glm_h192_triton.py` was excluded because it queries CUDA
- shell syntax: both tracked entrypoints, the exact outer renderer output, and
  both exact decoded wrappers passed `bash -n`; shellcheck was unavailable
- no Login/Slurm/tracker/GPU/model-weight access occurred
