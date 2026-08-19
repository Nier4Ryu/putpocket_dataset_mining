# GLM-5.2 vLLM native-DSA diagnostic

This is a two-job feasibility diagnostic, not a SWE-bench quality run. It
builds vLLM from pinned source in a CPU Slurm allocation and permits the exact
four-H200 run only through `afterok:<BUILD_JOB_ID>`. It executes one hard-pinned
SWE-bench Pro instance, records the unchanged official one-row evaluation
result, and can never select the full split or calculate the 40% acceptance
threshold.

For this exact vLLM pin, official source establishes that native DeepGEMM CUDA
kernels have unavoidable first-use runtime JIT and AOT is explicitly a TODO.
After validating the CPU-built SM90 bundle, the H200 job permits only that
technically inseparable native startup JIT. It uses a fresh run-local cache,
audited compiler wrappers, exact timestamps/environment/server command/logs,
the vLLM post-warmup JIT monitor in error mode, and a checksummed post-JIT
manifest. It never rebuilds vLLM or general project sources on H200. An
unallowlisted compiler command, identity mismatch, non-native cache component,
or missing provenance is classified BLOCKED.

## Immutable contract

- vLLM source: `4a3447d200e5aa428d68d1a00aa00f1a19a1a729`
- model: `nvidia/GLM-5.2-NVFP4` at
  `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`
- source build base: `nvidia/cuda:13.0.3-devel-ubuntu22.04` at OCI index
  digest `sha256:3869b846a8cc495ce11c172d87cfc0da8874b910d14a9810bec6b6182e9ee9f8`
  (`linux/amd64` manifest
  `sha256:dfb10241e392163c6ac5b8de0e55ef486b99f4e82075e6f5619cef243ef2aafb`)
- build: Python 3.12, PyTorch 2.13.0, CUDA 13.0.3,
  `TORCH_CUDA_ARCH_LIST=9.0`, CMake architecture 90, target `cuda`, and no
  precompiled vLLM wheel
- runtime: TP4, `modelopt_fp4`, `--linear-backend marlin`,
  `--attention-backend FLASHMLA_SPARSE`, context 4096, concurrency one,
  prefix caching disabled, eager execution, and no CPU offload/speculation/PD
- case:
  `instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5`
- row SHA-256:
  `78ff3ac298f276dfafaa311c26b7ace35be7c52d9094b4a6f658de2e7b5e25d1`
- exact serialized prompt: 2,071 tokens / 8,629 UTF-8 bytes / SHA-256
  `25d6597314bbb6c4df5afa886b064bebbdb7b57d27414d1e584e6a0127eeeab5`

The source/build lock is
`configs/cluster/glm52_vllm_diagnostic.lock.json`. The Herdr site file records
the measured CPU allocation contract (`cpu-max24`, `gsai-account`, `nogpu`, 24
CPUs, 192G, six hours), `/usr/bin/docker`, and node-local build scratch below
`/local-data/user-data/jslee202403`. The CLI still requires these values on the
render command so an orchestrator must state, and can audit, the exact site
contract used for each submission.

## Native instrumentation

The tracked patch instruments only
`vllm/model_executor/layers/sparse_attn_indexer.py`; its separate build-only
Dockerfile hunk removes upstream's local `VLLM_USE_PRECOMPILED` shortcut so
the CPU allocation builds the complete final wheel directly from source. At
the pinned commit the indexer file exposes the native prefill
`fp8_fp4_mqa_logits` tensor immediately before
`top_k_per_row_prefill`, and the native paged-decode
`fp8_fp4_paged_mqa_logits` tensor immediately before the Hopper
`cooperative_topk` branch. The hook records those already-produced logits and
the already-selected logical indices; it does not replace attention, change
top-k, or recompute scores. Decode capture refuses persistent/generic top-k,
DCP, non-SM90 dispatch, invalid logical coordinates, or an unexposed raw row.

Only prefill-last and existing decode steps 0, 1, 8, and 32 are captured for
all 21 full-indexer layers on all four TP ranks. The 57 shared-layer mapping is
locked and hashed. Rank-local JSONL records include the complete causal raw
vector, all 2,048 native IDs and corresponding scores, backend/revision
identity, and coordinate semantics. Finalization verifies finite values,
length/bounds, ID-score equality, mathematical top-k consistency, coverage,
and deterministic gzip/checksums. Missing native exposure produces a preserved
`BLOCKED`, never a pass.

## Build bundle gate

The CPU entrypoint validates the exact vLLM source file hashes, applies the
tracked patch, validates post-patch hashes, and uses the pinned official vLLM
Dockerfile to build the wheel and runtime image from scratch. It publishes an
atomic bundle below:

`/home2/jslee202403/putpocket-builds/vllm/vllm-4a3447d200e5-sm90-cu1303-py312-torch2130-patch-fc2f3734-image-3869b846`

The bundle contains the wheel, patched source bundle, saved runtime image,
build logs, resolved build/runtime package inventories, nvcc identities,
`SHA256SUMS`, manifest, and `SUCCESS`. Reuse is allowed only after every
identity, recorded byte count, provenance file, and file digest is revalidated.
The H200 entrypoint validates the bundle and SM90 compiled imports before model
config or weight download.

The compiled-import probe writes
`<run-artifacts>/phase1/compiled_import_probe.log`. It labels the container,
runtime nvcc, torch, vLLM extension, sparse-indexer, ModelOpt W4A16, diagnostic
hook, and SM90-capability steps independently. If any step exits nonzero, the
entrypoint replays the complete log between
`COMPILED_SM90_IMPORT_PROBE_LOG_BEGIN/END` markers to the shared Slurm stderr
before returning exit 31. This changes observability only: every original
import, the CUDA 13.0 check, and the SM90 assertion remain fail closed.

The CPU build deliberately passes upstream Docker build argument
`RUN_WHEEL_CHECK=false`. That disables only vLLM's 500 MB release-packaging
size policy for this intentional CUDA 13.0.3, SM90-only wheel; it does not
disable compilation, wheel cardinality/existence checks, compiled-SM90
inspection, runtime-image creation, or bundle validation. Before the runtime
image build, `logs/wheel_artifact.json` records the wheel's exact byte size and
SHA-256. `build_manifest.json` repeats those values under
`wheel_release_policy`, and validation requires them to match the immutable
`files.vllm_wheel` entry and `SHA256SUMS`.

## Runtime gates

The H200 job first proves one node, exactly four visible physical H200 GPUs,
full 141GB-class memory, MIG disabled, SM90, typed Slurm GRES, UUIDs,
topology/NVLink, driver, CUDA identities, and Slurm TRES. It then performs the
weightless build/runtime identity probe, arms the run-local native-JIT audit
before model weights, resolves the immutable model revision, loads all-resident
TP4, and samples per-GPU HBM. Startup JIT artifacts are accepted only when the
compiler/cache manifest proves the pinned vLLM/patch/image/Torch/CUDA/SM90
identity; the cache is then made read-only before inference.

The diagnostic path sends the same live server trace OFF and trace ON with the same tokenized
prompt, seed zero, greedy decoding, and generation bound. Prefix caching is
disabled and `/reset_prefix_cache` must succeed before each serial request.
Exact output token IDs and hashes must match. The server is stopped before the
model action is applied in the official row image and unchanged pinned
`swe_bench_pro_eval.py` runs for that row only.

## Rendering with measured site inventory

From an exact project checkout:

```bash
PYTHONPATH=src python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli \
  render-wrap \
  --site configs/cluster/sites/herdr_vllm_diagnostic.json \
  --project-url https://github.com/Nier4Ryu/putpocket_dataset_mining.git \
  --project-commit <exact-40-char-project-commit> \
  --cpu-partition cpu-max24 \
  --cpu-account gsai-account \
  --cpu-qos nogpu \
  --cpu-cpus-per-task 24 \
  --cpu-memory 192G \
  --cpu-wall-time 06:00:00 \
  --cpu-local-scratch-root /local-data/user-data/jslee202403/putpocket-vllm-build-scratch \
  --container-executable /usr/bin/docker
```

The one-line result creates only the shared build/log parent directories on
Login, submits the CPU job, validates its parsable ID, submits the exact H200
job with `afterok`, and prints `BUILD_JOB_ID` and `RUN_JOB_ID`. Clone, patch,
build, downloads, installs, image operations, inference, and evaluation occur
only inside allocated compute jobs. Before the H200 wrapper clones the project,
it requires the measured `/local-data/user-data` parent to exist and be
writable, creates the configured work/artifact roots, and verifies both are
writable. A node with a different or unavailable local-storage layout exits
before project, model metadata, or weight download.

Expected paths:

- build stdout/stderr:
  `/home2/jslee202403/putpocket-slurm/pp-vllm-sm90-build-<BUILD_JOB_ID>.{out,err}`
- run stdout/stderr:
  `/home2/jslee202403/putpocket-slurm/pp-glm52-vllm-dsa-<RUN_JOB_ID>.{out,err}`
- run artifacts:
  `/local-data/user-data/jslee202403/putpocket-glm52-vllm-diagnostic/artifacts/<RUN_JOB_ID>`
- result: `<run-artifacts>/diagnostic_manifest.json`
- runtime JIT provenance:
  `<run-artifacts>/phase2/runtime_jit/runtime_jit_manifest.json`
- runtime JIT checksums:
  `<run-artifacts>/phase2/runtime_jit/runtime_jit_SHA256SUMS`

For an exit-31 `COMPILED_SM90_IMPORT_PROBE_FAILED`, retain the shared run stderr
and recover these node-local files before diagnosing a dependency, driver, or
extension issue: `phase1/compiled_import_probe.log`, `phase1/image_load.log`,
`phase1/runtime_image_identity.txt`, `phase1/build_bundle_validation.json`,
`phase0/gpu_inventory.csv`, `phase0/driver_versions.csv`, and
`diagnostic_manifest.json`. The generic failure class alone does not identify
which command or import failed.

The older SGLang renderer and entrypoints are retained only as historical,
unreachable package evidence. They are not called by this renderer or either
vLLM entrypoint.
