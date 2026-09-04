# GLM-5.3 A6000 compatibility-gated image

This package builds current upstream vLLM commit
`9cd956c7e6cf54efa366b803cafa15ec6c2df827` for CUDA architecture 8.6 and
applies the latest PutPocket GLM-5.3 native-indexer capture and
stateful-edit-v3 overlay. It is a real vLLM source containing registered
`Glm5NextForConditionalGeneration` code. It is **not** a runnable GLM-5.3
server on RTX A6000 today.

## Compatibility verdict

The official GLM-5.3-Flash recipe requires Hopper or newer. At the pinned
source, the sparse FlashAttention MLA backend accepts compute-capability major
9, FlashMLA sparse accepts majors 9 or 10, and the remaining NVIDIA sparse
backend is specific to newer hardware. Dense Triton MLA cannot be substituted:
vLLM validates sparse and non-sparse attention as different contracts. No
official SM86 native compressed-pool indexer execution path is available.

Accordingly, the image doctor reports the exact blocker and `serve` exits
before model loading. The default lane contains no unmerged research Triton
patch. vLLM v0.28.0 is recorded only as rejected comparison evidence because
it does not register the GLM5Next architecture.

## Model and stateful contract

The only model binding in this package is official
`zai-org/GLM-5.3-Flash-BF16` revision
`a5b45eb41df6402735dedc900be14a42e8d5e538`. Its indexed weight payload is
642,646,653,816 bytes across 120 shards; weights are not downloaded or copied
into the image. A single 48-GB A6000 cannot hold it. This package does not
invent a GPU-count or tensor-parallel configuration because there is no
supported SM86 backend to validate such a topology.

The frozen scientific semantics remain:

- 45 layers: 11 MLA/indexer layers `3,7,...,43`, 34 KDA layers;
- native compressed indexer pools use `kpool=4`, and only complete selected
  pools may be transplanted; incomplete tails remain target-computed;
- KDA state remains target-computed;
- selector positions and provenance are digest-attested;
- stateful edit is default OFF and fail-closed;
- the implementation is full target prefill followed by selected donor-row
  overwrite, an accuracy ablation with no compute-saving or speedup claim.

The source hook is wired at current vLLM cache/indexer/model signatures and
then refuses activation on SM86. This preserves a directly inspectable port
without pretending that an unsupported kernel exists.

## CPU-only build

The source preparation applies two checksum-bound patches in order: the
SM86-only packaging patch, then the default-OFF runtime hook patch. The build
profile sets all three controls below and fails if the resulting vLLM wheel
contains task-built CUDA payloads for any other architecture:

- `PUTPOCKET_SM86_ONLY=1`;
- `TORCH_CUDA_ARCH_LIST=8.6`;
- `CMAKE_CUDA_ARCHITECTURES=86`.

This profile does not fetch, build, copy, or install DeepEP. It also excludes
bundled FA2/FA3/FA4, FlashMLA, FlashKDA, DeepGEMM, QuTLASS, the SM100 FMHA
package, and the optional TML-FA4 package from the wheel targets. Bundled FA2
is omitted because its pinned external project deliberately lowers an 8.6
target to `sm_80` plus `compute_80` PTX. Remaining vLLM CUDA sources with an
Ampere-compatible implementation are compiled specifically for `sm_86`;
loose lower-Ampere PTX selection is not allowed. Prebuilt FlashInfer
cubins/JIT cache, Cutlass-DSL/QuACK,
Tokenspeed-MLA, and Humming kernel packages are omitted from this gated lane.
The thin compatibility wrapper also removes the upstream CUTLASS-DSL runtime
distribution family because this blocked SM86 lane cannot use its optional
newer-architecture kernels.

The binary proof is deliberately scoped to CUDA code compiled by this task
into the vLLM wheel. The upstream PyTorch wheel, CUDA toolkit/runtime, and
NVIDIA dependency distributions remain third-party vendor binaries and may be
multi-architecture. The package does not make an unverifiable whole-image
single-architecture claim about those vendor files. It records them as an
explicit excluded provenance category while proving every task-built vLLM
CUDA ELF/PTX payload is `sm_86` with static `cuobjdump` inspection.

Prepare a fresh source tree. The task-local clone intentionally retains its
own `.git` directory because the upstream Dockerfile bind-mounts it while
building Rust and version metadata:

```bash
./scripts/glm53_a6000/prepare_vllm_source.sh \
  /path/to/clean/vllm-at-9cd956c7 \
  /new/task-local/prepared-vllm
```

Build without passing any NVIDIA device or GPU option:

```bash
MAX_JOBS=8 NVCC_THREADS=2 \
  ./scripts/glm53_a6000/build_image.sh /new/task-local/prepared-vllm
```

The upstream build receives the complete SM86 profile above. The downstream
image contains no model weights, credentials, host paths, or implicit model
download. `scripts/glm53_a6000/audit_sm86_build_log.py` verifies the successful
log's SM86 configuration/omission markers and rejects any visible non-8.6
compiler flag. Ninja need not print full commands, so the embedded wheel
`cuobjdump` report—not log verbosity—is the binary-architecture authority.

Static doctor (no torch/CUDA import and no device probe):

```bash
docker run --rm --network none \
  putpocket/glm53-a6000-gated:9cd956c7 doctor
```

The checked-in launch template exits without starting Docker. A future launch
must not be enabled until an official SM86 sparse MLA/indexer backend exists,
the lock is revised, and an explicit read-only model mount, GPU topology,
backend, quantization, and KV dtype have been reviewed. The intended future
OpenAI endpoint is `http://127.0.0.1:8000/v1`; the stateful proxy would be
`http://127.0.0.1:18000/v1` only after that gate is removed.
