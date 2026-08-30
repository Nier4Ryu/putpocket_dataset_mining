# GLM-5.3-Flash on three RTX PRO 6000 GPUs

This package deploys a bounded GLM-5.3-Flash NVFP4 endpoint on a single
three-GPU SM120 host without changing the canonical PutPocket runtime or the
GLM-5.2 environment. It is task-local, offline at serve time, and fail-closed
on model, runtime, GPU ownership, or parallel-layout drift.

## Immutable selection and provenance

The machine authority is
`configs/models/glm53_flash_nvfp4_montblanc.lock.json`.

The official Z-AI repositories audited on 2026-08-30 were:

| Checkpoint | Revision | Safetensor weight bytes | License evidence | Three-card result |
| --- | --- | ---: | --- | --- |
| `zai-org/GLM-5.3` (native FP8) | `e0b07fd2751b42d5efa199cc02c2b271deadc516` | 755,632,050,320 | HF tag `other` | cannot fit |
| `zai-org/GLM-5.3-BF16` | `304b805e31454b8417c1c69fba3df65a24583dc7` | 1,506,667,387,408 | HF tag `other` | cannot fit |
| `zai-org/GLM-5.3-Flash` (native FP8) | `04c4e9e95c5da8862dced7e5056455116f83a7e0` | 328,337,455,672 | MIT | cannot fit |
| `zai-org/GLM-5.3-Flash-BF16` | `f12e0fe852d7c7f0fe61fac952fc998a096e3814` | 642,652,070,880 | MIT | cannot fit |

The selected derivative is
[`RedHatAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/RedHatAI/GLM-5.3-Flash-NVFP4)
at `36c184c6cda000a481711306df5adde42f63321a`. Its model card and checked-in
LLM Compressor recipe identify the immutable official Flash checkpoint as the
base and describe compressed-tensors NVFP4 W4A4 quantization. The selected
revision has no derivative `LICENSE` file; the base revision has an MIT file
with SHA-256
`30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1`.
That is retained as a provenance caveat rather than being presented as a new
license grant.

The ten main shards, exact tokenizer/config artifacts, and the index-referenced
7,618,560,424-byte MTP shard are downloaded: 197,881,155,500 bytes (184.291
GiB). The MTP file is required because the immutable upstream index maps 1,753
layer-45 tensor names to it. MTP execution remains disabled, so these stored
weights are not included in the 45-layer serving payload estimate.

## Why this layout fits

The host contract is three `NVIDIA RTX PRO 6000 Blackwell Server Edition`
cards, 97,887 MiB each, compute capability 12.0. Header-level tensor accounting
for the selected shards gives:

- routed-expert payload: 171,228,556,800 bytes;
- replicated non-expert payload: 18,975,948,412 bytes;
- TP1/DP3/EP3 payload estimate per rank: 76,052,134,012 bytes (70.829 GiB).

TP=3 is invalid because 64 attention heads and hidden size 4096 are not
divisible by three. The pinned GLM implementation explicitly gates pipeline
parallelism off. The supported three-card plan is therefore TP=1, DP=3 with
expert parallelism, 96 of 288 experts per rank, and early expert-weight
filtering. Attention and non-expert weights are replicated. The smoke bounds
context to 4,096, one sequence, FP8 MLA KV, no MTP, and no prefix cache.

## SM120 runtime boundary

The first dedicated vLLM GLM-5.3 image cannot serve Flash's RoPE-free MLA on
SM120; [vLLM issue 53963](https://github.com/vllm-project/vllm/issues/53963)
records the exact missing kernel shape. This package builds
[vLLM PR 53906](https://github.com/vllm-project/vllm/pull/53906) at
`878631b6079d2cf9fb80830ef9cb41b43aded098` and overlays
[FlashInfer PR 4802](https://github.com/flashinfer-ai/flashinfer/pull/4802) at
`c2eec117219457e45126fa4fa87e7240dd4ea620`. The latter supplies the SM120
`GLM53_NOPE` query-512 path with effective top-k 2,176 (2,048 selection plus
the 128-wide tail tile). The pinned vLLM PR's Dockerfile references unpublished
FlashInfer 0.6.18rc10 release assets; a checksum-bound packaging-only patch
skips both the base install and the post-vLLM-wheel restore steps. The exact
FlashInfer PR source wheel is then
installed without an AOT cache, so its module is JIT-compiled into a task-local
cache. The patch does not change vLLM Python or CUDA runtime source.

The image is deliberately host-specific: vLLM's supported
`torch_cuda_arch_list` build argument is fixed to `12.0`. This retains the
SM120 NVFP4, Marlin MoE, router, cache, and extension intersections selected by
upstream CMake while avoiding seven unused architecture variants. FlashAttention
keeps upstream's per-kernel forward-compatible architecture defaults; it is not
the GLM-5.3 main-attention authority, which is the separately pinned
`FLASHINFER_MLA_SPARSE_SM120` source path. Image inspection fails unless
`TORCH_CUDA_ARCH_LIST=12.0` is embedded in the final environment.

The compressed-tensors checkpoint and Marlin MoE backend are deliberate.
[vLLM issue 54150](https://github.com/vllm-project/vllm/issues/54150) reports
clean output for this RedHatAI checkpoint on the same SM120 GPU family and
invalid output for the compared ModelOpt conversions. This package does not
use third-party desktop-client repositories or prebuilt community runtime
images.

## End-to-end runbook

All mutating setup commands acquire the repository advisory `build` lock.
They require the explicit task-worktree production override authorized for
this deployment.

```bash
export PUTPOCKET_ALLOW_TASK_PRODUCTION=1
export GLM53_RUNTIME_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/putpocket-runtime/T20260830-001__glm53-montblanc-deployment"
export GLM53_MODEL_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/putpocket-models/T20260830-001__glm53-montblanc-deployment/RedHatAI--GLM-5.3-Flash-NVFP4--36c184c6"

./scripts/glm53/download_model.sh
./scripts/glm53/bootstrap_runtime.sh

RUN_DIR="$(./scripts/glm53/launch_server.sh)"
./scripts/glm53/run_smoke.sh "$RUN_DIR"
./scripts/glm53/stop_server.sh "$RUN_DIR"
```

`launch_server.sh` refuses an occupied port, non-idle GPUs, model drift,
runtime-label drift, or an existing run directory. It publishes only on
`127.0.0.1:8137`. The server has offline Hugging Face/Transformers flags and
sees the checkpoint read-only. The stop script resolves one exact CID, verifies
both task and run labels, archives logs/telemetry, and uses SIGTERM with an
infinite Docker stop timeout so it never escalates to SIGKILL.

Montblanc's Docker daemon has no NVIDIA Container Toolkit or CDI runtime. The
launch therefore fails closed on driver drift and passes only the six required
NVIDIA character devices plus five resolved compute-driver libraries into the
container read-only. This task-local, driver-580.159.03 contract avoids a
machine-wide Docker daemon reconfiguration. A minimal Torch probe must see
exactly three SM120 devices before the production launch is accepted.

The smoke performs three checks against the same ready server:

1. `/v1/models` exposes `glm-5.3-flash-nvfp4`;
2. two seeded temperature-zero `/v1/chat/completions` calls match exactly;
3. PutPocket's `OpenAICompatibleHTTPGenerationEngine` sends the exact locally
   chat-templated rendered prompt through `/v1/completions` without applying a
   second server chat template.

The example integration config is
`configs/execution/server2_glm53_flash_nvfp4_ep3.example.yaml`. It documents a
server endpoint; it does not replace the existing local-Python vLLM engine for
other jobs.
