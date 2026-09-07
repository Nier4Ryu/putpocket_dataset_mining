# GLM-5.3 RunPod SM90 + SM120 image

This package builds one Linux/amd64 image for four H100/H200 GPUs (Hopper,
SM90) or four RTX PRO 6000 Blackwell GPUs (SM120). It contains vLLM source at
`9cd956c7e6cf54efa366b803cafa15ec6c2df827`, the native GLM5Next model,
compressed-pool indexer path, PutPocket selector attestations, proxy, and the
stateful-edit-v3 accuracy ablation. The Montblanc build is CPU-only. Neither
GPU profile is considered validated until its separate RunPod smoke passes.

## Frozen model and runtime

The default model is `Intel/GLM-5.3-Flash-W4A16-AutoRound` at revision
`5eee1846f0321058ed73745f9aa16f2aaf0fc0a0`. The repository payload is
181,505,393,058 bytes, of which 181,472,931,028 bytes are 36 safetensors
files. It is AutoRound INT4 W4A16, group size 128, symmetric, GPTQ-packed;
vLLM maps the publisher's `auto-round` metadata to its `inc` loader. The model
card reports 99.84% relative average accuracy to BF16; that publisher result
has not been reproduced by PutPocket.

Weights are never in the image and silent downloads are disabled. Download
the exact revision into a persistent RunPod Network Volume before launch:

```bash
hf download Intel/GLM-5.3-Flash-W4A16-AutoRound \
  --revision 5eee1846f0321058ed73745f9aa16f2aaf0fc0a0 \
  --local-dir /workspace/models/glm53-flash-w4a16-autoround
```

Authentication, if Hugging Face requires it, is supplied by the operator; no
token is embedded. Compare metadata hashes against
`configs/models/glm53_flash_w4a16_runpod_sm90_sm120.lock.json` before loading.

The validated topology contract is TP=4, PP=1, DP=1, EP=4 using
`allgather_reducescatter`. Sixty-four attention heads, 32 indexer heads, and
288 experts are divisible by four. The under-200GB weight payload leaves
aggregate memory for runtime/KV on 4x80GB H100, 4x141GB H200, or 4x96GB RTX
PRO 6000, but actual per-rank workspace and KV headroom remain GPU-smoke
measurements rather than CPU-build claims.

## Architecture profiles

`PUTPOCKET_GLM53_RUNTIME_PROFILE=sm90` forces
`FLASHINFER_MLA_SPARSE_SM90`, `fp8_e4m3` KV, and block size 128. The `sm120`
profile forces `FLASHINFER_MLA_SPARSE_SM120`, `fp8_ds_mla` KV, and block size
512. Runtime doctor rejects mixed/wrong compute capabilities and any GPU count
other than four. It also validates the mounted model metadata before vLLM is
started. No backend, quantization, KV dtype, topology, or model fallback is
allowed.

The source overlay builds task-owned vLLM code for `9.0a 12.0a`. DeepGEMM is
required for native pre-top-k indexer logits and FlashKDA for the 34 KDA
layers. FlashInfer 0.6.18 provides the sparse MLA backends/JIT cache. The
validated SM120 NoPE cache and active-top-k-length fixes are applied to the
current source. Bundled FA2/FA3, FlashMLA, QuTLASS, fmha_sm100, tml_fa4, and
DeepEP are omitted because the forced GLM profiles do not use them. Vendor
PyTorch/CUDA/FlashInfer distributions can themselves be multi-architecture;
the SM90+SM120 binary audit is scoped to task-built vLLM/PutPocket shared
objects. DeepGEMM's native binding is built only for the image's Python 3.12
runtime rather than packaging unused bindings for every supported Python.
The audit requires native SM90 and SM120 code and rejects unknown targets; it
also records the upstream-supported SM80 Marlin/Marlin-MoE and SM89 c2x
compatibility SASS/PTX used by the W4A16 path. Those compatibility objects are
not described as native Hopper or Blackwell kernels.

## Build without a GPU

```bash
./scripts/glm53_runpod/prepare_vllm_source.sh \
  /path/to/clean/vllm-9cd956c7 \
  /new/task-local/vllm-glm53-sm90-sm120
GLM53_RUNPOD_IMAGE_TAG=putpocket/glm53-runpod-sm90-sm120-w4a16:9cd956c7-v1 \
  ./scripts/glm53_runpod/build_image.sh \
  /new/task-local/vllm-glm53-sm90-sm120
docker run --rm putpocket/glm53-runpod-sm90-sm120-w4a16:9cd956c7-v1 doctor
```

The final command is a static hash/source/install check. It does not import
torch, invoke a CUDA runtime API, inspect devices, or load weights.

## RunPod start commands

Mount a pre-downloaded model read-only, immutable episode/harness data
read-only, and results read-write. H100/H200 profile:

```bash
docker run --rm --gpus all --network host \
  -e PUTPOCKET_GLM53_RUNTIME_PROFILE=sm90 \
  -v /workspace/models/glm53-flash-w4a16-autoround:/models/glm53-flash-w4a16-autoround:ro \
  -v /workspace/putpocket-data:/data:ro \
  -v /workspace/results:/results \
  IMAGE runtime-doctor

docker run --rm --gpus all --network host \
  -e PUTPOCKET_GLM53_RUNTIME_PROFILE=sm90 \
  -v /workspace/models/glm53-flash-w4a16-autoround:/models/glm53-flash-w4a16-autoround:ro \
  -v /workspace/putpocket-data:/data:ro \
  -v /workspace/results:/results \
  IMAGE serve
```

For four RTX PRO 6000 Blackwell GPUs change only the profile to `sm120`.
Runtime doctor must pass before `serve`. The OpenAI-compatible endpoint is
`http://HOST:8000/v1`, model health is `http://HOST:8000/health`, and the
stateful proxy endpoint is `http://HOST:18000/v1`.

For the complete server -> proxy -> frozen multi-turn harness flow, mount an
executable `/data/run_putpocket_harness.sh`, `/data/phase.json`, and
`/data/frozen-episode.json`, then run:

```bash
docker run --rm --gpus all --network host \
  -e PUTPOCKET_GLM53_RUNTIME_PROFILE=sm90 \
  -e PUTPOCKET_RUN_ID=unique-run-id \
  -v /workspace/models/glm53-flash-w4a16-autoround:/models/glm53-flash-w4a16-autoround:ro \
  -v /workspace/putpocket-data:/data:ro \
  -v /workspace/results:/results \
  IMAGE experiment
```

The image deliberately does not bundle SWE-bench Pro repositories, gold
outcomes, model weights, or a mutable evaluation workspace. PutPocket authors
the frozen A1-execute-Q2 trajectory and system edit; SWE-bench Pro supplies the
problem repository and evaluator.

## Scientific boundary

The stateful hook is default OFF and fail closed. When explicitly armed it
runs a complete target prefill and then overwrites selector-chosen target
cache rows with private donor rows. The 11 MLA/indexer layers are
3,7,...,43; only complete four-token indexer pools can be transplanted; all 34
KDA layers and incomplete indexer tails stay target-computed. This is an
accuracy ablation, not true partial prefill, skipped compute, or a latency
speedup. Q2 and later tokens run normally after the overwrite. Existing RoPE
behavior is unchanged (GLM-5.3 sparse MLA is configured NoPE).
