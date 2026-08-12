# RunPod CUDA 12.9 Editable Runtime

The development image provides Ubuntu 22.04, CUDA 12.9.1, nvcc, compiler tools,
SSH/rsync, Vim, Zellij, Node.js LTS with npm/npx, the OpenAI Codex CLI, a
minimal OS Python for bootstrap dispatch, and a pinned `uv` binary.
It deliberately does not include the project source, project Python environment,
torch, vLLM, LMCache, model weights, caches, credentials, or experiment outputs.

`bubblewrap` is installed for general Linux compatibility. RunPod blocks the
user namespace it requires (`No permissions to create new namespace`), so it
does not provide an inner Codex sandbox here. The startup helper reconciles the
persistent non-secret Codex config to `sandbox_mode = "danger-full-access"`
and `approval_policy = "on-request"`; the outer RunPod Docker container is the
security boundary. Existing unrelated config and file-backed authentication
remain on the Network Volume.

Build the image from the repository root:

```bash
docker buildx build \
  --platform linux/amd64 \
  --file cloud/runpod/Dockerfile.dev-base \
  --tag "${PUTPOCKET_RUNPOD_IMAGE_REPO}:${PUTPOCKET_RUNPOD_IMAGE_VERSION}" \
  --tag "${PUTPOCKET_RUNPOD_IMAGE_REPO}:git-$(git rev-parse --short HEAD)" \
  --load \
  .
```

Use immutable tags such as:

```text
cuda12.9.1-ubuntu22.04-agent-v2
git-<short-sha>
```

Do not use `latest` as the only deployment tag.

The Network Volume owns the editable runtime:

```bash
cd /workspace/putpocket_dataset_mining
./scripts/env/bootstrap_sr.sh --preset runpod-dev
source scripts/env/env_activate.sh
```

For planning without mutation:

```bash
./scripts/env/bootstrap_sr.sh --preset runpod-dev --dry-run
```

The container startup helper is inert. It creates `/workspace/.private/codex`
with mode `0700` when `/workspace` is mounted, writes only non-secret Codex CLI
configuration, prints tool versions, and then runs `sleep infinity`. It does
not clone the repository, build vLLM, download models, start a server, run
Codex, or authenticate Codex.

For the published v2 image, leave the RunPod Start Command blank. The Docker
image CMD already runs `/usr/local/bin/putpocket-runpod-start`.

Codex authentication is a runtime action on a trusted private Pod:

```bash
codex login --device-auth
```

`CODEX_HOME` is:

```text
/workspace/.private/codex
```

Expected policy:

- `auth.json` contains access and refresh tokens.
- Never copy `auth.json` into the image.
- Never commit it, upload it to Docker Hub, print it in logs, or include it in
  a Docker build context.
- Keep `CODEX_HOME` mode `0700`; after login, keep `auth.json` mode `0600`.
- Reuse the same private Network Volume for serialized later Pod launches.
- Do not share one `auth.json` across concurrently active Pods or machines.
- Reauthenticate only when refresh fails, access is revoked, or the credential
  file is removed.

Alternative automation mode is `OPENAI_API_KEY` supplied through a RunPod
Secret at runtime. Do not place API keys in Dockerfile `ENV`/`ARG`, labels,
template plaintext, or logs.

Python-only vLLM changes are visible through the editable install. C++ or CUDA
changes require an incremental native rebuild:

```bash
cd /workspace/putpocket_dataset_mining/externals/vllm
export TORCH_CUDA_ARCH_LIST="8.6 9.0 10.0 12.0"
export PUTPOCKET_BUILD_JOBS="$(nproc)"
export MAX_JOBS="${PUTPOCKET_BUILD_JOBS}"
export CMAKE_BUILD_PARALLEL_LEVEL="${PUTPOCKET_BUILD_JOBS}"
export NVCC_THREADS=1
python use_existing_torch.py
uv pip install -r requirements/build/cuda.txt
CCACHE_NOHASHDIR=true uv pip install --no-build-isolation -e .
```

The default `runpod-dev` architecture profile is portable across the current
NVIDIA targets:

```bash
export PUTPOCKET_CUDA_ARCH_PROFILE=portable-nvidia
export PUTPOCKET_CUDA_ARCH_LIST="8.6 9.0 10.0 12.0"
export TORCH_CUDA_ARCH_LIST="${PUTPOCKET_CUDA_ARCH_LIST}"
```

Architecture profiles:

| Profile | GPU target | TORCH_CUDA_ARCH_LIST |
| --- | --- | --- |
| `portable-nvidia` | RTX 3090, H100/H200, B200/GB200, RTX PRO 6000 Blackwell | `8.6 9.0 10.0 12.0` |
| `rtx3090` | RTX 3090 | `8.6` |
| `hopper` | H100/H200 | `9.0` |
| `blackwell-datacenter` | B200/GB200-class Blackwell | `10.0` |
| `blackwell-rtx` | RTX PRO 6000 Blackwell Server Edition | `12.0` |
| `native` | Explicit visible-GPU detection | detected supported capability only |

The portable build is slower and produces larger native artifacts than a
single-architecture development build. Use `blackwell-rtx` for the first RTX
PRO 6000 smoke and `hopper` for H100/H200-specific iteration:

```bash
./scripts/env/bootstrap_sr.sh --preset runpod-dev --cuda-arch-profile blackwell-rtx --dry-run
./scripts/env/bootstrap_sr.sh --preset runpod-dev --cuda-arch-profile blackwell-rtx --build-jobs "$(nproc)"
./scripts/env/bootstrap_sr.sh --preset runpod-dev --cuda-arch-profile hopper
```

`--cuda-arch-list` has highest precedence and is useful for one-off narrow
builds:

```bash
./scripts/env/bootstrap_sr.sh --preset runpod-dev --cuda-arch-list "9.0"
```

The same source tree does not make a native artifact portable across Python
ABI, torch version, CUDA runtime, glibc, or base-image digest changes. A
runtime/build manifest must match before reusing a persistent native build.

The first bootstrap must fail closed until the torch `2.10.0+cu129` wheel/index
provenance is resolved in `configs/env/torch/torch_2_10_cu129.lock.yaml`.
