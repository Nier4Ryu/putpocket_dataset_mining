# RunPod CUDA 12.9 Editable Runtime

The development image provides Ubuntu 22.04, CUDA 12.9.1, nvcc, compiler tools,
SSH/rsync, a minimal OS Python for bootstrap dispatch, and a pinned `uv` binary.
It deliberately does not include the project source, project Python environment,
torch, vLLM, LMCache, model weights, caches, or experiment outputs.

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

Python-only vLLM changes are visible through the editable install. C++ or CUDA
changes require an incremental native rebuild:

```bash
cd /workspace/putpocket_dataset_mining/externals/vllm
export TORCH_CUDA_ARCH_LIST="8.6 9.0 10.0 12.0"
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
