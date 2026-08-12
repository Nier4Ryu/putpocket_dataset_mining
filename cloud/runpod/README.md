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
python use_existing_torch.py
uv pip install -r requirements/build/cuda.txt
CCACHE_NOHASHDIR=true uv pip install --no-build-isolation -e .
```

The default architecture profile is portable across the current NVIDIA targets:

```bash
export PUTPOCKET_CUDA_ARCH_PROFILE=portable-nvidia
export PUTPOCKET_CUDA_ARCH_LIST="8.6 9.0 10.0 12.0"
export TORCH_CUDA_ARCH_LIST="${PUTPOCKET_CUDA_ARCH_LIST}"
```

The first bootstrap must fail closed until the torch `2.10.0+cu129` wheel/index
provenance is resolved in `configs/env/torch/torch_2_10_cu129.lock.yaml`.
