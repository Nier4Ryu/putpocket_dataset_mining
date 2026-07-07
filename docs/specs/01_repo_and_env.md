# Repo and Env Spec

This repo is standalone and owns its own environment.

## Required layout

```text
Putpocket_env/
scripts/env/env_activate.sh
externals/vllm/
externals/lmcache/
externals/cline/
src/putpocket_dataset_mining/
configs/dataset_mining/
docker/default_python/Dockerfile
data/dataset_mining/
```

## Env versions

- Python 3.13
- CUDA `/usr/local/cuda-12.8`
- PyTorch `2.10.0+cu128`
- Ray `2.55.1`
- vLLM branch `Putpocket-v0.19.1` under `externals/vllm`
- LMCache branch `Putpocket-v0.4.4` under `externals/lmcache`
- Cline under `externals/cline` as read-only reference

## Activation script

`env_activate.sh` must be source-able:

```bash
source scripts/env/env_activate.sh
```

It should:

- resolve repo root from script location,
- activate `Putpocket_env`,
- export CUDA path variables,
- expose repo-local source package,
- not rely on `.bashrc`.

## Shared model cache

Use constants for shared HF cache path.
Do not mutate global HF env vars.


## Shared server resource limits

This empty dataset-mining repo is sharing the server with other users.

vLLM / CUDA extension builds must be capped at 32 CPU build threads. The implementation must set or respect:

```bash
export PUTPOCKET_BUILD_THREADS=32
export MAX_JOBS=32
export CMAKE_BUILD_PARALLEL_LEVEL=32
export CARGO_BUILD_JOBS=32
export NVCC_THREADS=1
```

Do not auto-detect all CPU threads for vLLM builds.

Runtime GPU allocation is also restricted. Dataset mining workers may use only:

```text
GPU 4, GPU 5, GPU 6, GPU 7
```

GPUs 0,1,2,3 are unavailable to this repo and must not be allocated by preflight or worker launch.
