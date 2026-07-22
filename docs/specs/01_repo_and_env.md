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
- CUDA `/usr/local/cuda-12.9`
- PyTorch `2.10.0+cu129`
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

This dataset-mining repo is running on the Blackwell server.

vLLM / CUDA extension builds must be capped at 16 CPU build threads by default. The implementation must set or respect:

```bash
export PUTPOCKET_BUILD_THREADS=16
export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export CARGO_BUILD_JOBS=16
export NVCC_THREADS=1
```

Do not auto-detect all CPU threads for vLLM builds. Retry vLLM only on
OOM-like build failures with 12 and then 8 threads.

Runtime GPU allocation is also restricted. Dataset mining workers may use only:

```text
GPU 4, GPU 5, GPU 6, GPU 7
```

GPUs 0,1,2,3 are unavailable to this repo and must not be allocated by preflight or worker launch.
