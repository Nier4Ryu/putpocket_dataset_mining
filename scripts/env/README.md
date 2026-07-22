# Putpocket Environment Scripts

## First-Time Setup Or Repair

```bash
./scripts/env/bootstrap_env.sh
```

`bootstrap_env.sh` creates or repairs the repo-local `Putpocket_env`, installs
Python dependencies, checks external repositories, builds editable vLLM/LMCache
when needed, ensures the default Docker image, and runs smoke checks. It writes
stage logs under `logs/env_setup/<timestamp>/`; `logs/env_setup/latest` points
to the newest run.

Useful options:

```bash
./scripts/env/bootstrap_env.sh --doctor-only
./scripts/env/bootstrap_env.sh --skip-docker
./scripts/env/bootstrap_env.sh --skip-vllm-build --skip-deepgemm-build --skip-lmcache-build
./scripts/env/bootstrap_env.sh --force-vllm-build
./scripts/env/bootstrap_env.sh --force-deepgemm-build
./scripts/env/bootstrap_env.sh --force-lmcache-build
./scripts/env/bootstrap_env.sh --force-docker-build
```

## Every Shell Session

```bash
source scripts/env/env_activate.sh
```

`env_activate.sh` only activates the existing `Putpocket_env` and exports
runtime/build environment variables. It does not install packages, build
extensions, clone repositories, run Docker, or start mining jobs.

Blackwell build parallelism defaults to 16 jobs and can be overridden before
activation:

```bash
export PUTPOCKET_BUILD_THREADS=16
export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export CARGO_BUILD_JOBS=16
source scripts/env/env_activate.sh
```

`bootstrap_env.sh` defaults to `CUDA_HOME=/usr/local/cuda-12.9` and installs
the CUDA 12.9 PyTorch wheel set (`cu129`) unless `TORCH_SPEC`,
`TORCH_CUDA_TAG`, or `TORCH_INDEX_URL` are overridden. If `python3.13` is not
available but `uv` can be installed, bootstrap installs a repo-local `uv` under
`.local_python/bin` and asks it for Python 3.13.

Editable vLLM builds start at 16 build threads. If the build log shows an
OOM-like failure, bootstrap retries with 12 threads and then 8 threads, writing
logs such as `logs/env_setup/<timestamp>/vllm_build_threads_16.log`.

After vLLM is available, bootstrap installs the vLLM-pinned DeepGEMM package
when `deep_gemm` is not already importable. GLM sparse MLA model loading needs
that package before vLLM can reach backend selection. Use
`--skip-deepgemm-build` only for dependency inspection or when DeepGEMM is known
to be unavailable on the host.

The activation script does not set `CUDA_VISIBLE_DEVICES`; GPU selection belongs
to runtime configs or per-command environment variables.
