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
./scripts/env/bootstrap_env.sh --skip-vllm-build --skip-lmcache-build
./scripts/env/bootstrap_env.sh --force-vllm-build
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

Build parallelism defaults to 32 jobs and can be overridden before activation:

```bash
export PUTPOCKET_BUILD_THREADS=16
export MAX_JOBS=16
export CMAKE_BUILD_PARALLEL_LEVEL=16
export CARGO_BUILD_JOBS=16
source scripts/env/env_activate.sh
```

The activation script does not set `CUDA_VISIBLE_DEVICES`; GPU selection belongs
to runtime configs or per-command environment variables.
