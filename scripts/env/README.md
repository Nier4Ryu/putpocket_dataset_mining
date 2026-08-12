# Putpocket Server-2 Environment

Server-2 has one active project/runtime environment:

```text
Putpocket_env
```

Use one setup command:

```bash
./scripts/env/bootstrap_sr.sh --preset server2
```

Use one activation command in every shell:

```bash
source scripts/env/env_activate.sh
```

Validate without installation or build:

```bash
./scripts/env/bootstrap_sr.sh --preset server2 --doctor-only
```

Preview the exact plan without mutation:

```bash
./scripts/env/bootstrap_sr.sh --preset server2 --dry-run
```

`bootstrap_sr.sh --preset server2` writes logs under
`logs/env_setup/<timestamp>/` and maintains `logs/env_setup/latest`.
It validates the uv-managed `Putpocket_env`, active Qwen vLLM/LMCache
externals, Docker image presence, project imports, and CLI doctor output.

`bootstrap_env.sh` is retained only as a compatibility wrapper that delegates
to the canonical preset. Do not add new setup logic there.

The local GLM bootstrap scripts are retired from the active Server-2 path:

```text
bootstrap_glm52_env.sh
bootstrap_glm52_v025_env.sh
env_activate_glm52.sh
env_activate_glm52_v025.sh
env_activate_ref.sh
```

They are preserved for historical inspection and require explicit opt-in where
execution is still possible. Future full GLM-5.2 inference should run on
RunPod Hopper runtimes, not in the active Server-2 Qwen environment.

Advanced static multi-host bootstrap flags such as `--server-profile`,
`--role`, `--stage`, and `--vllm-profile` remain available for Server-1 and
RunPod planning. Normal Server-2 users should use `--preset server2`.

CUDA architecture selection is explicit bootstrap/build configuration, not
ordinary shell activation. Supported profiles:

```text
portable-nvidia          8.6 9.0 10.0 12.0
rtx3090                  8.6
hopper                   9.0
blackwell-datacenter     10.0
blackwell-rtx            12.0
native                   explicitly detected visible GPU capability
```

Preset defaults:

```text
server1_rtx3090                 rtx3090
runpod_hopper                   hopper
runpod-dev                      portable-nvidia
server2_blackwell               blackwell-rtx
server2_rtxpro6000_blackwell    blackwell-rtx
```

Override precedence is:

```text
--cuda-arch-list
--cuda-arch-profile
PUTPOCKET_CUDA_ARCH_LIST
PUTPOCKET_CUDA_ARCH_PROFILE
preset default
```

Portable editable vLLM builds set:

```bash
TORCH_CUDA_ARCH_LIST="8.6 9.0 10.0 12.0"
PUTPOCKET_BUILD_JOBS="$(nproc)"
MAX_JOBS="${PUTPOCKET_BUILD_JOBS}"
CMAKE_BUILD_PARALLEL_LEVEL="${PUTPOCKET_BUILD_JOBS}"
NVCC_THREADS=1
```

For `runpod-dev`, native build jobs resolve as `--build-jobs`, then
`PUTPOCKET_BUILD_JOBS`, then `nproc`. The resolved value controls vLLM,
CMake/Ninja, LMCache, and future native extension builds. `NVCC_THREADS`
remains `1` to avoid multiplying build-level and NVCC-level parallelism.

The portable profile covers RTX 3090, H100/H200, B200/GB200-class Blackwell,
and RTX PRO 6000 Blackwell Server Edition, but it is slower and larger than a
single-architecture development build.
