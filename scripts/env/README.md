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
