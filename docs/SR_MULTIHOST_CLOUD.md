# SR Multi-Host Cloud Execution

`master` is the canonical branch for SR bootstrap and cloud execution work. New installations should use:

```bash
git fetch origin --prune
git checkout master
git pull --ff-only origin master
```

`blackwell` remains a temporary compatibility alias and is kept fast-forwarded to the same tested implementation commit.

## Topology

- Server 1: 4 x NVIDIA RTX 3090, SM86. Phase 1 role: remote Docker verifier host. Later roles: optional local controller and analysis host.
- Server 2: 3 x NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120. Phase 1 role: controller/process host and local Docker development host.
- RunPod: Hopper H100/H200, SM90-family. Later role: real GLM-5.2 model server and custom vLLM/SR runtime.

Phase 1 remotes hidden verification only. The agent episode Docker workspace remains local to Server 2 for the first deployment.

## Bootstrap

Canonical entrypoint:

```bash
./scripts/env/bootstrap_sr.sh
```

Stages: `preflight`, `system`, `core`, `verifier`, `vllm_source`, `vllm_build`, `validate`, `all`.

Roles: `controller`, `verifier`, `model_server`, `development`.

Server profile architecture defaults:

- `server1_rtx3090`: `TORCH_CUDA_ARCH_LIST=8.6`
- `server2_blackwell`: `TORCH_CUDA_ARCH_LIST=12.0`
- `server2_rtxpro6000_blackwell`: `TORCH_CUDA_ARCH_LIST=12.0`
- `runpod_hopper`: `TORCH_CUDA_ARCH_LIST=9.0`
- `custom`: explicit `TORCH_CUDA_ARCH_LIST` or safe detection required

User-provided `TORCH_CUDA_ARCH_LIST` overrides profile defaults.

Server 2 no-GPU core bootstrap:

```bash
CUDA_VISIBLE_DEVICES="" ./scripts/env/bootstrap_sr.sh \
  --server-profile server2_blackwell \
  --role development \
  --stage core
```

Server 1 verifier dry run:

```bash
./scripts/env/bootstrap_sr.sh \
  --server-profile server1_rtx3090 \
  --role verifier \
  --stage all \
  --vllm-profile skip \
  --dry-run
```

RunPod model-server dry run:

```bash
./scripts/env/bootstrap_sr.sh \
  --server-profile runpod_hopper \
  --role model_server \
  --stage all \
  --vllm-profile patched \
  --dry-run
```

No bootstrap stage downloads full GLM weights, starts vLLM, runs model inference, mines data, or runs H100/H200 checks unless explicitly added in a later task.

## Remote Verifier Configuration

Controller-side environment example for direct Server 2 -> Server 1 verification:

```bash
export SR_EXECUTION_ROLE=controller
export SR_WORKSPACE_BACKEND=local_docker
export SR_VERIFIER_BACKEND=remote_ssh_docker
export SR_REMOTE_HOST=10.0.0.5
export SR_REMOTE_USER=dyryu
export SR_REMOTE_PORT=42
export SR_REMOTE_ROUTE=direct
export SR_REMOTE_REPOSITORY_ROOT=/home/dyryu/putpocket_dataset_mining
export SR_REMOTE_JOB_ROOT=/home/dyryu/putpocket_dataset_mining/data/remote_verifier
export SR_REMOTE_WRAPPER=/home/dyryu/putpocket_dataset_mining/Putpocket_env/bin/putpocket-remote-verifier
export SR_REMOTE_DOCKER_IMAGE=putpocket-classeval-python:ubuntu22.04-py313-v1
export SR_VERIFIER_TIMEOUT_SEC=3600
```

Do not store passwords, private keys, tokens, or unverified known_hosts entries in the repository.

Sanitized route examples:

- `configs/remote_verifier/server2_to_server1.direct.example.yaml`
- `configs/remote_verifier/server2_to_server1.proxy_jump.example.yaml`
- `configs/remote_verifier/operator_to_server2.proxy_jump.example.yaml`

## SSH Routes

Direct default target:

```text
dyryu@10.0.0.5 -p 42
```

ProxyJump to Server 1:

```text
ProxyJump dyryu@141.223.145.88:4500,dyryu@141.223.25.156:42 -> dyryu@10.0.0.5:42
```

Operator route to Server 2:

```text
ProxyJump dyryu@141.223.145.88:4500,dyryu@141.223.25.155:42 -> dyryu@10.0.0.47:42
```

The transport renders these from structured config; it does not edit `~/.ssh/config`.

## Repository-Integrated Remote Wrapper

Install the repository on Server 1 and make this console entry point available from the project environment:

```bash
putpocket-remote-verifier protocol-version
```

Commands:

- `protocol-version`
- `preflight`
- `ensure-image`
- `promote`
- `verify`
- `result-status`
- `cleanup`

The wrapper uses `SR_REMOTE_JOB_ROOT` and retains job directories by default. It rejects invalid job IDs, path traversal, and symlink escapes. Commands run only inside Docker verifier containers.

Use the absolute wrapper path in non-interactive SSH configurations because a remote shell may not load the project virtualenv on `PATH`:

```yaml
wrapper: /home/dyryu/putpocket_dataset_mining/Putpocket_env/bin/putpocket-remote-verifier
```

## Remote Job Lifecycle

Controller lifecycle:

1. Copy the existing agent workspace snapshot into a verifier-only workspace.
2. Materialize hidden tests only in that verifier workspace.
3. Assert the original agent-visible workspace has no hidden tests.
4. Generate a sanitized job ID and manifest.
5. Calculate workspace checksum.
6. Transfer to `incoming/<job_id>.partial/`.
7. Promote to `ready/<job_id>/`.
8. Invoke `putpocket-remote-verifier verify --job-id <job_id>`.
9. Retrieve `completed/<job_id>/result.json`, `stdout.txt`, and `stderr.txt`.
10. Verify/adapt the structured result to the existing local verifier result interface.
11. Write local-compatible `checklist.json`, `stdout.txt`, and `stderr.txt`.

Partial jobs are never executed. Completed jobs are immutable by job ID and input checksum.

## Docker Image Behavior

Expected verifier image:

```text
putpocket-classeval-python:ubuntu22.04-py313-v1
```

Expected Dockerfile:

```text
docker/classeval_python/Dockerfile
```

`putpocket-remote-verifier ensure-image` reuses the image when present and builds it from the repository Dockerfile when absent. Image build locking prevents concurrent rebuilds. Verification preserves network-disabled Docker execution, CPU/memory limits, timeout, and `/workspace` semantics.

## Failure Classification

Remote SSH, rsync, Docker daemon, image build, protocol, and integrity failures are classified as `infra_failed`. The controller does not silently fall back to local Docker when `remote_ssh_docker` is configured.

Verifier statuses:

- `passed`
- `failed`
- `timeout`
- `infra_failed`

The higher-level verifier receives equivalent local fields: return code, stdout, stderr, and timeout flag.

## vLLM Profiles

`third_party/vllm_glm52_v025/manifest.yaml` pins upstream vLLM to:

- tag: `v0.25.1`
- commit: `752a3a504485790a2e8491cacbb35c137339ad34`

Profiles:

- `clean`: exact upstream commit.
- `patched`: same commit plus `patches/0001-tiny-glm-h192-sm120.patch`.

Build manifests are keyed by vLLM commit, profile, patch digest, Python, torch, CUDA, and target architecture list.

## Next Server 1 Commands

These commands are for the next approved live task. They were not executed in Phase 0/1.

Server 1 safe inspection:

```bash
ssh -p 42 dyryu@10.0.0.5 'cd /home/dyryu/putpocket_dataset_mining && pwd && git status -sb && git branch --show-current && git rev-parse HEAD'
```

Server 1 checkout of canonical master:

```bash
ssh -p 42 dyryu@10.0.0.5 'cd /home/dyryu/putpocket_dataset_mining && git fetch origin --prune && git checkout master && git pull --ff-only origin master'
```

Server 1 verifier bootstrap:

```bash
ssh -p 42 dyryu@10.0.0.5 'cd /home/dyryu/putpocket_dataset_mining && ./scripts/env/bootstrap_sr.sh --server-profile server1_rtx3090 --role verifier --stage all --vllm-profile skip --allow-docker-build'
```

Dedicated SSH key and pinned known_hosts setup must be approved separately.

Direct Server 2 -> Server 1 preflight:

```bash
putpocket-dataset-mining remote-preflight \
  --config configs/remote_verifier/server2_to_server1.direct.example.yaml
```

ProxyJump preflight uses values in `configs/remote_verifier/server2_to_server1.proxy_jump.example.yaml`.

Dry-run the reusable disposable pass/fail/timeout fixture command before live connection:

```bash
putpocket-dataset-mining remote-test \
  --config configs/remote_verifier/server2_to_server1.direct.example.yaml \
  --fixtures pass,fail,timeout \
  --timeout-fixture-sec 2 \
  --dry-run
```

After live connection is approved, remove `--dry-run` to transfer disposable verifier-only fixture jobs to Server 1. Pass/fail fixtures use the production verifier timeout default of 3600 seconds; the timeout fixture explicitly overrides to 2 seconds. One real read-only verifier job should be a separate approved task.

The SSH connection timeout remains short, for example 10 seconds. The remote wrapper command timeout must be at least verifier timeout plus the configured grace period, currently 3600 + 120 = 3720 seconds.

## Hidden-Test Policy

Hidden tests may exist on Server 2/controller and in verifier job bundles. They must not appear in model prompts, agent-visible episode workspaces, or normal analysis artifacts.

## Retention

All remote job directories are retained by default. Cleanup is manual-only and supports dry-run semantics.
