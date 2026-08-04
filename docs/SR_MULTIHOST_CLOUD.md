# SR Multi-Host Cloud Execution

`master` is the canonical branch for new SR bootstrap and cloud execution work.
`blackwell` remains a temporary compatibility alias.

## Topology

- Server 1: 4 x NVIDIA RTX 3090, SM86. Local controller, Docker workspace host,
  verifier host, analysis destination, and SM86 build/import validation.
- Server 2: 3 x NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120. Development
  server, local controller, Docker workspace/verifier host, Tiny GLM h192/SM120
  validation, and analysis destination.
- RunPod: Hopper H100/H200, SM90-family. Real GLM-5.2 inference and custom
  vLLM/SR runtime. Use 8 x H200 only for full GLM-5.2 execution.

## Execution Modes

Mode A, `local_controller`:

- local server runs the controller, episode Docker workspace, hidden verifier,
  and artifact storage;
- RunPod can provide only a remote model endpoint when in-process vLLM/SR hooks
  are not required.

Mode B, `cloud_controller`:

- RunPod runs the controller, model, and in-process SR/vLLM hooks;
- local server runs episode Docker workspaces and hidden verification through
  `remote_ssh_docker`.

In `cloud_controller` mode, `local_docker` is forbidden and fails early with
`E_CLOUD_LOCAL_DOCKER_FORBIDDEN`.

## Configuration

Environment variables:

```bash
export SR_EXECUTION_ROLE=cloud_controller
export SR_WORKSPACE_BACKEND=remote_ssh_docker
export SR_VERIFIER_BACKEND=remote_ssh_docker
export SR_HARDWARE_PROFILE=sm90
export SR_SERVER_PROFILE=runpod_hopper
export SR_REMOTE_HOST=verifier.example.internal
export SR_REMOTE_USER=sr-verifier
export SR_REMOTE_PORT=22
export SR_REMOTE_ROOT=/srv/putpocket-sr-verifier
export SR_REMOTE_IDENTITY_FILE=/run/secrets/sr_verifier_ssh_key
export SR_REMOTE_KNOWN_HOSTS_FILE=/run/secrets/sr_verifier_known_hosts
export SR_REMOTE_DOCKER_IMAGE=putpocket-classeval-python:ubuntu22.04-py313-v1
```

No passwords, private keys, or tokens belong in source control.

## Remote Worker

Install the repository on the verifier host and make this command available:

```bash
python -m putpocket_dataset_mining.remote_worker preflight
```

Supported wrapper commands:

- `preflight`
- `fixture-pass`
- `fixture-fail`
- `workspace-create`
- `workspace-exec`
- `workspace-snapshot`
- `workspace-destroy`
- `verify`
- `result-status`
- `cleanup-stale`

The wrapper validates job/session IDs, uses a fixed remote root, rejects path
traversal, runs commands only inside Docker containers, and records structured
JSON results.

Run a local-to-local preflight later:

```bash
putpocket-dataset-mining remote-preflight \
  --docker-image putpocket-classeval-python:ubuntu22.04-py313-v1
```

## Bootstrap

CPU phase:

```bash
CUDA_VISIBLE_DEVICES="" ./scripts/env/bootstrap_sr.sh \
  --phase cpu \
  --hardware-profile cpu \
  --execution-role local_controller \
  --verifier-backend disabled \
  --dry-run
```

Server 1:

```bash
./scripts/smoke/server1_sm86.sh
```

Server 2:

```bash
./scripts/smoke/server2_sm120_tiny_glm.sh
```

RunPod one-H100:

```bash
./scripts/smoke/runpod_h100_sm90.sh
```

RunPod 8 x H200 full GLM:

```bash
export SR_MODEL_PATH=/network-volume/models/GLM-5.2-FP8
export SR_ALLOW_EXPENSIVE_GPU_RUN=1
./scripts/smoke/runpod_8xh200_full_glm.sh
```

The 8 x H200 command refuses to download weights and refuses to run without
`SR_ALLOW_EXPENSIVE_GPU_RUN=1`.

## vLLM Profiles

`third_party/vllm_glm52_v025/manifest.yaml` pins upstream vLLM to:

- tag: `v0.25.1`
- commit: `752a3a504485790a2e8491cacbb35c137339ad34`

Profiles:

- `clean`: exact upstream commit.
- `patched`: same commit plus `patches/0001-tiny-glm-h192-sm120.patch`.

Use per-host CUDA arch lists:

- Server 1: `TORCH_CUDA_ARCH_LIST=8.6`
- Server 2: `TORCH_CUDA_ARCH_LIST=12.0`
- RunPod Hopper: `TORCH_CUDA_ARCH_LIST=9.0`

Do not assume an SM120-only wheel is valid on SM90.

## Tiny GLM Isolation

Tiny GLM h192 custom routing is allowed only for:

- SM120/SM121;
- GLM MoE/DSA family;
- `kv_lora_rank=128`;
- `qk_rope_head_dim=64`;
- derived head size 192;
- `v_head_dim=128`;
- `index_n_heads=8`;
- `index_head_dim=64`;
- custom kernel availability.

Real GLM h576/Hopper runs should set:

```bash
export SR_ASSERT_NO_TINY_GLM_KERNEL=1
```

## Prefix Cache Policy

- Smoke 0: built-in prefix caching may be disabled for exactly one clean engine
  execution.
- Normal runs: built-in prefix caching remains enabled.
- Full recompute references should use request-level bypass, cache reset, or an
  isolated engine rather than leaving prefix caching globally disabled.
- Built-in exact-prefix counters and SR selective-reuse metrics are recorded in
  separate fields. Missing SR hooks are represented as `null` plus an explicit
  unavailable reason.

## Selective Artifact Replication

Build a manifest:

```bash
putpocket-dataset-mining sync-artifacts \
  --source-root data/model_evaluation/runs/example \
  --profile analysis_minimal \
  --manifest-out sync_manifest.json \
  --dry-run
```

Profiles:

- `analysis_minimal`
- `analysis_with_workspaces`
- `analysis_with_selected_kv`
- `verifier_input`
- `verifier_output`

Default analysis profiles exclude hidden tests, reference solutions, credentials,
model weights, Hugging Face caches, virtual environments, build caches, Docker
layers, complete raw KV pools, and allocated KV pools.

## Hidden-Test Policy

Hidden tests may be present on the controller and in verifier job bundles. They
must not appear in model prompts, agent-visible episode workspaces, or normal
analysis replication profiles.

## Rollback and Cleanup

- Remote job and session directories are rooted under `SR_REMOTE_ROOT`.
- Completed verifier jobs are immutable by job ID and checksum.
- Duplicate cleanup requests are safe.
- No `--delete` is used by selective artifact replication by default.
