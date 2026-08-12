# GLM-5.2 AWQ PP3/TP2 Agent resume report

## Executive summary

The newly published frozen dataset is present and exact. Checkpoint structural validation, canonical source ownership, six-GPU health, NCCL, and static PP3/TP2 compatibility all pass. Two concrete infrastructure blockers remain: RunPod has no usable local Docker workspace backend, and strict Server-1 SSH cannot authenticate or validate the route because its approved runtime credential files are absent. Per policy, no 442-GiB-class model load was attempted.

## Classification

`GLM52_AWQ_PP3_TP2_LOCAL_WORKSPACE_BLOCKED`

The independent remote status is `SERVER1_WRAPPER_PREFLIGHT_FAILED`.

## Source and convergence

- RunPod `master` and `origin/master`: `ec2b22642a1976501f06e560968153da2af090ca`
- Frozen dataset commit: `c7273c2d33443613f2bec476931baf006ce18ea9`
- Publication metadata commit: `ec2b22642a1976501f06e560968153da2af090ca`
- Server-2 publication handoff proves the dataset originated through its canonical publication workflow; no direct Pod-to-Server-2 route was required.
- Server-1 source SHA was not reached because strict SSH route setup failed before login.
- Integrated PP3/TP2 timing and distributed profile source are present semantically.

## Frozen dataset

- Tracked path: `data/dataset_mining/datasets/classeval_stateful_working_v0/accepted.jsonl`
- Rows: 18, all valid and unique
- SHA-256: `6031d368ee8359c9dfc3c7b785d5c30e4db9ae5b2969bfba3a7e09512a46b30d`
- Ordered IDs: 76, 37, 30, 21, 80, 83, 8, 47, 63, 60, 99, 7, 19, 64, 11, 13, 86, 3 (all prefixed `test_ClassEval_`)
- Dataset was not modified.

## Checkpoint

- 83/83 referenced shards; zero structurally bad
- Index and parsed tensor payload: 474,194,040,288 bytes
- Physical files: 474,223,144,408 bytes
- Header overhead: 29,104,120 bytes
- Structural gate: PASS; no redownload or cache deletion

## Hardware, NCCL, and model topology

- 6 × NVIDIA RTX PRO 6000 Blackwell Server Edition, SM120
- PP=3, TP=2, DP=1, world size 6
- NCCL 2.27.5 rank/device mapping, barriers, all-reduce, and shutdown: PASS in 24.651768 seconds; all ranks observed 21.0
- `GlmMoeDsaForCausalLM`, 78 layers, 6144 hidden, 64 attention/KV heads, 256 experts, compressed-tensors
- Expected PP allocation: 26/26/26; head and hidden dimensions divide by TP=2
- vLLM registry maps the architecture to `GlmMoeDsaForCausalLM`; its model registry/parallel config expose PP support validation and compressed-tensors support.

## Workspace diagnosis

Configured backend is `local_docker`. Fresh diagnosis found no Docker CLI, dockerd, containerd, podman, nerdctl, runc, or crun; no Docker socket, `DOCKER_HOST`, or daemon process exists. The container is seccomp-filtered. Source contains `remote_ssh_docker`, but selecting it would move workspace execution off RunPod, contradict the fixed topology and profile validation. No safe equivalent local backend exists.

## Server-1 live preflight

The canonical combined preflight used strict host-key policy and the required Proxy-A → Proxy-C → Server-1 route. It failed because `/root/.ssh/putpocket_server1_ed25519` and the configured route known-hosts material are absent. Only the GitHub key exists. No private key was printed, no permissive host-key option was used, and no remote job was submitted.

## Context sizing

Requested and effective maximum are 65,536. Tokenizer-only preprocessing found raw Query-1 and Query-1-plus-Query-2 maxima of 81 and 99 tokens. Stored generated History-1 is not published; because each history permits up to 30 generations of 2,048 tokens, 65,536 is the smallest capped bucket that safely covers the configured upper bound.

## Model and evaluation

Model load, tiny generation, smoke samples, and the 18-sample run were not attempted. No engine, CUDA compute process, telemetry collector, or remote verifier job remains active. Local hidden-verifier fallback stayed disabled.

## Recovery

Provide a supported local isolated workspace backend in the RunPod image (normally Docker CLI plus a usable daemon/socket), and materialize the approved Server-1 private key and pinned Proxy-A/Proxy-C/Server-1 known-hosts under `/root/.ssh`. Then rerun tokenizer sizing and the combined preflight under a new run ID.
