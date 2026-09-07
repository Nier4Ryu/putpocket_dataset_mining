# T20260907-001__glm53-runpod-sm90

task identity: T20260907-001__glm53-runpod-sm90
objective: package the latest GLM-5.3 PutPocket stateful-edit-v3 accuracy-ablation stack as one CPU-built RunPod image for H100/H200 SM90 and RTX PRO 6000 Blackwell SM120 using an under-200GB checkpoint
status: completed_cpu_package_gpu_validation_pending
base tip: cf6e02a529eea92ada871c1a90e0a45a3a643ad0
branch: agent/T20260907-001__glm53-runpod-sm90
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260907-001__glm53-runpod-sm90
runtime mode: isolated-native
write scope:
  - task-local SM90+SM120 locks, source overlays, Docker/package scripts, docs, tests, task evidence, and handoff
forbidden paths:
  - Putpocket_env/
  - data/
  - logs/
  - models/
  - .ssh/
fixed decisions:
  - canonical runtime checkout is /home/${USER}/putpocket_dataset_mining or /workspace/putpocket_dataset_mining
  - task worktrees live under /home/${USER}/putpocket_dataset_mining_worktrees or /workspace/putpocket_dataset_mining_worktrees
  - no nvidia-smi, CUDA runtime execution, model loading/inference, NVIDIA device passthrough, or GPU jobs on Montblanc
  - Docker compilation and static CPU-only container checks are allowed without a device
  - canonical master, DSV4, and every pre-existing worktree/image/tag are read-only
  - base source is exact A6000-package commit cf6e02a529eea92ada871c1a90e0a45a3a643ad0, but the new image is a distinct SM90+SM120 runtime package
  - model payload must be below 200,000,000,000 bytes and is never embedded in the image
  - default checkpoint candidate is Intel/GLM-5.3-Flash-W4A16-AutoRound only if immutable metadata and current-vLLM loader/kernel compatibility pass
  - targets are linux/amd64 Hopper compute capability 9.0 (RunPod H100/H200) and Blackwell compute capability 12.0 (RTX PRO 6000 Blackwell)
  - preserve default-OFF full-target-prefill then donor-row-overwrite accuracy-ablation semantics; never claim true selective prefill or compute saving
  - no silent model download, attention backend, quantization, KV dtype, or topology fallback
  - Vikunja and AFFiNE are out of scope
plan:
  - freeze immutable under-200GB checkpoint and current-vLLM/SM90 kernel compatibility
  - apply the generalized PutPocket stateful hook and validated SM120 NoPE fixes to the exact vLLM source and bind postimage hashes
  - add dual-architecture Dockerfile, target-specific doctor/serve profiles, proxy/experiment commands, volume and RunPod contracts
  - build and inspect the image without a GPU device or CUDA runtime probe
  - run focused/full CPU/static tests and collision-safe Git/Docker Hub pushes
  - record exact build, registry, and remaining RunPod-only GPU evidence
completion criteria:
  - static doctor, CLI help, all PutPocket imports, source/postimage, model metadata, and SM90+SM120 CUDA artifact audits pass
  - local image is immutable, model-free, and offers fail-closed doctor/serve/proxy/experiment entrypoint contracts
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - CPU-only vLLM build passed with MAX_JOBS=4 and NVCC_THREADS=1; native targets were 9.0a and 12.0a
  - task-built wheel audit passed for required SM90/SM120 native code and declared SM80/SM89 W4A16 compatibility objects
  - static doctor, no-device PutPocket imports, exact INC AutoRound loader source inspection, no-weight check, and fresh patch-chain hashes passed
  - immutable linux/amd64 image was pushed under a new Docker Hub tag; local/registry index digest is sha256:1410ccdb3a3418ebbeb9124ed6977eeb6570cd4af09cbfeec0ac1b5a14b231ef
  - focused tests passed: 24 tests and 204 subtests
  - full CPU/static tests passed: 243 tests and 268 subtests; GPU-probing collection was explicitly excluded
  - runtime GPU validation remains pending separately on SM90 and SM120 RunPod hosts
artifacts:
  - agent/tasks/T20260907-001__glm53-runpod-sm90/
  - agent/tasks/T20260907-001__glm53-runpod-sm90/evidence/cpu-build-v1.json
  - /home/dyryu/.cache/putpocket-artifacts/T20260907-001__glm53-runpod-sm90/image-sm90-sm120-w4a16-9cd956c7-v1 (external build evidence; not committed)
  - /home/dyryu/.cache/putpocket-artifacts/T20260907-001__glm53-runpod-sm90/final-image-9964b170-9cd956c7-v1 (final image/registry evidence; not committed)
commits:
  - 9964b1705972fcf208e48d57696a066c493355b1 (implementation and CPU-built package)
  - final task metadata/handoff commit: see branch tip
final handoff link: agent/tasks/T20260907-001__glm53-runpod-sm90/handoffs/TO_GPT_20260907.md
