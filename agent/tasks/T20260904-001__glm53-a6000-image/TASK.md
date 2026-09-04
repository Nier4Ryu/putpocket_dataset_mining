# T20260904-001__glm53-a6000-image

task identity: T20260904-001__glm53-a6000-image
objective: package the latest PutPocket GLM-5.3 stateful-edit-v3 accuracy-ablation stack in a conservative CPU-built Docker image for NVIDIA RTX A6000 / SM86
status: complete
base tip: 2c5bb088019b8e50ea335297ab5ef5d164f9fd48
branch: agent/T20260904-001__glm53-a6000-image
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260904-001__glm53-a6000-image
runtime mode: isolated-native
write scope:
  - task-local A6000 locks, source overlays, Docker/package scripts, docs, tests, task evidence, and handoff
forbidden paths:
  - Putpocket_env/
  - data/
  - logs/
  - models/
  - .ssh/
fixed decisions:
  - no nvidia-smi, CUDA runtime execution, model loading/inference, NVIDIA device passthrough, or GPU jobs
  - Docker build is CPU-only; nvcc compilation for architecture 8.6 is permitted without a device
  - DeepEP and every non-SM86 optional extension are excluded from the task build; bundled FA2 is also omitted because pinned upstream lowers SM86 to SM80 SASS/compute_80 PTX
  - the SM86 binary claim covers task-built vLLM/PutPocket CUDA artifacts; vendor PyTorch/CUDA/NVIDIA distributions are separately attested and may be multi-architecture
  - canonical master and every pre-existing worktree are read-only
  - Vikunja and AFFiNE are out of scope
  - source base is exact GLM-5.3 retarget SHA 2c5bb088019b8e50ea335297ab5ef5d164f9fd48
  - model weights are never embedded or downloaded into the image
  - NVFP4 and FLASHINFER_MLA_SPARSE_SM120 are not valid A6000 capability claims
  - preserve default-OFF full-target-prefill then donor-row-overwrite accuracy-ablation semantics; never claim true selective prefill or compute saving
plan:
  - establish an official-source vLLM and GLM-5.3 Flash Ampere compatibility boundary
  - bind a cleanly applicable PutPocket overlay to the selected upstream source and postimage hashes
  - add SM86-only Dockerfile, build/launch/doctor/preflight contracts, manifest, docs, and tests
  - build and inspect the image without any NVIDIA device or CUDA runtime probe
  - archive only when conservative disk preflight permits it
  - run focused/full CPU/static tests, commit, normal-push, and audit isolation
completion criteria:
  - image build succeeds CPU-only or a precise upstream/disk blocker is recorded without false compatibility claims
  - runtime refuses non-SM86, unspecified topology/model, backend/quantization/KV fallback, and unarmed stateful controls
  - all repository/package/source/image hashes are machine-readable and verified
  - dedicated branch is clean and equal to its GitHub tracking ref

completion evidence (2026-09-04 UTC):
  - current vLLM `9cd956c7e6cf54efa366b803cafa15ec6c2df827` built without a GPU device; the CUDA wheel build used only architecture 8.6
  - embedded cuobjdump report observed only `sm_86` in task-built vLLM CUDA payloads; host-only shared objects were recorded separately
  - DeepEP, bundled FA2, all other non-SM86 optional extension targets, FlashInfer JIT cache, and CUTLASS-DSL runtime distributions are absent from the final compatibility wrapper
  - final local image `putpocket/glm53-a6000-gated:9cd956c7-sm86-v1` is `sha256:d51e84188fc153621fe60b01867b9617c4f6d7cd00434f96ac8011bb29890677` and 5,230,080,976 bytes
  - static container doctor passed with no torch/CUDA import or device probe; GLM-5.3 serving remains intentionally fail-closed on SM86
  - Docker archive is stored outside Git at `/home/dyryu/.cache/putpocket-artifacts/T20260904-001__glm53-a6000-image/image-sm86-v1/putpocket-glm53-a6000-gated-9cd956c7-sm86-v1.docker-image.tar.zst` (5,160,566,828 bytes, SHA-256 `20d26204cd16336a3f8d16abc045d1718482211b94f6074dd5beff4cf838c483`)
  - focused tests: 13 passed plus 120 subtests; full CPU/static suite: 234 passed plus 184 subtests
  - no model weights were downloaded or embedded; official metadata-only provenance remains pinned to `zai-org/GLM-5.3-Flash-BF16@a5b45eb41df6402735dedc900be14a42e8d5e538`
