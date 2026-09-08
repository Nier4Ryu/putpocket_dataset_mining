# T20260907-002__glm53-runpod-fa-fix

task identity: T20260907-002__glm53-runpod-fa-fix
objective: repair the dual SM90/SM120 GLM-5.3 RunPod image by packaging the upstream common vLLM FA2 and FA3 import dependencies and adapting the SM120 FlashInfer GLM entry point to the model's native NoPE query while preserving the forced sparse-MLA runtime path
status: in_progress
base tip: 96c59224e79bc5589c977439717badf92864947e
branch: agent/T20260907-002__glm53-runpod-fa-fix
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260907-002__glm53-runpod-fa-fix
runtime mode: isolated-native
write scope:
  - task-local dual-architecture build patch, audits, doctor/runtime gate, lock, Docker labels, docs, tests, evidence, and handoff
forbidden paths:
  - Putpocket_env/
  - data/
  - logs/
  - models/
  - .ssh/
fixed decisions:
  - canonical master, DSV4, GLM-5.2, T20260907-001, and all historical worktrees/images/tags are read-only
  - no nvidia-smi, CUDA runtime API, GPU device passthrough, model load/inference, or GPU jobs on Montblanc
  - exact vLLM source remains 9cd956c7e6cf54efa366b803cafa15ec6c2df827
  - exact external model remains Intel/GLM-5.3-Flash-W4A16-AutoRound revision 5eee1846f0321058ed73745f9aa16f2aaf0fc0a0; weights are not downloaded or embedded
  - include upstream cmake/external_projects/vllm_flash_attn.cmake last so _vllm_fa2_C and _vllm_fa3_C are packaged as common import dependencies
  - FA2/FA3 are not the selected GLM sparse attention backend; forced profiles remain FLASHINFER_MLA_SPARSE_SM90 and FLASHINFER_MLA_SPARSE_SM120
  - retain DeepGEMM and FlashKDA; continue omitting FlashMLA, QuTLASS, fmha_sm100, tml_fa4, and DeepEP
  - task-built CUDA audit requires SM90 and SM120 native products and both FA extension prefixes while retaining documented upstream W4A16 compatibility products
  - runtime doctor must fail early with VLLM_FLASH_ATTN_EXTENSION_MISSING when either installed extension is absent
  - preserve default-OFF full-target-prefill then donor-row-overwrite accuracy-ablation semantics; do not claim true partial prefill or compute saving
  - preserve the model's qk_rope_head_dim=0 setting; the SM120 adapter may append zero query channels only to select FlashInfer's 656-byte GLM kernel ABI and must not change RoPE/model semantics
  - after selecting the 576-wide GLM v32 ABI, pass active sparse lengths only through seq_lens; do not also forward the native-NoPE TRTLLM-GEN-only sparse_mla_top_k_lens argument
  - reuse the explicitly retained stopped RunPod Pod volume for image-only retries so the exact 181 GB model is not downloaded again; terminate the Pod and attached volume after final success or abandonment
  - failed Docker v1 tag/digest is immutable and must not be overwritten
  - Docker and Git pushes are normal, non-force, and limited to the new tag and this task branch
plan:
  - reproduce and bind the missing-extension failure contract in CPU/static tests
  - update the pinned source patch, audit, doctor, lock, labels, and documentation
  - validate a fresh exact patch chain and commit/push source repair
  - rebuild CPU-only from existing Docker cache with MAX_JOBS=4 and NVCC_THREADS=1
  - overlay the hash-locked Python-only SM120 NoPE query adapter without rebuilding unchanged native CUDA products
  - validate static doctor, no-device imports/preflight, INC registration, both FA extensions, CUDA architecture audit, platform, size, and no weights
  - publish a collision-free v2 image and record local/registry evidence
completion criteria:
  - focused and full CPU/static tests, compile, shell, JSON, diff, source/postimage hashes, and clean fresh patch-chain checks pass
  - final image contains both _vllm_fa2_C and _vllm_fa3_C shared objects and passes every CPU/static image gate
  - Git branch and Docker Hub tag are pushed normally and remote identities are verified
  - task-local final handoff records the remaining RunPod-only GPU validation boundary
validation:
  - pending
artifacts:
  - agent/tasks/T20260907-002__glm53-runpod-fa-fix/
commits:
  - pending
final handoff link: pending
