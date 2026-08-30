# T20260830-001__glm53-montblanc-deployment

task identity: T20260830-001__glm53-montblanc-deployment
objective: glm53-montblanc-deployment
status: in_progress
base tip: 6e8f8920ff5074ffeb7073223fa88fc3ea65ee0d
branch: agent/T20260830-001__glm53-montblanc-deployment
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260830-001__glm53-montblanc-deployment
runtime mode: isolated-native
write scope:
  - source/docs/tests required for this task
forbidden paths:
  - Putpocket_env/
  - data/
  - logs/
  - models/
  - .ssh/
fixed decisions:
  - canonical runtime checkout is /home/${USER}/putpocket_dataset_mining or /workspace/putpocket_dataset_mining
  - task worktrees live under /home/${USER}/putpocket_dataset_mining_worktrees or /workspace/putpocket_dataset_mining_worktrees
plan:
  - audit host, source, official models, and current GLM-5.2 boundary
  - lock GLM-5.3-Flash compressed-tensors NVFP4 and TP1/DP3/EP3 capacity plan
  - implement task-local doctor/download/build/lifecycle/client package
  - download and verify immutable selected files
  - build pinned vLLM plus FlashInfer SM120 NoPE runtime
  - launch real three-GPU endpoint and run OpenAI plus PutPocket smoke
  - stop only the task-owned server, validate, commit, push, and hand off
fixed task decisions:
  - selected checkpoint is RedHatAI/GLM-5.3-Flash-NVFP4 at 36c184c6cda000a481711306df5adde42f63321a
  - official full GLM-5.3 FP8/BF16 and Flash FP8/BF16 do not fit three 97,887 MiB cards
  - complete indexed checkpoint is 197,881,155,500 bytes; its MTP shard is stored for upstream-index completeness but MTP execution remains disabled
  - TP3 is illegal because 64 attention heads and hidden size 4096 are not divisible by 3
  - PP3 is unsupported by the pinned Glm5Next implementation
  - supported plan is TP1/DP3/EP3 with 96 of 288 experts per rank and early EP weight filtering
  - vLLM PR 53906 head 878631b6079d2cf9fb80830ef9cb41b43aded098 supplies GLM-5.3 integration
  - FlashInfer PR 4802 head c2eec117219457e45126fa4fa87e7240dd4ea620 supplies SM120 GLM53_NOPE
  - compressed-tensors uses Marlin MoE; MTP and prefix caching remain off for the bounded smoke
  - the vLLM image is SM120-only via the official torch_cuda_arch_list=12.0 build arg; FlashAttention retains upstream per-kernel compatibility defaults while GLM main attention uses the pinned FlashInfer SM120 NoPE path
  - do not exclude vllm-flash-attn from this wheel: Glm5Next imports its multimodal module eagerly, MMEncoderAttention imports fa_utils, and CUDA fa_utils imports vllm.vllm_flash_attn; setup.py also declares the FA2/FA3 extension targets
  - the checksum-bound packaging patch removes both unavailable 0.6.18rc10 release-wheel install sites; the pinned FlashInfer PR source remains the only task runtime overlay authority
completion criteria:
  - immutable model files pass exact size and SHA-256 verification
  - runtime image source labels and GLM53_NOPE symbols pass doctor checks
  - actual three-GPU vLLM endpoint becomes ready
  - /v1/models, deterministic chat, and PutPocket rendered-prompt HTTP smoke pass
  - task server stops via ownership-checked SIGTERM without hard kill
  - focused/full CPU tests pass and task-local TO_GPT handoff exists
validation:
  - focused CPU/static: 16 passed, 26 subtests passed (pre-runtime); 16 passed after two-site packaging correction
  - lock/package hashes: pass (pre-runtime)
  - shell syntax/compileall/git diff --check: pass (pre-runtime)
  - initial SM120 CUDA build reached wheel packaging; it then failed closed because the first patch omitted a second unpublished FlashInfer restore site (exit 2, not masked)
  - corrected patch applies to the exact upstream preimage and yields the locked postimage; cached rebuild and GPU runtime remain pending
artifacts:
  - task source: agent/tasks/T20260830-001__glm53-montblanc-deployment/
  - runtime evidence root: task-local cache selected by GLM53_RUNTIME_ROOT (not committed)
commits:
  - pending
final handoff link: pending
