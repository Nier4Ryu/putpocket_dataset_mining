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
  - actual model warmup proved the pinned generic fp8_ds_mla cache writer rejected GLM-5.3 NoPE at pe_dim=0; a second checksum-bound vLLM patch retains the 656-byte ABI, accepts only 0 or 64, and suppresses the RoPE copy warp only for NoPE
  - the following real warmup proved the SM120 Python backend omitted FlashInfer's mandatory per-query native-NoPE sparse_mla_top_k_lens; a checksum-bound post-wheel overlay mirrors the pinned generic backend's valid-count, empty-row, and exact-length contract without rebuilding CUDA or changing selection
  - the isolated Python 3.13 smoke client explicitly pins transformers 5.15.0 and jinja2 3.1.6; ambient client dependencies are not accepted
  - v4 observed an effective attention page size of 8704 despite requested block size 512 because hybrid Mamba requires a common page; both values must remain visible in evidence
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
  - first three-GPU server attempt loaded all weights and allocated KV, then failed closed in concat_and_cache_mla before readiness because upstream required pe_dim=64; no generation was claimed
  - the NoPE cache rebuild completed with image doctor and binary guard evidence; the next unique run passed that prior pe_dim boundary, then failed closed because sparse_mla_top_k_lens was absent; no generation was claimed
  - real launch exposed and fixed an ldconfig/pipefail SIGPIPE preflight bug and a transient connection-reset readiness bug; both failures are retained in task-local runtime evidence
  - v4 image sha256:b79bacf76a107fc9ddd67fcc84851e3f5ae222fe6fd6a2f187723297e1400b1e reached all three API workers ready; first worker readiness was 343.667 seconds after container start
  - v4 passed both prior warmup blockers (NoPE pe_dim=0 and native sparse_mla_top_k_lens); weights loaded in 67.76 seconds and model loading used 71.02 GiB per rank
  - requested block size remained 512; runtime effective attention page was 8704; KV capacities were 165,888/165,888/153,600 tokens
  - first v4 client attempt failed closed on missing jinja2 before generation; report SHA-256 fcf10ef00e1deaa936a25ee384f3321175ec24393b26e59e810e83d7265d331c
  - a post-fix same-server diagnostic report exists and is preserved (SHA-256 6ecfa7ddfe78030c1062852ad8792deefd6499e9b83e14a85be0753c6c5f4bd7), but is not accepted as the final end-to-end run because it did not begin from a fresh launch of the committed dependency fix
  - CPU-only continuation must not launch or attach GPUs; one fresh unique three-GPU launch/smoke/owned-stop cycle remains after explicit GPU re-authorization
  - CPU-only final focused GLM-5.3 suite: 21 passed in 0.021 seconds
  - full repository CPU/static suite: 205 passed in 2.717 seconds
  - no-device image import/lock/model-size checks: pass; exact fresh-source patch pre/postimages: pass
  - shell syntax, Python compileall, 38 JSON files, and git diff --check: pass
artifacts:
  - task source: agent/tasks/T20260830-001__glm53-montblanc-deployment/
  - runtime evidence root: task-local cache selected by GLM53_RUNTIME_ROOT (not committed)
  - v4 run evidence: ${GLM53_RUNTIME_ROOT}/runs/glm53-smoke-20260830T144600Z-afb94b4-topklens-v4
  - CPU-only finalization evidence: ${GLM53_RUNTIME_ROOT}/evidence/cpu-only-finalization-20260831T043000Z-v2 (SHA256SUMS SHA-256 f7da6dce9205b4e018caf3f7c75ced1c768239cafe51183f9fa85c65f51b1b7e)
commits:
  - 2e7cbec: initial isolated GLM-5.3 Flash deployment package
  - 1853b33..3b66633: model-index, upstream Dockerfile, FlashInfer packaging, SM120 build, and safe device-passthrough fixes
  - 046a0bc: NoPE fp8_ds_mla cache write fix
  - afb94b4: native SM120 sparse top-k length fix
  - final CPU-only dependency/evidence commit: this task-finalization commit (exact SHA in the final report)
remaining acceptance:
  - after explicit GPU re-authorization only, run one fresh unique launch -> smoke -> ownership-checked stop cycle from the pushed branch; no other implementation slice remains
final handoff link: agent/tasks/T20260830-001__glm53-montblanc-deployment/handoffs/TO_GPT_20260831-042717.md
