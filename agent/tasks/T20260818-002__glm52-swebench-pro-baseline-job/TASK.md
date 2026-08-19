# T20260818-002__glm52-swebench-pro-baseline-job

task identity: T20260818-002__glm52-swebench-pro-baseline-job
objective: >
  Deliver the authoritative vLLM-based, exactly-four-H200 native GLM-5.2 DSA
  single-instance diagnostic as a two-stage CPU-build/H200-run Slurm package.
  The fixed official SWE-bench Pro row is diagnostic only and never permits a
  full benchmark or quality-threshold claim.
status: vllm_two_stage_package_complete_with_measured_site_values
stacked base tip: 819854e28ae170ef43722118dfc3d2a53f43c7ce
superseded diagnostic tip: c1d4569de8089f41f60761b577232a37ff3aa451
measured-site source tip: 186d096a99bbe7a86c8eb6dff5302f88774c9133
branch: agent/T20260818-001__glm52-cluster-package-foundation
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260818-001__glm52-cluster-package-foundation
stacking note: >
  The repository task CLI has no same-worktree resume/start operation. This
  task remains separately recorded while intentionally sharing the phase-1
  branch/worktree; no earlier history is rewritten.
runtime mode: isolated-native
write scope:
  - active vLLM diagnostic source, configs, patch, instrumentation, docs, tests, and task-local handoff
forbidden paths:
  - Putpocket_env/
  - data/
  - logs/
  - models/
  - .ssh/
fixed decisions:
  - no Login, Slurm, Vikunja, AFFiNE, GPU, or model-weight access from Montblanc
  - vLLM source commit 4a3447d200e5aa428d68d1a00aa00f1a19a1a729 only
  - CPU Slurm build from source for SM90; no GPU request and no precompiled vLLM wheel
  - H200 job depends afterok on the successful immutable checksummed build bundle
  - exact model nvidia/GLM-5.2-NVFP4 at aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
  - TP4, ModelOpt FP4 to Marlin W4A16, native sparse indexer, no MTP/PD/offload/fallback
  - one node, exactly --gres=gpu:H200:4, H200/gsai-account/hpgpu, CPU32, 512G, 06:00:00
  - exact pinned instance and prompt identities in glm52_vllm_diagnostic.lock.json
  - same live server trace OFF/ON, exact token-ID equality, prefix cache disabled/reset
  - native 21-full/57-shared bounded raw/top-k capture or classified BLOCKED
  - pinned source requires DeepGEMM first-use runtime JIT and has no AOT CUDA-kernel path;
    only that inseparable native startup JIT is allowed after CPU-bundle
    validation, with run-local audited/checksummed caches and no reuse
  - general source/project/vLLM compilation on H200 remains forbidden and any
    unproven or out-of-scope compiler/cache activity is classified BLOCKED
  - unchanged official one-row evaluator only; no full selection or >=40 calculation
site state:
  - CPU build is cpu-max24/gsai-account/nogpu, 24 CPUs, 192G, 06:00:00
  - CPU build scratch is /local-data/user-data/jslee202403/putpocket-vllm-build-scratch and Docker is /usr/bin/docker
  - H200 resources remain H200/gsai-account/hpgpu, exactly four typed H200s, CPU32, 512G, 06:00:00
  - H200 work and artifact roots are configured below the measured writable /local-data/user-data parent
  - the H200 allocation validates and creates those roots before any clone or model access
plan:
  - validate exact official vLLM source capabilities and immutable build identities
  - patch only native sparse_attn_indexer score/top-k exposure and build inside CPU allocation
  - atomically publish and revalidate a source/wheel/runtime-image bundle under shared storage
  - validate bundle and SM90 imports before any model metadata/weight access on H200
  - run exact OFF/ON diagnostic, validate bounded artifacts, then unchanged one-row evaluator
  - render a compact Login control command with two sbatch calls and afterok dependency
completion criteria:
  - focused and broad CPU/static suites pass without GPU/model/container/cluster activity
  - both tracked scripts and the rendered one-line command pass bash syntax checks
  - active renderer and entrypoints contain no SGLang/full-transition path
  - source changes committed and pushed normally on the stacked branch
submission boundary: >
  Codex does not connect or submit. The orchestrator renders the exact project
  commit with the recorded measured inventory and submits both jobs.
final handoff link: agent/tasks/T20260818-002__glm52-swebench-pro-baseline-job/handoffs/TO_GPT_20260819_VLLM.md
