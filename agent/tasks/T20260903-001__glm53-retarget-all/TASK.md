# T20260903-001__glm53-retarget-all

task identity: T20260903-001__glm53-retarget-all
objective: retarget the active GLM-5.2 stateful-edit-v3 experiment package to GLM-5.3 without changing its scientific semantics
status: cpu_port_complete_gpu_validation_pending
base tip: 61da82cf29c2c0319b7f90dba211713b15a2de06
branch: agent/T20260903-001__glm53-retarget-all
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260903-001__glm53-retarget-all
runtime mode: isolated-native
write scope:
  - task-local source, configs, scripts, tests, docs, inventory, and handoff required for the GLM-5.3 retarget
forbidden paths:
  - Putpocket_env/
  - data/
  - logs/
  - models/
  - .ssh/
fixed decisions:
  - GPU use, CUDA execution, model loading, GPU jobs, and NVIDIA device passthrough are forbidden for this task
  - canonical master and all historical task worktrees are read-only
  - GLM-5.3 deployment base is 61da82cf29c2c0319b7f90dba211713b15a2de06
  - preserve existing stateful-edit-v3 semantics; do not add new SWE-bench multi-turn meaning or benchmark instances
  - preserve the full-target-prefill then donor-row-overwrite backend as an accuracy ablation only; do not call it true compute-saving partial prefill
  - preserve GLM-5.2 results and artifacts as historical/superseded provenance; never relabel them as GLM-5.3 evidence
plan:
  - inventory active GLM-5.2 stateful implementation, configs, scripts, tests, and task packages
  - map every reusable/model-specific source to its GLM-5.3 target and explicit disposition
  - port the existing harness, proxy, evaluator, ratio isolation, cache/indexer contract, and tests to the locked GLM-5.3 Flash NVFP4 runtime
  - statically validate architecture, layer mapping, tokenizer/chat template, cache products, indexer dimensions, and vLLM source pins
  - run focused and full CPU/static verification plus source/package hash checks
  - create final handoff, commit, and normally push only the dedicated branch
completion criteria:
  - machine-readable inventory and source-to-target matrix cover every active GLM-5.2 stateful path reviewed
  - GLM-5.3 package consumes the existing immutable deployment lock and does not duplicate model authority
  - default-off accuracy-ablation semantics and exact single-instance episode provenance remain explicit
  - focused/full CPU tests and all static/hash/scope checks pass
  - dedicated branch is clean, committed, and equal to its GitHub tracking ref
validation:
  - focused: `pytest -q tests/test_glm53_stateful_edit.py` -> 14 passed, 6 subtests passed
  - full: `pytest -q` -> 221 passed, 64 subtests passed
  - Python compile: GLM-5.3 CLI, proxy, runtime hook, and focused tests passed
  - shell syntax: all five new `scripts/glm53/*stateful*` scripts passed `bash -n`
  - JSON/schema: experiment lock, authoring template, four new schemas, inventory, and CPU metadata audit parsed; schema assertions passed in tests
  - fresh patch chain: exact vLLM `878631b` archive plus deployment patches and stateful patch applied fail-fast in `overlay-validation-008`; all four postimages and hook matched lock hashes
  - model/config static check: exact downloaded metadata produced `model-contract-validation-final.json` SHA-256 `a90d3be7cde9db6bc8dd6cb099249fdea5bc78b1be2274817ee9e9cadf2d9a43`
  - `git diff --check` passed
  - no GPU query, CUDA execution, weight load, device passthrough, job, Vikunja, or AFFiNE action occurred
artifacts:
  - repository task package: agent/tasks/T20260903-001__glm53-retarget-all/
  - CPU metadata audit cache: `/home/dyryu/.cache/putpocket-runtime/T20260903-001__glm53-retarget-all/metadata-audit/`
  - final fresh overlay evidence: `/home/dyryu/.cache/putpocket-runtime/T20260903-001__glm53-retarget-all/overlay-validation-008/`
  - current model cache: absent; 169,755,856,896 free bytes was below the locked 197,881,155,500-byte weight payload, so no unsafe download was attempted
  - historical deployment model-verification evidence remains read-only and is not stateful GLM-5.3 success evidence
commits:
  - implementation commit: f10da9cecaabc11b84b8218df50dcfd28c062c3e
  - validation handoff commit: 067f4ed11fb0b1f1eb652535e901e91a8129f5e5
final handoff link: agent/tasks/T20260903-001__glm53-retarget-all/handoffs/TO_GPT_20260903_FINAL.md
