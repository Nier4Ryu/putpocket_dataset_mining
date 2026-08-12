# T20260812-001__runpod-agent-dev-image

task identity: T20260812-001__runpod-agent-dev-image
objective: runpod-agent-dev-image
status: closed
base tip: 487bc28e4e799daca4c6c73134110662ce3dd335
branch: agent/T20260812-001__runpod-agent-dev-image
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260812-001__runpod-agent-dev-image
runtime mode: shared-python-overlay
write scope:
  - source/docs/tests required for this task
forbidden paths:
  - Putpocket_env/
  - Putpocket_env_glm52/
  - Putpocket_env_glm52_v025/
  - data/
  - logs/
  - models/
  - .ssh/
fixed decisions:
  - canonical runtime checkout is /home/${USER}/putpocket_dataset_mining or /workspace/putpocket_dataset_mining
  - task worktrees live under /home/${USER}/putpocket_dataset_mining_worktrees or /workspace/putpocket_dataset_mining_worktrees
plan:
  - pin Node, Zellij, Codex, and uv tool contract
  - extend RunPod CUDA 12.9.1 dev Dockerfile with agent tools
  - add inert startup helper and private CODEX_HOME policy
  - strengthen Docker build context exclusions
  - update RunPod template and documentation
  - build and smoke-test local image
  - run secret leakage audit
  - integrate source to master
completion criteria:
  - tests pass
  - task-local TO_GPT handoff exists
  - local image build passes
  - CPU-only smoke passes
  - secret leakage audit passes
  - Docker Hub publication attempted only when explicit repo/auth are available
validation:
  - git diff --check: PASS
  - bash -n scripts/env/*.sh and cloud/runpod/*.sh: PASS
  - compileall src tests: PASS
  - unittest discover: PASS, 140 tests, 6 skipped
  - focused pytest: PASS, 33 tests, 12 subtests
  - local Docker build: PASS
  - CPU-only smoke: PASS
  - startup helper CODEX_HOME mode/no-auth check: PASS
  - secret leakage audit: PASS
  - Docker Hub publication: BLOCKED, DOCKERHUB_USERNAME/PUTPOCKET_RUNPOD_IMAGE_REPO/PUTPOCKET_RUNPOD_IMAGE_VERSION unset and no Docker Hub username detected
artifacts:
  - agent/tasks/T20260812-001__runpod-agent-dev-image/
  - /tmp/putpocket_runpod_agent_smoke.YPaSEl
  - /tmp/putpocket_runpod_agent_audit_refined.oZvEQq
commits:
  - 3076015a0b4a3e725105135a0fe064627fee4bdb feat(runpod): add agent development image contract
  - pending closeout commit
final handoff link: agent/tasks/T20260812-001__runpod-agent-dev-image/handoffs/TO_GPT_20260812-221849.md
