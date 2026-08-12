# T20260813-001__advisory-lock-queue-control

task identity: T20260813-001__advisory-lock-queue-control
objective: advisory-lock-queue-control
status: implementation_validated
base tip: 3225e44c262375604a0b820be8cfe63f8b4a9e7d
branch: agent/T20260813-001__advisory-lock-queue-control
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260813-001__advisory-lock-queue-control
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
  - implement .git/putpocket-locks advisory locks
  - wire locks into task start, task integrate, runtime sync, and mutating bootstrap/build flows
  - expose lock status and pending request inspection
  - document required pre-task collaboration checks
  - validate focused regressions
  - handoff
completion criteria:
  - tests pass
  - task-local TO_GPT handoff exists
validation:
  - git diff --check: PASS
  - bash -n scripts/env/*.sh cloud/runpod/*.sh: PASS
  - compileall src tests with PYTHONPYCACHEPREFIX=/tmp/putpocket_pycache_lock_task: PASS
  - pytest tests/test_agent_control.py: PASS, 11 tests
  - focused pytest agent/bootstrap/build/runpod minus local /workspace-dependent case: PASS, 37 passed, 1 deselected, 2 subtests
  - putpocket_dataset_mining.agent_cli locks status: PASS, no active locks, no pending requests
  - merged latest origin/master 563bde9979aab22ebc2e3f5f9e7f88a817635c3a and repeated validation: PASS
artifacts:
  - agent/tasks/T20260813-001__advisory-lock-queue-control/
commits:
  - 42bdd8378cdd34a3b05a447397f8be3e0080045c feat(agent): add advisory coordination locks
  - pending post-merge metadata update
final handoff link: agent/tasks/T20260813-001__advisory-lock-queue-control/handoffs/TO_GPT_20260813-014600.md
