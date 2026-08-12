# T20260812-001__agent-worktree-control-plane

task identity: T20260812-001__agent-worktree-control-plane
objective: Adopt durable Agent task/worktree control plane for SR repository.
status: in_progress
base tip: 120353c988c0ef0b411bf67eec3cff8b1e0b4abf
branch: agent/T20260812-001__agent-worktree-control-plane
worktree: /home/dyryu/putpocket_dataset_mining_worktrees/T20260812-001__agent-worktree-control-plane
runtime mode: shared-python-overlay
write scope:
  - AGENTS.md
  - agent/
  - examples/agent.toml.example
  - scripts/env/env_activate.sh
  - src/putpocket_dataset_mining/agent_control.py
  - src/putpocket_dataset_mining/agent_cli.py
  - pyproject.toml
  - tests/
forbidden paths:
  - Putpocket_env/
  - Putpocket_env_glm52/
  - Putpocket_env_glm52_v025/
  - data/
  - logs/
  - models/
  - accepted.jsonl
fixed decisions:
  - canonical runtime root is /home/4USER/putpocket_dataset_mining locally
  - task worktree root is /home/4USER/putpocket_dataset_mining_worktrees locally
  - integration is fast-forward only
plan:
  - implement agent control plane
  - update activation context detection
  - inventory legacy worktrees
  - validate with static/unit/focused tests
  - commit, push, integrate, runtime sync
completion criteria:
  - helper CLI works
  - activation distinguishes canonical and task contexts
  - legacy worktrees inventoried
  - tests pass
  - handoff exists under this task
validation:
  - git diff --check: PASS
  - bash -n scripts/env/*.sh: PASS
  - compileall: PASS
  - unittest discover: PASS
  - focused pytest tests/test_agent_control.py: PASS
  - agent doctor task context: PASS
artifacts:
  - agent/tasks/T20260812-001__agent-worktree-control-plane/
  - agent/inventories/legacy-worktrees.md
commits:
  - pending implementation commit
final handoff link: pending
