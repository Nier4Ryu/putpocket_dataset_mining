# Agent Operating Contract

Mandatory read order for every source-changing task:

1. `AGENTS.md`
2. `agent/PROJECT_STRUCTURE.md`
3. `agent/PROJECT_STATUS.md`
4. `agent/IMPLEMENTATION_PLAN.md`
5. relevant files under `agent/policies/` and `agent/decisions/`
6. current `agent/tasks/<TASK>/TASK.md`
7. current source and tests
8. latest final task-local `agent/tasks/<TASK>/handoffs/TO_GPT_*.md`

Evidence precedence:

current source, accepted decisions, canonical status, Git history, and test
evidence outrank old handoffs and reports.

Canonical runtime checkout:

- local: `/home/${USER}/putpocket_dataset_mining`
- RunPod: `/workspace/putpocket_dataset_mining`
- branch: `master`
- HEAD: must equal `origin/master` after runtime sync

New task worktrees:

- local root: `/home/${USER}/putpocket_dataset_mining_worktrees`
- RunPod root: `/workspace/putpocket_dataset_mining_worktrees`
- branch pattern: `agent/T<YYYYMMDD>-<NNN>__<short-topic>`
- worktree pattern: `${PUTPOCKET_WORKTREE_ROOT}/T<YYYYMMDD>-<NNN>__<short-topic>`

Do not create new task worktrees under `/home/${USER}/worktrees`.
