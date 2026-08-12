# Task Worktree Policy

Every meaningful task owns one branch and one linked worktree.

Task ID format:

`T<YYYYMMDD>-<NNN>__<short-topic>`

Branch:

`agent/T<YYYYMMDD>-<NNN>__<short-topic>`

Worktree:

`${PUTPOCKET_WORKTREE_ROOT}/T<YYYYMMDD>-<NNN>__<short-topic>`

Runtime modes:

- `shared-python-overlay`: canonical environment plus task source overlay; no environment mutation.
- `isolated-native`: task-specific environment and external worktrees; required for vLLM, LMCache, torch, CUDA/C++, dependency, and lock changes.
- `audit-only`: no source or runtime mutation.

Task worktrees must not run durable production workflows unless the task policy explicitly allows bounded validation.
