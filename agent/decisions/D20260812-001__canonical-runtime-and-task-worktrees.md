# Decision D20260812-001: Canonical Runtime and Task Worktrees

Accepted.

The canonical runtime checkout is `/home/${USER}/putpocket_dataset_mining` locally and `/workspace/putpocket_dataset_mining` on RunPod.

All new Agent task worktrees use `/home/${USER}/putpocket_dataset_mining_worktrees` locally and `/workspace/putpocket_dataset_mining_worktrees` on RunPod.

The canonical uv environment may be activated from a task worktree only as a shared Python runtime with an explicit task source overlay. Activation must not mutate editable metadata.

Production operations are allowed by default only from canonical-runtime context.
