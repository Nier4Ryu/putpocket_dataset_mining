# Project Status

Canonical branch: `master`

Canonical runtime checkout:

- local: `/home/${USER}/putpocket_dataset_mining`
- RunPod: `/workspace/putpocket_dataset_mining`

The canonical runtime checkout must be clean, on `master`, and fast-forwarded to `origin/master` before production execution.

Disposable task worktrees may use the canonical uv environment through a source overlay for CPU/static/focused tests. They must not silently change editable install metadata in the canonical environment.

Known limitation at creation of this policy: existing historical worktrees under `/home/${USER}/worktrees` are inventoried but not deleted.
