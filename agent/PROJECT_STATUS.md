# Project Status

## 2026-08-12: Server-A / Server-B execution contract

Accepted decision `D20260812-002` defines Server-A as inference/controller only and Server-B as isolated agent workspace plus hidden verifier/Judge. The canonical RunPod-to-Server-1 profile uses a dedicated SSH remote Docker workspace backend and does not require Docker on RunPod.

Canonical branch: `master`

Canonical runtime checkout:

- local: `/home/${USER}/putpocket_dataset_mining`
- RunPod: `/workspace/putpocket_dataset_mining`

The canonical runtime checkout must be clean, on `master`, and fast-forwarded to `origin/master` before production execution.

Disposable task worktrees may use the canonical uv environment through a source overlay for CPU/static/focused tests. They must not silently change editable install metadata in the canonical environment.

Known limitation at creation of this policy: existing historical worktrees under `/home/${USER}/worktrees` are inventoried but not deleted.
