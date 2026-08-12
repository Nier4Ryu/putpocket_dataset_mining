# Artifact Policy

Task-local durable handoffs live under:

`agent/tasks/<TASK>/handoffs/TO_GPT_<YYYYMMDD-HHMMSS>.md`

Do not create new root-level `TO_GPT` files.

Runtime artifacts, model outputs, datasets, Docker exports, credentials, virtual environments, and caches must not be committed.

Use evidence labels in final handoffs:

- `CODE_CONFIRMED`
- `TEST_CONFIRMED`
- `REAL_EXECUTION_CONFIRMED`
- `ARTIFACT_CONFIRMED`
- `HANDOFF_REPORTED`
- `INFERENCE`
- `BLOCKED`
