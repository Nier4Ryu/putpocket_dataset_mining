# Project Structure

`/home/${USER}/putpocket_dataset_mining` is the local canonical runtime and deployment checkout.

`/workspace/putpocket_dataset_mining` is the RunPod canonical runtime and deployment checkout.

`agent/` contains the Agent control plane:

- `policies/`: durable project rules
- `decisions/`: accepted architecture decisions
- `templates/`: task and handoff templates
- `tasks/`: task-local TASK.md and TO_GPT handoffs
- `inventories/`: read-only inventories of legacy worktrees and environment state

Task worktrees live outside the canonical checkout under:

- `/home/${USER}/putpocket_dataset_mining_worktrees`
- `/workspace/putpocket_dataset_mining_worktrees`

The canonical runtime checkout owns `Putpocket_env`, canonical externals, production inference, dataset mining/evaluation, runtime Docker usage, and Cloud deploy source.
