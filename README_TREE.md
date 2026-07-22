# Dataset Mining Codex Feed v0.5

This directory contains the objective/specification capsule to hand to Codex for implementing the standalone Dataset Mining repo.

## Current purpose

The repo that receives this feed is a standalone empty dataset-mining repo.
It owns its own env, externals, Docker specs, mining code, run artifacts, and local materialized datasets.
It is not the fixing-code repo and must not depend on that repo.

## Visual tree

```text
dataset_mining_codex_feed_v0_5/
├── AGENT_FEED_REVIEW_PROMPT.md
├── AGENT_FEED_IMPLEMENT_PROMPT.md
├── README_TREE.md
├── configs/
│   └── dataset_mining/
│       ├── mbpp_stateful_multi.yaml
│       └── mbpp_stateful_single.yaml
├── docs/
│   └── specs/
│       ├── 00_decision_log.md
│       ├── 01_repo_and_env.md
│       ├── 02_single_sample_episode.md
│       ├── 03_prompt_rendering_and_history.md
│       ├── 04_docker_workspace.md
│       ├── 05_headless_cline_runtime.md
│       ├── 06_verifier_and_judge.md
│       ├── 07_multi_sample_master_worker.md
│       ├── 08_dataset_storage_structure.md
│       └── 09_implementation_breakdown.md
└── tasks/
    └── objectives/
        └── dataset_mining/
            ├── START_HERE.md
            ├── objective.yaml
            └── state.yaml
```

## Recommended review flow

Copy this feed into the repo root, then ask Codex to review in read-only mode first:

```bash
codex exec \
  --cd /path/to/empty-dataset-mining-repo \
  --sandbox read-only \
  --ask-for-approval never \
  - < AGENT_FEED_REVIEW_PROMPT.md
```

After human review, change:

```yaml
implementation_lock:
  status: human_review_required
```

in `tasks/objectives/dataset_mining/state.yaml` to:

```yaml
implementation_lock:
  status: ready_for_implementation
```

Then run implementation mode:

```bash
codex exec \
  --cd /path/to/empty-dataset-mining-repo \
  --sandbox workspace-write \
  --ask-for-approval on-request \
  - < AGENT_FEED_IMPLEMENT_PROMPT.md
```

## Important scope boundary

Do not design dataset movement into another repo.
Do not create a portable package/export as a required deliverable.
The goal is to finish dataset mining under this repo and document the generated dataset structure.


## Blackwell server constraints in v0.5

- vLLM/CUDA extension builds must use at most 16 CPU build threads by default.
- Runtime dataset-mining workers may use only GPUs 0,1,2.
- `full_server` profile therefore uses 3 workers.

## How to place this feed in the repo

The files in this feed are intended to live at the empty repo root. Prefer one of:

```bash
# from repo root, unpack directly into root
tar -xzf /path/to/dataset_mining_codex_feed_v0_5.tar.gz --strip-components=1
```

or:

```bash
# if already extracted as dataset_mining_codex_feed_v0_5/
rsync -a dataset_mining_codex_feed_v0_5/ ./
```

After that, run Codex from the repo root using:

```bash
codex exec   --cd .   --sandbox read-only   --ask-for-approval never   - < AGENT_FEED_REVIEW_PROMPT.md
```

or, after unlocking `state.yaml`, implementation mode:

```bash
codex exec   --cd .   --sandbox workspace-write   --ask-for-approval on-request   - < AGENT_FEED_IMPLEMENT_PROMPT.md
```

If you keep the feed nested under `dataset_mining_codex_feed_v0_5/` and run from the parent root, the relative paths inside `START_HERE.md` will not match the repo layout unless Codex first copies the feed contents into the root.
