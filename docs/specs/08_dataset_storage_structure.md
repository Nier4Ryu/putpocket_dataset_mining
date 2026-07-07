# Dataset Storage Structure

This repo stores local mining artifacts and local materialized datasets.
Do not design cross-repo import/export now.

## Roots

```text
data/dataset_mining/runs/
data/dataset_mining/mining_index.sqlite
data/dataset_mining/datasets/
```

## runs/

Raw attempt artifacts live under:

```text
data/dataset_mining/runs/<run_id>/samples/<sample_id>/<attempt_id>/
```

Required files include:

```text
episode_timeline.md
episode_timeline.jsonl
source_task.json
scenario_config.yaml
prepared/
serving/
trajectories/
workspace_snapshots/
verification/
judge/
episode_summary.json
```

## mining_index.sqlite

Local controller DB for pass/fail/cache lookup.
Master is the only DB writer.

## datasets/

Local materialized dataset views live under:

```text
data/dataset_mining/datasets/<dataset_version>/
```

Each dataset contains:

```text
dataset_manifest.yaml
accepted.jsonl
rejected.jsonl
uncertain.jsonl
artifact_index.jsonl
```

Rows may reference local `runs/` artifact paths.
How another repo later consumes or copies these datasets is out of scope.

## Dataset versions

```yaml
mbpp_stateful_debug_v0:
  debug profile output; not for real evaluation

mbpp_stateful_parallel_smoke_v0:
  2-worker E2E sanity dataset

mbpp_stateful_working_v0:
  20 accepted samples for initial working experiments
```
