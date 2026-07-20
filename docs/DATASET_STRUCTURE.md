# Dataset Structure

Dataset Mining stores raw execution artifacts and local materialized dataset views under this repo only.

## Raw Runs

Raw attempt artifacts live at:

```text
data/dataset_mining/runs/<run_id>/samples/<sample_id>/<attempt_id>/
```

Each attempt writes:

```text
episode_timeline.md
episode_timeline.jsonl
source_task.json
scenario_config.yaml
prepared/
serving/
trajectories/
workspace/
workspace_snapshots/
verification/
judge/
episode_summary.json
```

`prepared/` contains semantic messages, static Cline rules, rendered prompt text, and tokenization metadata. `.clinerules` is not written into the Docker workspace.

`workspace/` is the agent-visible host-mounted workspace. It starts with `solution.py` only. MBPP unit tests are not materialized here.

`verification/<history>/workspace/` is a copied verifier-only workspace. Hidden MBPP pytest files are injected there as `tests/test_solution.py`.

## SQLite Index

The mining index is:

```text
data/dataset_mining/mining_index.sqlite
```

The multi-sample controller keeps SQLite writes in the master process only. Workers return attempt summaries to the master.

## Materialized Views

Local dataset views are written to:

```text
data/dataset_mining/datasets/<dataset_version>/
```

Each dataset view contains:

```text
dataset_manifest.yaml
accepted.jsonl
rejected.jsonl
uncertain.jsonl
artifact_index.jsonl
```

Rows reference local run artifact paths. Packaging, copying, or importing this data into another repo is out of scope.
