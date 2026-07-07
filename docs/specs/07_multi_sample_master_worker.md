# Multi-sample Master-Worker Spec

Multi-sample mining is not a sequential loop around single-sample mining.
It is a parallel master-worker controller.

## Master responsibilities

- Load config.
- Preflight GPU/TP/PP slots.
- Spawn workers.
- Lazily generate sample/config attempts.
- Check SQLite index for prior success/failure.
- Assign jobs to idle workers.
- Aggregate results.
- Master-only DB write.
- Incrementally materialize accepted samples.
- Handle stop signals.
- Cleanup cancelled/running attempts.

## Worker responsibilities

- Own one GPU slot or TP/PP group.
- Load local vLLM Python engine once per worker lifetime.
- Mine one sample attempt at a time.
- Use one Docker episode container per attempt.
- Return result to master.

## Profiles

```yaml
debug:
  num_workers: 1
  target_accepted: 1
  max_attempts: 5
  dataset_version: mbpp_stateful_debug_v0

first_parallel:
  num_workers: 2
  target_accepted: 2
  max_attempts: 10
  dataset_version: mbpp_stateful_parallel_smoke_v0

full_server:
  num_workers: 4
  target_accepted: 20
  max_attempts: 100
  dataset_version: mbpp_stateful_working_v0
```

## GPU policy

Initial Qwen3.5-9B policy:

- TP=1
- PP=1
- 4 workers on GPUs 4,5,6,7 for full_server

Preflight must verify the config is assignable before spawning workers.

## Stop

Support:

- SIGINT/SIGTERM
- stop file
- CLI stop command

Default stop is graceful.
Hard stop deletes partial trajectories, workspace snapshots, and serving logs, while keeping stop summary logs.


## Shared server GPU policy

This repo can only use GPUs 4,5,6,7. Full-server mining therefore means four parallel workers, not eight.

```yaml
full_server:
  num_workers: 4
  target_accepted: 20
  max_attempts: 100

gpu:
  allowed_cuda_devices: [4, 5, 6, 7]
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
  full_server_slots:
    - [4]
    - [5]
    - [6]
    - [7]
```

Preflight must reject any config that tries to allocate GPU 0,1,2,3 or that assigns overlapping GPU slots.
