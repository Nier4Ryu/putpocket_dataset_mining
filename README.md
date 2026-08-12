# putpocket-dataset-mining

Standalone Dataset Mining implementation for MBPP-stateful Cline-style mining episodes.

The implementation follows `tasks/objectives/dataset_mining/objective.yaml` and keeps raw run artifacts under `data/dataset_mining/runs/`, the SQLite index at `data/dataset_mining/mining_index.sqlite`, and local materialized datasets under `data/dataset_mining/datasets/`.

Basic commands:

```bash
./scripts/env/bootstrap_sr.sh --preset server2
source scripts/env/env_activate.sh
./scripts/env/bootstrap_sr.sh --preset server2 --doctor-only
putpocket-dataset-mining doctor
putpocket-dataset-mining docker ensure-image
putpocket-dataset-mining single --config configs/dataset_mining/mbpp_stateful_single.yaml --sample-index 0
putpocket-dataset-mining multi --config configs/dataset_mining/mbpp_stateful_multi.yaml --profile debug
```

Runtime mining requires Docker, Codex CLI, HuggingFace dataset/model access, Transformers, and local vLLM Python engine availability. vLLM generation is invoked with already-rendered prompt strings; vLLM is never asked to apply a chat template internally.

Server-2 uses one active uv-managed environment at `Putpocket_env`. The canonical setup entrypoint is `scripts/env/bootstrap_sr.sh --preset server2`; `scripts/env/bootstrap_env.sh` is retained only as a compatibility wrapper. Legacy local GLM bootstrap environments are retired from the active Server-2 path.
