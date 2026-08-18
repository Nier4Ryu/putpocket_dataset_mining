# putpocket-dataset-mining

Standalone Dataset Mining implementation for MBPP-stateful Cline-style mining episodes.

The repository also carries a provider-neutral, Git-delivered Cluster Center
experiment-package foundation. `putpocket-cluster` validates GLM-5.2 profiles,
renders Slurm jobs without submitting them, enforces compute-allocation guards,
and performs staged readiness checks. See `docs/CLUSTER_GLM52_PHASE1.md`.

The active diagnostic package builds pinned vLLM from source for SM90 in a CPU
Slurm job, then gates one exact four-H200 GLM-5.2 native-DSA trace run through
an `afterok` dependency. Its single official SWE-bench Pro row result is never
quality-score eligible and cannot transition to the full split. See
`docs/CLUSTER_GLM52_DSA_DIAGNOSTIC.md`. Montblanc support remains CPU/static
only; builds, downloads, GPU work, and evaluation are allocation guarded.

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
putpocket-cluster profiles validate
putpocket-cluster readiness --profile glm52_nvfp4_tp1_pcp4_ep --stage static
putpocket-swebench-pro validate
```

Runtime mining requires Docker, Codex CLI, HuggingFace dataset/model access, Transformers, and local vLLM Python engine availability. vLLM generation is invoked with already-rendered prompt strings; vLLM is never asked to apply a chat template internally.

Server-2 uses one active uv-managed environment at `Putpocket_env`. The canonical setup entrypoint is `scripts/env/bootstrap_sr.sh --preset server2`; the canonical activation entrypoint is `source scripts/env/env_activate.sh`. Root-level generated `PROMPT`, `COMMANDS`, and `TO_GPT` Markdown files are not part of the active workflow; task reports belong under `agent/tasks/<TASK>/handoffs/`.
