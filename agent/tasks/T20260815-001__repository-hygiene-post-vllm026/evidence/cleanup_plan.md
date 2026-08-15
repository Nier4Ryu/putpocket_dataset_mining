# Repository Hygiene Cleanup Plan

## Canonical Runtime Protected

- `/home/dyryu/putpocket_dataset_mining/Putpocket_env`
- vLLM `v0.26.0` at `568afb3a13806beb53bb2e6bd518269357b237c0`
- LMCache at `72eb0e375bcf0739a45046433f46ee32be361656`
- `configs/env/server2_blackwell.lock.yaml`
- `configs/env/runpod_dev.lock.yaml`
- frozen ClassEval package under `data/dataset_mining/datasets/classeval_stateful_working_v0/`

## Tracked Cleanup

- Delete root-level generated prompt/report/command Markdown files.
- Delete retired `scripts/env` compatibility wrappers and archived legacy environment scripts.
- Delete stale `configs/legacy` GLM serving/evaluation configs that point at removed local legacy environments.
- Keep only `scripts/env/bootstrap_sr.sh` and `scripts/env/env_activate.sh` as the source-controlled env shell interface.
- Remove active source references to deleted legacy environment directory names.

## Post-Integration Filesystem Cleanup

Delete these exact directories only after the tracked cleanup is integrated and validated on canonical `master`:

```text
/home/dyryu/putpocket_dataset_mining/Putpocket_env.pre_unification_20260812_143359
/home/dyryu/putpocket_dataset_mining/Putpocket_env_glm52
/home/dyryu/putpocket_dataset_mining/Putpocket_env_glm52_v025
```

Before deletion, re-check realpath, active process use, and that the path is not the canonical `Putpocket_env`.
