# Decision Log — Dataset Mining v0.5

## Resolved decisions

- This is a standalone empty dataset-mining repo.
- The repo owns its own `Putpocket_env`, `externals/`, Docker specs, mining code, data, and artifacts.
- Python package import root is `putpocket_dataset_mining`.
- Env activation script is `scripts/env/env_activate.sh`.
- Model is `Qwen/Qwen3.5-9B`.
- Shared HF cache path and `RANDOM_SEED=42` are managed in constants.
- Dataset source is HuggingFace MBPP, preferring `google-research-datasets/mbpp`, fallback `Muennighoff/mbpp`.
- MBPP `code` field is reference/gold solution, not initial code.
- Agent-visible initial workspace contains only `solution.py` stub.
- MBPP unit tests are hidden verifier-only and injected only into verifier containers.
- `.clinerules` is not created in Docker workspace.
- Cline rules are static prompt artifacts and are rendered into system prompts.
- Prompt rendering uses explicit tokenizer chat template in Putpocket code.
- vLLM must receive rendered prompt strings; no internal chat template rendering.
- Generation is deterministic greedy by default, `temperature=0.0`.
- `mining_seed` defaults to 42 and is distinct from `RANDOM_SEED=42` for later evaluation.
- Headless Cline uses original Cline tool format from `externals/cline`.
- Compact Cline prompt is default for 9B model.
- Docker default base is `ubuntu:22.04` with Python 3.13 installed into the image.
- Docker workspace is host-mounted at `/workspace` and containers run as host UID/GID.
- Judge is Codex CLI with read-only sandbox and approval never.
- Multi-sample mining is master-worker parallel controller, not a sequential loop.
- Master is the only SQLite DB writer.
- Multi-sample profiles are debug, first_parallel, full_server.
- vLLM builds are capped at 16 CPU build threads by default on the Blackwell server.
- Runtime workers may use only GPUs 0,1,2; `full_server` uses 3 workers.
- Local materialized dataset views are under `data/dataset_mining/datasets/`.
- Cross-repo dataset import/export is out of scope.

## No remaining blocking human decisions

The implementation may proceed once `state.yaml` lock is changed to `ready_for_implementation`.
