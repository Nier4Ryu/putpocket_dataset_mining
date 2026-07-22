# Env Setup Audit Report

## Executive Summary

Reusable environment setup code exists, but it is partial rather than a single complete Blackwell bootstrap.

The repo already has:
- repo-local env creation at `Putpocket_env` via `scripts/env/bootstrap_env.sh`
- activation and build-limit exports via `scripts/env/env_activate.sh`
- external checkout and editable-install helpers through `putpocket-dataset-mining externals ...`
- Docker image management through `putpocket-dataset-mining docker ensure-image`
- setup validation through `putpocket-dataset-mining doctor`

What is missing: a single script or lockfile that installs the exact full runtime stack from scratch, especially `torch==2.10.0+cu128`, `ray==2.55.1`, editable vLLM, editable LMCache, and any Blackwell/GLM sparse-MLA/DeepGEMM-specific checks.

## Repo Context

- Repo path: `/home/dyryu/putpocket_dataset_mining`
- Git branch: `master`
- Git HEAD inspected: `40d1515c4c87487b843596706e3d05e9edb5358f`
- Remote: `origin https://github.com/Nier4Ryu/putpocket_dataset_mining.git`
- Git status at audit start: `## master...origin/master`
- Package/import root: `src/putpocket_dataset_mining`
- CLI entrypoint: `putpocket-dataset-mining = putpocket_dataset_mining.cli:main`

## Found Env Setup Files

| Purpose | Path | Exists? | Notes |
|---|---:|---:|---|
| Env bootstrap | `scripts/env/bootstrap_env.sh` | yes | Creates `Putpocket_env`, sources activation, upgrades pip tooling, installs repo editable with `[dev]`. |
| Env activation | `scripts/env/env_activate.sh` | yes | Exports repo root, CUDA 12.8 paths, `PYTHONPATH`, HF cache env, seed, and build caps; sources `Putpocket_env/bin/activate` if present. |
| External checkout wrapper | `scripts/externals/checkout_externals.sh` | yes | Sources activation and runs `putpocket-dataset-mining externals checkout all`. |
| External metadata/install logic | `src/putpocket_dataset_mining/externals.py` | yes | Defines vLLM/LMCache/Cline URLs, branches, paths, checkout, and editable install with `--no-build-isolation`. |
| CLI setup commands | `src/putpocket_dataset_mining/cli.py` | yes | Provides `doctor`, `docker ensure-image`, `externals checkout`, `externals install-editable`. |
| Doctor command | `src/putpocket_dataset_mining/doctor.py` | yes | Checks `docker`, `codex`, `git`, `python3`, Python modules, config paths, and externals. |
| Python package config | `pyproject.toml` | yes | Requires Python `>=3.13`; dependencies: `datasets`, `pyyaml`, `transformers`; optional `dev`, `vllm`. |
| Dataset mining configs | `configs/dataset_mining/*.yaml` | yes | Holds model id, Docker image, GPU slots, profiles, build-resource env overrides. |
| Dockerfile | `docker/default_python/Dockerfile` | yes | Ubuntu 22.04 image that builds Python 3.13.1 and installs pytest. |
| Objective/spec source | `tasks/objectives/dataset_mining/objective.yaml` | yes | Records intended env policy including Python 3.13, CUDA 12.8, Torch 2.10.0+cu128, Ray 2.55.1. |
| Runtime validation state | `tasks/objectives/dataset_mining/state.yaml` | yes | Records this server as validated with `Putpocket_env`, editable vLLM/LMCache, Docker image, GPUs 4,5,6,7. |
| Env spec docs | `docs/specs/01_repo_and_env.md` | yes | Documents required layout and shared-server CPU/GPU limits. |
| Makefile | `Makefile` | no | No Make targets found. |
| Requirements files | `requirements*.txt` | no | No repo-level requirements files found. |
| uv lockfile | `uv.lock` | no | No lockfile found. |
| setup scripts directory | `scripts/setup/` | no | No setup subdirectory found. |
| integrations directory | `integrations/` | no | Not present. |
| common package | `src/putpocket_dataset_mining/common/` | no | Not present. |

## Existing Setup Flow

The intended setup order is reconstructible, but split across shell scripts, CLI commands, docs, and `state.yaml`:

1. Provide Python 3.13.
2. Create repo-local `Putpocket_env` with `scripts/env/bootstrap_env.sh`.
3. Activate with `source scripts/env/env_activate.sh`.
4. Install missing runtime packages not covered by `pyproject.toml` bootstrap, notably Torch 2.10.0+cu128 and Ray 2.55.1.
5. Checkout externals with `scripts/externals/checkout_externals.sh` or `putpocket-dataset-mining externals checkout all`.
6. Editable-install vLLM and LMCache using `putpocket-dataset-mining externals install-editable ...`.
7. Verify Python imports with doctor/import smoke.
8. Build or reuse Docker image with `putpocket-dataset-mining docker ensure-image`.
9. Run compile/unit tests.
10. Only after setup: run mining/evaluation smoke commands.

There is no single canonical script that performs steps 1-10 end-to-end.

## Exact Commands Found In Repo

From `scripts/env/bootstrap_env.sh`:

```bash
"${python_bin}" -m venv "${venv_dir}"
source "${repo_root}/scripts/env/env_activate.sh"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${repo_root}[dev]"
```

From `scripts/env/env_activate.sh`:

```bash
export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="${CUDA_HOME}/bin:${PUTPOCKET_DATASET_MINING_ROOT}/Putpocket_env/bin:${PUTPOCKET_DATASET_MINING_ROOT}/.local_python/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PUTPOCKET_DATASET_MINING_ROOT}/src:${PYTHONPATH:-}"
export PUTPOCKET_HF_HUB_CACHE_DIR="${PUTPOCKET_HF_HUB_CACHE_DIR:-/data/shared/hf_cache/hub}" # if present, otherwise HOME cache
export RANDOM_SEED="${RANDOM_SEED:-42}"
export PUTPOCKET_BUILD_THREADS="${PUTPOCKET_BUILD_THREADS:-32}"
export MAX_JOBS="${MAX_JOBS:-32}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-32}"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-32}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
source "${PUTPOCKET_DATASET_MINING_ROOT}/Putpocket_env/bin/activate"
```

From `scripts/externals/checkout_externals.sh`:

```bash
source "${repo_root}/scripts/env/env_activate.sh"
putpocket-dataset-mining externals checkout all
```

From `src/putpocket_dataset_mining/cli.py`:

```bash
putpocket-dataset-mining doctor
putpocket-dataset-mining docker ensure-image
putpocket-dataset-mining externals checkout all
putpocket-dataset-mining externals install-editable vllm --python python
putpocket-dataset-mining externals install-editable lmcache --python python
```

From `src/putpocket_dataset_mining/externals.py`, editable install runs:

```bash
python -m pip install --no-build-isolation -e <external_path>
```

From `src/putpocket_dataset_mining/docker_workspace.py`, Docker build runs:

```bash
docker build -t putpocket-default-python:ubuntu22.04-py313-v1 -f docker/default_python/Dockerfile docker
```

From `README.md`:

```bash
source scripts/env/env_activate.sh
putpocket-dataset-mining doctor
putpocket-dataset-mining docker ensure-image
putpocket-dataset-mining single --config configs/dataset_mining/mbpp_stateful_single.yaml --sample-index 0
putpocket-dataset-mining multi --config configs/dataset_mining/mbpp_stateful_multi.yaml --profile debug
```

From `tasks/objectives/dataset_mining/state.yaml`, completed validation commands include:

```bash
source scripts/env/env_activate.sh && putpocket-dataset-mining doctor --json
source scripts/env/env_activate.sh && python -m compileall src tests
source scripts/env/env_activate.sh && python -m unittest discover -s tests -v
source scripts/env/env_activate.sh && python -c "import datasets, transformers, vllm, lmcache, putpocket_dataset_mining"
CUDA_VISIBLE_DEVICES=4 putpocket-dataset-mining single --config configs/dataset_mining/mbpp_stateful_single.yaml --run-id single_validation_path_20260707T175646Z
CUDA_VISIBLE_DEVICES=4 putpocket-dataset-mining multi --config configs/dataset_mining/mbpp_stateful_multi.yaml --profile debug --run-id debug_validation_20260707T175646Z
CUDA_VISIBLE_DEVICES=4,5 putpocket-dataset-mining multi --config configs/dataset_mining/mbpp_stateful_multi.yaml --profile first_parallel --run-id first_parallel_validation_20260707T175646Z
CUDA_VISIBLE_DEVICES=4,5,6,7 putpocket-dataset-mining multi --config configs/dataset_mining/mbpp_stateful_multi.yaml --profile full_server --run-id full_server_validation_20260707T175646Z
```

## Externals / Branches

Configured in `src/putpocket_dataset_mining/externals.py`:

| External | Configured path | Configured URL | Configured branch | Current exists? | Current branch | Current commit |
|---|---|---|---|---:|---|---|
| vLLM | `externals/vllm` | `https://github.com/Nier4Ryu/vllm_mod.git` | `Putpocket-v0.19.1` | yes | `Putpocket-v0.19.1` | `b65d39ddbab966bb72110056a481d17e4726892b` |
| LMCache | `externals/lmcache` | `https://github.com/Nier4Ryu/LMCache_mod.git` | `Putpocket-v0.4.4` | yes | `Putpocket-v0.4.4` | `72eb0e375bcf0739a45046433f46ee32be361656` |
| Cline | `externals/cline` | `https://github.com/Nier4Ryu/cline_mod.git` | none | yes | `main` | `03f47045f338dcb6ac45b1ac1d6279a78be2b118` |

No repo-level `.gitmodules`, lock manifest, or pinned external commit file was found. Branches are encoded in Python source and expected state is recorded in `state.yaml`; only the local checked-out directories provide exact commits.

## Build Parallelism / CPU Limits

Build parallelism is configured in four places:

- `scripts/env/env_activate.sh`
- `src/putpocket_dataset_mining/constants.py`
- `configs/dataset_mining/mbpp_stateful_multi.yaml`
- `tasks/objectives/dataset_mining/objective.yaml`

Current defaults:

```bash
PUTPOCKET_BUILD_THREADS=32
MAX_JOBS=32
CMAKE_BUILD_PARALLEL_LEVEL=32
CARGO_BUILD_JOBS=32
NVCC_THREADS=1
```

How to set CPU cap to 32 manually:

```bash
export PUTPOCKET_BUILD_THREADS=32
export MAX_JOBS=32
export CMAKE_BUILD_PARALLEL_LEVEL=32
export CARGO_BUILD_JOBS=32
export NVCC_THREADS=1
```

Important Blackwell note: `env_activate.sh` respects preexisting values through `${VAR:-32}`, but `externals.py::build_env()` force-overwrites subprocess build env with `BUILD_ENV_OVERRIDES` from `constants.py`. If the Blackwell server should use more or fewer than 32 jobs, change `BUILD_ENV_OVERRIDES` and the multi config rather than relying only on shell exports.

## GPU Runtime Config

GPU policy is configured in:

- `src/putpocket_dataset_mining/constants.py`
  - `ALLOWED_CUDA_DEVICES = (4, 5, 6, 7)`
  - `DISALLOWED_CUDA_DEVICES = (0, 1, 2, 3)`
- `configs/dataset_mining/mbpp_stateful_multi.yaml`
  - `allowed_cuda_devices: [4, 5, 6, 7]`
  - `debug_slots: [[4]]`
  - `first_parallel_slots: [[4], [5]]`
  - `full_server_slots: [[4], [5], [6], [7]]`
  - `tensor_parallel_size: 1`
  - `pipeline_parallel_size: 1`
- `src/putpocket_dataset_mining/multi.py`
  - validates the YAML allowed set against `ALLOWED_CUDA_DEVICES`
  - exports `CUDA_VISIBLE_DEVICES` per worker
- `src/putpocket_dataset_mining/model_evaluation/glm_eval.py`
  - defaults smoke to GPU `4`, full to `4,5,6,7`
  - validates each eval worker has one allowed GPU

To adapt from old GPUs `4,5,6,7` to the Blackwell server:

1. If Blackwell should also use physical GPU ids `4,5,6,7`, no code/config change is needed.
2. If Blackwell should use different ids, update both `ALLOWED_CUDA_DEVICES` in `constants.py` and the YAML slots in `configs/dataset_mining/mbpp_stateful_multi.yaml`.
3. Update model-evaluation CLI usage with `--gpu-slots <ids>`; note that validation still uses `ALLOWED_CUDA_DEVICES`.
4. Keep `tp=1`, `pp=1` unless the runner is extended; GLM eval currently rejects other TP/PP values.

## Docker Setup

- Dockerfile path: `docker/default_python/Dockerfile`
- Base image: `ubuntu:22.04`
- Python inside Docker: `ARG PYTHON_VERSION=3.13.1`, installed into `/opt/python/3.13`
- Python build jobs: `ARG PYTHON_BUILD_JOBS=8`
- Image name/tag from constants and config: `putpocket-default-python:ubuntu22.04-py313-v1`
- Installed image packages: bash, build-essential, curl, git, jq, patch, ripgrep, tree, wget, Python build dependencies.
- Python packages in image: upgraded pip/setuptools/wheel and pytest.
- Runtime workspace policy from config/code:
  - workspace root: `/workspace`
  - network: `none`
  - dependency install in workspace: disabled by config
  - host UID/GID used for mounted workspace

Build command found via CLI/code:

```bash
source scripts/env/env_activate.sh
putpocket-dataset-mining docker ensure-image
```

Equivalent code-level Docker command:

```bash
docker build -t putpocket-default-python:ubuntu22.04-py313-v1 -f docker/default_python/Dockerfile docker
```

## Python Package / Dependency Setup

`pyproject.toml`:

```toml
requires-python = ">=3.13"
dependencies = [
  "datasets>=2.20",
  "pyyaml>=6.0",
  "transformers>=4.43",
]

[project.optional-dependencies]
dev = ["pytest>=8.0"]
vllm = ["vllm"]
```

Notably absent from `pyproject.toml`:

- `torch==2.10.0+cu128`
- `ray==2.55.1`
- pinned vLLM/LMCache packages
- CUDA/Blackwell-specific build dependencies

Current original server env inspection:

- Python: `3.13.13`
- Torch: `2.10.0+cu128`
- Ray: `2.55.1`
- vLLM: `0.1.dev15375+gb65d39ddb.cu128`, editable from `externals/vllm`
- LMCache: `0.1.dev1451`, editable from `externals/lmcache`
- datasets: `5.0.0`
- transformers: `5.12.1`
- PyYAML: `6.0.3`
- pytest: `9.1.1`

The repo uses plain `venv`/pip. It can use `uv python find 3.13` as a Python discovery fallback in `bootstrap_env.sh`, but there is no `uv.lock` and no `uv sync`-based setup.

## Model Cache / Constants

From `src/putpocket_dataset_mining/constants.py`:

- `RANDOM_SEED = 42`
- `MINING_SEED_DEFAULT = 42`
- `SHARED_HF_HUB_CACHE_DIR = Path(os.environ.get("PUTPOCKET_HF_HUB_CACHE_DIR", "/data/shared/hf_cache/hub"))`
- `DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"`
- `GLM52_08B_MODEL_ID = "inference-optimization/GLM-5.2-0.8B-A0.8B"`
- `MODEL_EVALUATION_ROOT = data/model_evaluation`

From `scripts/env/env_activate.sh`:

- if `/data/shared/hf_cache/hub` exists, `PUTPOCKET_HF_HUB_CACHE_DIR` defaults there
- otherwise it defaults to `${HOME}/.cache/huggingface/hub`

From `src/putpocket_dataset_mining/serving.py`:

- `LocalVLLMEngine` accepts arbitrary `model_id`
- passes `download_dir=str(cache_dir)` to `vllm.LLM`
- sets `trust_remote_code=True`
- applies `CUDA_VISIBLE_DEVICES` from the given GPU slot
- uses rendered prompt strings only; chat template rendering is outside vLLM

## Doctor / Smoke / Validation Commands

Existing lightweight setup validation:

```bash
source scripts/env/env_activate.sh
putpocket-dataset-mining doctor --json
python -m compileall src tests
python -m unittest discover -s tests -v
python -c "import datasets, transformers, vllm, lmcache, putpocket_dataset_mining"
```

Observed current `doctor --json` result on this server:

- `docker`, `codex`, `git`, `python3`: found
- `yaml`, `datasets`, `transformers`, `vllm`: present
- Dockerfile and dataset configs: present
- externals: all present

Existing runtime smoke commands recorded in state/README, to run only after env and Docker are ready:

```bash
CUDA_VISIBLE_DEVICES=4 putpocket-dataset-mining single --config configs/dataset_mining/mbpp_stateful_single.yaml --sample-index 0
CUDA_VISIBLE_DEVICES=4 putpocket-dataset-mining multi --config configs/dataset_mining/mbpp_stateful_multi.yaml --profile debug
```

Existing GLM eval readiness commands found in repo artifacts/code:

```bash
python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --profile smoke \
  --gpu-slots 4 \
  --workers 1
```

For full GLM eval after smoke:

```bash
python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --profile full \
  --gpu-slots 4,5,6,7 \
  --workers 4
```

On the original RTX 3090 server, `TO_GPT.md` reports GLM was blocked by sparse MLA CUDA backend support even after DeepGEMM install. The smallest Blackwell readiness check is therefore a GLM smoke run after vLLM/LMCache setup, not a full mining run.

## Blackwell Server Setup Plan

### Commands directly found in repo

```bash
git clone https://github.com/Nier4Ryu/putpocket_dataset_mining.git
cd putpocket_dataset_mining

# Create repo-local env. If python3.13 is on PATH:
scripts/env/bootstrap_env.sh

# Or force a specific Python 3.13:
PYTHON_BIN=/path/to/python3.13 scripts/env/bootstrap_env.sh

source scripts/env/env_activate.sh

putpocket-dataset-mining externals checkout all
putpocket-dataset-mining externals install-editable vllm --python python
putpocket-dataset-mining externals install-editable lmcache --python python

putpocket-dataset-mining doctor --json
python -m compileall src tests
python -m unittest discover -s tests -v
python -c "import datasets, transformers, vllm, lmcache, putpocket_dataset_mining"

putpocket-dataset-mining docker ensure-image
```

### Inferred/proposed commands not fully encoded in repo

These are required to reproduce the observed original-server env because the repo does not provide a full dependency lock/install script.

```bash
cd ~/putpocket_dataset_mining

# If Python 3.13 is not available, install/provide it first.
# With uv, if allowed on the Blackwell server:
uv python install 3.13
PYTHON_BIN="$(uv python find 3.13)" scripts/env/bootstrap_env.sh

source scripts/env/env_activate.sh

# Install runtime packages missing from pyproject bootstrap.
# Verify the exact PyTorch index/wheel availability on the Blackwell server.
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple "torch==2.10.0+cu128"
python -m pip install "ray==2.55.1"

# Repo core deps are handled by bootstrap, but this is safe if rerun:
python -m pip install -e ".[dev]"

# Checkout and build editable externals with the current CPU cap.
putpocket-dataset-mining externals checkout all
PUTPOCKET_BUILD_THREADS=32 MAX_JOBS=32 CMAKE_BUILD_PARALLEL_LEVEL=32 CARGO_BUILD_JOBS=32 NVCC_THREADS=1 \
  putpocket-dataset-mining externals install-editable vllm --python python
PUTPOCKET_BUILD_THREADS=32 MAX_JOBS=32 CMAKE_BUILD_PARALLEL_LEVEL=32 CARGO_BUILD_JOBS=32 NVCC_THREADS=1 \
  putpocket-dataset-mining externals install-editable lmcache --python python

# Optional/conditional for GLM sparse MLA if smoke says DeepGEMM is missing.
PUTPOCKET_BUILD_THREADS=32 MAX_JOBS=32 CMAKE_BUILD_PARALLEL_LEVEL=32 CARGO_BUILD_JOBS=32 NVCC_THREADS=1 CUDA_VISIBLE_DEVICES=<blackwell_gpu_id> \
  bash externals/vllm/tools/install_deepgemm.sh

# Validate.
putpocket-dataset-mining doctor --json
python -m compileall src tests
python -m unittest discover -s tests -v
python -c "import torch, ray, datasets, transformers, vllm, lmcache, putpocket_dataset_mining; print(torch.__version__); print(ray.__version__)"

# Docker image for verifier/workspace.
putpocket-dataset-mining docker ensure-image

# GLM readiness smoke on one Blackwell GPU after all setup.
python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --profile smoke \
  --gpu-slots <blackwell_gpu_id> \
  --workers 1
```

### Blackwell-specific config choices

- If Blackwell GPU ids are not `4,5,6,7`, update `ALLOWED_CUDA_DEVICES`, mining YAML slots, and GLM eval command slots before running.
- If Blackwell is dedicated and can use more than 32 CPU build jobs, update `BUILD_ENV_OVERRIDES` in `constants.py` and `build_resources.env_overrides` in `configs/dataset_mining/mbpp_stateful_multi.yaml`; CLI editable installs currently force the constants values.
- If `/data/shared/hf_cache/hub` does not exist, either create/mount it or set `PUTPOCKET_HF_HUB_CACHE_DIR` explicitly before activation.

## Missing / Unclear Items

- No single end-to-end Blackwell setup script.
- No repo-level `requirements.txt`, `constraints.txt`, `uv.lock`, or lock manifest.
- No recorded exact pip command for installing `torch==2.10.0+cu128`.
- No recorded exact pip command for installing `ray==2.55.1`.
- No explicit Blackwell GPU id policy; old code assumes allowed ids `4,5,6,7`.
- Build CPU cap is partly configurable in shell but hardcoded for CLI external installs through `BUILD_ENV_OVERRIDES`.
- No explicit Blackwell sparse MLA / DeepGEMM smoke script; only the prior GLM report and vLLM tool script indicate the likely check.
- No `.gitmodules`; externals are cloned directories managed by helper code and ignored by the main repo.
- Docker image build is available, but not Blackwell-specific and does not install project dependencies into the verifier image beyond pytest.
- `doctor` does not check `torch`, `ray`, `lmcache`, CUDA version, GPU visibility, DeepGEMM, or GLM backend support.
- The repo contains model-evaluation code and `TO_GPT.md` showing GLM failed on RTX 3090 due to sparse MLA backend support; Blackwell should rerun smoke rather than assuming success.

## Recommended Next Action

Smallest next action for the human:

1. On the Blackwell server, clone the repo and run `scripts/env/bootstrap_env.sh`.
2. Install the missing pinned runtime packages `torch==2.10.0+cu128` and `ray==2.55.1`.
3. Run external checkout/editable installs for vLLM and LMCache.
4. Run:

```bash
source scripts/env/env_activate.sh
putpocket-dataset-mining doctor --json
python -c "import torch, ray, datasets, transformers, vllm, lmcache, putpocket_dataset_mining; print(torch.__version__); print(ray.__version__)"
```

5. Before full evaluation, run exactly one GLM smoke sample on one Blackwell GPU. If it reports missing DeepGEMM or sparse MLA backend errors, fix that first and do not run the full 20-sample evaluation yet.

## Appendix: Commands Run

```bash
pwd
git status -sb --untracked-files=normal
git branch --show-current
find . -maxdepth 4 -type f | sort
find scripts -maxdepth 4 -type f | sort
find configs -maxdepth 4 -type f | sort
find docker -maxdepth 4 -type f | sort
find tasks/objectives -maxdepth 4 -type f | sort
find docs -maxdepth 4 -type f | sort
find . -maxdepth 3 \( -name 'Makefile' -o -name 'requirements*.txt' -o -name 'uv.lock' -o -name 'AGENTS.md' -o -name 'README*' -o -name 'pyproject.toml' \) -type f | sort
rg -n "Putpocket_env|env_activate|run_with_putpocket_env|UV_PROJECT_ENVIRONMENT|CUDA_HOME|MAX_JOBS|CMAKE_BUILD_PARALLEL_LEVEL|CARGO_BUILD_JOBS|NVCC_THREADS|vllm|LMCache|lmcache|torch|ray|datasets|transformers|Dockerfile|Qwen|GLM|gpu_slots|CUDA_VISIBLE_DEVICES" . -g '!externals/**' -g '!data/**' -g '!Putpocket_env/**' -g '!codex_runs/**' -g '!**/__pycache__/**' -g '!*.pyc'
sed -n '1,220p' scripts/env/bootstrap_env.sh
sed -n '1,220p' scripts/env/env_activate.sh
sed -n '1,220p' scripts/externals/checkout_externals.sh
sed -n '1,220p' pyproject.toml
sed -n '1,260p' docker/default_python/Dockerfile
sed -n '1,240p' README.md
sed -n '1,260p' src/putpocket_dataset_mining/externals.py
sed -n '1,260p' src/putpocket_dataset_mining/cli.py
sed -n '1,240p' src/putpocket_dataset_mining/doctor.py
sed -n '1,240p' src/putpocket_dataset_mining/constants.py
sed -n '1,260p' src/putpocket_dataset_mining/serving.py
sed -n '1,360p' src/putpocket_dataset_mining/multi.py
sed -n '1,240p' configs/dataset_mining/mbpp_stateful_single.yaml
sed -n '1,260p' configs/dataset_mining/mbpp_stateful_multi.yaml
sed -n '35,115p' tasks/objectives/dataset_mining/objective.yaml
sed -n '55,95p' tasks/objectives/dataset_mining/state.yaml
sed -n '1,140p' docs/specs/01_repo_and_env.md
sed -n '1,120p' docs/specs/09_implementation_breakdown.md
sed -n '1,180p' src/putpocket_dataset_mining/docker_workspace.py
sed -n '1,260p' src/putpocket_dataset_mining/model_evaluation/glm_eval.py
sed -n '560,760p' src/putpocket_dataset_mining/model_evaluation/glm_eval.py
sed -n '920,1020p' src/putpocket_dataset_mining/model_evaluation/glm_eval.py
sed -n '1,180p' src/putpocket_dataset_mining/model_evaluation/dataset_loader.py
sed -n '1,140p' tests/test_glm_eval.py
sed -n '1,160p' TO_GPT.md
sed -n '1,260p' AGENT_GLM_EVAL_GOAL_RUN_COMMANDS.md
git -C externals/vllm status -sb
git -C externals/vllm branch --show-current
git -C externals/vllm rev-parse HEAD
git -C externals/vllm remote -v
git -C externals/lmcache status -sb
git -C externals/lmcache branch --show-current
git -C externals/lmcache rev-parse HEAD
git -C externals/lmcache remote -v
git -C externals/cline status -sb
git -C externals/cline branch --show-current
git -C externals/cline rev-parse HEAD
git -C externals/cline remote -v
sed -n '1,80p' Putpocket_env/pyvenv.cfg
Putpocket_env/bin/python --version
Putpocket_env/bin/python -m pip show torch ray vllm lmcache datasets transformers pyyaml pytest
/usr/local/cuda-12.8/bin/nvcc --version
bash -lc 'source scripts/env/env_activate.sh && putpocket-dataset-mining doctor --json'
```
