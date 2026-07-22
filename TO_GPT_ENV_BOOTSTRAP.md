# Env Bootstrap Implementation Report

## Executive Summary

Implemented the canonical repo-local environment flow:

- First-time setup or repair: `./scripts/env/bootstrap_env.sh`
- Every shell session: `source scripts/env/env_activate.sh`

Validation passed on the existing server/env using `--doctor-only`: Python 3.13.13, torch 2.10.0+cu128, Ray 2.55.1, datasets, transformers, vLLM, LMCache with `lmcache.c_ops`, repo import, CLI doctor, compileall, unittest discovery, CUDA visibility, and Docker image inspection all passed. A full destructive/reinstall bootstrap was not run because `Putpocket_env`, externals, and Docker image already existed; the accepted validation path for an existing env was run.

## Files Changed

| Path | Purpose |
| --- | --- |
| `scripts/env/bootstrap_env.sh` | Canonical idempotent setup/provisioning script with stage logs, Python 3.13 env creation, dependency install, externals checkout, editable vLLM/LMCache install, Docker image ensure, doctor/smoke checks, and useful skip/force options. |
| `scripts/env/env_activate.sh` | Source-only session activation script; activates repo-local `Putpocket_env` and exports runtime/build defaults without installing/building/checking out anything. |
| `scripts/env/README.md` | Short operator docs for bootstrap vs activation behavior. |
| `docker/default_python/Dockerfile` | Docker Python build job default changed to 32 to match repo build policy. |
| `src/putpocket_dataset_mining/externals.py` | External checkout is now idempotent: verifies existing git checkouts, fetches/switches/fast-forwards configured branches, records current branch/commit/remote in doctor status, and respects user build env overrides. |
| `src/putpocket_dataset_mining/docker_workspace.py` | Docker image builds pass `PYTHON_BUILD_JOBS=${PUTPOCKET_BUILD_THREADS:-32}`. |
| `src/putpocket_dataset_mining/doctor.py` | Doctor now checks `torch`, `ray`, and `lmcache` module availability in addition to existing modules. |
| `.gitignore` | Ignores generated `logs/env_setup/` bootstrap logs. |

Existing untracked files not modified by this implementation: `TO_GPT_ENV_SETUP_AUDIT.md`, `scripts/env/env_activate_ref.sh`.

## Final User Workflow

First-time setup or repair:

```bash
./scripts/env/bootstrap_env.sh
```

Every new shell session:

```bash
source scripts/env/env_activate.sh
```

Useful repair/validation variants:

```bash
./scripts/env/bootstrap_env.sh --doctor-only
./scripts/env/bootstrap_env.sh --skip-docker
./scripts/env/bootstrap_env.sh --skip-vllm-build --skip-lmcache-build
./scripts/env/bootstrap_env.sh --force-vllm-build
./scripts/env/bootstrap_env.sh --force-lmcache-build
./scripts/env/bootstrap_env.sh --force-docker-build
```

## Implemented Bootstrap Stages

| Stage | Implemented? | Notes |
| --- | --- | --- |
| Resolve repo root | Yes | Works from repo root or subdirectories by resolving from script path. |
| Create logs | Yes | Writes `logs/env_setup/<timestamp>/`; `logs/env_setup/latest` points to latest run. |
| GPU report | Yes | Reports visible GPUs with `nvidia-smi -L` if available; does not set `CUDA_VISIBLE_DEVICES`. |
| Resolve Python 3.13 | Yes | Uses `PYTHON_BIN`, `python3.13`, `.local_python/bin/python3.13`, then `uv python find/install 3.13`; refuses Python 3.10/3.11. |
| Create/reuse `Putpocket_env` | Yes | Reuses existing Python 3.13 env; fails clearly if existing env is not Python 3.13. |
| Activate env for bootstrap | Yes | Sources `scripts/env/env_activate.sh` in the current shell so later stages use the venv. |
| Install dependencies | Yes | Upgrades pip/setuptools/wheel/build helpers; installs torch 2.10.0+cu128, Ray 2.55.1, datasets, transformers, pyyaml, pytest, and editable repo `[dev]`. |
| Checkout externals | Yes | Ensures vLLM, LMCache, and Cline paths; existing git checkouts are verified/fetched/fast-forwarded. |
| Install vLLM editable | Yes | Uses `putpocket-dataset-mining externals install-editable vllm --python <venv python>` with no build isolation; skips if import works unless forced. |
| Install LMCache editable | Yes | Uses same editable external install path; skips if import works unless forced. |
| Record external lock | Yes | Writes `externals.lock` in each setup log dir. |
| Ensure Dockerfile/image | Yes | Creates Dockerfile if missing; builds/ensures image unless skipped; skips rebuild when image exists unless forced. |
| Doctor/smoke | Yes | Runs imports, CLI doctor, compileall, unittest discovery, and Docker image inspect. |
| Fail clearly | Yes | Failure output includes failed stage, command, log path, and rerun command. |

## Implemented Activation Behavior

`scripts/env/env_activate.sh` must be sourced. Direct execution prints:

```text
source this file instead of executing it.
Usage: source scripts/env/env_activate.sh
```

When sourced, it:

- resolves repo root from script location,
- checks `Putpocket_env/bin/activate` exists,
- tells the user to run `./scripts/env/bootstrap_env.sh first` if missing,
- activates `Putpocket_env`,
- exports `CUDA_HOME=/usr/local/cuda-12.8` unless already set,
- adds CUDA bin/lib64 and `src` to runtime paths when those paths exist,
- exports `UV_PROJECT_ENVIRONMENT` to repo-local `Putpocket_env` unless already set,
- exports `PYTHONNOUSERSITE=1`,
- exports `TZ=Asia/Seoul` unless already set,
- exports HF cache policy using `/data/shared/hf_cache/hub` when present,
- exports build cap defaults without overriding user values,
- prints a concise activation summary.

It does not install packages, build vLLM/LMCache, checkout externals, run Docker, run mining, run evaluation, or set `CUDA_VISIBLE_DEVICES`.

## Build Limits

Default build settings:

```bash
PUTPOCKET_BUILD_THREADS=32
MAX_JOBS=32
CMAKE_BUILD_PARALLEL_LEVEL=32
CARGO_BUILD_JOBS=32
NVCC_THREADS=1
```

User-provided values take precedence:

```bash
export PUTPOCKET_BUILD_THREADS=16
source scripts/env/env_activate.sh
```

Validated result:

```text
PUTPOCKET_BUILD_THREADS=16
CUDA_VISIBLE_DEVICES=unset
```

`src/putpocket_dataset_mining/externals.py` now uses `env.setdefault(...)`, so editable external builds also respect user overrides.

## Docker Image

- Image: `putpocket-default-python:ubuntu22.04-py313-v1`
- Dockerfile: `docker/default_python/Dockerfile`
- Base image: `ubuntu:22.04`
- Python: built from Python 3.13.1 source into `/opt/python/3.13`
- Includes: `bash`, `git`, `coreutils`, `ripgrep`, `tree`, `jq`, Python 3.13, `pytest`
- Runtime workspace root: `/workspace`
- Runtime network policy: existing `DockerWorkspace` and verifier containers use `--network none`
- Build behavior: bootstrap inspects the image first and only builds when missing or `--force-docker-build` is passed
- Current local image inspect passed:
  `sha256:a9a82a604441d3b4980f92940fa4bc8885c93df513e7e41e6e262917dd96d075`

## Externals

Current externals recorded in `logs/env_setup/latest/externals.lock`:

| External | Path | Branch | Commit | Remote |
| --- | --- | --- | --- | --- |
| vLLM | `externals/vllm` | `Putpocket-v0.19.1` | `b65d39ddbab966bb72110056a481d17e4726892b` | `https://github.com/Nier4Ryu/vllm_mod.git` |
| LMCache | `externals/lmcache` | `Putpocket-v0.4.4` | `72eb0e375bcf0739a45046433f46ee32be361656` | `https://github.com/Nier4Ryu/LMCache_mod.git` |
| Cline | `externals/cline` | `main` | `03f47045f338dcb6ac45b1ac1d6279a78be2b118` | `https://github.com/Nier4Ryu/cline_mod.git` |

vLLM and LMCache are editable build dependencies. Cline remains a read-only prompt/tool reference checkout.

## Commands Run

Inspection:

```bash
git status -sb --untracked-files=normal
sed -n '1,240p' scripts/env/bootstrap_env.sh
sed -n '1,220p' scripts/env/env_activate.sh
sed -n '1,260p' src/putpocket_dataset_mining/externals.py
sed -n '1,220p' src/putpocket_dataset_mining/doctor.py
sed -n '1,260p' src/putpocket_dataset_mining/cli.py
sed -n '1,240p' src/putpocket_dataset_mining/constants.py
sed -n '1,220p' pyproject.toml
sed -n '1,120p' .gitignore
find scripts -maxdepth 3 -type f | sort
rg -n "class DockerImageManager|ensure_image|DEFAULT_DOCKER_IMAGE|docker build|network" src/putpocket_dataset_mining docker scripts Makefile README.md docs -S
rg --files src/putpocket_dataset_mining | sort
sed -n '1,260p' src/putpocket_dataset_mining/docker_workspace.py
sed -n '1,220p' scripts/externals/checkout_externals.sh
find docs -maxdepth 2 -type f | sort
rg -n "gpu_slots|allowed_cuda|CUDA_VISIBLE_DEVICES|full_server|worker_profiles|tp|tensor_parallel|ALLOWED_CUDA_DEVICES" configs src tests docs -S
sed -n '1,260p' configs/dataset_mining/mbpp_stateful_multi.yaml
find configs -maxdepth 3 -type f | sort
git diff --stat
git diff -- scripts/env/bootstrap_env.sh scripts/env/env_activate.sh src/putpocket_dataset_mining/externals.py src/putpocket_dataset_mining/doctor.py src/putpocket_dataset_mining/docker_workspace.py docker/default_python/Dockerfile .gitignore scripts/env/README.md
```

Validation:

```bash
ls -l scripts/env/bootstrap_env.sh scripts/env/env_activate.sh
bash -n scripts/env/bootstrap_env.sh scripts/env/env_activate.sh
python -m compileall -q src/putpocket_dataset_mining/externals.py src/putpocket_dataset_mining/doctor.py src/putpocket_dataset_mining/docker_workspace.py
chmod +x scripts/env/bootstrap_env.sh scripts/env/env_activate.sh
Putpocket_env/bin/python -m compileall -q src/putpocket_dataset_mining/externals.py src/putpocket_dataset_mining/doctor.py src/putpocket_dataset_mining/docker_workspace.py
./scripts/env/bootstrap_env.sh --help
scripts/env/env_activate.sh
bash -lc 'source scripts/env/env_activate.sh; python - <<'"'"'PY'"'"'
import sys
print(sys.executable)
assert "Putpocket_env" in sys.executable
PY'
bash -lc 'source scripts/env/env_activate.sh; python - <<'"'"'PY'"'"'
import torch, ray, datasets, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("ray", ray.__version__)
print("datasets/transformers ok")
PY'
bash -lc 'source scripts/env/env_activate.sh; python - <<'"'"'PY'"'"'
import vllm, lmcache, putpocket_dataset_mining
print("vllm/lmcache/package import ok")
PY'
./scripts/env/bootstrap_env.sh --doctor-only --skip-docker
readlink -f logs/env_setup/latest
sed -n '1,260p' logs/env_setup/latest/doctor-smoke_.log
sed -n '1,120p' logs/env_setup/latest/record-externals_.log
sed -n '1,160p' logs/env_setup/latest/gpu-report_.log
sed -n '1,120p' logs/env_setup/latest/summary.txt
find logs/env_setup/latest -maxdepth 1 -type f -printf '%f\n' | sort
docker image inspect putpocket-default-python:ubuntu22.04-py313-v1
bash -lc 'source scripts/env/env_activate.sh; putpocket-dataset-mining doctor --json'
head -n 1 Putpocket_env/bin/putpocket-dataset-mining
bash -lc 'source scripts/env/env_activate.sh; which python; which python3; python -V; python3 -V; echo "$PATH"'
Putpocket_env/bin/python -c 'import sys; print(sys.executable)'
./scripts/env/bootstrap_env.sh --doctor-only
sed -n '1,260p' logs/env_setup/latest/doctor-smoke.log
sed -n '1,120p' logs/env_setup/latest/activate-env.log
sed -n '1,120p' logs/env_setup/latest/record-externals.log
docker image inspect putpocket-default-python:ubuntu22.04-py313-v1 --format '{{.Id}} {{.Created}} {{.Size}}'
bash -lc 'unset CUDA_VISIBLE_DEVICES; export PUTPOCKET_BUILD_THREADS=16; source scripts/env/env_activate.sh; echo "PUTPOCKET_BUILD_THREADS=${PUTPOCKET_BUILD_THREADS}"; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-unset}"'
```

One early `python -m compileall ...` command failed because bare `python` is not on the non-activated shell PATH; the same compileall check passed with `Putpocket_env/bin/python`. One early Docker inspect failed before the permission mode changed because the sandbox could not access `/var/run/docker.sock`; Docker inspect later passed.

## Validation Results

| Check | Result | Notes |
| --- | --- | --- |
| Script executable bits | Pass | `bootstrap_env.sh` and `env_activate.sh` are executable. |
| Bash syntax | Pass | `bash -n scripts/env/bootstrap_env.sh scripts/env/env_activate.sh`. |
| Help output | Pass | `./scripts/env/bootstrap_env.sh --help`. |
| Direct activation execution guard | Pass | Direct execution exits with source-only message. |
| Sourced activation | Pass | Python resolves to `/home/dyryu/putpocket_dataset_mining/Putpocket_env/bin/python`. |
| User build override precedence | Pass | `PUTPOCKET_BUILD_THREADS=16` remained 16 after activation. |
| `CUDA_VISIBLE_DEVICES` safety | Pass | Activation did not set it. |
| Python version | Pass | Python 3.13.13. |
| torch/Ray/import smoke | Pass | torch 2.10.0+cu128, CUDA 12.8, CUDA available true, 8 GPUs visible; Ray 2.55.1. |
| datasets/transformers import | Pass | datasets 5.0.0, transformers 5.12.1. |
| vLLM import | Pass | `vllm import ok`. |
| LMCache import | Pass | `lmcache import ok`, backend `lmcache.c_ops`. |
| Repo package import | Pass | `putpocket_dataset_mining import ok`. |
| CLI doctor | Pass | Module and required path checks passed. |
| compileall | Pass | `src` and `tests` compiled in bootstrap smoke. |
| unittest discovery | Pass | 13 tests passed. |
| Docker image inspect | Pass | `putpocket-default-python:ubuntu22.04-py313-v1` exists. |
| Full mining/evaluation jobs | Not run | Intentionally out of scope for env bootstrap. |

## Logs

Latest successful bootstrap validation:

```text
/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T072357Z
/home/dyryu/putpocket_dataset_mining/logs/env_setup/latest
```

Key logs:

```text
logs/env_setup/latest/gpu-report.log
logs/env_setup/latest/activate-env.log
logs/env_setup/latest/record-externals.log
logs/env_setup/latest/externals.lock
logs/env_setup/latest/doctor-smoke.log
logs/env_setup/latest/summary.txt
```

No new Codex run log was identified or needed for this env bootstrap goal.

## Blackwell Fresh Setup Instructions

On the new Blackwell server:

```bash
git clone https://github.com/Nier4Ryu/putpocket_dataset_mining.git
cd putpocket_dataset_mining
```

If Python 3.13 is already installed:

```bash
./scripts/env/bootstrap_env.sh
source scripts/env/env_activate.sh
```

If Python 3.13 is missing but `uv` is installed, the bootstrap script will try `uv python find/install 3.13` automatically.

If neither Python 3.13 nor `uv` is available, install one of them, then rerun:

```bash
./scripts/env/bootstrap_env.sh
```

Optional Blackwell/server-specific overrides before bootstrap:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PUTPOCKET_BUILD_THREADS=32
export MAX_JOBS=32
export CMAKE_BUILD_PARALLEL_LEVEL=32
export CARGO_BUILD_JOBS=32
export NVCC_THREADS=1
./scripts/env/bootstrap_env.sh
```

After setup, each shell session only needs:

```bash
cd putpocket_dataset_mining
source scripts/env/env_activate.sh
```

GPU runtime slots are not controlled by activation. Current mining runtime slots live in `configs/dataset_mining/mbpp_stateful_multi.yaml` under:

```yaml
gpu:
  allowed_cuda_devices: [4, 5, 6, 7]
  debug_slots:
    - [4]
  first_parallel_slots:
    - [4]
    - [5]
  full_server_slots:
    - [4]
    - [5]
    - [6]
    - [7]
```

The code-level guard is `ALLOWED_CUDA_DEVICES = (4, 5, 6, 7)` in `src/putpocket_dataset_mining/constants.py`, with runtime validation in `src/putpocket_dataset_mining/multi.py` and model-evaluation GPU validation in `src/putpocket_dataset_mining/model_evaluation/glm_eval.py`. Adapt those runtime configs/code guards for a different Blackwell GPU layout; do not put GPU IDs in `env_activate.sh`.

## Known Blockers / Remaining Work

No blocker remains for the implemented env bootstrap/activation flow on this server.

Remaining Blackwell-specific checks to do on the actual Blackwell host:

- Run the full first-time command: `./scripts/env/bootstrap_env.sh`
- Confirm PyTorch 2.10.0+cu128 supports that host/driver stack.
- Confirm the Putpocket vLLM branch builds and loads on Blackwell.
- Confirm LMCache builds with CUDA ops on Blackwell.
- Update runtime GPU slot config/code guard if Blackwell GPU IDs differ from `[4,5,6,7]`.
- Run a lightweight GLM/vLLM smoke later, not as part of env bootstrap:
  `source scripts/env/env_activate.sh` then the repo's GLM evaluation smoke command once model download/GPU allocation is approved.

The earlier sandbox-only Docker permission failure is no longer active; `docker image inspect putpocket-default-python:ubuntu22.04-py313-v1` passed after permissions changed.

## Safety Notes

- `scripts/env/env_activate.sh` is source-only and does not install, build, clone, checkout, run Docker, run mining, or run evaluation.
- `scripts/env/env_activate.sh` does not hardcode or export `CUDA_VISIBLE_DEVICES`.
- Build parallelism defaults to 32 jobs and can be lowered by the user before bootstrap or activation.
- `scripts/env/bootstrap_env.sh` is idempotent by design: it reuses `Putpocket_env`, verifies/fetches existing externals, skips external editable installs when imports already work unless forced, and skips Docker rebuild when the image already exists unless forced.
- Mined dataset data, `codex_runs`, and existing artifact directories were not deleted or modified.
