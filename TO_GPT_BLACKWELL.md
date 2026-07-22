# Blackwell Env + GLM Evaluation Report

## Executive Summary
- Branch `blackwell` was created from `09906fc63f71bfc950a33b506b0c9441e25ae6df` and pushed to `origin/blackwell` before source changes.
- Environment setup code was updated for Blackwell defaults: GPUs `0,1,2`, CUDA `/usr/local/cuda-12.9`, Python 3.13, PyTorch `2.10.0+cu129`, and build cap `16`.
- A real bootstrap run built the environment, installed torch/Ray/repo deps, checked out externals, built vLLM editable with retry fallback, and built LMCache editable. It failed only at Docker image setup because the `docker` command is missing.
- vLLM build retry worked as required: `16` failed OOM-like, `12` failed OOM-like, `8` passed.
- A skip-Docker idempotent bootstrap pass completed successfully and validated torch CUDA, vLLM, LMCache, repo import, doctor, and tests.
- GLM smoke was attempted on GPU `0`; it failed before model load/sample execution because Docker is missing. Full GLM evaluation was not attempted because smoke did not pass.

## Branch / Git
- Branch name: `blackwell`
- Upstream: `origin/blackwell`
- Base commit before Blackwell changes: `09906fc63f71bfc950a33b506b0c9441e25ae6df`
- Initial branch push: completed before implementation.
- Source/config/report commit: `e043f774fdf6385baacd4c1badfdfb93c6067a2b`
- Pushed status for source/config/report commit: pushed to `origin/blackwell`.
- Report metadata update commit: this report line is part of a follow-up commit; final branch HEAD is reported in the assistant handoff after push.

## Hardware Detected
- GPUs:
  - `0`: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability `12.0`, `97887 MiB`, driver `580.159.03`
  - `1`: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability `12.0`, `97887 MiB`, driver `580.159.03`
  - `2`: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability `12.0`, `97887 MiB`, driver `580.159.03`
- CUDA path: `/usr/local/cuda-12.9`
- CUDA version: `12.9`, `nvcc` release `12.9`, `V12.9.41`
- CPU cores: `64`
- CPU RAM: `62Gi`
- Preflight notes:
  - `nvcc` was not initially on PATH, but `/usr/local/cuda-12.9/bin/nvcc` exists and env activation exports it.
  - `python3.13` was not initially on PATH; bootstrap installed/found Python 3.13 through repo-local uv tooling.
  - `docker` is missing.

## Env Setup Changes
- Modified environment/bootstrap files:
  - `scripts/env/bootstrap_env.sh`
  - `scripts/env/env_activate.sh`
  - `scripts/env/env_activate_ref.sh`
  - `scripts/env/README.md`
- Modified runtime/config/source defaults:
  - `configs/dataset_mining/mbpp_stateful_multi.yaml`
  - `configs/model_evaluation/glm52_08b_blackwell.yaml`
  - `src/putpocket_dataset_mining/constants.py`
  - `src/putpocket_dataset_mining/multi.py`
  - `src/putpocket_dataset_mining/model_evaluation/glm_eval.py`
  - `src/putpocket_dataset_mining/externals.py`
  - `src/putpocket_dataset_mining/docker_workspace.py`
  - `src/putpocket_dataset_mining/cli.py`
- Python version: `Python 3.13.14`
- CUDA path: `/usr/local/cuda-12.9`
- Torch: `2.10.0+cu129`
- `torch.version.cuda`: `12.9`
- `torch.cuda.is_available()`: `True`
- Ray: `2.55.1`
- vLLM: `0.1.dev15375+gb65d39ddb`
- LMCache: `0.1.dev1451`, imported with `lmcache.c_ops`
- Build cap defaults confirmed:
  - `PUTPOCKET_BUILD_THREADS=16`
  - `MAX_JOBS=16`
  - `CMAKE_BUILD_PARALLEL_LEVEL=16`
  - `CARGO_BUILD_JOBS=16`
  - `NVCC_THREADS=1`
- PyTorch/CUDA decision: installed official PyTorch CUDA 12.9 wheel `torch==2.10.0+cu129` from `https://download.pytorch.org/whl/cu129`; CPU torch was not accepted.

## vLLM Build Retry Summary
- Retry log root: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z`
- Attempted thread counts:
  - `16`: fail, OOM-like
    - Log: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/vllm_build_threads_16.log`
  - `12`: fail, OOM-like
    - Log: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/vllm_build_threads_12.log`
  - `8`: pass
    - Log: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/vllm_build_threads_8.log`
- Final successful thread count: `8`
- Summary file: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/vllm_build_retry_summary.tsv`
- No remaining vLLM build blocker after the 8-thread retry passed.

## Bootstrap Usage
- First-time command:

```bash
./scripts/env/bootstrap_env.sh
```

- This real run failed at Docker only:
  - Failed stage: `ensure-docker-image`
  - Failing command: `ensure_docker_image`
  - Log path: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/ensure-docker-image.log`
  - Error: `docker command is missing. Install Docker or rerun with --skip-docker.`
- Idempotent validation command after Docker blocker:

```bash
./scripts/env/bootstrap_env.sh --skip-docker --skip-vllm-build --skip-lmcache-build
```

- Successful skip-Docker log root: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T122119Z`
- Activation command:

```bash
source scripts/env/env_activate.sh
```

## Validation Commands Run
- `bash -n scripts/env/bootstrap_env.sh scripts/env/env_activate.sh scripts/env/env_activate_ref.sh`: pass
- `./scripts/env/bootstrap_env.sh`: failed only at Docker image setup after env/vLLM/LMCache succeeded
- `./scripts/env/bootstrap_env.sh --skip-docker --skip-vllm-build --skip-lmcache-build`: pass
- `source scripts/env/env_activate.sh && python -V`: pass, `Python 3.13.14`
- `source scripts/env/env_activate.sh && python` executable assertion contains `Putpocket_env`: pass
- torch/Ray/datasets/transformers import check: pass
- repo package import check: pass
- vLLM/LMCache import check: pass
- `putpocket-dataset-mining doctor --json`: pass, with `docker: null`
- `python -m compileall src tests`: pass
- `python -m unittest discover -s tests -v`: pass, `13` tests
- `putpocket-dataset-mining docker ensure-image`: fail, exit `2`, `docker command is missing. Install Docker or rerun with --skip-docker where supported.`

## Dataset Found
- Dataset version: `mbpp_stateful_working_v0`
- Accepted count: `20`
- Accepted path: `/home/dyryu/putpocket_dataset_mining/data/dataset_mining/datasets/mbpp_stateful_working_v0/accepted.jsonl`
- Artifact paths resolve: yes, `0` missing paths across accepted rows.

## GLM Evaluation Code Status
- GLM evaluation code exists and was updated for Blackwell defaults.
- Main CLI module: `src/putpocket_dataset_mining/model_evaluation/glm_eval.py`
- Added config: `configs/model_evaluation/glm52_08b_blackwell.yaml`
- CLI help command passed:

```bash
source scripts/env/env_activate.sh
python -m putpocket_dataset_mining.model_evaluation.glm_eval --help
```

- Smoke command used:

```bash
CUDA_VISIBLE_DEVICES=0 python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --profile smoke \
  --workers 1 \
  --gpu-slots 0 \
  --run-id eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_blackwell_20260722T122033Z
```

## GLM Smoke Result
- Status: failed before model load/sample execution due missing Docker.
- Output path: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/runs/eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_blackwell_20260722T122033Z`
- Log path: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/logs/glm_smoke_blackwell_20260722T122033Z.log`
- Files produced:
  - `eval_config.yaml`
  - `dataset_audit.json`
  - empty `results.jsonl`
- Final error:

```json
{
  "error": "InfraError: docker command is missing. Install Docker or rerun with --skip-docker where supported.",
  "status": "failed"
}
```

## GLM Full Evaluation Result
- Status: not attempted because smoke failed before model load/sample execution.
- Pending command after Docker is installed and smoke passes:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --profile full \
  --workers 3 \
  --gpu-slots 0,1,2
```

- Number of samples planned: `20`
- accepted/rejected/failed_infra/uncertain: unavailable for this run because full evaluation did not start.
- Failure stage histogram: unavailable for this run because smoke stopped before sample execution.

## Logs
- Preflight log dir: `/home/dyryu/putpocket_dataset_mining/logs/blackwell_preflight`
- Real bootstrap log dir: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z`
- Successful skip-Docker bootstrap log dir: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T122119Z`
- vLLM retry summary: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/vllm_build_retry_summary.tsv`
- Docker failure log: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/ensure-docker-image.log`
- GLM smoke log: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/logs/glm_smoke_blackwell_20260722T122033Z.log`
- GLM smoke run path: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/runs/eval_glm52_08b_on_mbpp_stateful_working_v0_smoke_blackwell_20260722T122033Z`

## Known Blockers
- Docker is missing on the host.
  - Failing bootstrap command: `./scripts/env/bootstrap_env.sh`
  - Failed stage: `ensure-docker-image`
  - Failing stage command: `ensure_docker_image`
  - Log path: `/home/dyryu/putpocket_dataset_mining/logs/env_setup/20260722T105947Z/ensure-docker-image.log`
  - Error: `docker command is missing. Install Docker or rerun with --skip-docker.`
- Docker image validation also fails:
  - Failing command: `source scripts/env/env_activate.sh && putpocket-dataset-mining docker ensure-image`
  - Exit code: `2`
  - Error: `docker command is missing. Install Docker or rerun with --skip-docker where supported.`
- GLM smoke is blocked by the same missing Docker dependency:
  - Failing command: see `## GLM Smoke Result`
  - Log path: `/home/dyryu/putpocket_dataset_mining/data/model_evaluation/logs/glm_smoke_blackwell_20260722T122033Z.log`
- Docker install continuation attempt on 2026-07-23:
  - Checked branch/state: `blackwell` at `e827fc575afcfe91e12e1909cc91c108cc12d5d0`, clean before this report update.
  - System install check: `sudo -n true` failed with `sudo: a password is required`.
  - Host OS: Ubuntu 22.04.3 LTS.
  - Rootless prerequisites present: `/etc/subuid` and `/etc/subgid` each contain `dyryu:558752:65536`; unprivileged user namespaces are enabled.
  - Rootless prerequisites missing: `newuidmap` and `newgidmap`.
  - Official rootless installer command attempted:

```bash
curl -fsSL https://get.docker.com/rootless -o /tmp/docker-rootless-install.sh
sh /tmp/docker-rootless-install.sh
```

  - Rootless installer log: `/home/dyryu/putpocket_dataset_mining/logs/docker_install/rootless_install_20260722T172904Z.log`
  - Rootless installer result: blocked by missing host `uidmap` package; installer requested:

```bash
sudo apt-get -y install uidmap
```

  - Exact package-install command attempted:

```bash
sudo -n apt-get -y install uidmap
```

  - Exact package-install failure: `sudo: a password is required`
- Smallest next action: an administrator must install the host `uidmap` package, or provide passwordless/interactive sudo for user `dyryu` long enough to run `sudo apt-get -y install uidmap`. Then rerun the rootless installer and Docker validation:

```bash
sh /tmp/docker-rootless-install.sh
export PATH="$HOME/bin:$PATH"
export DOCKER_HOST="unix:///run/user/1007/docker.sock"
source scripts/env/env_activate.sh
putpocket-dataset-mining docker ensure-image
CUDA_VISIBLE_DEVICES=0 python -m putpocket_dataset_mining.model_evaluation.glm_eval \
  --dataset-version mbpp_stateful_working_v0 \
  --model-id inference-optimization/GLM-5.2-0.8B-A0.8B \
  --eval-name eval_glm52_08b_on_mbpp_stateful_working_v0 \
  --profile smoke \
  --workers 1 \
  --gpu-slots 0
```

## Next Recommended Action
Install host `uidmap` for user `dyryu` so rootless Docker can be installed without rootful daemon access, or install/enable a rootful Docker daemon for this user. Then rerun Docker image setup and the GLM smoke command. If smoke passes, run the pending full evaluation with GPUs `0,1,2`.
