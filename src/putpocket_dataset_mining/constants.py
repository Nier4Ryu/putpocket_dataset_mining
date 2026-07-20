from __future__ import annotations

import os
from pathlib import Path

RANDOM_SEED = 42
MINING_SEED_DEFAULT = 42

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
REPO_ROOT = SRC_ROOT.parent

DATA_ROOT = REPO_ROOT / "data" / "dataset_mining"
RUNS_ROOT = DATA_ROOT / "runs"
DATASETS_ROOT = DATA_ROOT / "datasets"
CONTROL_ROOT = DATA_ROOT / "control"
INDEX_DB = DATA_ROOT / "mining_index.sqlite"

SHARED_HF_HUB_CACHE_DIR = Path(os.environ.get("PUTPOCKET_HF_HUB_CACHE_DIR", "/data/shared/hf_cache/hub"))
DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"

ALLOWED_CUDA_DEVICES = (4, 5, 6, 7)
DISALLOWED_CUDA_DEVICES = (0, 1, 2, 3)

VLLM_BUILD_MAX_CPU_THREADS = 32
BUILD_ENV_OVERRIDES = {
    "PUTPOCKET_BUILD_THREADS": "32",
    "MAX_JOBS": "32",
    "CMAKE_BUILD_PARALLEL_LEVEL": "32",
    "CARGO_BUILD_JOBS": "32",
    "NVCC_THREADS": "1",
}

DEFAULT_DOCKER_IMAGE = "putpocket-default-python:ubuntu22.04-py313-v1"
DEFAULT_DOCKERFILE = REPO_ROOT / "docker" / "default_python" / "Dockerfile"
DOCKER_WORKSPACE_ROOT = "/workspace"
CONTAINER_HOME = "/tmp/putpocket-home"

FINAL_STATUSES = {"accepted", "rejected", "failed_infra", "uncertain"}


def ensure_data_dirs() -> None:
    for path in (DATA_ROOT, RUNS_ROOT, DATASETS_ROOT, CONTROL_ROOT):
        path.mkdir(parents=True, exist_ok=True)
