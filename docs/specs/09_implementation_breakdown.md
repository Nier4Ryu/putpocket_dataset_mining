# Implementation Breakdown

Implement in phases. Do not skip to a fake completed state.

## Phase 0 — Repo substrate

- pyproject
- package root `src/putpocket_dataset_mining`
- constants module with `RANDOM_SEED=42` and shared HF cache path
- `scripts/env/env_activate.sh`
- basic doctor command

## Phase 1 — Env and externals

- external checkout helpers for vLLM, LMCache, Cline
- editable install path
- vLLM build must cap CPU build threads at 16 via MAX_JOBS/CMAKE_BUILD_PARALLEL_LEVEL/CARGO_BUILD_JOBS/NVCC_THREADS
- import smoke

## Phase 2 — Docker default Python image

- `docker/default_python/Dockerfile`
- Ubuntu 22.04 + Python 3.13 + pytest/git/bash/coreutils/ripgrep/tree/jq
- build-if-missing image manager
- host-mounted workspace permissions with host UID/GID

## Phase 3 — Dataset source adapter

- HuggingFace MBPP loader
- field mapping/normalization
- source task schema
- MBPP test materializer for hidden verifier

## Phase 4 — Prompt rendering

- Cline compact/full prompt profiles
- static Cline rules v1/v2 artifacts
- query1 wrapper
- query2 candidate generator
- explicit tokenizer chat template rendering
- rendered prompt and tokenization metadata saving

## Phase 5 — Headless Cline runtime

- original Cline tool parser ported from `externals/cline`
- Docker tool backend
- multi-step history rollout
- attempt_completion finish handling
- parse failure retry policy
- trajectory/timeline writer

## Phase 6 — Serving connector

- local vLLM Python engine wrapper
- worker-lifetime engine reuse support
- greedy sampling default
- request/response logging

## Phase 7 — Verifier and judge

- fresh verifier container from workspace snapshot
- hidden test injection
- checklist writer
- Codex CLI judge backend read-only approval-never

## Phase 8 — Single-sample runner

- full single-sample attempt orchestration
- final status labels
- required artifacts
- SQLite update interface to master

## Phase 9 — Multi-sample master-worker

- master process
- GPU slot preflight restricted to CUDA devices 0,1,2
- worker pool
- lazy job allocation
- stop interfaces
- master-only DB writer
- incremental materialization

## Phase 10 — Dataset structure docs and acceptance

- `docs/DATASET_STRUCTURE.md`
- debug profile run
- first_parallel profile run
- full_server profile definition
