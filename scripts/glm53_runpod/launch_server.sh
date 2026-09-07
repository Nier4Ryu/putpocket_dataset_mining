#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/putpocket
LOCK="${ROOT}/configs/models/glm53_flash_w4a16_runpod_sm90_sm120.lock.json"
PROFILE="${PUTPOCKET_GLM53_RUNTIME_PROFILE:-sm90}"
MODEL_PATH="${MODEL_PATH:-/models/glm53-flash-w4a16-autoround}"
MODEL_ID="${MODEL_ID:-Intel/GLM-5.3-Flash-W4A16-AutoRound}"
MODEL_REVISION="${MODEL_REVISION:-5eee1846f0321058ed73745f9aa16f2aaf0fc0a0}"
TP="${TENSOR_PARALLEL_SIZE:-4}"
PP="${PIPELINE_PARALLEL_SIZE:-1}"
PORT="${VLLM_PORT:-8000}"
MAX_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY="${GPU_MEMORY_UTILIZATION:-0.92}"

case "${PROFILE}" in
  sm90)
    BACKEND=FLASHINFER_MLA_SPARSE_SM90
    KV_DTYPE=fp8_e4m3
    BLOCK_SIZE=128
    ;;
  sm120)
    BACKEND=FLASHINFER_MLA_SPARSE_SM120
    KV_DTYPE=fp8_ds_mla
    BLOCK_SIZE=512
    ;;
  *)
    echo "unsupported PUTPOCKET_GLM53_RUNTIME_PROFILE=${PROFILE}" >&2
    exit 2
    ;;
esac

[[ "${MODEL_ID}" == "Intel/GLM-5.3-Flash-W4A16-AutoRound" ]] || { echo "MODEL_ID mismatch" >&2; exit 2; }
[[ "${MODEL_REVISION}" == "5eee1846f0321058ed73745f9aa16f2aaf0fc0a0" ]] || { echo "MODEL_REVISION mismatch" >&2; exit 2; }
[[ "${TP}" == "4" && "${PP}" == "1" ]] || { echo "validated topology is TP=4 PP=1 only" >&2; exit 2; }
[[ -d "${MODEL_PATH}" ]] || { echo "MODEL_PATH must be a pre-downloaded read-only volume: ${MODEL_PATH}" >&2; exit 2; }
[[ -d /results && -w /results ]] || { echo "/results must be a writable external volume" >&2; exit 2; }
[[ -z "${PUTPOCKET_GLM53_STATEFUL_EDIT_CONTROL:-}" || -f "${PUTPOCKET_GLM53_STATEFUL_EDIT_CONTROL}" ]] || { echo "stateful control path is not a file" >&2; exit 2; }

export PUTPOCKET_GLM53_RUNTIME_PROFILE="${PROFILE}"
export VLLM_ATTENTION_BACKEND="${BACKEND}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

python3 -m putpocket_dataset_mining.glm53_runpod_image runtime \
  --lock "${LOCK}" --root "${ROOT}" --profile "${PROFILE}" \
  --model-path "${MODEL_PATH}" --output "/results/runtime-doctor-${PROFILE}.json"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --tensor-parallel-size "${TP}" \
  --pipeline-parallel-size "${PP}" \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --quantization inc \
  --dtype bfloat16 \
  --kv-cache-dtype "${KV_DTYPE}" \
  --block-size "${BLOCK_SIZE}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --disable-frontend-multiprocessing \
  --no-enable-prefix-caching \
  --no-enable-chunked-prefill \
  --disable-log-requests \
  "$@"
