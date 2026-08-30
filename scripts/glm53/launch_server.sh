#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_ID="T20260830-001__glm53-montblanc-deployment"
RUNTIME_ROOT="${GLM53_RUNTIME_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/putpocket-runtime/${TASK_ID}}"
MODEL_DIR="${GLM53_MODEL_DIR:-${XDG_CACHE_HOME:-${HOME}/.cache}/putpocket-models/${TASK_ID}/RedHatAI--GLM-5.3-Flash-NVFP4--36c184c6}"
IMAGE="${GLM53_IMAGE:-putpocket/glm53-flash-sm120:878631b-c2eec1}"
PORT="${GLM53_PORT:-8137}"
RUN_ID="${GLM53_RUN_ID:-glm53-smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_DIR="${GLM53_RUN_DIR:-${RUNTIME_ROOT}/runs/${RUN_ID}}"
LOCK_PATH="${REPO_ROOT}/configs/models/glm53_flash_nvfp4_montblanc.lock.json"
CONTAINER_NAME="putpocket-${TASK_ID%%__*}-glm53-${RUN_ID//[^a-zA-Z0-9_.-]/-}"

if [[ "$(git -C "${REPO_ROOT}" branch --show-current)" != "agent/${TASK_ID}" ]]; then
  echo "Refusing launch outside agent/${TASK_ID}" >&2
  exit 2
fi
if [[ "${PUTPOCKET_ALLOW_TASK_PRODUCTION:-0}" != "1" ]]; then
  echo "Set PUTPOCKET_ALLOW_TASK_PRODUCTION=1 for this user-authorized task server." >&2
  exit 2
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to overwrite existing run directory ${RUN_DIR}" >&2
  exit 2
fi
if ss -ltnH "sport = :${PORT}" | grep -q .; then
  echo "Refusing to use occupied TCP port ${PORT}" >&2
  exit 2
fi

mkdir -p "${RUN_DIR}" "${RUNTIME_ROOT}/flashinfer-workspace"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m putpocket_dataset_mining.glm53_deployment --lock "${LOCK_PATH}" \
  host-doctor --model-dir "${MODEL_DIR}" --output "${RUN_DIR}/host_doctor.json"
python3 -m putpocket_dataset_mining.glm53_deployment --lock "${LOCK_PATH}" \
  verify-model --sizes-only --model-dir "${MODEL_DIR}" \
  --output "${RUN_DIR}/model_sizes_and_metadata.json"
python3 -m putpocket_dataset_mining.glm53_deployment --lock "${LOCK_PATH}" \
  inspect-image --image "${IMAGE}" --output "${RUN_DIR}/runtime_image.json"

EXPECTED_DRIVER_VERSION="$(python3 - "${LOCK_PATH}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["runtime"]["nvidia_driver_version"])
PY
)"
HOST_DRIVER_VERSIONS="$(
  nvidia-smi --query-gpu=driver_version --format=csv,noheader |
    sed '/^[[:space:]]*$/d' | sort -u
)"
if [[ "${HOST_DRIVER_VERSIONS}" != "${EXPECTED_DRIVER_VERSION}" ]]; then
  echo "Host NVIDIA driver does not match the locked version ${EXPECTED_DRIVER_VERSION}." >&2
  exit 2
fi

GPU_PASSTHROUGH_ARGS=()
for device in \
  /dev/nvidia0 /dev/nvidia1 /dev/nvidia2 /dev/nvidiactl \
  /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
  if [[ ! -c "${device}" ]]; then
    echo "Required NVIDIA character device is absent: ${device}" >&2
    exit 2
  fi
  GPU_PASSTHROUGH_ARGS+=(--device "${device}")
done

DRIVER_LIB_SONAMES=(
  libcuda.so.1
  libnvidia-ml.so.1
  libnvidia-ptxjitcompiler.so.1
  libnvidia-nvvm.so.4
  "libnvidia-gpucomp.so.${EXPECTED_DRIVER_VERSION}"
)
for soname in "${DRIVER_LIB_SONAMES[@]}"; do
  host_path="$(
    /sbin/ldconfig -p |
      awk -v expected="${soname}" '
        $1 == expected && first == "" {first = $NF}
        END {if (first != "") print first}
      '
  )"
  host_path="$(readlink -f "${host_path}")"
  if [[ -z "${host_path}" || ! -f "${host_path}" ]]; then
    echo "Required NVIDIA driver library is absent: ${soname}" >&2
    exit 2
  fi
  GPU_PASSTHROUGH_ARGS+=(
    --volume "${host_path}:/usr/local/nvidia/lib64/${soname}:ro"
  )
done

cat > "${RUN_DIR}/launch_args.json" <<JSON
{
  "run_id": "${RUN_ID}",
  "launch_requested_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "image": "${IMAGE}",
  "served_model_name": "glm-5.3-flash-nvfp4",
  "port": ${PORT},
  "parallelism": {"tp": 1, "dp": 3, "ep": 3, "pp": 1},
  "max_model_len": 4096,
  "max_num_seqs": 1,
  "gpu_memory_utilization": 0.94,
  "kv_cache_dtype": "fp8_ds_mla",
  "block_size": 512,
  "attention_backend": "FLASHINFER_MLA_SPARSE_SM120",
  "moe_backend": "marlin",
  "container_gpu_passthrough": "explicit_devices_and_driver_libs",
  "nvidia_driver_version": "${EXPECTED_DRIVER_VERSION}",
  "mtp": false,
  "prefix_caching": false
}
JSON

docker run --detach --rm \
  --name "${CONTAINER_NAME}" \
  --cidfile "${RUN_DIR}/container.cid" \
  --label "putpocket.task_id=${TASK_ID}" \
  --label "putpocket.run_id=${RUN_ID}" \
  "${GPU_PASSTHROUGH_ARGS[@]}" \
  --shm-size 24g \
  --publish "127.0.0.1:${PORT}:8000" \
  --env CUDA_VISIBLE_DEVICES=0,1,2 \
  --env LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/local/cuda/lib64 \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env FLASHINFER_DISABLE_VERSION_CHECK=1 \
  --env FLASHINFER_WORKSPACE_BASE=/runtime/flashinfer \
  --env VLLM_LOGGING_LEVEL=INFO \
  --volume "${MODEL_DIR}:/model:ro" \
  --volume "${RUNTIME_ROOT}/flashinfer-workspace:/runtime/flashinfer:rw" \
  "${IMAGE}" \
  /model \
  --served-model-name glm-5.3-flash-nvfp4 \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --quantization compressed-tensors \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 3 \
  --data-parallel-backend mp \
  --enable-expert-parallel \
  --enable-ep-weight-filter \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 512 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.94 \
  --enforce-eager \
  --no-enable-prefix-caching \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_SM120"}' \
  --kernel-config '{"moe_backend":"marlin","linear_backend":"auto","enable_flashinfer_autotune":false}' \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  > "${RUN_DIR}/docker_run_stdout.txt"

docker inspect "$(cat "${RUN_DIR}/container.cid")" > "${RUN_DIR}/container_start_inspect.json"
printf '%s\n' "${RUN_DIR}"
