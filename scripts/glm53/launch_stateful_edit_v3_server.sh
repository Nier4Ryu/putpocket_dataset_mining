#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_ID="T20260903-001__glm53-retarget-all"
RUNTIME_ROOT="${GLM53_RUNTIME_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/putpocket-runtime/${TASK_ID}}"
MODEL_DIR="${GLM53_MODEL_DIR:?Set GLM53_MODEL_DIR to the fully verified exact-revision model directory}"
IMAGE="${GLM53_IMAGE:-putpocket/glm53-flash-sm120-stateful-edit-v3:878631b}"
PORT="${GLM53_PORT:-8137}"
RUN_ID="${GLM53_RUN_ID:-glm53-stateful-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_DIR="${GLM53_RUN_DIR:-${RUNTIME_ROOT}/runs/${RUN_ID}}"
STATEFUL_SHARED_ROOT="${GLM53_STATEFUL_SHARED_ROOT:-${RUNTIME_ROOT}/stateful-control}"
STATEFUL_CONTROL_PATH="${STATEFUL_SHARED_ROOT}/control.json"
SERVER_MODE="${GLM53_STATEFUL_SERVER_MODE:-accuracy}"
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
docker image inspect "${IMAGE}" > "${RUN_DIR}/runtime_image.json"
stateful_base_id="$(docker image inspect "${IMAGE}" --format '{{index .Config.Labels "putpocket.glm53.deployment_base_image_id"}}')"
if [[ "${stateful_base_id}" != "sha256:b79bacf76a107fc9ddd67fcc84851e3f5ae222fe6fd6a2f187723297e1400b1e" ]]; then
  echo "Stateful image is not derived from the locked GLM-5.3 deployment image." >&2
  exit 2
fi
docker run --rm --entrypoint python3 "${IMAGE}" -c '
import hashlib, importlib, pathlib, vllm
root = pathlib.Path(vllm.__file__).resolve().parent
expected = {
    "model_executor/layers/attention/mla_attention.py": "ae008425588218b94628058988eadc0a39018ea92851dadb26a5ff9d2554d89a",
    "model_executor/layers/sparse_attn_indexer_kpool.py": "1accf858644f7ed08a9fa952d9b906f9a32e167a8c389ce0cfc4ecf0e54c9b5c",
    "models/glm5next/nvidia/model.py": "bce7d0fb2816715977b3b15d7c41adcbc3034b4aba094731996addefba9f8d0d",
    "model_executor/layers/glm53_stateful_edit_accuracy_ablation.py": "94cbc2d49f557236813156c439bb76c25f80f6b1537e15b44758958a9f1cd7f0",
}
for relative, digest in expected.items():
    assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest
module = importlib.import_module("vllm.model_executor.layers.glm53_stateful_edit_accuracy_ablation")
assert module.ATTENTION_BACKEND == "FLASHINFER_MLA_SPARSE_SM120"
assert module.KV_CACHE_DTYPE == "fp8_ds_mla"
assert module.accuracy_ablation_armed() is False
print("STATEFUL_IMAGE_CPU_DOCTOR_PASSED")
' > "${RUN_DIR}/stateful_image_cpu_doctor.txt"
if [[ "${SERVER_MODE}" == accuracy && -e "${STATEFUL_CONTROL_PATH}" ]]; then
  echo "Refusing stale stateful control: ${STATEFUL_CONTROL_PATH}" >&2
  exit 2
fi
mkdir -p "${STATEFUL_SHARED_ROOT}"
EXPERIMENT_ENV_ARGS=()
case "${SERVER_MODE}" in
  accuracy)
    EXPERIMENT_ENV_ARGS+=(--env "PUTPOCKET_GLM53_STATEFUL_EDIT_CONTROL=${STATEFUL_CONTROL_PATH}")
    ;;
  selector-capture)
    SELECTOR_CAPTURE_CONTROL="${GLM53_SELECTOR_CAPTURE_CONTROL:?Set the absolute selector capture control path}"
    case "${SELECTOR_CAPTURE_CONTROL}" in "${STATEFUL_SHARED_ROOT}"/*) ;; *)
      echo "Selector capture control must be inside GLM53_STATEFUL_SHARED_ROOT." >&2
      exit 2
    esac
    [[ -f "${SELECTOR_CAPTURE_CONTROL}" ]] || {
      echo "Selector capture control does not exist." >&2
      exit 2
    }
    EXPERIMENT_ENV_ARGS+=(--env "PUTPOCKET_GLM53_BASE_SELECTOR_CAPTURE=${SELECTOR_CAPTURE_CONTROL}")
    ;;
  *)
    echo "GLM53_STATEFUL_SERVER_MODE must be accuracy or selector-capture." >&2
    exit 2
    ;;
esac

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
  "prefix_caching": false,
  "chunked_prefill": false,
  "stateful_edit_mode": "default_off_full_target_prefill_then_donor_overwrite_accuracy_ablation",
  "server_mode": "${SERVER_MODE}",
  "true_partial_prefill": false
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
  "${EXPERIMENT_ENV_ARGS[@]}" \
  --volume "${STATEFUL_SHARED_ROOT}:${STATEFUL_SHARED_ROOT}:rw" \
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
  --no-enable-chunked-prefill \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_SM120"}' \
  --kernel-config '{"moe_backend":"marlin","linear_backend":"auto","enable_flashinfer_autotune":false}' \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  > "${RUN_DIR}/docker_run_stdout.txt"

docker inspect "$(cat "${RUN_DIR}/container.cid")" > "${RUN_DIR}/container_start_inspect.json"
printf '%s\n' "${RUN_DIR}"
