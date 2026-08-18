#!/usr/bin/env bash
set -euo pipefail
umask 077

PROJECT_COMMIT=${1:-}
SOURCE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
STORAGE_ROOT=/local-data/jslee202403/putpocket-glm52-sglang-gate
RUN_ROOT="$STORAGE_ROOT/artifacts/${SLURM_JOB_ID:-unknown}"
CACHE_ROOT="$STORAGE_ROOT/cache"
METADATA_ROOT="$CACHE_ROOT/model-metadata/${SLURM_JOB_ID:-unknown}"
BOOTSTRAP_PYTHON=/home2/jslee202403/miniconda3/bin/python3
GIT=/usr/bin/git
NVIDIA_SMI=/usr/bin/nvidia-smi
IMAGE_HUMAN_TAG=lmsysorg/sglang:latest
IMAGE_DIGEST=sha256:3be8803490a8b899a44f7ab2e22d8f6a1fb877cab52faeb400769a1555317db4
IMAGE="lmsysorg/sglang@$IMAGE_DIGEST"
MODEL_ID=nvidia/GLM-5.2-NVFP4
MODEL_REF=main
PORT=30000
SERVER_CONTAINER="pp-glm52-gate-${SLURM_JOB_ID:-unknown}"
SERVER_PID=
SAMPLER_PID=
PHASE=allocation_inventory
FAILURE_CLASS=UNCLASSIFIED_FAILURE
JOB_STATUS=FAIL

write_failure_manifest() {
  rc=$1
  mkdir -p "$RUN_ROOT" 2>/dev/null || true
  printf '{"schema_version":1,"status":"FAIL","gate":"glm52_sglang_minimal_feasibility","phase":"%s","failure_class":"%s","returncode":%s,"job_id":"%s","project_commit":"%s","model_id":"nvidia/GLM-5.2-NVFP4","fallback_attempted":false,"offload_attempted":false}\n' \
    "$PHASE" "$FAILURE_CLASS" "$rc" "${SLURM_JOB_ID:-unknown}" "${PROJECT_COMMIT:-unknown}" > "$RUN_ROOT/gate_manifest.json.partial" 2>/dev/null || true
  mv "$RUN_ROOT/gate_manifest.json.partial" "$RUN_ROOT/gate_manifest.json" 2>/dev/null || true
}

stop_children() {
  if [[ -n $SAMPLER_PID ]]; then
    kill "$SAMPLER_PID" >/dev/null 2>&1 || true
    wait "$SAMPLER_PID" >/dev/null 2>&1 || true
    SAMPLER_PID=
  fi
  if [[ -n $SERVER_PID ]]; then
    docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
    SERVER_PID=
  fi
}

finish() {
  rc=$?
  trap - EXIT INT TERM
  stop_children
  if [[ $JOB_STATUS != PASS ]]; then
    write_failure_manifest "$rc"
  fi
  exit "$rc"
}
trap finish EXIT INT TERM

fail() {
  FAILURE_CLASS=$1
  echo "$1" >&2
  exit "${2:-2}"
}

failure_class_from_log() {
  local log_path=$1
  local fallback=$2
  "$BOOTSTRAP_PYTHON" - "$log_path" "$fallback" <<'PY'
import json
import re
import sys

path, fallback = sys.argv[1:]
try:
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
except OSError:
    print(fallback)
    raise SystemExit(0)
for line in reversed(lines):
    try:
        error = json.loads(line).get("error", "")
    except (json.JSONDecodeError, AttributeError):
        continue
    failure = error.split(":", 1)[0]
    if re.fullmatch(r"[A-Z][A-Z0-9_]+", failure):
        print(failure)
        raise SystemExit(0)
print(fallback)
PY
}

[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail SLURM_ALLOCATION_REQUIRED 20
[[ ${SLURM_JOB_NUM_NODES:-0} == 1 && -n ${SLURM_JOB_NODELIST:-} ]] || fail SLURM_NODE_COUNT_MISMATCH 20
[[ ${SLURM_GPUS_ON_NODE:-0} == 4 ]] || fail SLURM_GPU_COUNT_MISMATCH 20
[[ $PROJECT_COMMIT =~ ^[0-9a-f]{40}$ ]] || fail PROJECT_COMMIT_INVALID 21
[[ -x $GIT && $($GIT -C "$SOURCE_ROOT" rev-parse HEAD) == "$PROJECT_COMMIT" ]] || fail PROJECT_COMMIT_MISMATCH 21
[[ -x $BOOTSTRAP_PYTHON ]] || fail COMPUTE_PYTHON_MISSING 21
[[ -x $NVIDIA_SMI ]] || fail NVIDIA_SMI_MISSING 21
[[ -d /local-data/jslee202403 ]] || fail COMPUTE_LOCAL_STORAGE_MISSING 21

mkdir -p "$RUN_ROOT/phase0" "$RUN_ROOT/phase1" "$RUN_ROOT/phase2" "$RUN_ROOT/phase3" "$CACHE_ROOT"
printf '%s\n' "$PROJECT_COMMIT" > "$RUN_ROOT/project_commit.txt"
export PYTHONPATH="$SOURCE_ROOT/src"
export HOME="$STORAGE_ROOT/home"
export HF_HOME="$CACHE_ROOT/huggingface"
export TRANSFORMERS_CACHE="$CACHE_ROOT/transformers"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TORCH_HOME="$CACHE_ROOT/torch"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TMPDIR="$STORAGE_ROOT/tmp"
export DOCKER_CONFIG="$STORAGE_ROOT/docker-config"
mkdir -p "$HOME" "$HF_HOME" "$TRANSFORMERS_CACHE" "$XDG_CACHE_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR" "$DOCKER_CONFIG"

# Initialize the site module command explicitly. The Login PATH exposed CUDA
# 13.1, while the tracked cluster lock requires the compute-side CUDA 12.9
# compiler module for any local compilation or probe.
if ! type module >/dev/null 2>&1; then
  if [[ -r /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  elif [[ -r /usr/share/Modules/init/bash ]]; then
    # shellcheck disable=SC1091
    source /usr/share/Modules/init/bash
  else
    fail CUDA_MODULE_SYSTEM_MISSING 24
  fi
fi
module purge || fail CUDA_MODULE_RESET_FAILED 24
module load cuda/12.9 || fail CUDA_12_9_MODULE_UNAVAILABLE 24
module -t list > "$RUN_ROOT/phase0/modules.txt" 2>&1 || true
NVCC=$(command -v nvcc) || fail NVCC_MISSING 24
"$NVCC" --version > "$RUN_ROOT/phase0/nvcc_version.txt" 2>&1
grep -Eq 'release 12\.9([,[:space:]]|$)' "$RUN_ROOT/phase0/nvcc_version.txt" || fail CUDA_12_9_REQUIRED 24

GPU_SELECTOR=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}
[[ -n $GPU_SELECTOR ]] || fail GPU_VISIBILITY_UNSPECIFIED 25
IFS=',' read -r -a GPU_IDS <<< "$GPU_SELECTOR"
[[ ${#GPU_IDS[@]} == 4 ]] || fail GPU_VISIBILITY_COUNT_MISMATCH 25
for gpu_id in "${GPU_IDS[@]}"; do
  [[ $gpu_id =~ ^(?:[0-9]+|GPU-[A-Za-z0-9-]+)$ ]] || fail GPU_SELECTOR_INVALID 25
done

"$NVIDIA_SMI" --id="$GPU_SELECTOR" \
  --query-gpu=index,uuid,name,memory.total,memory.free,mig.mode.current,compute_cap \
  --format=csv,noheader,nounits > "$RUN_ROOT/phase0/gpu_inventory.csv" || fail GPU_INVENTORY_QUERY_FAILED 25
"$NVIDIA_SMI" -L > "$RUN_ROOT/phase0/nvidia_smi_listing.txt" || fail GPU_LISTING_QUERY_FAILED 25
"$NVIDIA_SMI" topo -m > "$RUN_ROOT/phase0/topology.txt" || fail GPU_TOPOLOGY_QUERY_FAILED 25
"$NVIDIA_SMI" nvlink --status -i "$GPU_SELECTOR" > "$RUN_ROOT/phase0/nvlink_status.txt" || fail NVLINK_QUERY_FAILED 25
"$NVIDIA_SMI" --id="$GPU_SELECTOR" --query-gpu=uuid,driver_version --format=csv,noheader > "$RUN_ROOT/phase0/driver_versions.csv" || fail DRIVER_QUERY_FAILED 25
"$NVIDIA_SMI" > "$RUN_ROOT/phase0/nvidia_smi.txt" || fail NVIDIA_SMI_QUERY_FAILED 25

if ! command -v scontrol >/dev/null 2>&1; then
  fail SLURM_CONTROL_COMMAND_MISSING 26
fi
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_ROOT/phase0/slurm_job.txt" || fail SLURM_JOB_INSPECTION_FAILED 26
grep -Eqi 'gres/gpu:H200=4|gres:gpu:H200:4' "$RUN_ROOT/phase0/slurm_job.txt" || fail SLURM_TYPED_GPU_TRES_MISMATCH 26
for variable in SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_NODELIST SLURM_JOB_NUM_NODES SLURM_NNODES SLURM_GPUS_ON_NODE SLURM_JOB_GPUS SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE SLURM_JOB_PARTITION CUDA_VISIBLE_DEVICES; do
  printf '%s=%s\n' "$variable" "${!variable-}"
done > "$RUN_ROOT/phase0/slurm_environment.txt"

if ! "$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_sglang_gate_cli validate-inventory \
  --csv "$RUN_ROOT/phase0/gpu_inventory.csv" \
  --listing "$RUN_ROOT/phase0/nvidia_smi_listing.txt" \
  --output "$RUN_ROOT/phase0/inventory_manifest.json" > "$RUN_ROOT/phase0/inventory_validation.log" 2>&1; then
  FAILURE_CLASS=$(failure_class_from_log "$RUN_ROOT/phase0/inventory_validation.log" ALLOCATION_INVENTORY_MISMATCH)
  exit 27
fi
printf '%s\n' passed > "$RUN_ROOT/phase0/PASSED"

PHASE=weightless_backend_config_probe
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  fail CONTAINER_RUNTIME_UNAVAILABLE 30
fi
docker version --format '{{json .}}' > "$RUN_ROOT/phase1/docker_version.json" 2>/dev/null || fail CONTAINER_RUNTIME_UNAVAILABLE 30
if ! docker pull "$IMAGE" > "$RUN_ROOT/phase1/image_pull.log" 2>&1; then
  fail IMMUTABLE_RUNTIME_IMAGE_UNAVAILABLE 31
fi
docker image inspect "$IMAGE" --format '{{json .RepoDigests}} {{.Id}} {{.Architecture}}' > "$RUN_ROOT/phase1/image_identity.txt"
grep -Fq "$IMAGE_DIGEST" "$RUN_ROOT/phase1/image_identity.txt" || fail RUNTIME_IMAGE_IDENTITY_MISMATCH 31
grep -Eq 'amd64' "$RUN_ROOT/phase1/image_identity.txt" || fail RUNTIME_IMAGE_ARCHITECTURE_MISMATCH 31
printf '%s\n' "$IMAGE_HUMAN_TAG" > "$RUN_ROOT/phase1/image_human_tag.txt"
printf '%s\n' "$IMAGE" > "$RUN_ROOT/phase1/image_immutable_reference.txt"

docker_env=(
  --env SLURM_JOB_ID
  --env SLURM_JOB_NODELIST
  --env SLURM_JOB_NUM_NODES
  --env SLURM_GPUS_ON_NODE
  --env HOME=/localdata/home
  --env HF_HOME=/localdata/cache/huggingface
  --env TRANSFORMERS_CACHE=/localdata/cache/transformers
  --env XDG_CACHE_HOME=/localdata/cache/xdg
  --env TORCH_HOME=/localdata/cache/torch
  --env PIP_CACHE_DIR=/localdata/cache/pip
  --env TRITON_CACHE_DIR=/localdata/cache/triton
  --env TMPDIR=/localdata/tmp
  --env PYTHONPATH=/workspace/project/src
  --volume "$SOURCE_ROOT:/workspace/project:ro"
  --volume "$STORAGE_ROOT:/localdata"
)

if ! docker run --rm --gpus "device=$GPU_SELECTOR" "${docker_env[@]}" --entrypoint python3 "$IMAGE" \
  -m putpocket_dataset_mining.glm52_sglang_gate_cli phase1 \
  --lock /workspace/project/configs/cluster/glm52_sglang_gate_sources.lock.json \
  --artifact-root /localdata/artifacts/"$SLURM_JOB_ID"/phase1 \
  --metadata-root /localdata/cache/model-metadata/"$SLURM_JOB_ID" > "$RUN_ROOT/phase1/probe.log" 2>&1; then
  FAILURE_CLASS=$(failure_class_from_log "$RUN_ROOT/phase1/probe.log" WEIGHTLESS_COMPATIBILITY_PROBE_FAILED)
  exit 32
fi
MODEL_REVISION=$(tr -d '\r\n' < "$RUN_ROOT/phase1/model_revision.txt")
[[ $MODEL_REVISION =~ ^[0-9a-f]{40}$ ]] || fail MODEL_REVISION_UNRESOLVED 32
MODEL_ROOT="$STORAGE_ROOT/models/$MODEL_REVISION"
printf '%s\n' passed > "$RUN_ROOT/phase1/PASSED"

PHASE=minimal_all_resident_load
mkdir -p "$MODEL_ROOT"
if [[ -f "$MODEL_ROOT/.putpocket_checkpoint_ready.json" ]]; then
  if ! docker run --rm --gpus "device=$GPU_SELECTOR" "${docker_env[@]}" --entrypoint python3 "$IMAGE" \
    -m putpocket_dataset_mining.glm52_sglang_gate_cli validate-checkpoint \
    --model-root /localdata/models/"$MODEL_REVISION" --revision "$MODEL_REVISION" \
    > "$RUN_ROOT/phase2/checkpoint_validation.log" 2>&1; then
    FAILURE_CLASS=$(failure_class_from_log "$RUN_ROOT/phase2/checkpoint_validation.log" CHECKPOINT_MARKER_MISMATCH)
    exit 40
  fi
else
  if ! docker run --rm --gpus "device=$GPU_SELECTOR" "${docker_env[@]}" --entrypoint python3 "$IMAGE" \
    -m putpocket_dataset_mining.glm52_sglang_gate_cli download-model \
    --revision-file /localdata/artifacts/"$SLURM_JOB_ID"/phase1/model_revision.txt \
    --model-root /localdata/models/"$MODEL_REVISION" > "$RUN_ROOT/phase2/checkpoint_download.log" 2>&1; then
    FAILURE_CLASS=$(failure_class_from_log "$RUN_ROOT/phase2/checkpoint_download.log" CHECKPOINT_DOWNLOAD_FAILED)
    exit 40
  fi
fi

HBM_SAMPLES="$RUN_ROOT/phase2/hbm_samples.csv"
printf '%s\n' 'timestamp,uuid,memory_total_mib,memory_used_mib,memory_free_mib' > "$HBM_SAMPLES"
sample_hbm() {
  while true; do
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    while IFS= read -r line; do
      printf '%s,%s\n' "$timestamp" "$line" >> "$HBM_SAMPLES"
    done < <("$NVIDIA_SMI" --id="$GPU_SELECTOR" --query-gpu=uuid,memory.total,memory.used,memory.free --format=csv,noheader,nounits)
    sleep 2
  done
}
sample_hbm &
SAMPLER_PID=$!

SERVER_LOG="$RUN_ROOT/phase2/server.log"
SERVER_COMMAND=(
  python3 -m sglang.launch_server
  --model-path /model
  --served-model-name "$MODEL_ID"
  --tp 4
  --quantization modelopt_fp4
  --fp4-gemm-backend marlin
  --moe-runner-backend marlin
  --dsa-prefill-backend flashmla_sparse
  --dsa-decode-backend fa3
  --dsa-topk-backend sgl-kernel
  --context-length 4096
  --max-running-requests 1
  --cpu-offload-gb 0
  --weight-cache-mode off
  --mem-fraction-static 0.90
  --host 127.0.0.1
  --port "$PORT"
)
printf '%q ' "${SERVER_COMMAND[@]}" > "$RUN_ROOT/phase2/exact_command.txt"
printf '\n' >> "$RUN_ROOT/phase2/exact_command.txt"

docker run --rm --name "$SERVER_CONTAINER" --gpus "device=$GPU_SELECTOR" --ipc=host --network=host \
  "${docker_env[@]}" --volume "$MODEL_ROOT:/model:ro" --entrypoint python3 "$IMAGE" \
  "${SERVER_COMMAND[@]:1}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

SERVER_READY=false
for _ in $(seq 1 360); do
  if curl --fail --silent --show-error "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    SERVER_READY=true
    break
  fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    FAILURE_CLASS=$("$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_sglang_gate_cli classify-startup --server-log "$SERVER_LOG" 2>/dev/null || printf MODEL_LOAD_FAILED)
    exit 41
  fi
  sleep 30
done
[[ $SERVER_READY == true ]] || fail MODEL_LOAD_TIMEOUT 42

if ! curl --fail --silent --show-error "http://127.0.0.1:$PORT/get_server_info" > "$RUN_ROOT/phase2/server_info.json"; then
  fail RUNTIME_BACKEND_INFO_UNAVAILABLE 43
fi
printf '%s\n' passed > "$RUN_ROOT/phase2/PASSED"

PHASE=correctness_sentinel
cat > "$RUN_ROOT/phase3/sentinel_request.json" <<'JSON'
{"model":"nvidia/GLM-5.2-NVFP4","messages":[{"role":"user","content":"State in one short sentence that the feasibility sentinel completed. Do not repeat words."}],"temperature":0,"max_tokens":64,"n":1,"stream":false}
JSON
if ! curl --fail --silent --show-error -H 'Content-Type: application/json' \
  --data-binary @"$RUN_ROOT/phase3/sentinel_request.json" \
  "http://127.0.0.1:$PORT/v1/chat/completions" > "$RUN_ROOT/phase3/sentinel_response.json"; then
  fail SENTINEL_REQUEST_FAILED 50
fi
sleep 4
kill "$SAMPLER_PID" >/dev/null 2>&1 || true
wait "$SAMPLER_PID" >/dev/null 2>&1 || true
SAMPLER_PID=

if ! "$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_sglang_gate_cli validate-runtime \
  --inventory "$RUN_ROOT/phase0/inventory_manifest.json" \
  --model-config "$RUN_ROOT/phase1/model_config.json" \
  --server-info "$RUN_ROOT/phase2/server_info.json" \
  --server-log "$SERVER_LOG" \
  --response "$RUN_ROOT/phase3/sentinel_response.json" \
  --hbm-samples "$HBM_SAMPLES" \
  --model-revision "$RUN_ROOT/phase1/model_revision.txt" \
  --project-commit "$PROJECT_COMMIT" \
  --source-lock-report "$RUN_ROOT/phase1/source_lock_validation.json" \
  --capability-report "$RUN_ROOT/phase1/capability_probe.json" \
  --exact-command "$RUN_ROOT/phase2/exact_command.txt" \
  --output "$RUN_ROOT/gate_manifest.json" > "$RUN_ROOT/phase3/validation.log" 2>&1; then
  FAILURE_CLASS=$(failure_class_from_log "$RUN_ROOT/phase3/validation.log" CORRECTNESS_OR_RUNTIME_EVIDENCE_FAILED)
  exit 51
fi

printf '%s\n' passed > "$RUN_ROOT/phase3/PASSED"
JOB_STATUS=PASS
PHASE=complete
stop_children
exit 0
