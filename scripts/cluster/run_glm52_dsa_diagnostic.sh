#!/usr/bin/env bash
set -euo pipefail
umask 077

PROJECT_COMMIT=${1:-}
SOURCE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
STORAGE_ROOT=/local-data/jslee202403/putpocket-glm52-sglang-gate
RUN_ROOT="$STORAGE_ROOT/artifacts/glm52-dsa-diagnostic/${SLURM_JOB_ID:-unknown}"
CACHE_ROOT="$STORAGE_ROOT/cache"
METADATA_ROOT="$CACHE_ROOT/model-metadata/${SLURM_JOB_ID:-unknown}"
EPHEMERAL_ROOT="$STORAGE_ROOT/tmp/diagnostic-${SLURM_JOB_ID:-unknown}"
SGLANG_COMMIT=83d7d453306977dd3aad4402c921c8a6b66d9a9d
SGLANG_ROOT="$CACHE_ROOT/sglang-source/$SGLANG_COMMIT-432f00fa4851-87d8f3090c84"
HARNESS_COMMIT=ca10a60a5fcae51e6948ffe1485d4153d421e6c5
HARNESS_ROOT="$CACHE_ROOT/swebench-pro/$HARNESS_COMMIT"
MODEL_ID=nvidia/GLM-5.2-NVFP4
MODEL_REF=main
MODEL_REVISION_REQUIRED=aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
IMAGE_HUMAN_TAG=lmsysorg/sglang:latest
IMAGE_DIGEST=sha256:3be8803490a8b899a44f7ab2e22d8f6a1fb877cab52faeb400769a1555317db4
IMAGE="lmsysorg/sglang@$IMAGE_DIGEST"
BOOTSTRAP_PYTHON=/home2/jslee202403/miniconda3/bin/python3
GIT=/usr/bin/git
NVIDIA_SMI=/usr/bin/nvidia-smi
PORT=30000
SERVER_CONTAINER="pp-glm52-dsa-${SLURM_JOB_ID:-unknown}"
AGENT_CONTAINER="pp-glm52-agent-${SLURM_JOB_ID:-unknown}"
SERVER_PID=
SAMPLER_PID=
PHASE=allocation_inventory
FAILURE_CLASS=UNCLASSIFIED_FAILURE
JOB_STATUS=FAIL

write_terminal_manifest() {
  rc=$1
  mkdir -p "$RUN_ROOT" 2>/dev/null || true
  status=$JOB_STATUS
  [[ $status == BLOCKED ]] || status=FAIL
  printf '{"schema_version":1,"status":"%s","diagnostic":"glm52_nvfp4_native_dsa_single_swebench_pro_instance","phase":"%s","failure_class":"%s","returncode":%s,"job_id":"%s","project_commit":"%s","model_id":"nvidia/GLM-5.2-NVFP4","fallback_attempted":false,"offload_attempted":false,"quality_score_eligible":false}\n' \
    "$status" "$PHASE" "$FAILURE_CLASS" "$rc" "${SLURM_JOB_ID:-unknown}" "${PROJECT_COMMIT:-unknown}" \
    > "$RUN_ROOT/diagnostic_manifest.json.partial" 2>/dev/null || true
  mv "$RUN_ROOT/diagnostic_manifest.json.partial" "$RUN_ROOT/diagnostic_manifest.json" 2>/dev/null || true
}

stop_sampler() {
  if [[ -n $SAMPLER_PID ]]; then
    kill "$SAMPLER_PID" >/dev/null 2>&1 || true
    wait "$SAMPLER_PID" >/dev/null 2>&1 || true
    SAMPLER_PID=
  fi
}

stop_children() {
  stop_sampler
  docker rm -f "$AGENT_CONTAINER" >/dev/null 2>&1 || true
  if [[ -n $SERVER_PID ]]; then
    docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
    wait "$SERVER_PID" >/dev/null 2>&1 || true
    SERVER_PID=
  fi
}

cleanup_ephemeral() {
  rm -f \
    "$EPHEMERAL_ROOT/completion_request.json" \
    "$EPHEMERAL_ROOT/official_raw_sample.jsonl" \
    "$EPHEMERAL_ROOT/official_image.txt" \
    "$EPHEMERAL_ROOT/agent_action.sh"
}

finish() {
  rc=$?
  trap - EXIT INT TERM
  stop_children
  cleanup_ephemeral
  if [[ $JOB_STATUS != PASS ]]; then
    write_terminal_manifest "$rc"
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
  "$BOOTSTRAP_PYTHON" - "$1" "$2" <<'PY'
import json, re, sys
path, fallback = sys.argv[1:]
try:
    lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
except OSError:
    print(fallback); raise SystemExit
for line in reversed(lines):
    try: error = json.loads(line).get("error", "")
    except (json.JSONDecodeError, AttributeError): continue
    failure = error.split(":", 1)[0]
    if re.fullmatch(r"[A-Z][A-Z0-9_]+", failure): print(failure); raise SystemExit
print(fallback)
PY
}

[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail SLURM_ALLOCATION_REQUIRED 20
[[ ${SLURM_JOB_NUM_NODES:-0} == 1 && -n ${SLURM_JOB_NODELIST:-} ]] || fail SLURM_NODE_COUNT_MISMATCH 20
[[ ${SLURM_GPUS_ON_NODE:-0} == 4 ]] || fail SLURM_GPU_COUNT_MISMATCH 20
[[ $PROJECT_COMMIT =~ ^[0-9a-f]{40}$ ]] || fail PROJECT_COMMIT_INVALID 21
[[ -x $GIT && $($GIT -C "$SOURCE_ROOT" rev-parse HEAD) == "$PROJECT_COMMIT" ]] || fail PROJECT_COMMIT_MISMATCH 21
[[ -x $BOOTSTRAP_PYTHON && -x $NVIDIA_SMI ]] || fail COMPUTE_TOOL_MISSING 21
[[ -d /local-data/jslee202403 ]] || fail COMPUTE_LOCAL_STORAGE_MISSING 21
command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 || fail CONTAINER_RUNTIME_UNAVAILABLE 30

mkdir -p "$RUN_ROOT/phase0" "$RUN_ROOT/phase1" "$RUN_ROOT/phase2" "$RUN_ROOT/phase3" "$RUN_ROOT/official" "$CACHE_ROOT" "$EPHEMERAL_ROOT"
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

if ! type module >/dev/null 2>&1; then
  for module_init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
    if [[ -r $module_init ]]; then
      # shellcheck disable=SC1090
      source "$module_init"
      type module >/dev/null 2>&1 && break
    fi
  done
fi
type module >/dev/null 2>&1 || fail CUDA_MODULE_SYSTEM_MISSING 24
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

"$NVIDIA_SMI" --id="$GPU_SELECTOR" --query-gpu=index,uuid,name,memory.total,memory.free,mig.mode.current,compute_cap --format=csv,noheader,nounits > "$RUN_ROOT/phase0/gpu_inventory.csv" || fail GPU_INVENTORY_QUERY_FAILED 25
"$NVIDIA_SMI" -L > "$RUN_ROOT/phase0/nvidia_smi_listing.txt" || fail GPU_LISTING_QUERY_FAILED 25
"$NVIDIA_SMI" topo -m > "$RUN_ROOT/phase0/topology.txt" || fail GPU_TOPOLOGY_QUERY_FAILED 25
"$NVIDIA_SMI" nvlink --status -i "$GPU_SELECTOR" > "$RUN_ROOT/phase0/nvlink_status.txt" || fail NVLINK_QUERY_FAILED 25
"$NVIDIA_SMI" --id="$GPU_SELECTOR" --query-gpu=uuid,driver_version --format=csv,noheader > "$RUN_ROOT/phase0/driver_versions.csv" || fail DRIVER_QUERY_FAILED 25
"$NVIDIA_SMI" > "$RUN_ROOT/phase0/nvidia_smi.txt" || fail NVIDIA_SMI_QUERY_FAILED 25
command -v scontrol >/dev/null 2>&1 || fail SLURM_CONTROL_COMMAND_MISSING 26
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_ROOT/phase0/slurm_job.txt" || fail SLURM_JOB_INSPECTION_FAILED 26
grep -Eqi 'gres/gpu:H200=4|gres:gpu:H200:4' "$RUN_ROOT/phase0/slurm_job.txt" || fail SLURM_TYPED_GPU_TRES_MISMATCH 26
for variable in SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_NODELIST SLURM_JOB_NUM_NODES SLURM_NNODES SLURM_GPUS_ON_NODE SLURM_JOB_GPUS SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE SLURM_JOB_PARTITION CUDA_VISIBLE_DEVICES; do
  printf '%s=%s\n' "$variable" "${!variable-}"
done > "$RUN_ROOT/phase0/slurm_environment.txt"
if ! "$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_sglang_gate_cli validate-inventory --csv "$RUN_ROOT/phase0/gpu_inventory.csv" --listing "$RUN_ROOT/phase0/nvidia_smi_listing.txt" --output "$RUN_ROOT/phase0/inventory_manifest.json" > "$RUN_ROOT/phase0/inventory_validation.log" 2>&1; then
  FAILURE_CLASS=$(failure_class_from_log "$RUN_ROOT/phase0/inventory_validation.log" ALLOCATION_INVENTORY_MISMATCH)
  exit 27
fi
printf '%s\n' passed > "$RUN_ROOT/phase0/PASSED"

PHASE=weightless_backend_config_probe
docker version --format '{{json .}}' > "$RUN_ROOT/phase1/docker_version.json" 2>/dev/null || fail CONTAINER_RUNTIME_UNAVAILABLE 30
docker pull "$IMAGE" > "$RUN_ROOT/phase1/image_pull.log" 2>&1 || fail IMMUTABLE_RUNTIME_IMAGE_UNAVAILABLE 31
docker image inspect "$IMAGE" --format '{{json .RepoDigests}} {{.Id}} {{.Architecture}}' > "$RUN_ROOT/phase1/image_identity.txt"
grep -Fq "$IMAGE_DIGEST" "$RUN_ROOT/phase1/image_identity.txt" || fail RUNTIME_IMAGE_IDENTITY_MISMATCH 31
grep -Eq 'amd64' "$RUN_ROOT/phase1/image_identity.txt" || fail RUNTIME_IMAGE_ARCHITECTURE_MISMATCH 31
printf '%s\n' "$IMAGE_HUMAN_TAG" > "$RUN_ROOT/phase1/image_human_tag.txt"
printf '%s\n' "$IMAGE" > "$RUN_ROOT/phase1/image_immutable_reference.txt"

if [[ ! -d $SGLANG_ROOT/.git ]]; then
  mkdir -p "$SGLANG_ROOT"
  "$GIT" -C "$SGLANG_ROOT" init
  "$GIT" -C "$SGLANG_ROOT" fetch --depth=1 https://github.com/sgl-project/sglang.git "$SGLANG_COMMIT"
  "$GIT" -C "$SGLANG_ROOT" checkout --detach FETCH_HEAD
fi
[[ $($GIT -C "$SGLANG_ROOT" rev-parse HEAD) == "$SGLANG_COMMIT" ]] || fail SGLANG_SOURCE_COMMIT_MISMATCH 32
PATCH_MARKER="$SGLANG_ROOT/.putpocket-dsa-patch-432f00fa4851-87d8f3090c84.ready"
if [[ ! -f $PATCH_MARKER ]]; then
  "$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_dsa_diagnostic_cli validate-patch --repository-root "$SOURCE_ROOT" --source-root "$SGLANG_ROOT" --output "$RUN_ROOT/phase1/patch_preflight.json" || fail SGLANG_PATCH_CONTEXT_DIGEST_MISMATCH 32
  "$GIT" -C "$SGLANG_ROOT" apply --unidiff-zero --check "$SOURCE_ROOT/patches/sglang/$SGLANG_COMMIT/glm52_native_dsa_bounded_dump.patch" || fail SGLANG_PATCH_CONTEXT_MISMATCH 32
  "$GIT" -C "$SGLANG_ROOT" apply --unidiff-zero "$SOURCE_ROOT/patches/sglang/$SGLANG_COMMIT/glm52_native_dsa_bounded_dump.patch" || fail SGLANG_PATCH_APPLICATION_FAILED 32
  install -m 0644 "$SOURCE_ROOT/instrumentation/sglang/dsa_diagnostic_dump.py" "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa/dsa_diagnostic_dump.py"
  "$GIT" -C "$SGLANG_ROOT" diff --check || fail SGLANG_PATCH_DIFF_INVALID 32
  printf '%s\n' 'sglang=83d7d453306977dd3aad4402c921c8a6b66d9a9d patch=432f00fa48519ab81d389993f54bbcf796bdc7ed8c9ed3760b6173fe7dbc1266 instrumentation=87d8f3090c84fc272c1c0aca7723c4b8110ab0d714562d79396c568871a6c50b' > "$PATCH_MARKER"
fi
grep -Fq 'patch=432f00fa48519ab81d389993f54bbcf796bdc7ed8c9ed3760b6173fe7dbc1266' "$PATCH_MARKER" || fail SGLANG_PATCH_CACHE_IDENTITY_MISMATCH 32
[[ $(sha256sum "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa/dsa_indexer.py" | cut -d' ' -f1) == 7985bffeb4f8e7b712e75b452b062fbcf02fd2299386dcf1b1f0a0864d28e050 ]] || fail SGLANG_PATCHED_TARGET_CACHE_MISMATCH 32
[[ $(sha256sum "$SGLANG_ROOT/python/sglang/srt/layers/attention/dsa/dsa_diagnostic_dump.py" | cut -d' ' -f1) == 87d8f3090c84fc272c1c0aca7723c4b8110ab0d714562d79396c568871a6c50b ]] || fail SGLANG_INSTRUMENTATION_CACHE_MISMATCH 32

TRACE_CONTROL="$RUN_ROOT/phase3/trace_control.json"
TRACE_PROVENANCE="$RUN_ROOT/phase3/trace_provenance.json"
TRACE_RAW="$RUN_ROOT/phase3/native_raw"
mkdir -p "$TRACE_RAW"
cat > "$TRACE_PROVENANCE.partial" <<JSON
{"schema_version":1,"backend_identities":{"quantization":"modelopt_fp4","fp4_gemm":"marlin_w4a16","dsa_prefill":"flashmla_sparse","dsa_decode":"fa3","dsa_topk":"sgl-kernel"},"revisions":{"model":"$MODEL_REVISION_REQUIRED","sglang":"$SGLANG_COMMIT","image":"$IMAGE_DIGEST","project":"$PROJECT_COMMIT"}}
JSON
mv "$TRACE_PROVENANCE.partial" "$TRACE_PROVENANCE"

docker_env=(
  --env SLURM_JOB_ID --env SLURM_JOB_NAME --env SLURM_JOB_NODELIST --env SLURM_JOB_NUM_NODES --env SLURM_GPUS_ON_NODE
  --env HOME=/localdata/home --env HF_HOME=/localdata/cache/huggingface --env TRANSFORMERS_CACHE=/localdata/cache/transformers
  --env XDG_CACHE_HOME=/localdata/cache/xdg --env TORCH_HOME=/localdata/cache/torch --env PIP_CACHE_DIR=/localdata/cache/pip
  --env TRITON_CACHE_DIR=/localdata/cache/triton --env TMPDIR=/localdata/tmp
  --env PYTHONPATH=/sglang-source/python:/workspace/project/src
  --env PUTPOCKET_DSA_TRACE_CONTROL=/localdata/artifacts/glm52-dsa-diagnostic/"$SLURM_JOB_ID"/phase3/trace_control.json
  --env PUTPOCKET_DSA_TRACE_PROVENANCE=/localdata/artifacts/glm52-dsa-diagnostic/"$SLURM_JOB_ID"/phase3/trace_provenance.json
  --env PUTPOCKET_DSA_TRACE_ROOT=/localdata/artifacts/glm52-dsa-diagnostic/"$SLURM_JOB_ID"/phase3/native_raw
  --volume "$SOURCE_ROOT:/workspace/project:ro" --volume "$SGLANG_ROOT:/sglang-source:ro" --volume "$STORAGE_ROOT:/localdata"
)

if ! docker run --rm --gpus "device=$GPU_SELECTOR" "${docker_env[@]}" --entrypoint python3 "$IMAGE" -m putpocket_dataset_mining.glm52_sglang_gate_cli phase1 --lock /workspace/project/configs/cluster/glm52_sglang_gate_sources.lock.json --artifact-root /localdata/artifacts/glm52-dsa-diagnostic/"$SLURM_JOB_ID"/phase1 --metadata-root /localdata/cache/model-metadata/"$SLURM_JOB_ID" > "$RUN_ROOT/phase1/probe.log" 2>&1; then
  FAILURE_CLASS=$(failure_class_from_log "$RUN_ROOT/phase1/probe.log" WEIGHTLESS_COMPATIBILITY_PROBE_FAILED)
  exit 33
fi
MODEL_REVISION=$(tr -d '\r\n' < "$RUN_ROOT/phase1/model_revision.txt")
[[ $MODEL_REVISION == "$MODEL_REVISION_REQUIRED" ]] || fail MODEL_REVISION_CHANGED_SINCE_PIN 33
MODEL_ROOT="$STORAGE_ROOT/models/$MODEL_REVISION"
printf '%s\n' passed > "$RUN_ROOT/phase1/PASSED"

PHASE=minimal_all_resident_load
mkdir -p "$MODEL_ROOT"
if [[ -f $MODEL_ROOT/.putpocket_checkpoint_ready.json ]]; then
  docker run --rm --gpus "device=$GPU_SELECTOR" "${docker_env[@]}" --entrypoint python3 "$IMAGE" -m putpocket_dataset_mining.glm52_sglang_gate_cli validate-checkpoint --model-root /localdata/models/"$MODEL_REVISION" --revision "$MODEL_REVISION" > "$RUN_ROOT/phase2/checkpoint_validation.log" 2>&1 || fail CHECKPOINT_MARKER_MISMATCH 40
else
  docker run --rm --gpus "device=$GPU_SELECTOR" "${docker_env[@]}" --entrypoint python3 "$IMAGE" -m putpocket_dataset_mining.glm52_sglang_gate_cli download-model --revision-file /localdata/artifacts/glm52-dsa-diagnostic/"$SLURM_JOB_ID"/phase1/model_revision.txt --model-root /localdata/models/"$MODEL_REVISION" > "$RUN_ROOT/phase2/checkpoint_download.log" 2>&1 || fail CHECKPOINT_DOWNLOAD_FAILED 40
fi

sample_hbm() {
  file=$1
  printf '%s\n' 'timestamp,uuid,memory_total_mib,memory_used_mib,memory_free_mib' > "$file"
  while true; do
    timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    while IFS= read -r line; do printf '%s,%s\n' "$timestamp" "$line" >> "$file"; done < <("$NVIDIA_SMI" --id="$GPU_SELECTOR" --query-gpu=uuid,memory.total,memory.used,memory.free --format=csv,noheader,nounits)
    sleep 2
  done
}

LOAD_HBM="$RUN_ROOT/phase2/hbm_samples.csv"
sample_hbm "$LOAD_HBM" &
SAMPLER_PID=$!
SERVER_LOG="$RUN_ROOT/phase2/server.log"
SERVER_COMMAND=(
  python3 -m sglang.launch_server --model-path /model --served-model-name "$MODEL_ID" --tp 4
  --quantization modelopt_fp4 --fp4-gemm-backend marlin --moe-runner-backend marlin
  --dsa-prefill-backend flashmla_sparse --dsa-decode-backend fa3 --dsa-topk-backend sgl-kernel
  --context-length 4096 --max-running-requests 1 --cpu-offload-gb 0 --weight-cache-mode off
  --mem-fraction-static 0.90 --disable-radix-cache --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled
  --host 127.0.0.1 --port "$PORT"
)
printf '%q ' "${SERVER_COMMAND[@]}" > "$RUN_ROOT/phase2/exact_command.txt"; printf '\n' >> "$RUN_ROOT/phase2/exact_command.txt"
docker run --rm --name "$SERVER_CONTAINER" --gpus "device=$GPU_SELECTOR" --ipc=host --network=host "${docker_env[@]}" --volume "$MODEL_ROOT:/model:ro" --entrypoint python3 "$IMAGE" "${SERVER_COMMAND[@]:1}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
SERVER_READY=false
for _ in $(seq 1 360); do
  if curl --fail --silent --show-error "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then SERVER_READY=true; break; fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    FAILURE_CLASS=$("$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_sglang_gate_cli classify-startup --server-log "$SERVER_LOG" 2>/dev/null || printf MODEL_LOAD_FAILED)
    exit 41
  fi
  sleep 30
done
[[ $SERVER_READY == true ]] || fail MODEL_LOAD_TIMEOUT 42
stop_sampler
curl --fail --silent --show-error "http://127.0.0.1:$PORT/get_server_info" > "$RUN_ROOT/phase2/server_info.json" || fail RUNTIME_BACKEND_INFO_UNAVAILABLE 43
printf '%s\n' passed > "$RUN_ROOT/phase2/PASSED"

PHASE=native_dsa_single_instance_diagnostic
if [[ ! -d $HARNESS_ROOT/.git ]]; then
  mkdir -p "$HARNESS_ROOT"
  "$GIT" -C "$HARNESS_ROOT" init
  "$GIT" -C "$HARNESS_ROOT" fetch --depth=1 https://github.com/scaleapi/SWE-bench_Pro-os.git "$HARNESS_COMMIT"
  "$GIT" -C "$HARNESS_ROOT" checkout --detach FETCH_HEAD
  "$GIT" -C "$HARNESS_ROOT" submodule update --init --recursive
fi
[[ $($GIT -C "$HARNESS_ROOT" rev-parse HEAD) == "$HARNESS_COMMIT" ]] || fail HARNESS_COMMIT_MISMATCH 50
[[ $($GIT -C "$HARNESS_ROOT/mini-swe-agent" rev-parse HEAD) == d74716a3c8104a113f77cc9ab94cf407ecdcf1e9 ]] || fail MINI_SWE_COMMIT_MISMATCH 50
[[ $($GIT -C "$HARNESS_ROOT/SWE-agent" rev-parse HEAD) == 402a7b8fdac8193f3f255bb53859ba274234f596 ]] || fail SWE_AGENT_COMMIT_MISMATCH 50

docker run --rm --gpus "device=$GPU_SELECTOR" "${docker_env[@]}" --volume "$MODEL_ROOT:/model:ro" --volume "$HARNESS_ROOT:/harness:ro" --entrypoint python3 "$IMAGE" -m putpocket_dataset_mining.glm52_dsa_diagnostic_cli prepare --model-root /model --harness-root /harness --ephemeral-root /localdata/tmp/diagnostic-"$SLURM_JOB_ID" --artifact-root /localdata/artifacts/glm52-dsa-diagnostic/"$SLURM_JOB_ID"/phase3 > "$RUN_ROOT/phase3/prepare.log" 2>&1 || fail PINNED_DIAGNOSTIC_PREPARATION_FAILED 51

flush_cache() {
  output=$1
  curl --fail --silent --show-error -X POST "http://127.0.0.1:$PORT/flush_cache" > "$output"
}

RUN_ID="slurm-${SLURM_JOB_ID}-glm52-dsa"
docker inspect --format '{{.Id}} {{.State.Pid}} {{.State.StartedAt}}' "$SERVER_CONTAINER" > "$RUN_ROOT/phase3/server_identity_before.txt" || fail LIVE_SERVER_IDENTITY_UNAVAILABLE 52
"$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_dsa_diagnostic_cli control --mode OFF --run-id "$RUN_ID" --output "$TRACE_CONTROL"
flush_cache "$RUN_ROOT/phase3/flush_before_off.json" || fail CACHE_ISOLATION_FLUSH_FAILED 52
OFF_HBM="$RUN_ROOT/phase3/hbm_off.csv"; sample_hbm "$OFF_HBM" & SAMPLER_PID=$!
start_ns=$(date +%s%N)
curl --fail --silent --show-error -H 'Content-Type: application/json' --data-binary @"$EPHEMERAL_ROOT/completion_request.json" "http://127.0.0.1:$PORT/v1/completions" > "$RUN_ROOT/phase3/off_response.json" || fail TRACE_OFF_INFERENCE_FAILED 53
OFF_DURATION_NS=$(( $(date +%s%N) - start_ns )); stop_sampler

"$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_dsa_diagnostic_cli control --mode ON --run-id "$RUN_ID" --output "$TRACE_CONTROL"
flush_cache "$RUN_ROOT/phase3/flush_before_on.json" || fail CACHE_ISOLATION_FLUSH_FAILED 52
ON_HBM="$RUN_ROOT/phase3/hbm_on.csv"; sample_hbm "$ON_HBM" & SAMPLER_PID=$!
start_ns=$(date +%s%N)
curl --fail --silent --show-error -H 'Content-Type: application/json' --data-binary @"$EPHEMERAL_ROOT/completion_request.json" "http://127.0.0.1:$PORT/v1/completions" > "$RUN_ROOT/phase3/on_response.json" || {
  if compgen -G "$TRACE_RAW/BLOCKED-*.json" >/dev/null; then JOB_STATUS=BLOCKED; FAILURE_CLASS=NATIVE_DSA_EXPOSURE_BLOCKED; exit 3; fi
  fail TRACE_ON_INFERENCE_FAILED 53
}
ON_DURATION_NS=$(( $(date +%s%N) - start_ns )); stop_sampler
docker inspect --format '{{.Id}} {{.State.Pid}} {{.State.StartedAt}}' "$SERVER_CONTAINER" > "$RUN_ROOT/phase3/server_identity_after.txt" || fail LIVE_SERVER_IDENTITY_UNAVAILABLE 52
cmp -s "$RUN_ROOT/phase3/server_identity_before.txt" "$RUN_ROOT/phase3/server_identity_after.txt" || fail LIVE_SERVER_PROCESS_CHANGED 52

"$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_dsa_diagnostic_cli trace-equivalence --off-response "$RUN_ROOT/phase3/off_response.json" --on-response "$RUN_ROOT/phase3/on_response.json" --off-duration-ns "$OFF_DURATION_NS" --on-duration-ns "$ON_DURATION_NS" --output "$RUN_ROOT/phase3/trace_equivalence.json" || fail TRACE_OUTPUT_MISMATCH 54
set +e
"$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_dsa_diagnostic_cli finalize-captures --raw-root "$TRACE_RAW" --output-root "$RUN_ROOT/phase3/capture" --trace-report "$RUN_ROOT/phase3/trace_equivalence.json" --run-id "$RUN_ID"
capture_rc=$?
set -e
if [[ $capture_rc == 3 ]]; then JOB_STATUS=BLOCKED; FAILURE_CLASS=NATIVE_DSA_EXPOSURE_BLOCKED; exit 3; fi
[[ $capture_rc == 0 ]] || fail DSA_CAPTURE_VALIDATION_FAILED 55

docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
wait "$SERVER_PID" >/dev/null 2>&1 || true
SERVER_PID=

"$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_dsa_diagnostic_cli extract-action --response "$RUN_ROOT/phase3/on_response.json" --action-output "$EPHEMERAL_ROOT/agent_action.sh" --metadata-output "$RUN_ROOT/phase3/agent_action_metadata.json" || fail AGENT_FORMAT_INVALID 56
OFFICIAL_IMAGE=$(tr -d '\r\n' < "$EPHEMERAL_ROOT/official_image.txt")
[[ $OFFICIAL_IMAGE == docker.io/jefzda/sweap-images:* ]] || fail OFFICIAL_DOCKERHUB_TAG_INVALID 56
docker pull "$OFFICIAL_IMAGE" > "$RUN_ROOT/official/instance_image_pull.log" 2>&1 || fail OFFICIAL_INSTANCE_IMAGE_UNAVAILABLE 57
docker run -d --name "$AGENT_CONTAINER" -w /testbed "$OFFICIAL_IMAGE" sleep 2h > "$RUN_ROOT/official/agent_container_id.txt" || fail AGENT_CONTAINER_START_FAILED 57
set +e
timeout 600 docker exec -i -w /testbed "$AGENT_CONTAINER" bash -s < "$EPHEMERAL_ROOT/agent_action.sh" > "$RUN_ROOT/official/agent_action.log" 2>&1
action_rc=$?
set -e
printf '%s\n' "$action_rc" > "$RUN_ROOT/official/agent_action_returncode.txt"
docker exec -w /testbed "$AGENT_CONTAINER" git diff --binary > "$RUN_ROOT/official/model_patch.diff" || fail PATCH_GATHER_FAILED 58
docker rm -f "$AGENT_CONTAINER" >/dev/null 2>&1 || true

"$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_dsa_diagnostic_cli make-prediction --patch "$RUN_ROOT/official/model_patch.diff" --output "$RUN_ROOT/official/preds.json" --official-pred-root "$RUN_ROOT/official/pred_inputs"
"$BOOTSTRAP_PYTHON" "$HARNESS_ROOT/helper_code/gather_patches.py" --directory "$RUN_ROOT/official/pred_inputs" --prefix glm52-dsa-diagnostic --output "$RUN_ROOT/official/patches.json" > "$RUN_ROOT/official/gather.log" 2>&1 || fail OFFICIAL_PATCH_GATHER_FAILED 58

UV_ROOT="$CACHE_ROOT/tools/uv-0.11.31"
UV="$UV_ROOT/bin/uv"
EVAL_ENV="$CACHE_ROOT/eval-env/pandas-2.3.3_tqdm-4.67.1_docker-7.1.0"
if [[ ! -x $UV ]]; then
  "$BOOTSTRAP_PYTHON" -m venv "$UV_ROOT"
  "$UV_ROOT/bin/python" -m pip install --disable-pip-version-check --no-input 'uv==0.11.31'
fi
if [[ ! -x $EVAL_ENV/bin/python ]]; then "$UV" venv --python "$BOOTSTRAP_PYTHON" "$EVAL_ENV"; fi
"$UV" pip install --python "$EVAL_ENV/bin/python" --no-input 'pandas==2.3.3' 'tqdm==4.67.1' 'docker==7.1.0' > "$RUN_ROOT/official/evaluator_install.log" 2>&1 || fail EVALUATOR_DEPENDENCY_INSTALL_FAILED 59
"$UV" pip freeze --python "$EVAL_ENV/bin/python" > "$RUN_ROOT/official/evaluator_freeze.txt"
mkdir -p "$RUN_ROOT/official/evaluation"
(
  cd "$HARNESS_ROOT"
  "$EVAL_ENV/bin/python" "$HARNESS_ROOT/swe_bench_pro_eval.py" --raw_sample_path "$EPHEMERAL_ROOT/official_raw_sample.jsonl" --patch_path "$RUN_ROOT/official/patches.json" --output_dir "$RUN_ROOT/official/evaluation" --scripts_dir "$HARNESS_ROOT/run_scripts" --num_workers 1 --dockerhub_username jefzda --use_local_docker
) > "$RUN_ROOT/official/evaluator.log" 2>&1 || fail OFFICIAL_SINGLE_ROW_EVALUATOR_FAILED 60
[[ -f $RUN_ROOT/official/evaluation/eval_results.json ]] || fail OFFICIAL_EVAL_RESULTS_MISSING 60

"$BOOTSTRAP_PYTHON" -m putpocket_dataset_mining.glm52_dsa_diagnostic_cli finalize-diagnostic --inventory "$RUN_ROOT/phase0/inventory_manifest.json" --model-config "$RUN_ROOT/phase1/model_config.json" --server-info "$RUN_ROOT/phase2/server_info.json" --server-log "$SERVER_LOG" --model-revision "$RUN_ROOT/phase1/model_revision.txt" --trace-report "$RUN_ROOT/phase3/trace_equivalence.json" --capture-manifest "$RUN_ROOT/phase3/capture/capture_manifest.json" --eval-results "$RUN_ROOT/official/evaluation/eval_results.json" --load-hbm "$LOAD_HBM" --off-hbm "$OFF_HBM" --on-hbm "$ON_HBM" --project-commit "$PROJECT_COMMIT" --output "$RUN_ROOT/diagnostic_manifest.json" || fail FINAL_DIAGNOSTIC_VALIDATION_FAILED 61

cleanup_ephemeral
printf '%s\n' passed > "$RUN_ROOT/phase3/PASSED"
JOB_STATUS=PASS
PHASE=complete
exit 0
