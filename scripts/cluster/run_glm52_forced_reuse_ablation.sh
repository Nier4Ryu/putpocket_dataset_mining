#!/usr/bin/env bash
set -euo pipefail
umask 077

fail() { printf 'BLOCKED_%s\n' "$1" >&2; exit "${2:-20}"; }
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [[ -n ${SAMPLER_PID:-} ]]; then kill "$SAMPLER_PID" 2>/dev/null || true; wait "$SAMPLER_PID" 2>/dev/null || true; fi
  if [[ -n ${PROXY_PID:-} ]]; then kill "$PROXY_PID" 2>/dev/null || true; wait "$PROXY_PID" 2>/dev/null || true; fi
  if [[ -n ${SERVER_NAME:-} && -n ${CONTAINER:-} ]]; then "$CONTAINER" rm -f "$SERVER_NAME" >/dev/null 2>&1 || true; fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail SLURM_ALLOCATION_REQUIRED
[[ ${SLURM_JOB_NUM_NODES:-0} == 1 ]] || fail EXACTLY_ONE_NODE_REQUIRED
[[ ${SLURM_GPUS_ON_NODE:-0} == 4 ]] || fail EXACTLY_FOUR_GPUS_REQUIRED
[[ -n ${SLURM_JOB_NODELIST:-} && -n ${CUDA_VISIBLE_DEVICES:-} ]] || fail SLURM_GPU_VISIBILITY_MISSING
[[ ${PUTPOCKET_UNSAFE_FORCED_REUSE_ACK:-} == I_UNDERSTAND_ZERO_SAFE_PAGES ]] || fail UNSAFE_ACK_MISSING

for name in PUTPOCKET_PACKAGE_ROOT PUTPOCKET_CONTAINER_EXECUTABLE PUTPOCKET_SHARED_BUILD_ROOT PUTPOCKET_EXPECTED_BUNDLE_KEY PUTPOCKET_H200_STORAGE_PARENT PUTPOCKET_H200_WORK_ROOT PUTPOCKET_RUN_ARTIFACT_ROOT PUTPOCKET_CAPTURE_INPUT_ROOT PUTPOCKET_EPISODE_ROOT PUTPOCKET_DURABLE_OUTPUT_ROOT PUTPOCKET_SWEEP_PROFILE; do
  [[ -n ${!name:-} ]] || fail "ENV_${name}_MISSING"
done
case "$PUTPOCKET_SWEEP_PROFILE" in
  episode) PUTPOCKET_RATIOS=episode ;;
  smoke) PUTPOCKET_RATIOS=0,10 ;;
  full) PUTPOCKET_RATIOS=0,10,20,30,40,50,60,70,80,90,100 ;;
  *) fail SWEEP_PROFILE_INVALID ;;
esac

PACKAGE=$(realpath "$PUTPOCKET_PACKAGE_ROOT")
CONTAINER=$(realpath "$PUTPOCKET_CONTAINER_EXECUTABLE")
BUNDLE=$(realpath "$PUTPOCKET_SHARED_BUILD_ROOT/$PUTPOCKET_EXPECTED_BUNDLE_KEY")
STORAGE_PARENT=$(realpath "$PUTPOCKET_H200_STORAGE_PARENT")
STORAGE=$(realpath "$PUTPOCKET_H200_WORK_ROOT")
ARTIFACT_PARENT=$(realpath "$PUTPOCKET_RUN_ARTIFACT_ROOT")
INPUTS=$(realpath "$PUTPOCKET_CAPTURE_INPUT_ROOT")
EPISODE_REQUESTED=$PUTPOCKET_EPISODE_ROOT
DURABLE_PARENT=$(realpath "$PUTPOCKET_DURABLE_OUTPUT_ROOT")
BASELINE_CAPTURE="$INPUTS/baseline"
EDITED_CAPTURE="$INPUTS/edited"
DONOR_PROMPT="$INPUTS/donor-prompt-token-ids.json"
EDITED_PROMPT="$INPUTS/edited-prompt-token-ids.json"
SELECTOR="$INPUTS/selector/selector.json"
VLLM_ARCHIVE="$PACKAGE/artifacts/vllm-4a3447d200e5aa428d68d1a00aa00f1a19a1a729.tar.gz"
MODEL_REVISION=aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
MODEL="$STORAGE/cache/models/$MODEL_REVISION"

[[ -d $PACKAGE && -x $CONTAINER && -d $BUNDLE && -d $STORAGE_PARENT && -d $STORAGE && -d $ARTIFACT_PARENT && -d $DURABLE_PARENT ]] || fail PINNED_SITE_PATH_INVALID
[[ -d $BASELINE_CAPTURE && -d $EDITED_CAPTURE && -f $DONOR_PROMPT && -f $EDITED_PROMPT && -f $SELECTOR && -f $INPUTS/SHA256SUMS ]] || fail TRACK_B_OR_PROMPT_ARTIFACT_MISSING
(cd "$INPUTS" && sha256sum --check SHA256SUMS >/dev/null) || fail TRACK_B_INPUT_DIGEST_MISMATCH
if [[ $PUTPOCKET_SWEEP_PROFILE == episode ]]; then
  EPISODE_PARENT=$(realpath "$(dirname "$EPISODE_REQUESTED")")
  [[ ! -e $EPISODE_REQUESTED ]] || fail EPISODE_ROOT_ALREADY_EXISTS
else
  EPISODE=$(realpath "$EPISODE_REQUESTED")
  [[ -d $EPISODE && -f $EPISODE/episode.json && -f $EPISODE/donor-history-token-ids.json && -f $EPISODE/edited-history-token-ids.json && -f $EPISODE/selector/selector.json && -f $EPISODE/SHA256SUMS ]] || fail FROZEN_EPISODE_INCOMPLETE
  (cd "$EPISODE" && sha256sum --check SHA256SUMS >/dev/null) || fail FROZEN_EPISODE_DIGEST_MISMATCH
fi
[[ -f $VLLM_ARCHIVE && -f $PACKAGE/artifacts/vllm-source.sha256 ]] || fail PINNED_VLLM_SOURCE_ARCHIVE_MISSING
(cd "$PACKAGE" && sha256sum -c artifacts/vllm-source.sha256) || fail PINNED_VLLM_SOURCE_ARCHIVE_DIGEST_MISMATCH
[[ -f $BUNDLE/build_manifest.json && -f $BUNDLE/runtime-image.tar ]] || fail IMMUTABLE_RUNTIME_BUNDLE_INCOMPLETE
python3 - "$BUNDLE/build_manifest.json" "$PUTPOCKET_EXPECTED_BUNDLE_KEY" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x.get('status')=='SUCCESS'
assert x.get('vllm_commit')=='4a3447d200e5aa428d68d1a00aa00f1a19a1a729'
assert x.get('bundle_key')==sys.argv[2]
assert x.get('general_h200_compilation_allowed') is False
assert x.get('h200_runtime_jit_scope')=='native_first_use_deepgemm_dsa_only'
assert x.get('runtime_gate')=='ALLOW_NATIVE_FIRST_USE_JIT_WITH_RUN_LOCAL_AUDIT'
PY

case "$STORAGE/" in "$STORAGE_PARENT"/*) ;; *) fail WORK_ROOT_OUTSIDE_STORAGE_PARENT ;; esac
case "$ARTIFACT_PARENT/" in "$STORAGE/"*) ;; *) fail ARTIFACT_ROOT_OUTSIDE_WORK_ROOT ;; esac
mapfile -t GPU_LINES < <(/usr/bin/nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits)
[[ ${#GPU_LINES[@]} == 4 ]] || fail GPU_COUNT_MISMATCH
for line in "${GPU_LINES[@]}"; do
  [[ $line == NVIDIA\ H200* ]] || fail GPU_MODEL_MISMATCH
  total=$(awk -F, '{gsub(/ /,"",$2); print $2}' <<<"$line")
  free=$(awk -F, '{gsub(/ /,"",$3); print $3}' <<<"$line")
  sm=$(awk -F, '{gsub(/ /,"",$4); print $4}' <<<"$line")
  (( total >= 140000 && free >= 135000 )) || fail GPU_HBM_PREFLIGHT_FAILED
  [[ $sm == 9.0 ]] || fail GPU_SM90_REQUIRED
done
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
(( available_kib >= 32 * 1024 * 1024 )) || fail HOST_RAM_32G_GATE_FAILED

RUN_ROOT="$ARTIFACT_PARENT/${SLURM_JOB_ID}-${PUTPOCKET_RATIOS//,/-}"
[[ ! -e $RUN_ROOT ]] || fail RUN_ROOT_ALREADY_EXISTS
mkdir -p "$RUN_ROOT"/{logs,source,overlay,runtime,results,prepared,config,jit/{deepgemm,flashinfer,triton,torchinductor,cuda,vllm,tmp}}
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_ROOT/slurm-job.txt"
/usr/bin/nvidia-smi -L > "$RUN_ROOT/gpu-list.txt"
/usr/bin/nvidia-smi topo -m > "$RUN_ROOT/gpu-topology.txt"

"$CONTAINER" load --input "$BUNDLE/runtime-image.tar" > "$RUN_ROOT/logs/image-load.log" 2>&1 || fail RUNTIME_IMAGE_LOAD_FAILED
RUNTIME_IMAGE_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_image_id"])' "$BUNDLE/build_manifest.json")
[[ $RUNTIME_IMAGE_ID == sha256:* ]] || fail RUNTIME_IMAGE_ID_INVALID
"$CONTAINER" image inspect "$RUNTIME_IMAGE_ID" --format '{{.Id}} {{.Architecture}}' > "$RUN_ROOT/runtime-image.txt"
grep -Fq "$RUNTIME_IMAGE_ID" "$RUN_ROOT/runtime-image.txt" || fail RUNTIME_IMAGE_ID_MISMATCH

if [[ ! -f $MODEL/.putpocket_model_revision ]]; then
  available_bytes=$(df --output=avail -B1 "$STORAGE" | tail -1 | tr -d ' ')
  (( available_bytes >= 550 * 1024 * 1024 * 1024 )) || fail MODEL_STAGE_SPACE_BELOW_550G
  mkdir -p "$STORAGE/cache/models"
  "$CONTAINER" run --rm --volume "$STORAGE:/storage" --entrypoint python3 "$RUNTIME_IMAGE_ID" - "$MODEL_REVISION" > "$RUN_ROOT/logs/model-stage.log" 2>&1 <<'PY' || fail MODEL_STAGE_FAILED
from huggingface_hub import snapshot_download
from pathlib import Path
import sys
revision=sys.argv[1]
target=Path('/storage/cache/models')/revision
snapshot_download(repo_id='nvidia/GLM-5.2-NVFP4',revision=revision,local_dir=target)
(target/'.putpocket_model_revision').write_text(revision+'\n')
PY
fi
[[ $(tr -d '\r\n' < "$MODEL/.putpocket_model_revision") == "$MODEL_REVISION" ]] || fail MODEL_REVISION_MISMATCH
[[ $(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}') == 19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d ]] || fail TOKENIZER_JSON_DIGEST_MISMATCH
[[ $(sha256sum "$MODEL/tokenizer_config.json" | awk '{print $1}') == 77af7d4769cd62c107b90495cac9b0ba81573c86486821bfba2980c04285ec7a ]] || fail TOKENIZER_CONFIG_DIGEST_MISMATCH
[[ $(sha256sum "$MODEL/chat_template.jinja" | awk '{print $1}') == 172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679 ]] || fail CHAT_TEMPLATE_DIGEST_MISMATCH
[[ -f $MODEL/model.safetensors.index.json ]] || fail MODEL_WEIGHT_INDEX_MISSING

tar -xzf "$VLLM_ARCHIVE" -C "$RUN_ROOT/source"
VLLM_SOURCE="$RUN_ROOT/source/vllm"
[[ -f $VLLM_SOURCE/VLLM_COMMIT && $(tr -d '\r\n' < "$VLLM_SOURCE/VLLM_COMMIT") == 4a3447d200e5aa428d68d1a00aa00f1a19a1a729 ]] || fail VLLM_ARCHIVE_COMMIT_MISMATCH
patch -d "$VLLM_SOURCE" -p1 --forward --batch < "$PACKAGE/patches/vllm/4a3447d200e5aa428d68d1a00aa00f1a19a1a729/glm52_forced_edit_reuse.patch" >/dev/null || fail VLLM_PATCH_APPLY_FAILED
install -m 0644 "$PACKAGE/instrumentation/vllm/glm52_forced_edit_reuse.py" "$VLLM_SOURCE/vllm/model_executor/layers/glm52_forced_edit_reuse.py"

"$CONTAINER" run --rm --volume "$RUN_ROOT/overlay:/export" --entrypoint /bin/bash "$RUNTIME_IMAGE_ID" -lc 'set -euo pipefail; cp -a /opt/venv/lib/python3.12/site-packages/vllm /export/vllm' > "$RUN_ROOT/logs/overlay-copy.log" 2>&1 || fail RUNTIME_OVERLAY_COPY_FAILED
for relative in model_executor/layers/attention/mla_attention.py model_executor/layers/sparse_attn_indexer.py model_executor/models/glm4_moe_lite.py model_executor/layers/glm52_forced_edit_reuse.py; do
  install -m 0644 "$VLLM_SOURCE/vllm/$relative" "$RUN_ROOT/overlay/vllm/$relative"
done

GPU_REQUEST="\"device=$CUDA_VISIBLE_DEVICES\""
common_mounts=(
  --volume "$PACKAGE:/project:ro" --volume "$STORAGE:/storage"
  --volume "$RUN_ROOT/overlay:/overlay:ro" --volume "$BASELINE_CAPTURE:/baseline:ro"
  --volume "$EDITED_CAPTURE:/edited:ro" --volume "$DONOR_PROMPT:/inputs/donor.json:ro"
  --volume "$EDITED_PROMPT:/inputs/edited.json:ro" --volume "$INPUTS/selector:/inputs/selector:ro"
)
if [[ $PUTPOCKET_SWEEP_PROFILE != episode ]]; then common_mounts+=(--volume "$EPISODE:/episode:ro"); fi
common_env=(
  --env PYTHONPATH=/overlay:/project/src --env HOME=/storage/home
  --env HF_HOME=/storage/cache/huggingface --env VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE
  --env VLLM_CACHE_ROOT=/storage/artifacts/"$SLURM_JOB_ID-${PUTPOCKET_RATIOS//,/-}"/jit/vllm
  --env DG_JIT_CACHE_DIR=/storage/artifacts/"$SLURM_JOB_ID-${PUTPOCKET_RATIOS//,/-}"/jit/deepgemm
  --env VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/storage/artifacts/"$SLURM_JOB_ID-${PUTPOCKET_RATIOS//,/-}"/jit/flashinfer
  --env TRITON_CACHE_DIR=/storage/artifacts/"$SLURM_JOB_ID-${PUTPOCKET_RATIOS//,/-}"/jit/triton
  --env TORCHINDUCTOR_CACHE_DIR=/storage/artifacts/"$SLURM_JOB_ID-${PUTPOCKET_RATIOS//,/-}"/jit/torchinductor
  --env CUDA_CACHE_PATH=/storage/artifacts/"$SLURM_JOB_ID-${PUTPOCKET_RATIOS//,/-}"/jit/cuda
  --env TMPDIR=/storage/artifacts/"$SLURM_JOB_ID-${PUTPOCKET_RATIOS//,/-}"/jit/tmp
)

"$CONTAINER" run --rm --gpus "$GPU_REQUEST" "${common_mounts[@]}" "${common_env[@]}" --entrypoint python3 "$RUNTIME_IMAGE_ID" - <<'PY' > "$RUN_ROOT/logs/runtime-probe.log" 2>&1
import hashlib,importlib,pathlib,torch,vllm
assert torch.cuda.device_count()==4 and all(torch.cuda.get_device_capability(i)==(9,0) for i in range(4))
assert pathlib.Path(vllm.__file__).resolve().is_relative_to(pathlib.Path('/overlay'))
importlib.import_module('vllm.model_executor.layers.glm52_forced_edit_reuse')
importlib.import_module('vllm.v1.attention.backends.mla.flashmla_sparse')
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4W4A16LinearMethod
assert ModelOptNvFp4W4A16LinearMethod
for path in ('/overlay/vllm/model_executor/layers/attention/mla_attention.py','/overlay/vllm/model_executor/layers/sparse_attn_indexer.py','/overlay/vllm/model_executor/models/glm4_moe_lite.py','/overlay/vllm/model_executor/layers/glm52_forced_edit_reuse.py'):
 print(hashlib.sha256(open(path,'rb').read()).hexdigest(),path)
print('torch',torch.__version__,'cuda',torch.version.cuda,'vllm',vllm.__version__)
PY

CONTROL_CONTAINER="/storage/artifacts/${SLURM_JOB_ID}-${PUTPOCKET_RATIOS//,/-}/control.json"

PORT=$((20000 + SLURM_JOB_ID % 20000))
SERVER_NAME="pp-glm52-forced-${SLURM_JOB_ID}"
SERVER_LOG="$RUN_ROOT/logs/server.log"
server_args=(
  serve /model --served-model-name nvidia/GLM-5.2-NVFP4 --revision "$MODEL_REVISION"
  --tensor-parallel-size 4 --pipeline-parallel-size 1 --quantization modelopt_fp4
  --linear-backend marlin --attention-backend FLASHMLA_SPARSE --kv-cache-dtype bfloat16
  --max-model-len 65536 --max-num-seqs 1 --max-num-batched-tokens 65536
  --cpu-offload-gb 0 --swap-space 0 --no-enable-prefix-caching --no-enable-chunked-prefill
  --enforce-eager --jit-monitor-mode warn --jit-monitor-verbose
  --host 127.0.0.1 --port "$PORT"
)
printf '%q ' vllm "${server_args[@]}" > "$RUN_ROOT/exact-server-command.txt"; printf '\n' >> "$RUN_ROOT/exact-server-command.txt"
server_env=()
if [[ $PUTPOCKET_SWEEP_PROFILE != episode ]]; then server_env+=(--env PUTPOCKET_GLM52_FORCED_REUSE_CONTROL="$CONTROL_CONTAINER"); fi
"$CONTAINER" run --rm --name "$SERVER_NAME" --gpus "$GPU_REQUEST" --ipc=host --network=host \
  "${common_mounts[@]}" "${common_env[@]}" "${server_env[@]}" --volume "$MODEL:/model:ro" \
  --env PUTPOCKET_SERVED_MODEL_NAME=nvidia/GLM-5.2-NVFP4 \
  --entrypoint vllm "$RUNTIME_IMAGE_ID" "${server_args[@]}" > "$SERVER_LOG" 2>&1 &

ready=false
for _ in $(seq 1 240); do
  if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=true; break; fi
  "$CONTAINER" inspect "$SERVER_NAME" >/dev/null 2>&1 || fail SERVER_EXITED_BEFORE_HEALTH
  sleep 30
done
[[ $ready == true ]] || fail SERVER_HEALTH_TIMEOUT

sample_hbm() { while true; do date -u +%Y-%m-%dT%H:%M:%SZ; /usr/bin/nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-gpu=uuid,memory.total,memory.used,memory.free --format=csv,noheader,nounits; sleep 5; done; }
sample_hbm > "$RUN_ROOT/hbm.csv" & SAMPLER_PID=$!

HARNESS_COMMIT=ca10a60a5fcae51e6948ffe1485d4153d421e6c5
HARNESS="$STORAGE/cache/harness/$HARNESS_COMMIT"
AGENT_ENV="$STORAGE/cache/agent-env/$HARNESS_COMMIT"
if [[ ! -d $HARNESS/.git ]]; then
  mkdir -p "$HARNESS"
  git -C "$HARNESS" init
  git -C "$HARNESS" fetch --depth=1 https://github.com/scaleapi/SWE-bench_Pro-os.git "$HARNESS_COMMIT"
  git -C "$HARNESS" checkout --detach FETCH_HEAD
  git -C "$HARNESS" submodule update --init --recursive
fi
[[ $(git -C "$HARNESS" rev-parse HEAD) == "$HARNESS_COMMIT" ]] || fail HARNESS_COMMIT_MISMATCH
if [[ ! -f $AGENT_ENV/READY ]]; then
  python3 -m venv "$AGENT_ENV"
  "$AGENT_ENV/bin/pip" install --disable-pip-version-check --no-input -r "$HARNESS/requirements.txt" > "$RUN_ROOT/logs/agent-env.log" 2>&1 || fail AGENT_ENV_REQUIREMENTS_FAILED
  "$AGENT_ENV/bin/pip" install --disable-pip-version-check --no-input -e "$HARNESS/mini-swe-agent" >> "$RUN_ROOT/logs/agent-env.log" 2>&1 || fail MINI_SWE_AGENT_INSTALL_FAILED
  printf 'harness=%s\n' "$HARNESS_COMMIT" > "$AGENT_ENV/READY"
fi
AGENT_PY="$AGENT_ENV/bin/python"
AGENT_BIN="$AGENT_ENV/bin/mini-extra"
[[ -x $AGENT_PY && -x $AGENT_BIN ]] || fail AGENT_ENV_INCOMPLETE
export PYTHONPATH="$PACKAGE/src"
"$AGENT_PY" -m putpocket_dataset_mining.swebench_pro_cli prepare --selection smoke --harness-root "$HARNESS" --output-root "$RUN_ROOT/prepared" > "$RUN_ROOT/logs/prepare-agent-case.log" 2>&1 || fail AGENT_CASE_PREPARE_FAILED
"$AGENT_PY" -m putpocket_dataset_mining.swebench_pro_cli agent-config --harness-root "$HARNESS" --runtime docker --output "$RUN_ROOT/config/base-agent.yaml" > "$RUN_ROOT/logs/agent-config.log" 2>&1 || fail BASE_AGENT_CONFIG_FAILED

PROXY_PORT=$((40000 + SLURM_JOB_ID % 15000))
PHASE_FILE="$RUN_ROOT/proxy-phase.json"
PROXY_LOG="$RUN_ROOT/proxy-usage.jsonl"
step_limit=0
[[ $PUTPOCKET_SWEEP_PROFILE == episode ]] && step_limit=1
"$AGENT_PY" -m putpocket_dataset_mining.glm52_stateful_cli configure-agent \
  --input "$RUN_ROOT/config/base-agent.yaml" --output "$RUN_ROOT/config/donor-agent.yaml" \
  --api-base "http://127.0.0.1:$PROXY_PORT/v1" --step-limit "$step_limit" || fail DONOR_AGENT_CONFIG_FAILED

if [[ $PUTPOCKET_SWEEP_PROFILE == episode ]]; then
  printf '{"phase_id":"episode-capture","ratio_percent":null,"mode":"capture_old_turn1"}\n' > "$PHASE_FILE"
  "$AGENT_PY" -m putpocket_dataset_mining.glm52_stateful_proxy \
    --listen-port "$PROXY_PORT" --backend-port "$PORT" --phase-file "$PHASE_FILE" \
    --log "$PROXY_LOG" --capture "$RUN_ROOT/episode-capture.json" > "$RUN_ROOT/logs/proxy.log" 2>&1 &
  PROXY_PID=$!
  for _ in $(seq 1 60); do curl --fail --silent "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1 && break; kill -0 "$PROXY_PID" >/dev/null 2>&1 || fail STATEFUL_PROXY_EXITED; sleep 1; done
  curl --fail --silent "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1 || fail STATEFUL_PROXY_HEALTH_FAILED
  OPENAI_API_KEY=local-vllm-no-auth "$AGENT_BIN" swebench \
    --subset "$RUN_ROOT/prepared/mini_dataset" --split test --workers 1 \
    --model openai/nvidia/GLM-5.2-NVFP4 --config "$RUN_ROOT/config/donor-agent.yaml" \
    --environment-class docker --output "$RUN_ROOT/episode-inference" > "$RUN_ROOT/logs/episode-agent.log" 2>&1 || fail EPISODE_AGENT_FAILED
  INSTANCE_ID=instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5
  TRAJECTORY="$RUN_ROOT/episode-inference/$INSTANCE_ID/$INSTANCE_ID.traj.json"
  [[ -f $TRAJECTORY && -f $RUN_ROOT/episode-capture.json ]] || fail EPISODE_ARTIFACT_MISSING
  "$CONTAINER" run --rm "${common_mounts[@]}" "${common_env[@]}" \
    --volume "$MODEL:/model:ro" --volume "$RUN_ROOT:/run" --entrypoint python3 "$RUNTIME_IMAGE_ID" \
    -m putpocket_dataset_mining.glm52_stateful_cli build-episode \
    --capture /run/episode-capture.json --trajectory "/run/episode-inference/$INSTANCE_ID/$INSTANCE_ID.traj.json" \
    --base-donor-prompt /inputs/donor.json --base-edited-prompt /inputs/edited.json \
    --base-selector /inputs/selector/selector.json --model-root /model --output-root /run/episode-export \
    > "$RUN_ROOT/logs/build-episode.log" 2>&1 || fail STATEFUL_EPISODE_BUILD_FAILED
  (cd "$RUN_ROOT/episode-export" && find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  EPISODE_STAGE="$EPISODE_REQUESTED.partial"
  [[ ! -e $EPISODE_STAGE && ! -e $EPISODE_REQUESTED ]] || fail EPISODE_EXPORT_COLLISION
  cp -a "$RUN_ROOT/episode-export" "$EPISODE_STAGE"
  chmod -R a-w "$EPISODE_STAGE"
  mv "$EPISODE_STAGE" "$EPISODE_REQUESTED"
  printf 'FROZEN_STATEFUL_EPISODE=%s\n' "$EPISODE_REQUESTED"
else
EXPERIMENT_ID="glm52-stateful-mid-edit-${SLURM_JOB_ID}"
write_control() {
  local mode=$1 phase=$2 side=$3 ratio=$4 evidence=$5
  "$CONTAINER" run --rm "${common_mounts[@]}" "${common_env[@]}" --entrypoint python3 "$RUNTIME_IMAGE_ID" \
    -m putpocket_dataset_mining.glm52_stateful_cli control \
    --mode "$mode" --experiment-id "$EXPERIMENT_ID" --phase-id "$phase" --prompt-side "$side" --ratio "$ratio" \
    --selector /episode/selector/selector.json --donor-history /episode/donor-history-token-ids.json --edited-history /episode/edited-history-token-ids.json \
    --evidence-dir "$evidence" --output "$CONTROL_CONTAINER"
}

write_control SNAPSHOT donor donor 100 "/storage/artifacts/${SLURM_JOB_ID}-${PUTPOCKET_RATIOS//,/-}/runtime/donor"
python3 - "$EPISODE/donor-history-token-ids.json" "$RUN_ROOT/donor-request.json" <<'PY'
import json,pathlib,sys
tokens=json.load(open(sys.argv[1]))['prompt']
pathlib.Path(sys.argv[2]).write_text(json.dumps({'model':'nvidia/GLM-5.2-NVFP4','prompt':tokens,'temperature':0,'top_p':1,'max_tokens':1,'n':1,'seed':0,'stream':False,'return_token_ids':True},separators=(',',':'))+'\n')
PY
curl --fail --silent --show-error -H 'Content-Type: application/json' --data-binary @"$RUN_ROOT/donor-request.json" "http://127.0.0.1:$PORT/v1/completions" > "$RUN_ROOT/donor-response.json" || fail DONOR_SNAPSHOT_REQUEST_FAILED
"$AGENT_PY" - "$RUN_ROOT/runtime/donor" "$EPISODE/episode.json" <<'PY' || fail DONOR_RUNTIME_COVERAGE_INVALID
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); episode=json.load(open(sys.argv[2])); expected=episode['eligible_end']-episode['eligible_start']
records=[]
for path in sorted(root.glob('runtime.rank-*.jsonl')):
 records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
ranks=sorted({int(record['rank']) for record in records})
assert ranks == [0,1,2,3], ranks
products={("mla_kv",layer) for layer in range(78)} | {("indexer_k",layer) for layer in (0,1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74)}
for rank in ranks:
 rows=[record for record in records if int(record['rank'])==rank and record.get('action')=='source_snapshot']
 assert {(record['product'],int(record['layer'])) for record in rows} == products
 assert all(int(record['row_count'])==expected for record in rows)
PY

printf '{"phase_id":"startup","ratio_percent":0,"mode":"replay_then_edit"}\n' > "$PHASE_FILE"
"$AGENT_PY" -m putpocket_dataset_mining.glm52_stateful_proxy --listen-port "$PROXY_PORT" --backend-port "$PORT" --phase-file "$PHASE_FILE" --log "$PROXY_LOG" --capture "$RUN_ROOT/unused-capture.json" --episode "$EPISODE/episode.json" > "$RUN_ROOT/logs/proxy.log" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 60); do curl --fail --silent "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1 && break; kill -0 "$PROXY_PID" >/dev/null 2>&1 || fail ACCURACY_PROXY_EXITED; sleep 1; done
curl --fail --silent "http://127.0.0.1:$PROXY_PORT/health" >/dev/null 2>&1 || fail ACCURACY_PROXY_HEALTH_FAILED

IFS=',' read -r -a RATIOS <<< "$PUTPOCKET_RATIOS"
RATIO_RESULTS=()
for ratio in "${RATIOS[@]}"; do
  printf -v phase 'ratio-%03d' "$ratio"
  ratio_root="$RUN_ROOT/results/$phase"
  evidence_container="/storage/artifacts/${SLURM_JOB_ID}-${PUTPOCKET_RATIOS//,/-}/runtime/$phase"
  mkdir -p "$ratio_root" "$RUN_ROOT/runtime/$phase"
  if (( ratio == 0 )); then write_control OFF "$phase" edited 0 "$evidence_container"; else write_control TRANSPLANT "$phase" edited "$ratio" "$evidence_container"; fi
  python3 - "$PHASE_FILE" "$phase" "$ratio" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); q=p.with_name(p.name+'.partial'); q.write_text(json.dumps({'phase_id':sys.argv[2],'ratio_percent':int(sys.argv[3]),'mode':'replay_then_edit'},separators=(',',':'))+'\n'); q.replace(p)
PY
  OPENAI_API_KEY=local-vllm-no-auth "$AGENT_BIN" swebench \
    --subset "$RUN_ROOT/prepared/mini_dataset" --split test --workers 1 \
    --model openai/nvidia/GLM-5.2-NVFP4 --config "$RUN_ROOT/config/donor-agent.yaml" \
    --environment-class docker --output "$ratio_root/inference" > "$ratio_root/agent.log" 2>&1 || fail "AGENT_INFERENCE_FAILED_${ratio}"
  "$AGENT_PY" -m putpocket_dataset_mining.swebench_pro_cli gather \
    --harness-root "$HARNESS" --inference-root "$ratio_root/inference" \
    --prefix "glm52-reuse-${phase}" --output "$ratio_root/patches.json" > "$ratio_root/gather.log" 2>&1 || fail "PATCH_GATHER_FAILED_${ratio}"
  mkdir -p "$ratio_root/evaluation"
  (cd "$HARNESS" && "$AGENT_PY" "$HARNESS/swe_bench_pro_eval.py" \
    --raw_sample_path "$RUN_ROOT/prepared/raw_samples.jsonl" --patch_path "$ratio_root/patches.json" \
    --output_dir "$ratio_root/evaluation" --scripts_dir "$HARNESS/run_scripts" \
    --num_workers 1 --dockerhub_username jefzda --use_local_docker) > "$ratio_root/evaluator.log" 2>&1 || fail "OFFICIAL_EVALUATOR_FAILED_${ratio}"
  "$AGENT_PY" -m putpocket_dataset_mining.glm52_stateful_cli finalize-ratio \
    --ratio "$ratio" --phase-id "$phase" --episode "$EPISODE/episode.json" --proxy-log "$PROXY_LOG" --evidence-dir "$RUN_ROOT/runtime/$phase" \
    --patches "$ratio_root/patches.json" --eval-results "$ratio_root/evaluation/eval_results.json" \
    --output "$ratio_root/ratio-result.json" || fail "RATIO_FINALIZATION_FAILED_${ratio}"
  RATIO_RESULTS+=("$ratio_root/ratio-result.json")
done
"$AGENT_PY" -m putpocket_dataset_mining.glm52_stateful_cli finalize-sweep --ratio-results "${RATIO_RESULTS[@]}" --expected-ratios "$PUTPOCKET_RATIOS" --output "$RUN_ROOT/results/sweep-report.json" || fail SWEEP_FINALIZATION_FAILED
if [[ $PUTPOCKET_SWEEP_PROFILE == smoke ]]; then
  "$AGENT_PY" - "$RUN_ROOT/results/ratio-000/ratio-result.json" <<'PY' || fail BASELINE_SCENARIO_NOT_SOLVED
import json,sys
raise SystemExit(0 if json.load(open(sys.argv[1]))['official_evaluator_resolved'] is True else 1)
PY
fi
kill "$PROXY_PID" >/dev/null 2>&1 || true
wait "$PROXY_PID" >/dev/null 2>&1 || true
PROXY_PID=
fi

curl --fail --silent "http://127.0.0.1:$PORT/health" > "$RUN_ROOT/server-health-after.txt"
/usr/bin/nvidia-smi -q > "$RUN_ROOT/nvidia-smi-after.txt"
find "$RUN_ROOT" -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/SHA256SUMS"
DURABLE_STAGE="$DURABLE_PARENT/${SLURM_JOB_ID}-${PUTPOCKET_SWEEP_PROFILE}.partial"
DURABLE_ROOT="$DURABLE_PARENT/${SLURM_JOB_ID}-${PUTPOCKET_SWEEP_PROFILE}"
[[ ! -e $DURABLE_STAGE && ! -e $DURABLE_ROOT ]] || fail DURABLE_OUTPUT_ALREADY_EXISTS
mkdir -p "$DURABLE_STAGE"
cp -a "$RUN_ROOT/results" "$RUN_ROOT/runtime" "$RUN_ROOT/config" "$RUN_ROOT/prepared" "$DURABLE_STAGE/"
cp "$RUN_ROOT/exact-server-command.txt" "$RUN_ROOT/SHA256SUMS" "$DURABLE_STAGE/"
[[ -f $RUN_ROOT/proxy-usage.jsonl ]] && cp "$RUN_ROOT/proxy-usage.jsonl" "$DURABLE_STAGE/"
[[ -f $RUN_ROOT/donor-response.json ]] && cp "$RUN_ROOT/donor-response.json" "$DURABLE_STAGE/"
cp "$RUN_ROOT/logs/server.log" "$RUN_ROOT/logs/proxy.log" "$DURABLE_STAGE/"
(cd "$DURABLE_STAGE" && find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
chmod -R a-w "$DURABLE_STAGE"
mv "$DURABLE_STAGE" "$DURABLE_ROOT"
printf 'DURABLE_ACCURACY_OUTPUT=%s\n' "$DURABLE_ROOT"
printf 'COMPLETED_CROSS_ENVIRONMENT_UNSAFE_ABLATION=%s\n' "$RUN_ROOT"
