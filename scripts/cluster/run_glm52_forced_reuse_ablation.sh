#!/usr/bin/env bash
set -euo pipefail
umask 077

fail() { printf 'BLOCKED_%s\n' "$1" >&2; exit "${2:-20}"; }
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [[ -n ${SAMPLER_PID:-} ]]; then kill "$SAMPLER_PID" 2>/dev/null || true; wait "$SAMPLER_PID" 2>/dev/null || true; fi
  if [[ -n ${SERVER_NAME:-} && -n ${CONTAINER:-} ]]; then "$CONTAINER" rm -f "$SERVER_NAME" >/dev/null 2>&1 || true; fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail SLURM_ALLOCATION_REQUIRED
[[ ${SLURM_JOB_NUM_NODES:-0} == 1 ]] || fail EXACTLY_ONE_NODE_REQUIRED
[[ ${SLURM_GPUS_ON_NODE:-0} == 4 ]] || fail EXACTLY_FOUR_GPUS_REQUIRED
[[ -n ${SLURM_JOB_NODELIST:-} && -n ${CUDA_VISIBLE_DEVICES:-} ]] || fail SLURM_GPU_VISIBILITY_MISSING
[[ ${PUTPOCKET_UNSAFE_FORCED_REUSE_ACK:-} == I_UNDERSTAND_ZERO_SAFE_PAGES ]] || fail UNSAFE_ACK_MISSING

for name in PUTPOCKET_PACKAGE_ROOT PUTPOCKET_CONTAINER_EXECUTABLE PUTPOCKET_SHARED_BUILD_ROOT PUTPOCKET_EXPECTED_BUNDLE_KEY PUTPOCKET_H200_STORAGE_PARENT PUTPOCKET_H200_WORK_ROOT PUTPOCKET_RUN_ARTIFACT_ROOT PUTPOCKET_BASELINE_CAPTURE_ROOT PUTPOCKET_EDITED_CAPTURE_ROOT PUTPOCKET_DONOR_PROMPT_TOKENS PUTPOCKET_EDITED_PROMPT_TOKENS PUTPOCKET_SWEEP_PROFILE; do
  [[ -n ${!name:-} ]] || fail "ENV_${name}_MISSING"
done
case "$PUTPOCKET_SWEEP_PROFILE" in
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
BASELINE_CAPTURE=$(realpath "$PUTPOCKET_BASELINE_CAPTURE_ROOT")
EDITED_CAPTURE=$(realpath "$PUTPOCKET_EDITED_CAPTURE_ROOT")
DONOR_PROMPT=$(realpath "$PUTPOCKET_DONOR_PROMPT_TOKENS")
EDITED_PROMPT=$(realpath "$PUTPOCKET_EDITED_PROMPT_TOKENS")
VLLM_ARCHIVE="$PACKAGE/artifacts/vllm-4a3447d200e5aa428d68d1a00aa00f1a19a1a729.tar.gz"
MODEL_REVISION=aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
MODEL="$STORAGE/cache/models/$MODEL_REVISION"

[[ -d $PACKAGE && -x $CONTAINER && -d $BUNDLE && -d $STORAGE_PARENT && -d $STORAGE && -d $ARTIFACT_PARENT ]] || fail PINNED_SITE_PATH_INVALID
[[ -d $BASELINE_CAPTURE && -d $EDITED_CAPTURE && -f $DONOR_PROMPT && -f $EDITED_PROMPT ]] || fail TRACK_B_OR_PROMPT_ARTIFACT_MISSING
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
mkdir -p "$RUN_ROOT"/{logs,source,overlay,selector,runtime,jit/{deepgemm,flashinfer,triton,torchinductor,cuda,vllm,tmp}}
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

GPU_REQUEST="device=$CUDA_VISIBLE_DEVICES"
common_mounts=(
  --volume "$PACKAGE:/project:ro" --volume "$STORAGE:/storage"
  --volume "$RUN_ROOT/overlay:/overlay:ro" --volume "$BASELINE_CAPTURE:/baseline:ro"
  --volume "$EDITED_CAPTURE:/edited:ro" --volume "$DONOR_PROMPT:/inputs/donor.json:ro"
  --volume "$EDITED_PROMPT:/inputs/edited.json:ro"
)
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

"$CONTAINER" run --rm "${common_mounts[@]}" "${common_env[@]}" --entrypoint python3 "$RUNTIME_IMAGE_ID" -m putpocket_dataset_mining.glm52_forced_reuse \
  --baseline-root /baseline --edited-root /edited \
  --baseline-prompt-token-ids /inputs/donor.json --edited-prompt-token-ids /inputs/edited.json \
  --output "/storage/artifacts/${SLURM_JOB_ID}-${PUTPOCKET_RATIOS//,/-}/selector/selector.json" > "$RUN_ROOT/logs/selector.log" 2>&1 || fail SELECTOR_BUILD_FAILED
sha256sum "$RUN_ROOT/selector/selector.json" > "$RUN_ROOT/selector/selector.json.sha256"

CONTROL_CONTAINER="/storage/artifacts/${SLURM_JOB_ID}-${PUTPOCKET_RATIOS//,/-}/control.json"
SELECTOR_CONTAINER="/storage/artifacts/${SLURM_JOB_ID}-${PUTPOCKET_RATIOS//,/-}/selector/selector.json"
"$CONTAINER" run --rm "${common_mounts[@]}" "${common_env[@]}" --entrypoint python3 "$RUNTIME_IMAGE_ID" - "$CONTROL_CONTAINER" "$SELECTOR_CONTAINER" <<'PY'
import hashlib,json,pathlib,sys
def ids(path):
 x=json.load(open(path)); return x.get('prompt',x.get('prompt_token_ids',x.get('token_ids',x))) if isinstance(x,dict) else x
donor=ids('/inputs/donor.json'); edited=ids('/inputs/edited.json')
digest=lambda x:hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
selector_sha=hashlib.sha256(open(sys.argv[2],'rb').read()).hexdigest()
p={"schema_version":1,"mode":"OFF","experiment_id":"glm52-forced-reuse","phase_id":"startup","prompt_side":"edited","unsafe_forced_reuse_ack":"I_UNDERSTAND_ZERO_SAFE_PAGES","production_default_enabled":False,"prompt_token_count":2071,"edit_position":114,"real_block_size":64,"main_layers":list(range(78)),"indexer_layers":[0,1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74],"donor_prompt_token_ids_sha256":digest(donor),"edited_prompt_token_ids_sha256":digest(edited),"selector_source_baseline_digest":digest(donor),"selector_source_edited_digest":digest(edited),"selector_path":sys.argv[2],"selector_sha256":selector_sha,"requested_ratio_percent":0,"evidence_dir":str(pathlib.Path(sys.argv[1]).parent/'runtime/off')}
pathlib.Path(sys.argv[1]).write_text(json.dumps(p,separators=(',',':'),sort_keys=True)+'\n')
PY

PORT=$((20000 + SLURM_JOB_ID % 20000))
SERVER_NAME="pp-glm52-forced-${SLURM_JOB_ID}"
SERVER_LOG="$RUN_ROOT/logs/server.log"
server_args=(
  serve /model --served-model-name nvidia/GLM-5.2-NVFP4 --revision "$MODEL_REVISION"
  --tensor-parallel-size 4 --pipeline-parallel-size 1 --quantization modelopt_fp4
  --linear-backend marlin --attention-backend FLASHMLA_SPARSE --kv-cache-dtype bfloat16
  --max-model-len 4096 --max-num-seqs 1 --cpu-offload-gb 0 --swap-space 0
  --no-enable-prefix-caching --enforce-eager --jit-monitor-mode warn --jit-monitor-verbose
  --host 127.0.0.1 --port "$PORT"
)
printf '%q ' vllm "${server_args[@]}" > "$RUN_ROOT/exact-server-command.txt"; printf '\n' >> "$RUN_ROOT/exact-server-command.txt"
"$CONTAINER" run --rm --name "$SERVER_NAME" --gpus "$GPU_REQUEST" --ipc=host --network=host \
  "${common_mounts[@]}" "${common_env[@]}" --volume "$MODEL:/model:ro" \
  --env PUTPOCKET_GLM52_FORCED_REUSE_CONTROL="$CONTROL_CONTAINER" \
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

"$CONTAINER" run --rm --network=host "${common_mounts[@]}" "${common_env[@]}" --entrypoint python3 "$RUNTIME_IMAGE_ID" -m putpocket_dataset_mining.glm52_forced_reuse_sweep \
  --endpoint "http://127.0.0.1:$PORT" --control "$CONTROL_CONTAINER" --selector "$SELECTOR_CONTAINER" \
  --donor-prompt /inputs/donor.json --edited-prompt /inputs/edited.json \
  --output-root "/storage/artifacts/${SLURM_JOB_ID}-${PUTPOCKET_RATIOS//,/-}/results" \
  --experiment-id "glm52-forced-reuse-${SLURM_JOB_ID}" --ratios "$PUTPOCKET_RATIOS" \
  --tp-size 4 --max-tokens 512 > "$RUN_ROOT/logs/sweep.log" 2>&1 || fail SWEEP_FAILED

curl --fail --silent "http://127.0.0.1:$PORT/health" > "$RUN_ROOT/server-health-after.txt"
/usr/bin/nvidia-smi -q > "$RUN_ROOT/nvidia-smi-after.txt"
find "$RUN_ROOT" -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/SHA256SUMS"
printf 'COMPLETED_CROSS_ENVIRONMENT_UNSAFE_ABLATION=%s\n' "$RUN_ROOT"
