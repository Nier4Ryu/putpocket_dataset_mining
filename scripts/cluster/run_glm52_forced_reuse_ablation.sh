#!/usr/bin/env bash
set -euo pipefail
umask 077

fail() {
  printf 'BLOCKED_%s\n' "$1" >&2
  exit "${2:-20}"
}

[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || fail SLURM_ALLOCATION_REQUIRED
[[ ${SLURM_JOB_NUM_NODES:-0} == 1 ]] || fail EXACTLY_ONE_NODE_REQUIRED
[[ ${SLURM_GPUS_ON_NODE:-0} == 4 ]] || fail EXACTLY_FOUR_GPUS_REQUIRED
[[ -n ${CUDA_VISIBLE_DEVICES:-} ]] || fail CUDA_ALLOCATION_NOT_VISIBLE
[[ ${PUTPOCKET_UNSAFE_FORCED_REUSE_ACK:-} == I_UNDERSTAND_ZERO_SAFE_PAGES ]] || fail UNSAFE_ACK_MISSING

for name in PUTPOCKET_PACKAGE_ROOT PUTPOCKET_VLLM_SOURCE PUTPOCKET_RUNTIME_PYTHON PUTPOCKET_MODEL_PATH PUTPOCKET_BASELINE_CAPTURE_ROOT PUTPOCKET_EDITED_CAPTURE_ROOT PUTPOCKET_DONOR_PROMPT_TOKENS PUTPOCKET_EDITED_PROMPT_TOKENS PUTPOCKET_ARTIFACT_ROOT PUTPOCKET_RATIOS; do
  [[ -n ${!name:-} ]] || fail "ENV_${name}_MISSING"
done

PACKAGE=$(realpath "$PUTPOCKET_PACKAGE_ROOT")
SOURCE=$(realpath "$PUTPOCKET_VLLM_SOURCE")
PYTHON=$(realpath "$PUTPOCKET_RUNTIME_PYTHON")
MODEL=$(realpath "$PUTPOCKET_MODEL_PATH")
ARTIFACT_PARENT=$(realpath "$PUTPOCKET_ARTIFACT_ROOT")
[[ -d $PACKAGE && -d $SOURCE && -x $PYTHON && -d $MODEL && -d $ARTIFACT_PARENT ]] || fail PINNED_ASSET_PATH_INVALID
[[ $(git -C "$SOURCE" rev-parse HEAD) == 4a3447d200e5aa428d68d1a00aa00f1a19a1a729 ]] || fail VLLM_COMMIT_MISMATCH
[[ -f $MODEL/config.json ]] || fail MODEL_CONFIG_MISSING
if [[ $(basename "$MODEL") != aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa ]]; then
  [[ -f $MODEL/.putpocket_model_revision ]] || fail MODEL_REVISION_ATTESTATION_MISSING
  [[ $(tr -d '\r\n' < "$MODEL/.putpocket_model_revision") == aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa ]] || fail MODEL_REVISION_MISMATCH
fi
[[ $(sha256sum "$MODEL/tokenizer.json" | awk '{print $1}') == 19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d ]] || fail TOKENIZER_JSON_DIGEST_MISMATCH
[[ $(sha256sum "$MODEL/tokenizer_config.json" | awk '{print $1}') == 77af7d4769cd62c107b90495cac9b0ba81573c86486821bfba2980c04285ec7a ]] || fail TOKENIZER_CONFIG_DIGEST_MISMATCH
[[ $(sha256sum "$MODEL/chat_template.jinja" | awk '{print $1}') == 172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679 ]] || fail CHAT_TEMPLATE_DIGEST_MISMATCH
[[ -f $MODEL/model.safetensors.index.json ]] || fail MODEL_WEIGHT_INDEX_MISSING

mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=name,memory.total,memory.free,compute_cap --format=csv,noheader,nounits)
[[ ${#GPU_LINES[@]} == 4 ]] || fail GPU_COUNT_MISMATCH
for line in "${GPU_LINES[@]}"; do
  [[ $line == NVIDIA\ H200* ]] || fail GPU_MODEL_MISMATCH
  memory_total=$(awk -F, '{gsub(/ /,"",$2); print $2}' <<<"$line")
  memory_free=$(awk -F, '{gsub(/ /,"",$3); print $3}' <<<"$line")
  compute_cap=$(awk -F, '{gsub(/ /,"",$4); print $4}' <<<"$line")
  (( memory_total >= 140000 && memory_free >= 135000 )) || fail GPU_HBM_PREFLIGHT_FAILED
  [[ $compute_cap == 9.0 ]] || fail GPU_SM90_REQUIRED
done
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
(( available_kib >= 32 * 1024 * 1024 )) || fail HOST_RAM_32G_GATE_FAILED

RUN_ROOT="$ARTIFACT_PARENT/${SLURM_JOB_ID}-${PUTPOCKET_RATIOS//,/-}"
[[ ! -e $RUN_ROOT ]] || fail RUN_ROOT_ALREADY_EXISTS
mkdir -p "$RUN_ROOT"/{logs,source,responses,runtime,selector}
cp -a "$SOURCE/." "$RUN_ROOT/source/vllm"
git -C "$RUN_ROOT/source/vllm" reset --hard 4a3447d200e5aa428d68d1a00aa00f1a19a1a729 >/dev/null
git -C "$RUN_ROOT/source/vllm" clean -ffd >/dev/null
git -C "$RUN_ROOT/source/vllm" apply "$PACKAGE/patches/vllm/4a3447d200e5aa428d68d1a00aa00f1a19a1a729/glm52_forced_edit_reuse.patch"
cp "$PACKAGE/instrumentation/vllm/glm52_forced_edit_reuse.py" "$RUN_ROOT/source/vllm/vllm/model_executor/layers/glm52_forced_edit_reuse.py"

export PYTHONPATH="$PACKAGE/src:$RUN_ROOT/source/vllm${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$RUN_ROOT/hf-cache"
export VLLM_CACHE_ROOT="$RUN_ROOT/vllm-cache"
export TRITON_CACHE_DIR="$RUN_ROOT/triton-cache"
export DEEP_GEMM_CACHE_DIR="$RUN_ROOT/deepgemm-cache"
export TMPDIR="$RUN_ROOT/tmp"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$DEEP_GEMM_CACHE_DIR" "$TMPDIR"
unset PUTPOCKET_VLLM_DSA_TRACE_CONTROL PUTPOCKET_VLLM_DSA_TRACE_ROOT

"$PYTHON" -m putpocket_dataset_mining.glm52_forced_reuse \
  --baseline-root "$PUTPOCKET_BASELINE_CAPTURE_ROOT" \
  --edited-root "$PUTPOCKET_EDITED_CAPTURE_ROOT" \
  --baseline-prompt-token-ids "$PUTPOCKET_DONOR_PROMPT_TOKENS" \
  --edited-prompt-token-ids "$PUTPOCKET_EDITED_PROMPT_TOKENS" \
  --output "$RUN_ROOT/selector/selector.json" | tee "$RUN_ROOT/logs/selector.log"
sha256sum "$RUN_ROOT/selector/selector.json" > "$RUN_ROOT/selector/selector.json.sha256"

CONTROL="$RUN_ROOT/control.json"
SELECTOR_SHA=$(awk '{print $1}' "$RUN_ROOT/selector/selector.json.sha256")
DONOR_SHA=$("$PYTHON" -c 'import hashlib,json,sys; x=json.load(open(sys.argv[1])); x=x.get("prompt",x.get("prompt_token_ids",x.get("token_ids",x))) if isinstance(x,dict) else x; print(hashlib.sha256(json.dumps(x,separators=(",", ":")).encode()).hexdigest())' "$PUTPOCKET_DONOR_PROMPT_TOKENS")
EDITED_SHA=$("$PYTHON" -c 'import hashlib,json,sys; x=json.load(open(sys.argv[1])); x=x.get("prompt",x.get("prompt_token_ids",x.get("token_ids",x))) if isinstance(x,dict) else x; print(hashlib.sha256(json.dumps(x,separators=(",", ":")).encode()).hexdigest())' "$PUTPOCKET_EDITED_PROMPT_TOKENS")
"$PYTHON" - "$CONTROL" "$RUN_ROOT/selector/selector.json" "$SELECTOR_SHA" "$DONOR_SHA" "$EDITED_SHA" "$RUN_ROOT/runtime/off" <<'PY'
import json,sys
from pathlib import Path
control,selector,selector_sha,donor,edited,evidence=sys.argv[1:]
payload={"schema_version":1,"mode":"OFF","experiment_id":"glm52-forced-reuse","phase_id":"startup","prompt_side":"edited","unsafe_forced_reuse_ack":"I_UNDERSTAND_ZERO_SAFE_PAGES","production_default_enabled":False,"prompt_token_count":2071,"edit_position":114,"real_block_size":64,"main_layers":list(range(78)),"indexer_layers":[0,1,2,6,10,14,18,22,26,30,34,38,42,46,50,54,58,62,66,70,74],"donor_prompt_token_ids_sha256":donor,"edited_prompt_token_ids_sha256":edited,"selector_source_baseline_digest":donor,"selector_source_edited_digest":edited,"selector_path":str(Path(selector).resolve()),"selector_sha256":selector_sha,"requested_ratio_percent":0,"evidence_dir":str(Path(evidence).resolve())}
Path(control).write_text(json.dumps(payload,separators=(",", ":"),sort_keys=True)+"\n")
PY

PORT=$((20000 + SLURM_JOB_ID % 20000))
export PUTPOCKET_GLM52_FORCED_REUSE_CONTROL="$CONTROL"
export PUTPOCKET_SERVED_MODEL_NAME=nvidia/GLM-5.2-NVFP4
export VLLM_ATTENTION_BACKEND=FLASHMLA_SPARSE
SERVER_LOG="$RUN_ROOT/logs/server.log"
"$PYTHON" -m vllm.entrypoints.openai.api_server \
  --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
  --served-model-name "$PUTPOCKET_SERVED_MODEL_NAME" \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 \
  --quantization modelopt_fp4 --kv-cache-dtype bfloat16 \
  --attention-backend FLASHMLA_SPARSE \
  --max-model-len 4096 --max-num-seqs 1 --enforce-eager \
  --no-enable-prefix-caching --cpu-offload-gb 0 --swap-space 0 \
  --trust-remote-code >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

"$PYTHON" -m putpocket_dataset_mining.glm52_forced_reuse_sweep \
  --endpoint "http://127.0.0.1:$PORT" --control "$CONTROL" \
  --selector "$RUN_ROOT/selector/selector.json" \
  --donor-prompt "$PUTPOCKET_DONOR_PROMPT_TOKENS" \
  --edited-prompt "$PUTPOCKET_EDITED_PROMPT_TOKENS" \
  --output-root "$RUN_ROOT/results" \
  --experiment-id "glm52-forced-reuse-${SLURM_JOB_ID}" \
  --ratios "$PUTPOCKET_RATIOS" --tp-size 4 --max-tokens 512 \
  | tee "$RUN_ROOT/logs/sweep.log"

curl --fail --silent "http://127.0.0.1:$PORT/health" > "$RUN_ROOT/server-health-after.txt"
nvidia-smi -q > "$RUN_ROOT/nvidia-smi-after.txt"
find "$RUN_ROOT" -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$RUN_ROOT/SHA256SUMS"
printf 'COMPLETED_CROSS_ENVIRONMENT_UNSAFE_ABLATION=%s\n' "$RUN_ROOT"
