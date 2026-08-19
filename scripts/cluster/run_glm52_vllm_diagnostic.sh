#!/usr/bin/env bash
set -euo pipefail
umask 077

PROJECT_COMMIT=${1:-}
SOURCE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
LOCK="$SOURCE_ROOT/configs/cluster/glm52_vllm_diagnostic.lock.json"
CONTAINER=${PUTPOCKET_CONTAINER_EXECUTABLE:-}
SHARED_ROOT=${PUTPOCKET_SHARED_BUILD_ROOT:-}
BUNDLE_KEY=${PUTPOCKET_EXPECTED_BUNDLE_KEY:-}
NVIDIA_SMI=${PUTPOCKET_NVIDIA_SMI:-/usr/bin/nvidia-smi}
BUNDLE="$SHARED_ROOT/$BUNDLE_KEY"
STORAGE_PARENT=${PUTPOCKET_H200_STORAGE_PARENT:-}
STORAGE=${PUTPOCKET_H200_WORK_ROOT:-}
ARTIFACT_ROOT=${PUTPOCKET_RUN_ARTIFACT_ROOT:-}
RUN_ROOT=
CACHE=
EPHEMERAL=
MODEL_REVISION=aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa
MODEL_ROOT=
HARNESS_COMMIT=ca10a60a5fcae51e6948ffe1485d4153d421e6c5
HARNESS=
PORT=8000
SERVER_NAME="pp-vllm-glm52-${SLURM_JOB_ID:-unknown}"
AGENT_NAME="pp-vllm-agent-${SLURM_JOB_ID:-unknown}"
SERVER_PID=
SAMPLER_PID=
PHASE=allocation_inventory
STATUS=FAIL
FAILURE_CLASS=UNCLASSIFIED_FAILURE

fail() { FAILURE_CLASS=$1; printf '%s\n' "$1" >&2; exit "${2:-2}"; }
stop_sampler() { if [[ -n $SAMPLER_PID ]]; then kill "$SAMPLER_PID" >/dev/null 2>&1 || true; wait "$SAMPLER_PID" >/dev/null 2>&1 || true; SAMPLER_PID=; fi; }
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  stop_sampler
  "$CONTAINER" rm -f "$SERVER_NAME" "$AGENT_NAME" >/dev/null 2>&1 || true
  if [[ -n $SERVER_PID ]]; then wait "$SERVER_PID" >/dev/null 2>&1 || true; fi
  if [[ $STATUS != PASS && -n $RUN_ROOT ]]; then
    mkdir -p "$RUN_ROOT" 2>/dev/null || true
    printf '{"schema_version":1,"status":"%s","phase":"%s","failure_class":"%s","returncode":%d,"quality_score_eligible":false,"full_selection_reachable":false,"fallback_attempted":false,"offload_attempted":false}\n' "$STATUS" "$PHASE" "$FAILURE_CLASS" "$rc" > "$RUN_ROOT/diagnostic_manifest.json.partial" 2>/dev/null || true
    mv "$RUN_ROOT/diagnostic_manifest.json.partial" "$RUN_ROOT/diagnostic_manifest.json" 2>/dev/null || true
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM

[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_JOB_NUM_NODES:-0} == 1 ]] || fail SLURM_ALLOCATION_REQUIRED 20
[[ ${SLURM_GPUS_ON_NODE:-0} == 4 ]] || fail SLURM_GPU_COUNT_MISMATCH 20
[[ $PROJECT_COMMIT =~ ^[0-9a-f]{40}$ && $(git -C "$SOURCE_ROOT" rev-parse HEAD) == "$PROJECT_COMMIT" ]] || fail PROJECT_COMMIT_MISMATCH 21
[[ -n $CONTAINER && -x $CONTAINER && -n $SHARED_ROOT && -n $BUNDLE_KEY && -x $NVIDIA_SMI && $STORAGE_PARENT == /* && $STORAGE == /* && $ARTIFACT_ROOT == /* ]] || fail RUN_SITE_CONFIGURATION_MISSING 21
"$CONTAINER" info >/dev/null 2>&1 || fail CONTAINER_RUNTIME_UNAVAILABLE 21
case "$STORAGE/" in "$STORAGE_PARENT"/*) ;; *) fail COMPUTE_WORK_ROOT_OUTSIDE_STORAGE_PARENT 21 ;; esac
[[ $STORAGE != "$STORAGE_PARENT" && $ARTIFACT_ROOT == "$STORAGE/artifacts" ]] || fail COMPUTE_ARTIFACT_ROOT_INVALID 21
[[ -d $STORAGE_PARENT && -w $STORAGE_PARENT ]] || fail COMPUTE_LOCAL_STORAGE_PARENT_UNWRITABLE 21
mkdir -p "$STORAGE" "$ARTIFACT_ROOT"
[[ -d $STORAGE && -w $STORAGE && -d $ARTIFACT_ROOT && -w $ARTIFACT_ROOT ]] || fail COMPUTE_RUN_ROOT_UNWRITABLE 21
RUN_ROOT="$ARTIFACT_ROOT/${SLURM_JOB_ID:-unknown}"
CACHE="$STORAGE/cache"
EPHEMERAL="$STORAGE/tmp/${SLURM_JOB_ID:-unknown}"
MODEL_ROOT="$CACHE/models/$MODEL_REVISION"
HARNESS="$CACHE/harness/$HARNESS_COMMIT"
mkdir -p "$RUN_ROOT/phase0" "$RUN_ROOT/phase1" "$RUN_ROOT/phase2" "$RUN_ROOT/phase3/native_raw" "$RUN_ROOT/official" "$CACHE" "$EPHEMERAL"
export PYTHONPATH="$SOURCE_ROOT/src"
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli validate-lock --lock "$LOCK" > "$RUN_ROOT/lock_validation.json"

GPU_SELECTOR=${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-}}
[[ -n $GPU_SELECTOR ]] || fail GPU_VISIBILITY_UNSPECIFIED 22
IFS=',' read -r -a GPU_IDS <<< "$GPU_SELECTOR"
[[ ${#GPU_IDS[@]} == 4 ]] || fail GPU_VISIBILITY_COUNT_MISMATCH 22
for value in "${GPU_IDS[@]}"; do [[ $value =~ ^([0-9]+|GPU-[A-Za-z0-9-]+)$ ]] || fail GPU_SELECTOR_INVALID 22; done
printf 'index,uuid,name,memory_total_mib,memory_free_mib,mig_mode,compute_capability\n' > "$RUN_ROOT/phase0/gpu_inventory.csv"
"$NVIDIA_SMI" --id="$GPU_SELECTOR" --query-gpu=index,uuid,name,memory.total,memory.free,mig.mode.current,compute_cap --format=csv,noheader,nounits >> "$RUN_ROOT/phase0/gpu_inventory.csv" || fail GPU_INVENTORY_QUERY_FAILED 22
"$NVIDIA_SMI" -L > "$RUN_ROOT/phase0/nvidia_smi_listing.txt" || fail GPU_LISTING_QUERY_FAILED 22
"$NVIDIA_SMI" topo -m > "$RUN_ROOT/phase0/topology.txt" || fail GPU_TOPOLOGY_QUERY_FAILED 22
"$NVIDIA_SMI" nvlink --status -i "$GPU_SELECTOR" > "$RUN_ROOT/phase0/nvlink_status.txt" || fail NVLINK_QUERY_FAILED 22
"$NVIDIA_SMI" --query-gpu=uuid,driver_version --format=csv,noheader > "$RUN_ROOT/phase0/driver_versions.csv" || fail DRIVER_QUERY_FAILED 22
if command -v nvcc >/dev/null 2>&1; then nvcc --version > "$RUN_ROOT/phase0/host_nvcc_version.txt"; else printf 'host nvcc unavailable; immutable runtime nvcc is validated before model metadata\n' > "$RUN_ROOT/phase0/host_nvcc_version.txt"; fi
scontrol show job "$SLURM_JOB_ID" -o > "$RUN_ROOT/phase0/slurm_job.txt" || fail SLURM_JOB_INSPECTION_FAILED 23
grep -Eqi 'gres/gpu:H200=4|gres:gpu:H200:4' "$RUN_ROOT/phase0/slurm_job.txt" || fail SLURM_TYPED_GPU_TRES_MISMATCH 23
for variable in SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_NODELIST SLURM_JOB_NUM_NODES SLURM_NNODES SLURM_GPUS_ON_NODE SLURM_JOB_GPUS SLURM_CPUS_PER_TASK SLURM_MEM_PER_NODE SLURM_JOB_PARTITION CUDA_VISIBLE_DEVICES; do printf '%s=%s\n' "$variable" "${!variable-}"; done > "$RUN_ROOT/phase0/slurm_environment.txt"
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli validate-inventory --csv "$RUN_ROOT/phase0/gpu_inventory.csv" --listing "$RUN_ROOT/phase0/nvidia_smi_listing.txt" --output "$RUN_ROOT/phase0/inventory_manifest.json" || fail ALLOCATION_INVENTORY_MISMATCH 23
printf 'passed\n' > "$RUN_ROOT/phase0/PASSED"

PHASE=immutable_build_bundle
[[ -f $BUNDLE/SUCCESS && -f $BUNDLE/build_manifest.json ]] || fail BUILD_BUNDLE_MISSING 30
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli validate-build-bundle --lock "$LOCK" --bundle-root "$BUNDLE" > "$RUN_ROOT/phase1/build_bundle_validation.json" || fail BUILD_BUNDLE_INVALID 30
"$CONTAINER" load --input "$BUNDLE/runtime-image.tar" > "$RUN_ROOT/phase1/image_load.log" 2>&1 || fail RUNTIME_IMAGE_LOAD_FAILED 30
RUNTIME_IMAGE_ID=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["runtime_image_id"])' "$BUNDLE/build_manifest.json")
[[ $RUNTIME_IMAGE_ID == sha256:* ]] || fail RUNTIME_IMAGE_ID_INVALID 30
"$CONTAINER" image inspect "$RUNTIME_IMAGE_ID" --format '{{.Id}} {{.Architecture}}' > "$RUN_ROOT/phase1/runtime_image_identity.txt"
grep -Fq "$RUNTIME_IMAGE_ID" "$RUN_ROOT/phase1/runtime_image_identity.txt" || fail RUNTIME_IMAGE_ID_MISMATCH 30
COMPILED_IMPORT_PROBE_LOG="$RUN_ROOT/phase1/compiled_import_probe.log"
printf 'PROBE_STEP_START=container_runtime\n' > "$COMPILED_IMPORT_PROBE_LOG"
set +e
"$CONTAINER" run --rm --gpus "device=$GPU_SELECTOR" --entrypoint /bin/bash "$RUNTIME_IMAGE_ID" -lc 'set -euo pipefail
printf "PROBE_STEP_START=runtime_nvcc\n"
if nvcc --version; then
  printf "PROBE_STEP_PASS=runtime_nvcc\n"
else
  rc=$?
  printf "PROBE_STEP_FAILED=runtime_nvcc exit_code=%d\n" "$rc"
  exit "$rc"
fi
python3 - <<"PY"
import importlib
import traceback


def probe(label, operation):
    print(f"PROBE_STEP_START={label}", flush=True)
    try:
        value = operation()
    except BaseException as exc:
        print(
            f"PROBE_STEP_FAILED={label} exception_type={type(exc).__name__} exception={exc}",
            flush=True,
        )
        traceback.print_exc()
        raise
    print(f"PROBE_STEP_PASS={label}", flush=True)
    return value


def imported_symbol(module_name, symbol_name):
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def require_sm90_device():
    capability = torch.cuda.get_device_capability(0)
    if tuple(capability) != (9, 0):
        raise RuntimeError(f"CUDA_DEVICE_CAPABILITY_MISMATCH:{capability!r}")
    return capability


torch = probe("import_torch", lambda: importlib.import_module("torch"))
vllm = probe("import_vllm", lambda: importlib.import_module("vllm"))
probe("import_vllm_C", lambda: importlib.import_module("vllm._C"))
sparse_attn_indexer = probe(
    "import_sparse_attn_indexer",
    lambda: imported_symbol(
        "vllm.model_executor.layers.sparse_attn_indexer", "sparse_attn_indexer"
    ),
)
modelopt_method = probe(
    "import_modelopt_nvfp4_w4a16",
    lambda: imported_symbol(
        "vllm.model_executor.layers.quantization.modelopt",
        "ModelOptNvFp4W4A16LinearMethod",
    ),
)
capture_hook = probe(
    "import_native_dsa_capture",
    lambda: imported_symbol(
        "vllm.model_executor.layers.vllm_dsa_diagnostic_dump",
        "maybe_capture_native_dsa",
    ),
)
capability = probe("validate_sm90_device_capability", require_sm90_device)
print(
    "PROBE_SUMMARY "
    f"torch={torch.__version__} torch_cuda={torch.version.cuda} "
    f"vllm={vllm.__version__} capability={capability} "
    f"symbols={sparse_attn_indexer.__name__},{modelopt_method.__name__},{capture_hook.__name__}",
    flush=True,
)
PY' >> "$COMPILED_IMPORT_PROBE_LOG" 2>&1
COMPILED_IMPORT_PROBE_RC=$?
set -e
if (( COMPILED_IMPORT_PROBE_RC != 0 )); then
  printf 'PROBE_STEP_FAILED=container_runtime exit_code=%d\n' "$COMPILED_IMPORT_PROBE_RC" >> "$COMPILED_IMPORT_PROBE_LOG"
  printf 'COMPILED_SM90_IMPORT_PROBE_LOG_BEGIN path=%s exit_code=%d\n' "$COMPILED_IMPORT_PROBE_LOG" "$COMPILED_IMPORT_PROBE_RC" >&2
  cat "$COMPILED_IMPORT_PROBE_LOG" >&2
  printf 'COMPILED_SM90_IMPORT_PROBE_LOG_END\n' >&2
  fail COMPILED_SM90_IMPORT_PROBE_FAILED 31
fi
printf 'PROBE_STEP_PASS=container_runtime\n' >> "$COMPILED_IMPORT_PROBE_LOG"
grep -Fq 'release 13.0' "$COMPILED_IMPORT_PROBE_LOG" || fail RUNTIME_CUDA_13_0_MISMATCH 31

container_env=(
  --env PYTHONPATH=/project/src --env HOME=/storage/home --env HF_HOME=/storage/cache/huggingface
  --env TRANSFORMERS_CACHE=/storage/cache/transformers --env XDG_CACHE_HOME=/storage/cache/xdg
  --env TORCH_HOME=/storage/cache/torch
  --env PUTPOCKET_VLLM_DSA_TRACE_CONTROL=/storage/artifacts/"$SLURM_JOB_ID"/phase3/trace_control.json
  --env PUTPOCKET_VLLM_DSA_TRACE_ROOT=/storage/artifacts/"$SLURM_JOB_ID"/phase3/native_raw
  --volume "$SOURCE_ROOT:/project:ro" --volume "$STORAGE:/storage"
)
mkdir -p "$CACHE/model-metadata/$SLURM_JOB_ID"
"$CONTAINER" run --rm --gpus "device=$GPU_SELECTOR" "${container_env[@]}" --entrypoint python3 "$RUNTIME_IMAGE_ID" -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli phase1 --lock /project/configs/cluster/glm52_vllm_diagnostic.lock.json --metadata-root /storage/cache/model-metadata/"$SLURM_JOB_ID" --artifact-root /storage/artifacts/"$SLURM_JOB_ID"/phase1 > "$RUN_ROOT/phase1/weightless_probe.log" 2>&1 || fail WEIGHTLESS_VLLM_COMPATIBILITY_FAILED 32
printf 'passed\n' > "$RUN_ROOT/phase1/PASSED"

PHASE=runtime_jit_provenance_preflight
JIT_ROOT="$RUN_ROOT/phase2/runtime_jit"
JIT_CACHE="$JIT_ROOT/cache"
AUDIT_CUDA="$RUN_ROOT/phase1/audit-cuda"
AUDIT_LOG="$JIT_ROOT/compiler_audit.jsonl"
[[ ! -e $JIT_ROOT ]] || fail RUNTIME_JIT_CACHE_REUSE_FORBIDDEN 40
mkdir -p "$JIT_CACHE/deep_gemm" "$JIT_CACHE/flashinfer" "$JIT_CACHE/triton" "$JIT_CACHE/torchinductor" "$JIT_CACHE/cuda" "$JIT_CACHE/vllm" "$JIT_CACHE/tmp" "$AUDIT_CUDA/bin"
[[ $(sha256sum "$SOURCE_ROOT/instrumentation/vllm/compiler_audit.sh" | cut -d' ' -f1) == e060c0b09e1c2eb4da90854ee81d284f7f6acce2b7375b9ee87ecf956a78f9ee ]] || fail COMPILER_AUDIT_DIGEST_MISMATCH 40
for tool in nvcc ptxas cc gcc c++ g++; do install -m 0755 "$SOURCE_ROOT/instrumentation/vllm/compiler_audit.sh" "$AUDIT_CUDA/bin/$tool"; done
ln -s /usr/local/cuda/include "$AUDIT_CUDA/include"
ln -s /usr/local/cuda/lib64 "$AUDIT_CUDA/lib64"
AUDIT_CUDA_CONTAINER="/storage/artifacts/$SLURM_JOB_ID/phase1/audit-cuda"
JIT_ROOT_CONTAINER="/storage/artifacts/$SLURM_JOB_ID/phase2/runtime_jit"
jit_env=(
  --env "PATH=$AUDIT_CUDA_CONTAINER/bin:/opt/venv/bin:/opt/uv/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  --env "CUDA_HOME=$AUDIT_CUDA_CONTAINER" --env "CUDACXX=$AUDIT_CUDA_CONTAINER/bin/nvcc" --env "CUDA_NVCC_EXECUTABLE=$AUDIT_CUDA_CONTAINER/bin/nvcc"
  --env "CC=$AUDIT_CUDA_CONTAINER/bin/gcc" --env "CXX=$AUDIT_CUDA_CONTAINER/bin/g++"
  --env "PUTPOCKET_COMPILER_AUDIT_LOG=$JIT_ROOT_CONTAINER/compiler_audit.jsonl"
  --env "VLLM_CACHE_ROOT=$JIT_ROOT_CONTAINER/cache/vllm" --env "DG_JIT_CACHE_DIR=$JIT_ROOT_CONTAINER/cache/deep_gemm"
  --env "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$JIT_ROOT_CONTAINER/cache/flashinfer"
  --env "TRITON_CACHE_DIR=$JIT_ROOT_CONTAINER/cache/triton" --env "TORCHINDUCTOR_CACHE_DIR=$JIT_ROOT_CONTAINER/cache/torchinductor"
  --env "CUDA_CACHE_PATH=$JIT_ROOT_CONTAINER/cache/cuda" --env "TMPDIR=$JIT_ROOT_CONTAINER/cache/tmp"
)
python3 - "$BUNDLE/build_manifest.json" "$JIT_ROOT/jit_policy.json" "$PROJECT_COMMIT" "$RUNTIME_IMAGE_ID" "$JIT_ROOT_CONTAINER/cache" <<'PY'
import json,pathlib,sys
manifest=json.load(open(sys.argv[1]))
if manifest.get('general_h200_compilation_allowed') is not False or manifest.get('h200_runtime_jit_scope')!='native_first_use_deepgemm_dsa_only' or manifest.get('runtime_jit_cache_reuse') is not False or manifest.get('runtime_gate')!='ALLOW_NATIVE_FIRST_USE_JIT_WITH_RUN_LOCAL_AUDIT': raise SystemExit('RUNTIME_JIT_POLICY_MANIFEST_INVALID')
payload={'schema_version':1,'status':'armed','scope':'native_first_use_deepgemm_dsa_only','project_commit':sys.argv[3],'vllm_commit':manifest['vllm_commit'],'patch_sha256':manifest['patch_sha256'],'base_image':manifest['base_image'],'runtime_image_id':sys.argv[4],'torch':manifest['torch'],'cuda':manifest['cuda'],'sm':'90','cache_root':sys.argv[5],'cache_reuse':False,'general_project_compilation_allowed':False,'compiler_audit_sha256':manifest['compiler_audit_sha256'],'evidence':manifest['runtime_jit_evidence']}
pathlib.Path(sys.argv[2]).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY

PHASE=model_download_and_tp4_load
mkdir -p "$MODEL_ROOT"
if [[ ! -f $MODEL_ROOT/.putpocket_model_revision ]]; then
  "$CONTAINER" run --rm "${container_env[@]}" --entrypoint python3 "$RUNTIME_IMAGE_ID" - "$MODEL_REVISION" <<'PY' > "$RUN_ROOT/phase2/model_download.log" 2>&1
import pathlib,sys
from huggingface_hub import snapshot_download
revision=sys.argv[1]
target=pathlib.Path('/storage/cache/models')/revision
snapshot_download(repo_id='nvidia/GLM-5.2-NVFP4',revision=revision,local_dir=target)
(target/'.putpocket_model_revision').write_text(revision+'\n')
PY
fi
[[ $(tr -d '\r\n' < "$MODEL_ROOT/.putpocket_model_revision") == "$MODEL_REVISION" ]] || fail CHECKPOINT_REVISION_MISMATCH 40

sample_hbm() {
  file=$1; printf 'timestamp,uuid,memory_total_mib,memory_used_mib,memory_free_mib\n' > "$file"
  while true; do timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ); while IFS= read -r line; do printf '%s,%s\n' "$timestamp" "$line" >> "$file"; done < <("$NVIDIA_SMI" --id="$GPU_SELECTOR" --query-gpu=uuid,memory.total,memory.used,memory.free --format=csv,noheader,nounits); sleep 2; done
}
LOAD_HBM="$RUN_ROOT/phase2/hbm_load.csv"; sample_hbm "$LOAD_HBM" & SAMPLER_PID=$!
SERVER_LOG="$RUN_ROOT/phase2/server.log"
server_args=(
  serve /model --served-model-name nvidia/GLM-5.2-NVFP4 --revision "$MODEL_REVISION"
  --tensor-parallel-size 4 --quantization modelopt_fp4 --linear-backend marlin
  --attention-backend FLASHMLA_SPARSE --max-model-len 4096 --max-num-seqs 1
  --cpu-offload-gb 0 --no-enable-prefix-caching --enforce-eager
  --jit-monitor-mode error --jit-monitor-verbose
  --host 127.0.0.1 --port "$PORT"
)
printf '%q ' vllm "${server_args[@]}" > "$RUN_ROOT/phase2/exact_command.txt"; printf '\n' >> "$RUN_ROOT/phase2/exact_command.txt"
printf '%s\n' \
  "scope=native_first_use_deepgemm_dsa_only" \
  "VLLM_CACHE_ROOT=$JIT_ROOT_CONTAINER/cache/vllm" \
  "DG_JIT_CACHE_DIR=$JIT_ROOT_CONTAINER/cache/deep_gemm" \
  "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=$JIT_ROOT_CONTAINER/cache/flashinfer" \
  "TRITON_CACHE_DIR=$JIT_ROOT_CONTAINER/cache/triton" \
  "TORCHINDUCTOR_CACHE_DIR=$JIT_ROOT_CONTAINER/cache/torchinductor" \
  "CUDA_CACHE_PATH=$JIT_ROOT_CONTAINER/cache/cuda" \
  "TMPDIR=$JIT_ROOT_CONTAINER/cache/tmp" \
  "CUDA_HOME=$AUDIT_CUDA_CONTAINER" \
  "CUDACXX=$AUDIT_CUDA_CONTAINER/bin/nvcc" \
  "CC=$AUDIT_CUDA_CONTAINER/bin/gcc" \
  "CXX=$AUDIT_CUDA_CONTAINER/bin/g++" > "$JIT_ROOT/jit_environment.txt"
JIT_STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
"$CONTAINER" run --rm --name "$SERVER_NAME" --gpus "device=$GPU_SELECTOR" --ipc=host --network=host "${container_env[@]}" "${jit_env[@]}" --volume "$MODEL_ROOT:/model:ro" --entrypoint vllm "$RUNTIME_IMAGE_ID" "${server_args[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
ready=false
for _ in $(seq 1 360); do
  if curl --fail --silent --show-error "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=true; break; fi
  if ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
    if grep -Eqi 'out of memory|CUDA OOM|repack.*OOM' "$SERVER_LOG"; then fail MODEL_LOAD_REPACK_OOM 41; fi
    fail MODEL_LOAD_OR_REPACK_FAILED 41
  fi
  sleep 30
done
[[ $ready == true ]] || fail MODEL_LOAD_TIMEOUT 41
stop_sampler
JIT_COMPLETED_UTC=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli finalize-runtime-jit --lock "$LOCK" --bundle-root "$BUNDLE" --cache-root "$JIT_CACHE" --audit-log "$AUDIT_LOG" --started-utc "$JIT_STARTED_UTC" --completed-utc "$JIT_COMPLETED_UTC" --project-commit "$PROJECT_COMMIT" --runtime-image-id "$RUNTIME_IMAGE_ID" --output "$JIT_ROOT/runtime_jit_manifest.json" || { STATUS=BLOCKED; fail RUNTIME_JIT_PROVENANCE_UNPROVEN 3; }
chmod -R a-w "$JIT_CACHE"
curl --fail --silent --show-error "http://127.0.0.1:$PORT/version" > "$RUN_ROOT/phase2/server_version.json" || fail SERVER_VERSION_UNAVAILABLE 42
for evidence in 'GlmMoeDsaForCausalLM' 'modelopt_fp4' 'marlin' 'FLASHMLA_SPARSE'; do grep -Fqi "$evidence" "$SERVER_LOG" || fail RUNTIME_BACKEND_EVIDENCE_AMBIGUOUS 42; done
if grep -Eqi 'fallback|cpu offload|nvme offload|dense attention' "$SERVER_LOG"; then fail RUNTIME_FALLBACK_OR_OFFLOAD_DETECTED 42; fi
printf 'passed\n' > "$RUN_ROOT/phase2/PASSED"

PHASE=fixed_one_row_trace_diagnostic
if [[ ! -d $HARNESS/.git ]]; then
  mkdir -p "$HARNESS"; git -C "$HARNESS" init; git -C "$HARNESS" fetch --depth=1 https://github.com/scaleapi/SWE-bench_Pro-os.git "$HARNESS_COMMIT"; git -C "$HARNESS" checkout --detach FETCH_HEAD; git -C "$HARNESS" submodule update --init --recursive
fi
[[ $(git -C "$HARNESS" rev-parse HEAD) == "$HARNESS_COMMIT" ]] || fail HARNESS_COMMIT_MISMATCH 50
"$CONTAINER" run --rm "${container_env[@]}" --volume "$MODEL_ROOT:/model:ro" --volume "$HARNESS:/harness:ro" --entrypoint python3 "$RUNTIME_IMAGE_ID" -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli prepare --lock /project/configs/cluster/glm52_vllm_diagnostic.lock.json --model-root /model --harness-root /harness --ephemeral-root /storage/tmp/"$SLURM_JOB_ID" --artifact-root /storage/artifacts/"$SLURM_JOB_ID"/phase3 > "$RUN_ROOT/phase3/prepare.log" 2>&1 || fail PINNED_CASE_PREPARATION_FAILED 50

reset_cache() { curl --fail --silent --show-error -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" > "$1"; }
RUN_ID="slurm-${SLURM_JOB_ID}-vllm-dsa"
"$CONTAINER" inspect --format '{{.Id}} {{.State.Pid}} {{.State.StartedAt}}' "$SERVER_NAME" > "$RUN_ROOT/phase3/server_identity_before.txt"
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli control --lock "$LOCK" --mode OFF --run-id "$RUN_ID" --output "$RUN_ROOT/phase3/trace_control.json" --project-commit "$PROJECT_COMMIT" --runtime-image-id "$RUNTIME_IMAGE_ID"
reset_cache "$RUN_ROOT/phase3/reset_before_off.json" || fail CACHE_ISOLATION_RESET_FAILED 51
OFF_HBM="$RUN_ROOT/phase3/hbm_off.csv"; sample_hbm "$OFF_HBM" & SAMPLER_PID=$!; start_ns=$(date +%s%N)
curl --fail --silent --show-error -H 'Content-Type: application/json' --data-binary @"$EPHEMERAL/completion_request.json" "http://127.0.0.1:$PORT/v1/completions" > "$RUN_ROOT/phase3/off_response.json" || fail TRACE_OFF_INFERENCE_FAILED 52
OFF_NS=$(( $(date +%s%N) - start_ns )); stop_sampler
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli control --lock "$LOCK" --mode ON --run-id "$RUN_ID" --output "$RUN_ROOT/phase3/trace_control.json" --project-commit "$PROJECT_COMMIT" --runtime-image-id "$RUNTIME_IMAGE_ID"
reset_cache "$RUN_ROOT/phase3/reset_before_on.json" || fail CACHE_ISOLATION_RESET_FAILED 51
ON_HBM="$RUN_ROOT/phase3/hbm_on.csv"; sample_hbm "$ON_HBM" & SAMPLER_PID=$!; start_ns=$(date +%s%N)
if ! curl --fail --silent --show-error -H 'Content-Type: application/json' --data-binary @"$EPHEMERAL/completion_request.json" "http://127.0.0.1:$PORT/v1/completions" > "$RUN_ROOT/phase3/on_response.json"; then
  if compgen -G "$RUN_ROOT/phase3/native_raw/BLOCKED*.json" >/dev/null; then STATUS=BLOCKED; fail NATIVE_DSA_EXPOSURE_BLOCKED 3; fi
  fail TRACE_ON_INFERENCE_FAILED 52
fi
ON_NS=$(( $(date +%s%N) - start_ns )); stop_sampler
(cd "$JIT_ROOT" && sha256sum --check runtime_jit_SHA256SUMS) > "$JIT_ROOT/post_warmup_cache_validation.log" 2>&1 || { STATUS=BLOCKED; fail POST_WARMUP_RUNTIME_JIT_OR_CACHE_MUTATION 3; }
printf 'passed\n' > "$JIT_ROOT/POST_WARMUP_CACHE_UNCHANGED"
"$CONTAINER" inspect --format '{{.Id}} {{.State.Pid}} {{.State.StartedAt}}' "$SERVER_NAME" > "$RUN_ROOT/phase3/server_identity_after.txt"
cmp -s "$RUN_ROOT/phase3/server_identity_before.txt" "$RUN_ROOT/phase3/server_identity_after.txt" || fail LIVE_SERVER_PROCESS_CHANGED 53
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli trace-equivalence --off-response "$RUN_ROOT/phase3/off_response.json" --on-response "$RUN_ROOT/phase3/on_response.json" --off-duration-ns "$OFF_NS" --on-duration-ns "$ON_NS" --output "$RUN_ROOT/phase3/trace_equivalence.json" || fail TRACE_OUTPUT_MISMATCH 53
set +e
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli finalize-captures --lock "$LOCK" --raw-root "$RUN_ROOT/phase3/native_raw" --output-root "$RUN_ROOT/phase3/capture" --trace-report "$RUN_ROOT/phase3/trace_equivalence.json"
capture_rc=$?
set -e
if [[ $capture_rc == 3 ]]; then STATUS=BLOCKED; fail NATIVE_DSA_EXPOSURE_BLOCKED 3; fi
[[ $capture_rc == 0 ]] || fail DSA_CAPTURE_VALIDATION_FAILED 54

"$CONTAINER" rm -f "$SERVER_NAME" >/dev/null 2>&1 || true; wait "$SERVER_PID" >/dev/null 2>&1 || true; SERVER_PID=
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli extract-action --response "$RUN_ROOT/phase3/on_response.json" --action-output "$EPHEMERAL/agent_action.sh" --metadata-output "$RUN_ROOT/phase3/agent_action_metadata.json" || fail AGENT_FORMAT_INVALID 55
OFFICIAL_IMAGE=$(tr -d '\r\n' < "$EPHEMERAL/official_image.txt")
[[ $OFFICIAL_IMAGE == docker.io/jefzda/sweap-images:* ]] || fail OFFICIAL_DOCKERHUB_TAG_INVALID 55
"$CONTAINER" pull "$OFFICIAL_IMAGE" > "$RUN_ROOT/official/instance_image_pull.log" 2>&1 || fail OFFICIAL_INSTANCE_IMAGE_UNAVAILABLE 56
"$CONTAINER" run -d --name "$AGENT_NAME" -w /testbed "$OFFICIAL_IMAGE" sleep 2h > "$RUN_ROOT/official/agent_container_id.txt" || fail AGENT_CONTAINER_START_FAILED 56
set +e; timeout 600 "$CONTAINER" exec -i -w /testbed "$AGENT_NAME" bash -s < "$EPHEMERAL/agent_action.sh" > "$RUN_ROOT/official/agent_action.log" 2>&1; action_rc=$?; set -e
printf '%s\n' "$action_rc" > "$RUN_ROOT/official/agent_action_returncode.txt"
"$CONTAINER" exec -w /testbed "$AGENT_NAME" git diff --binary > "$RUN_ROOT/official/model_patch.diff" || fail PATCH_GATHER_FAILED 57
"$CONTAINER" rm -f "$AGENT_NAME" >/dev/null 2>&1 || true
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli make-prediction --patch "$RUN_ROOT/official/model_patch.diff" --output "$RUN_ROOT/official/preds.json" --official-pred-root "$RUN_ROOT/official/pred_inputs"
python3 "$HARNESS/helper_code/gather_patches.py" --directory "$RUN_ROOT/official/pred_inputs" --prefix glm52-vllm-diagnostic --output "$RUN_ROOT/official/patches.json" > "$RUN_ROOT/official/gather.log" 2>&1 || fail OFFICIAL_PATCH_GATHER_FAILED 57

EVAL_ENV="$CACHE/eval-env/pandas-2.3.3_tqdm-4.67.1_docker-7.1.0"
if [[ ! -x $EVAL_ENV/bin/python ]]; then python3 -m venv "$EVAL_ENV"; "$EVAL_ENV/bin/pip" install --disable-pip-version-check --no-input 'pandas==2.3.3' 'tqdm==4.67.1' 'docker==7.1.0' > "$RUN_ROOT/official/evaluator_install.log" 2>&1; fi
mkdir -p "$RUN_ROOT/official/evaluation"
(cd "$HARNESS" && "$EVAL_ENV/bin/python" "$HARNESS/swe_bench_pro_eval.py" --raw_sample_path "$EPHEMERAL/official_raw_sample.jsonl" --patch_path "$RUN_ROOT/official/patches.json" --output_dir "$RUN_ROOT/official/evaluation" --scripts_dir "$HARNESS/run_scripts" --num_workers 1 --dockerhub_username jefzda --use_local_docker) > "$RUN_ROOT/official/evaluator.log" 2>&1 || fail OFFICIAL_SINGLE_ROW_EVALUATOR_FAILED 58
[[ -f $RUN_ROOT/official/evaluation/eval_results.json ]] || fail OFFICIAL_EVAL_RESULTS_MISSING 58

if grep -Eiwq 'nan|inf|infinity' "$SERVER_LOG"; then fail NONFINITE_RUNTIME_LOG_DETECTED 59; fi
python3 - "$RUN_ROOT" "$PROJECT_COMMIT" "$MODEL_REVISION" "$RUNTIME_IMAGE_ID" <<'PY'
import csv,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); result=json.load(open(root/'official/evaluation/eval_results.json'))
instance='instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5'
if set(result)!={instance} or not isinstance(result[instance],bool): raise SystemExit('OFFICIAL_SINGLE_ROW_RESULT_INVALID')
def hbm(path):
 rows=list(csv.DictReader(open(path))); by={}
 for row in rows:
  uuid=row['uuid']; total=int(float(row['memory_total_mib'])); used=int(float(row['memory_used_mib'])); free=int(float(row['memory_free_mib']))
  if not (0<=used<=total and 0<free<=total): raise SystemExit('HBM_SAMPLE_INVALID')
  state=by.setdefault(uuid,{'total_mib':total,'peak_used_mib':0,'minimum_free_mib':total}); state['peak_used_mib']=max(state['peak_used_mib'],used); state['minimum_free_mib']=min(state['minimum_free_mib'],free)
 if len(by)!=4 or any(v['minimum_free_mib']<=0 for v in by.values()): raise SystemExit('POSITIVE_FOUR_GPU_HBM_HEADROOM_REQUIRED')
 return by
hbm_report={'load':hbm(root/'phase2/hbm_load.csv'),'trace_off':hbm(root/'phase3/hbm_off.csv'),'trace_on':hbm(root/'phase3/hbm_on.csv')}
minimum_headroom=min(v['minimum_free_mib'] for phase in hbm_report.values() for v in phase.values())
payload={'schema_version':1,'status':'PASS','diagnostic':'glm52_nvfp4_vllm_native_dsa_single_swepro_instance','quality_score_eligible':False,'acceptance_threshold_evaluated':False,'full_selection_reachable':False,'instance_id':instance,'official_evaluation_resolved':result[instance],'project_commit':sys.argv[2],'model':{'id':'nvidia/GLM-5.2-NVFP4','revision':sys.argv[3],'unmodified':True},'vllm_commit':'4a3447d200e5aa428d68d1a00aa00f1a19a1a729','runtime_image_id':sys.argv[4],'tensor_parallel':4,'offload':False,'fallback_attempted':False,'runtime_jit':json.load(open(root/'phase2/runtime_jit/runtime_jit_manifest.json')),'trace_equivalence':json.load(open(root/'phase3/trace_equivalence.json')),'capture_manifest':json.load(open(root/'phase3/capture/capture_manifest.json')),'hbm':hbm_report,'minimum_positive_headroom_mib':minimum_headroom}
(root/'diagnostic_manifest.json.partial').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); (root/'diagnostic_manifest.json.partial').replace(root/'diagnostic_manifest.json')
PY
printf 'passed\n' > "$RUN_ROOT/phase3/PASSED"
STATUS=PASS; PHASE=complete; exit 0
