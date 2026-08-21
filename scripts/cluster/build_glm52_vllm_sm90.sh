#!/usr/bin/env bash
set -euo pipefail
umask 077

BUILD_SOURCE_COMMIT=${PUTPOCKET_BUILD_SOURCE_COMMIT:-}
RUNTIME_SOURCE_COMMIT=${PUTPOCKET_RUNTIME_SOURCE_COMMIT:-}
WRAPPER_SOURCE_COMMIT=${PUTPOCKET_WRAPPER_SOURCE_COMMIT:-}
ALLOW_RUNTIME_SOURCE_SPLIT=${PUTPOCKET_ALLOW_RUNTIME_SOURCE_SPLIT:-}
IMMUTABLE_BUNDLE_REUSE_ONLY=${PUTPOCKET_IMMUTABLE_BUNDLE_REUSE_ONLY:-}
SOURCE_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
LOCK="$SOURCE_ROOT/configs/cluster/glm52_vllm_diagnostic.lock.json"
CONTAINER=${PUTPOCKET_CONTAINER_EXECUTABLE:-}
CPU_SCRATCH=${PUTPOCKET_CPU_LOCAL_SCRATCH_ROOT:-}
SHARED_ROOT=${PUTPOCKET_SHARED_BUILD_ROOT:-}
BUNDLE_KEY=${PUTPOCKET_EXPECTED_BUNDLE_KEY:-}
VLLM_COMMIT=4a3447d200e5aa428d68d1a00aa00f1a19a1a729
BASE_IMAGE='nvidia/cuda:13.0.3-devel-ubuntu22.04@sha256:3869b846a8cc495ce11c172d87cfc0da8874b910d14a9810bec6b6182e9ee9f8'
PATCH_REL='patches/vllm/4a3447d200e5aa428d68d1a00aa00f1a19a1a729/glm52_native_dsa_bounded_dump.patch'
INSTRUMENT_REL='instrumentation/vllm/vllm_dsa_diagnostic_dump.py'
TARGET="$SHARED_ROOT/$BUNDLE_KEY"
WORK="$CPU_SCRATCH/putpocket-vllm-native-build/${SLURM_JOB_ID:-unknown}"
VLLM_ROOT="$WORK/vllm"
STAGING="$SHARED_ROOT/.${BUNDLE_KEY}.${SLURM_JOB_ID:-unknown}.partial"
BUILD_TAG="putpocket/vllm-build:${SLURM_JOB_ID:-unknown}"
RUNTIME_TAG="putpocket/vllm-glm52:${SLURM_JOB_ID:-unknown}"
BUILD_CONTAINER="pp-vllm-wheel-${SLURM_JOB_ID:-unknown}"

fail() { printf '%s\n' "$1" >&2; exit "${2:-2}"; }
cleanup() { "$CONTAINER" rm -f "$BUILD_CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

[[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ && ${SLURM_JOB_NUM_NODES:-0} == 1 ]] || fail CPU_SLURM_ALLOCATION_REQUIRED 20
[[ -z ${SLURM_JOB_GPUS:-} && -z ${SLURM_GPUS:-} && -z ${CUDA_VISIBLE_DEVICES:-} ]] || fail CPU_BUILD_GPU_ALLOCATION_FORBIDDEN 20
[[ $BUILD_SOURCE_COMMIT =~ ^[0-9a-f]{40}$ && $RUNTIME_SOURCE_COMMIT =~ ^[0-9a-f]{40}$ && $WRAPPER_SOURCE_COMMIT =~ ^[0-9a-f]{40}$ ]] || fail SOURCE_PROVENANCE_FULL_SHA_REQUIRED 21
[[ $ALLOW_RUNTIME_SOURCE_SPLIT =~ ^[01]$ && $IMMUTABLE_BUNDLE_REUSE_ONLY =~ ^[01]$ ]] || fail SOURCE_PROVENANCE_MODE_INVALID 21
[[ -n $CONTAINER && -x $CONTAINER && -n $CPU_SCRATCH && -n $SHARED_ROOT && -n $BUNDLE_KEY ]] || fail CPU_BUILD_SITE_CONFIGURATION_MISSING 21
[[ $BUNDLE_KEY == vllm-4a3447d200e5-sm90-cu1303-py312-torch2130-patch-fc2f3734-image-3869b846 ]] || fail BUNDLE_KEY_MISMATCH 21
[[ $TARGET == /home2/jslee202403/putpocket-builds/vllm/vllm-4a3447d200e5-sm90-cu1303-py312-torch2130-patch-fc2f3734-image-3869b846 ]] || fail IMMUTABLE_BUNDLE_ROOT_MISMATCH 21
"$CONTAINER" info >/dev/null 2>&1 || fail CPU_CONTAINER_RUNTIME_UNAVAILABLE 21
OBSERVED_RUNTIME_SOURCE_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
[[ $OBSERVED_RUNTIME_SOURCE_COMMIT == "$RUNTIME_SOURCE_COMMIT" ]] || fail RUNTIME_SOURCE_COMMIT_MISMATCH 22
export PYTHONPATH="$SOURCE_ROOT/src"
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli validate-lock --lock "$LOCK" >/dev/null
provenance_args=(
  --expected-build-source-commit "$BUILD_SOURCE_COMMIT"
  --runtime-source-commit "$RUNTIME_SOURCE_COMMIT"
  --observed-runtime-source-commit "$OBSERVED_RUNTIME_SOURCE_COMMIT"
  --wrapper-source-commit "$WRAPPER_SOURCE_COMMIT"
)
if [[ $ALLOW_RUNTIME_SOURCE_SPLIT == 1 ]]; then provenance_args+=(--allow-runtime-source-split); fi

if [[ -f $TARGET/SUCCESS ]]; then
  python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli validate-build-bundle --lock "$LOCK" --bundle-root "$TARGET" "${provenance_args[@]}"
  printf 'REUSED_BUILD_BUNDLE=%s\n' "$TARGET"
  exit 0
fi
[[ $IMMUTABLE_BUNDLE_REUSE_ONLY == 0 ]] || fail IMMUTABLE_BUILD_BUNDLE_MISSING_REBUILD_FORBIDDEN 30
[[ $BUILD_SOURCE_COMMIT == "$RUNTIME_SOURCE_COMMIT" && $RUNTIME_SOURCE_COMMIT == "$WRAPPER_SOURCE_COMMIT" && $ALLOW_RUNTIME_SOURCE_SPLIT == 0 ]] || fail NEW_BUILD_SOURCE_SPLIT_FORBIDDEN 30
if [[ -e $TARGET ]]; then
  mv "$TARGET" "$TARGET.invalid-${SLURM_JOB_ID}"
fi
mkdir -p "$VLLM_ROOT" "$STAGING/wheels" "$STAGING/source" "$STAGING/logs" "$SHARED_ROOT"

git -C "$VLLM_ROOT" init
git -C "$VLLM_ROOT" fetch --depth=1 https://github.com/vllm-project/vllm.git "$VLLM_COMMIT"
git -C "$VLLM_ROOT" checkout --detach FETCH_HEAD
[[ $(git -C "$VLLM_ROOT" rev-parse HEAD) == "$VLLM_COMMIT" ]] || fail VLLM_SOURCE_COMMIT_MISMATCH 30
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli validate-source --lock "$LOCK" --project-root "$SOURCE_ROOT" --source-root "$VLLM_ROOT" > "$STAGING/logs/source_preflight.json"
git -C "$VLLM_ROOT" apply --unidiff-zero --check "$SOURCE_ROOT/$PATCH_REL"
git -C "$VLLM_ROOT" apply --unidiff-zero "$SOURCE_ROOT/$PATCH_REL"
install -m 0644 "$SOURCE_ROOT/$INSTRUMENT_REL" "$VLLM_ROOT/vllm/model_executor/layers/vllm_dsa_diagnostic_dump.py"
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli validate-patched --lock "$LOCK" --source-root "$VLLM_ROOT" > "$STAGING/logs/source_post_patch.json"
git -C "$VLLM_ROOT" diff --check

export DOCKER_BUILDKIT=1
build_args=(
  --file "$VLLM_ROOT/docker/Dockerfile"
  --build-arg "BUILD_BASE_IMAGE=$BASE_IMAGE"
  --build-arg "FINAL_BASE_IMAGE=$BASE_IMAGE"
  --build-arg CUDA_VERSION=13.0.3
  --build-arg PYTHON_VERSION=3.12
  --build-arg torch_cuda_arch_list=9.0
  --build-arg vllm_target_device=cuda
  --build-arg RUN_WHEEL_CHECK=false
  --build-arg "max_jobs=${SLURM_CPUS_PER_TASK:-1}"
  --build-arg nvcc_threads=2
)
"$CONTAINER" build "${build_args[@]}" --target build --tag "$BUILD_TAG" "$VLLM_ROOT" > "$STAGING/logs/build-wheel-image.log" 2>&1
grep -Fq 'CUDA target architectures: 9.0' "$STAGING/logs/build-wheel-image.log" || fail SM90_CMAKE_CONFIGURATION_NOT_PROVEN 31
if grep -Eqi 'VLLM_USE_PRECOMPILED=(1|true)|precompiled wheel' "$STAGING/logs/build-wheel-image.log"; then fail PREBUILT_VLLM_SUBSTITUTION_DETECTED 31; fi
"$CONTAINER" create --name "$BUILD_CONTAINER" "$BUILD_TAG" /bin/true >/dev/null
"$CONTAINER" cp "$BUILD_CONTAINER:/workspace/dist/." "$STAGING/wheels"
wheel_count=$(find "$STAGING/wheels" -maxdepth 1 -type f -name 'vllm-*.whl' | wc -l)
[[ $wheel_count == 1 ]] || fail VLLM_WHEEL_CARDINALITY_MISMATCH 32
WHEEL=$(find "$STAGING/wheels" -maxdepth 1 -type f -name 'vllm-*.whl')
python3 - "$STAGING" "$WHEEL" <<'PY'
import hashlib,json,pathlib,sys
root,wheel=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2])
h=hashlib.sha256()
with wheel.open('rb') as stream:
    for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
audit={
 'schema_version':1,
 'run_wheel_check':False,
 'upstream_release_wheel_limit_mb':500,
 'exception_scope':'intentional_sm90_cuda13_source_build_only',
 'wheel_path':str(wheel.relative_to(root)),
 'wheel_bytes':wheel.stat().st_size,
 'wheel_sha256':h.hexdigest(),
}
(root/'logs/wheel_artifact.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
PY
"$CONTAINER" run --rm --entrypoint /bin/bash "$BUILD_TAG" -lc 'set -euo pipefail; mkdir -p /tmp/inspect; python3 -m zipfile -e /workspace/dist/vllm-*.whl /tmp/inspect; find /tmp/inspect -type f -name "*.so" -print0 | xargs -0 -r cuobjdump -lelf' > "$STAGING/logs/compiled_arches.txt" 2>&1
grep -Fq 'sm_90' "$STAGING/logs/compiled_arches.txt" || fail COMPILED_SM90_EVIDENCE_MISSING 32
"$CONTAINER" run --rm --entrypoint python3 "$BUILD_TAG" -c 'import importlib.metadata as m,json,platform,sys,torch; p=sorted(f"{d.metadata.get(chr(78)+chr(97)+chr(109)+chr(101),d.name)}=={d.version}" for d in m.distributions()); print(json.dumps({"python":platform.python_version(),"python_major_minor":f"{sys.version_info.major}.{sys.version_info.minor}","torch":torch.__version__,"torch_base":torch.__version__.split("+",1)[0],"torch_cuda":torch.version.cuda,"resolved_packages":p},sort_keys=True))' > "$STAGING/logs/build_environment.json"
"$CONTAINER" run --rm --entrypoint nvcc "$BUILD_TAG" --version > "$STAGING/logs/build_nvcc.txt"

"$CONTAINER" build "${build_args[@]}" --target vllm-openai --tag "$RUNTIME_TAG" "$VLLM_ROOT" > "$STAGING/logs/build-runtime-image.log" 2>&1
"$CONTAINER" image inspect "$RUNTIME_TAG" --format '{{.Id}}' > "$STAGING/runtime_image_id.txt"
"$CONTAINER" run --rm --entrypoint python3 "$RUNTIME_TAG" -c 'import importlib.metadata as m,json,platform,sys,torch; p=sorted(f"{d.metadata.get(chr(78)+chr(97)+chr(109)+chr(101),d.name)}=={d.version}" for d in m.distributions()); print(json.dumps({"python":platform.python_version(),"python_major_minor":f"{sys.version_info.major}.{sys.version_info.minor}","torch":torch.__version__,"torch_base":torch.__version__.split("+",1)[0],"torch_cuda":torch.version.cuda,"transformers":m.version("transformers"),"vllm":m.version("vllm"),"resolved_packages":p},sort_keys=True))' > "$STAGING/logs/runtime_environment.json"
"$CONTAINER" run --rm --entrypoint nvcc "$RUNTIME_TAG" --version > "$STAGING/logs/runtime_nvcc.txt"
"$CONTAINER" save --output "$STAGING/runtime-image.tar" "$RUNTIME_TAG"
git -C "$VLLM_ROOT" archive --format=tar.gz --output="$STAGING/source/vllm-source.tar.gz" HEAD
install -m 0644 "$SOURCE_ROOT/$PATCH_REL" "$STAGING/source/"
install -m 0644 "$SOURCE_ROOT/$INSTRUMENT_REL" "$STAGING/source/"
tar -C "$STAGING/source" -czf "$STAGING/vllm-source-bundle.tar.gz" .

python3 - "$STAGING" "$WHEEL" "$BUILD_SOURCE_COMMIT" <<'PY'
import hashlib,json,pathlib,sys
root,wheel,project=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]),sys.argv[3]
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
files={
 'runtime_image_tar':root/'runtime-image.tar',
 'vllm_wheel':wheel,
 'source_bundle':root/'vllm-source-bundle.tar.gz',
}
provenance_paths=[
 root/'logs/source_preflight.json',root/'logs/source_post_patch.json',
 root/'logs/build-wheel-image.log',root/'logs/compiled_arches.txt',
 root/'logs/wheel_artifact.json',
 root/'logs/build_environment.json',root/'logs/build_nvcc.txt',
 root/'logs/build-runtime-image.log',root/'logs/runtime_environment.json',
 root/'logs/runtime_nvcc.txt',
]
build_identity=json.loads((root/'logs/build_environment.json').read_text())
runtime_identity=json.loads((root/'logs/runtime_environment.json').read_text())
wheel_artifact=json.loads((root/'logs/wheel_artifact.json').read_text())
wheel_sha256=digest(wheel)
if wheel_artifact != {
 'schema_version':1,
 'run_wheel_check':False,
 'upstream_release_wheel_limit_mb':500,
 'exception_scope':'intentional_sm90_cuda13_source_build_only',
 'wheel_path':str(wheel.relative_to(root)),
 'wheel_bytes':wheel.stat().st_size,
 'wheel_sha256':wheel_sha256,
}:
    raise SystemExit('WHEEL_ARTIFACT_AUDIT_MISMATCH')
for label,identity in [('build',build_identity),('runtime',runtime_identity)]:
    if identity['python_major_minor']!='3.12' or identity['torch_base']!='2.13.0' or identity['torch_cuda']!='13.0':
        raise SystemExit(f'{label.upper()}_ENVIRONMENT_IDENTITY_MISMATCH')
if tuple(int(v) for v in runtime_identity['transformers'].split('.')[:2]) < (5,3):
    raise SystemExit('RUNTIME_TRANSFORMERS_5_3_REQUIRED')
manifest={
 'schema_version':1,'status':'SUCCESS','project_commit':project,
 'vllm_commit':'4a3447d200e5aa428d68d1a00aa00f1a19a1a729',
 'bundle_key':'vllm-4a3447d200e5-sm90-cu1303-py312-torch2130-patch-fc2f3734-image-3869b846',
 'patch_sha256':'fc2f3734225c077fd9cfaf08341e2eaf01955a8cfd1cf1bee3c1747accfe5a9b',
 'patch_target_post_sha256':'65ef4a917b35cc5298ab6e93c7351db3347cbe891f0ebc39cb115e86bb49b3dd',
 'build_patch_target_post_sha256':'f7c56f7c9100285057388cff5b7b074571853f6a3e552ee9cbdebe3221d4f71d',
 'instrumentation_sha256':'1dc2872ddb58fa719290e1f954c643819f4409c2ab7b1a1f78d701c13848c516',
 'compiler_audit_sha256':'e060c0b09e1c2eb4da90854ee81d284f7f6acce2b7375b9ee87ecf956a78f9ee',
 'base_image':'nvidia/cuda:13.0.3-devel-ubuntu22.04@sha256:3869b846a8cc495ce11c172d87cfc0da8874b910d14a9810bec6b6182e9ee9f8',
 'python':'3.12','torch':'2.13.0','cuda':'13.0.3','torch_cuda_arch_list':'9.0',
 'cmake_cuda_architectures':'90','vllm_target_device':'cuda','vllm_use_precompiled':False,
 'wheel_release_policy':wheel_artifact,
 'general_h200_compilation_allowed':False,'h200_runtime_jit_scope':'native_first_use_deepgemm_dsa_only','pinned_source_runtime_jit_required':True,'runtime_jit_cache_reuse':False,
 'prebuilt_vllm_wheel_used':False,'built_from_scratch':True,'compiled_arch_evidence':['sm_90'],
 'runtime_gate':'ALLOW_NATIVE_FIRST_USE_JIT_WITH_RUN_LOCAL_AUDIT',
 'runtime_jit_evidence':['cmake/external_projects/deepgemm.cmake:AOT CUDA kernels TODO','docker/Dockerfile:runtime nvcc required for FlashInfer/DeepGEMM/EP','vllm/model_executor/warmup/deep_gemm_warmup.py:DeepGEMM JITs kernels'],
 'runtime_image_id':(root/'runtime_image_id.txt').read_text().strip(),
 'build_environment':build_identity,'runtime_environment':runtime_identity,
 'files':{name:{'path':str(path.relative_to(root)),'sha256':digest(path),'bytes':path.stat().st_size} for name,path in files.items()},
 'provenance_files':{path.name:{'path':str(path.relative_to(root)),'sha256':digest(path),'bytes':path.stat().st_size} for path in provenance_paths},
}
(root/'build_manifest.json.partial').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
(root/'build_manifest.json.partial').replace(root/'build_manifest.json')
(root/'SHA256SUMS').write_text(''.join(f"{item['sha256']}  {item['path']}\n" for item in sorted((item for group in ('files','provenance_files') for item in manifest[group].values()),key=lambda item:item['path'])))
PY
printf 'SUCCESS\n' > "$STAGING/SUCCESS"
python3 -m putpocket_dataset_mining.glm52_vllm_diagnostic_cli validate-build-bundle --lock "$LOCK" --bundle-root "$STAGING" "${provenance_args[@]}" > "$STAGING/logs/bundle_validation.json"
mv "$STAGING" "$TARGET"
printf 'BUILD_BUNDLE=%s\n' "$TARGET"
