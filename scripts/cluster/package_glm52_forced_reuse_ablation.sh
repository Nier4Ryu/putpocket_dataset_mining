#!/usr/bin/env bash
set -euo pipefail
umask 077

[[ $# == 1 ]] || { echo 'usage: package_glm52_forced_reuse_ablation.sh OUTPUT_PARENT' >&2; exit 2; }
OUTPUT_PARENT=$(realpath "$1")
SOURCE_ROOT=$(git rev-parse --show-toplevel)
PROJECT_COMMIT=$(git -C "$SOURCE_ROOT" rev-parse HEAD)
[[ -z $(git -C "$SOURCE_ROOT" status --porcelain) ]] || { echo BLOCKED_DIRTY_PROJECT_SOURCE >&2; exit 20; }
VLLM_SOURCE=${PUTPOCKET_EXACT_VLLM_SOURCE:-/tmp/putpocket-vllm-4a3447d.xgvh3O}
[[ $(git -C "$VLLM_SOURCE" rev-parse HEAD) == 4a3447d200e5aa428d68d1a00aa00f1a19a1a729 ]] || { echo BLOCKED_VLLM_SOURCE_COMMIT_MISMATCH >&2; exit 20; }

PACKAGE_NAME="glm52-forced-reuse-${PROJECT_COMMIT:0:12}"
STAGING="$OUTPUT_PARENT/$PACKAGE_NAME.partial"
PACKAGE="$OUTPUT_PARENT/$PACKAGE_NAME"
ARCHIVE="$OUTPUT_PARENT/$PACKAGE_NAME.tar.gz"
[[ ! -e $STAGING && ! -e $PACKAGE && ! -e $ARCHIVE ]] || { echo BLOCKED_PACKAGE_TARGET_EXISTS >&2; exit 20; }
mkdir -p "$STAGING/artifacts"
git -C "$SOURCE_ROOT" archive "$PROJECT_COMMIT" | tar -x -C "$STAGING"

VLLM_STAGE="$OUTPUT_PARENT/$PACKAGE_NAME.vllm-source.partial"
[[ ! -e $VLLM_STAGE ]] || { echo BLOCKED_VLLM_STAGE_EXISTS >&2; exit 20; }
mkdir -p "$VLLM_STAGE/vllm"
git -C "$VLLM_SOURCE" archive 4a3447d200e5aa428d68d1a00aa00f1a19a1a729 | tar -x -C "$VLLM_STAGE/vllm"
printf '%s\n' 4a3447d200e5aa428d68d1a00aa00f1a19a1a729 > "$VLLM_STAGE/vllm/VLLM_COMMIT"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner -czf "$STAGING/artifacts/vllm-4a3447d200e5aa428d68d1a00aa00f1a19a1a729.tar.gz" -C "$VLLM_STAGE" vllm
rm -r -- "$VLLM_STAGE"
(cd "$STAGING" && sha256sum artifacts/vllm-4a3447d200e5aa428d68d1a00aa00f1a19a1a729.tar.gz > artifacts/vllm-source.sha256)

python3 - "$STAGING" "$PROJECT_COMMIT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1])
payload={
 'schema_version':1,
 'status':'content_addressed_source_package_no_weights',
 'project_commit':sys.argv[2],
 'vllm_commit':'4a3447d200e5aa428d68d1a00aa00f1a19a1a729',
 'model':{'id':'nvidia/GLM-5.2-NVFP4','revision':'aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa','weights_in_package':False},
 'runtime_bundle_key':'vllm-4a3447d200e5-sm90-cu1303-py312-torch2130-patch-fc2f3734-image-3869b846',
 'unsafe_research_ablation':True,
 'production_default_enabled':False,
 'entrypoints':{
   'submit':'scripts/cluster/submit_glm52_forced_reuse_ablation.sh',
   'compute':'scripts/cluster/run_glm52_forced_reuse_ablation.sh'
 },
}
(root/'PACKAGE-MANIFEST.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
(cd "$STAGING" && find . -type f -not -path './SHA256SUMS' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
mv "$STAGING" "$PACKAGE"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner -czf "$ARCHIVE" -C "$OUTPUT_PARENT" "$PACKAGE_NAME"
sha256sum "$ARCHIVE"
printf 'PACKAGE_ROOT=%s\nPACKAGE_ARCHIVE=%s\n' "$PACKAGE" "$ARCHIVE"
