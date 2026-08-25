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
project_commit=sys.argv[2]
def sha(relative):
 digest=hashlib.sha256()
 with (root/relative).open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
 return digest.hexdigest()
payload={
 'schema_version':2,
 'status':'stateful_edit_v3_content_addressed_source_package_no_weights',
 'project_commit':project_commit,
 'vllm_commit':'4a3447d200e5aa428d68d1a00aa00f1a19a1a729',
 'model':{'id':'nvidia/GLM-5.2-NVFP4','revision':'aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa','weights_in_package':False},
 'runtime_bundle_key':'vllm-4a3447d200e5-sm90-cu1303-py312-torch2130-patch-fc2f3734-image-3869b846',
 'unsafe_research_ablation':True,
 'production_default_enabled':False,
 'scientific_scope':{
   'accuracy_ablation_only':True,
   'compute_semantics':'full_target_prefill_then_selected_frozen_history_rows_overwritten_from_private_donor_snapshots',
   'true_partial_prefill':False,
   'latency_or_skipped_flop_claim':False
 },
 'entrypoints':{
   'submit':'scripts/cluster/submit_glm52_forced_reuse_ablation.sh',
   'compute':'scripts/cluster/run_glm52_forced_reuse_ablation.sh',
   'stateful_cli':'putpocket_dataset_mining.glm52_stateful_cli',
   'stateful_proxy':'putpocket_dataset_mining.glm52_stateful_proxy'
 },
}
(root/'PACKAGE-MANIFEST.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
tracked={relative:sha(relative) for relative in (
 'PACKAGE-MANIFEST.json',
 'configs/cluster/glm52_forced_reuse_ablation.lock.json',
 'docs/CLUSTER_GLM52_STATEFUL_EDIT_V3.md',
 'instrumentation/vllm/glm52_forced_edit_reuse.py',
 'scripts/cluster/run_glm52_forced_reuse_ablation.sh',
 'scripts/cluster/submit_glm52_forced_reuse_ablation.sh',
 'src/putpocket_dataset_mining/glm52_stateful_cli.py',
 'src/putpocket_dataset_mining/glm52_stateful_proxy.py',
)}
stateful={
 'schema_version':2,
 'status':'stateful_edit_v3_pre_execution_default_off',
 'project_commit':project_commit,
 'derived_from_login1_source_ancestor':'faa1b4b6096a06eadecc5d058d1a5531c13a9d4f',
 'port_base':'b9d44af05a464993001ecaf19acc07f188f072a7',
 'scenario':'SYS_old+Q1 -> frozen A1/action/Q2; equal-token system edit; SYS_new+Q1+A1+Q2 -> A2',
 'eligible_history_range':'[115, dynamic frozen history end), including A1',
 'q2_and_later_extensions':'always recomputed',
 'fresh_repository_container_per_ratio':True,
 'primary_outcome':'official pinned single-instance SWE-bench Pro resolved boolean',
 'compute_semantics_limitation':'compute_then_overwrite; no true partial prefill and no speedup claim',
 'tracked_sha256':tracked,
}
(root/'STATEFUL-EDIT-V3-MANIFEST.json').write_text(json.dumps(stateful,indent=2,sort_keys=True)+'\n')
PY
(cd "$STAGING" && find . -type f -not -path './SHA256SUMS' -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
chmod -R a-w "$STAGING"
mv "$STAGING" "$PACKAGE"
tar --sort=name --mtime='UTC 1970-01-01' --owner=0 --group=0 --numeric-owner -czf "$ARCHIVE" -C "$OUTPUT_PARENT" "$PACKAGE_NAME"
sha256sum "$ARCHIVE"
printf 'PACKAGE_ROOT=%s\nPACKAGE_ARCHIVE=%s\n' "$PACKAGE" "$ARCHIVE"
