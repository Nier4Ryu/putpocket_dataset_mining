#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bootstrap_glm52_attention_indexer.sh VLLM_ROOT [EVIDENCE_JSON]" >&2
  exit 2
fi

PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
VLLM_ROOT=$(realpath "$1")
EVIDENCE_JSON=${2:-$VLLM_ROOT/putpocket-overlay-bootstrap.json}
PYTHON_BIN=${PYTHON_BIN:-python3}
LOCK=$PROJECT_ROOT/configs/runpod/glm52_attention_indexer_package.lock.json
PATCH_ROOT=$PROJECT_ROOT/patches/vllm/4a3447d200e5aa428d68d1a00aa00f1a19a1a729
EXPECTED_VLLM=4a3447d200e5aa428d68d1a00aa00f1a19a1a729

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

[[ -d $VLLM_ROOT/.git ]] || { echo VLLM_GIT_CHECKOUT_REQUIRED >&2; exit 20; }
[[ $(git -C "$VLLM_ROOT" rev-parse HEAD) == "$EXPECTED_VLLM" ]] || { echo VLLM_COMMIT_MISMATCH >&2; exit 20; }
[[ -z $(git -C "$VLLM_ROOT" status --porcelain) ]] || { echo VLLM_SOURCE_MUST_START_CLEAN >&2; exit 20; }
PATCH_TMPDIR=$VLLM_ROOT/.putpocket-patch-tmp
mkdir "$PATCH_TMPDIR"

"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli validate-package \
  --lock "$LOCK" --project-root "$PROJECT_ROOT" --phase project_artifacts >/dev/null
"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli validate-package \
  --lock "$LOCK" --project-root "$PROJECT_ROOT" --vllm-root "$VLLM_ROOT" \
  --phase upstream_base >/dev/null

TMPDIR=$PATCH_TMPDIR patch -d "$VLLM_ROOT" -p1 --dry-run --forward --batch \
  < "$PATCH_ROOT/glm52_forced_edit_reuse.patch" >/dev/null
TMPDIR=$PATCH_TMPDIR patch -d "$VLLM_ROOT" -p1 --forward --batch \
  < "$PATCH_ROOT/glm52_forced_edit_reuse.patch" >/dev/null
rmdir "$PATCH_TMPDIR"
install -m 0644 "$PROJECT_ROOT/instrumentation/vllm/glm52_forced_edit_reuse.py" \
  "$VLLM_ROOT/vllm/model_executor/layers/glm52_forced_edit_reuse.py"
"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli validate-package \
  --lock "$LOCK" --project-root "$PROJECT_ROOT" --vllm-root "$VLLM_ROOT" \
  --phase post_legacy >/dev/null

git -C "$VLLM_ROOT" apply --check --unidiff-zero "$PATCH_ROOT/putpocket_true_partial_prefill.patch"
git -C "$VLLM_ROOT" apply --unidiff-zero "$PATCH_ROOT/putpocket_true_partial_prefill.patch"
install -m 0644 "$PROJECT_ROOT/instrumentation/vllm/putpocket_true_partial_prefill.py" \
  "$VLLM_ROOT/vllm/v1/putpocket_true_partial_prefill.py"
"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli validate-package \
  --lock "$LOCK" --project-root "$PROJECT_ROOT" --vllm-root "$VLLM_ROOT" \
  --phase post_true_partial >/dev/null

git -C "$VLLM_ROOT" apply --check --unidiff-zero "$PATCH_ROOT/glm52_attention_indexer_score_diagnostic.patch"
git -C "$VLLM_ROOT" apply --unidiff-zero "$PATCH_ROOT/glm52_attention_indexer_score_diagnostic.patch"
install -m 0644 "$PROJECT_ROOT/instrumentation/vllm/glm52_attention_indexer_scores.py" \
  "$VLLM_ROOT/vllm/model_executor/layers/glm52_attention_indexer_scores.py"
"$PYTHON_BIN" -m putpocket_dataset_mining.glm52_runpod_cli validate-package \
  --lock "$LOCK" --project-root "$PROJECT_ROOT" --vllm-root "$VLLM_ROOT" \
  --phase post_score_diagnostic >/dev/null

"$PYTHON_BIN" -m py_compile \
  "$VLLM_ROOT/vllm/model_executor/layers/glm52_attention_indexer_scores.py" \
  "$VLLM_ROOT/vllm/model_executor/layers/glm52_forced_edit_reuse.py" \
  "$VLLM_ROOT/vllm/model_executor/layers/mla.py" \
  "$VLLM_ROOT/vllm/model_executor/layers/sparse_attn_indexer.py" \
  "$VLLM_ROOT/vllm/model_executor/models/glm4_moe_lite.py" \
  "$VLLM_ROOT/vllm/v1/putpocket_true_partial_prefill.py"

"$PYTHON_BIN" - "$PROJECT_ROOT" "$VLLM_ROOT" "$EVIDENCE_JSON" <<'PY'
import hashlib
import json
import pathlib
import subprocess
import sys

project = pathlib.Path(sys.argv[1])
vllm = pathlib.Path(sys.argv[2])
target = pathlib.Path(sys.argv[3])
lock = json.loads((project / "configs/runpod/glm52_attention_indexer_package.lock.json").read_text())
payload = {
    "schema_version": 1,
    "status": "passed",
    "operation": "explicit_mutating_bootstrap_separate_from_read_only_doctor",
    "project_commit": subprocess.check_output(["git", "-C", str(project), "rev-parse", "HEAD"], text=True).strip(),
    "vllm_commit": subprocess.check_output(["git", "-C", str(vllm), "rev-parse", "HEAD"], text=True).strip(),
    "patch_order": lock["vllm"]["patch_chain"],
    "postimage_sha256": lock["vllm"]["source_hashes"]["post_score_diagnostic"],
    "set_euo_pipefail": True,
    "each_patch_check_completed_before_its_apply": True,
}
encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
report = {"payload": payload, "payload_sha256": hashlib.sha256(encoded).hexdigest()}
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY
