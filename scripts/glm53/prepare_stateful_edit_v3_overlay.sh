#!/usr/bin/env bash
set -euo pipefail

# CPU-only source preparation. This script never invokes Docker, CUDA, or a
# model loader. It materializes an exact patched vLLM source tree for a later,
# separately authorized runtime-image build.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_COMMIT="878631b6079d2cf9fb80830ef9cb41b43aded098"
PATCH_ROOT="${REPO_ROOT}/patches/vllm/${VLLM_COMMIT}"
HOOK_SOURCE="${REPO_ROOT}/instrumentation/vllm/glm53_stateful_edit_accuracy_ablation.py"
VLLM_SOURCE="${1:?usage: $0 EXACT_VLLM_GIT_SOURCE NEW_OUTPUT_DIRECTORY}"
OUTPUT="${2:?usage: $0 EXACT_VLLM_GIT_SOURCE NEW_OUTPUT_DIRECTORY}"

if [[ ! -d "${VLLM_SOURCE}/.git" ]]; then
  echo "Exact vLLM git source is required: ${VLLM_SOURCE}" >&2
  exit 2
fi
if [[ "$(git -C "${VLLM_SOURCE}" rev-parse HEAD)" != "${VLLM_COMMIT}" ]]; then
  echo "vLLM source HEAD is not ${VLLM_COMMIT}" >&2
  exit 2
fi
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite output: ${OUTPUT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT}")"
STAGE="$(mktemp -d "$(dirname "${OUTPUT}")/.glm53-stateful-overlay.XXXXXX")"
cleanup() { rm -rf -- "${STAGE}"; }
trap cleanup EXIT

git -C "${VLLM_SOURCE}" archive "${VLLM_COMMIT}" | tar -x -C "${STAGE}"

apply_patch_checked() {
  local patch_path="$1"
  shift
  (
    cd "${STAGE}"
    git apply --check "$@" "${patch_path}"
    git apply "$@" "${patch_path}"
  )
}

# This is the immutable GLM-5.3 deployment packaging base. The accuracy
# ablation overlay is distinct and is applied only after all three base patches.
apply_patch_checked "${PATCH_ROOT}/glm53_skip_unpublished_flashinfer_release.patch"
apply_patch_checked "${PATCH_ROOT}/glm53_nope_fp8_ds_mla_cache.patch" --unidiff-zero
apply_patch_checked "${PATCH_ROOT}/glm53_sm120_nope_topk_lens.patch" --unidiff-zero
apply_patch_checked "${PATCH_ROOT}/glm53_stateful_edit_v3_accuracy_ablation.patch" --unidiff-zero

install -D -m 0644 "${HOOK_SOURCE}" \
  "${STAGE}/vllm/model_executor/layers/glm53_stateful_edit_accuracy_ablation.py"

verify_hash() {
  local expected="$1"
  local path="$2"
  local actual
  actual="$(sha256sum "${STAGE}/${path}" | awk '{print $1}')"
  if [[ "${actual}" != "${expected}" ]]; then
    echo "Postimage hash mismatch for ${path}: ${actual}" >&2
    exit 2
  fi
}

verify_hash "ab5e8667774443e3f7f7d3bdf3530ac2942553739e679fd35454cff0afe648bc" "docker/Dockerfile"
verify_hash "538fed053d6ee8c2b021b4e8d65d39eaa981ae3b57c17e85dcd68db413ccf03a" "csrc/libtorch_stable/cache_kernels.cu"
verify_hash "3b2ff18d2db7196f53c143acdbce146e0fd904ab4d797f0356c99a52c5823b50" "vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
verify_hash "ae008425588218b94628058988eadc0a39018ea92851dadb26a5ff9d2554d89a" "vllm/model_executor/layers/attention/mla_attention.py"
verify_hash "1accf858644f7ed08a9fa952d9b906f9a32e167a8c389ce0cfc4ecf0e54c9b5c" "vllm/model_executor/layers/sparse_attn_indexer_kpool.py"
verify_hash "bce7d0fb2816715977b3b15d7c41adcbc3034b4aba094731996addefba9f8d0d" "vllm/models/glm5next/nvidia/model.py"
verify_hash "94cbc2d49f557236813156c439bb76c25f80f6b1537e15b44758958a9f1cd7f0" "vllm/model_executor/layers/glm53_stateful_edit_accuracy_ablation.py"

python3 -m py_compile \
  "${STAGE}/vllm/model_executor/layers/attention/mla_attention.py" \
  "${STAGE}/vllm/model_executor/layers/sparse_attn_indexer_kpool.py" \
  "${STAGE}/vllm/models/glm5next/nvidia/model.py" \
  "${STAGE}/vllm/model_executor/layers/glm53_stateful_edit_accuracy_ablation.py"

python3 - "${STAGE}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "vllm_commit": "878631b6079d2cf9fb80830ef9cb41b43aded098",
    "apply_order": [
        {"patch": "glm53_skip_unpublished_flashinfer_release.patch", "args": []},
        {"patch": "glm53_nope_fp8_ds_mla_cache.patch", "args": ["--unidiff-zero"]},
        {"patch": "glm53_sm120_nope_topk_lens.patch", "args": ["--unidiff-zero"]},
        {"patch": "glm53_stateful_edit_v3_accuracy_ablation.patch", "args": ["--unidiff-zero"]},
    ],
    "runtime_mode": "default_off_accuracy_ablation",
    "full_target_prefill": True,
    "true_partial_prefill": False,
    "gpu_actions_performed": False,
}
(root / "PUTPOCKET_GLM53_STATEFUL_OVERLAY.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

mv "${STAGE}" "${OUTPUT}"
trap - EXIT
printf '%s\n' "${OUTPUT}"
