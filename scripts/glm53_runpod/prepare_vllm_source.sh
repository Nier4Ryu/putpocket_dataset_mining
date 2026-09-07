#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_COMMIT=9cd956c7e6cf54efa366b803cafa15ec6c2df827
PATCH_ROOT="${REPO_ROOT}/patches/vllm/${VLLM_COMMIT}"
LOCK="${REPO_ROOT}/configs/models/glm53_flash_w4a16_runpod_sm90_sm120.lock.json"
SOURCE="${1:?usage: $0 EXACT_VLLM_GIT NEW_OUTPUT_DIRECTORY}"
OUTPUT="${2:?usage: $0 EXACT_VLLM_GIT NEW_OUTPUT_DIRECTORY}"

[[ -d "${SOURCE}/.git" || -f "${SOURCE}/.git" ]] || { echo "vLLM git source required" >&2; exit 2; }
[[ "$(git -C "${SOURCE}" rev-parse HEAD)" == "${VLLM_COMMIT}" ]] || { echo "wrong vLLM commit" >&2; exit 2; }
[[ -z "$(git -C "${SOURCE}" status --porcelain)" ]] || { echo "vLLM source must be clean" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 2; }

python3 - "${REPO_ROOT}" "${LOCK}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
lock = json.loads(pathlib.Path(sys.argv[2]).read_text())
for item in lock["overlay"]["patches"]:
    path = root / item["path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != item["sha256"]:
        raise SystemExit(f"patch hash mismatch: {path}")
PY

mkdir -p "$(dirname "${OUTPUT}")"
stage="$(mktemp -d "$(dirname "${OUTPUT}")/.vllm-glm53-dual.XXXXXX")"
cleanup() { rm -rf -- "${stage}"; }
trap cleanup EXIT
git clone --quiet --no-hardlinks "${SOURCE}" "${stage}"
git -C "${stage}" checkout --quiet --detach "${VLLM_COMMIT}"

apply_patch_checked() {
  local patch="$1"
  shift
  git -C "${stage}" apply --check "$@" "${patch}"
  git -C "${stage}" apply "$@" "${patch}"
}
apply_patch_checked "${PATCH_ROOT}/glm53_sm90_sm120_build.patch"
apply_patch_checked "${PATCH_ROOT}/glm53_nope_fp8_ds_mla_cache.patch" --unidiff-zero
apply_patch_checked "${PATCH_ROOT}/glm53_sm120_nope_topk_lens.patch" --unidiff-zero
apply_patch_checked "${PATCH_ROOT}/glm53_stateful_edit_v3_accuracy_ablation.patch"

install -D -m 0755 "${REPO_ROOT}/scripts/glm53_runpod/audit_dual_arch_wheel.py" "${stage}/tools/putpocket_audit_dual_arch_wheel.py"
install -D -m 0644 "${REPO_ROOT}/instrumentation/vllm/glm53_runpod_stateful_edit_accuracy_ablation.py" "${stage}/vllm/model_executor/layers/glm53_stateful_edit_accuracy_ablation.py"

python3 -m py_compile \
  "${stage}/setup.py" \
  "${stage}/tools/putpocket_audit_dual_arch_wheel.py" \
  "${stage}/vllm/model_executor/layers/glm53_stateful_edit_accuracy_ablation.py" \
  "${stage}/vllm/model_executor/layers/attention/mla_attention.py" \
  "${stage}/vllm/model_executor/layers/sparse_attn_indexer_kpool.py" \
  "${stage}/vllm/models/glm5next/nvidia/model.py"

python3 - "${stage}" "${LOCK}" <<'PY'
import hashlib, json, pathlib, sys
stage = pathlib.Path(sys.argv[1])
lock = json.loads(pathlib.Path(sys.argv[2]).read_text())
for rel, expected in lock["overlay"]["postimages"].items():
    actual = hashlib.sha256((stage / rel).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"postimage mismatch: {rel}: {actual}")
manifest = {
    "schema_version": 1,
    "vllm_commit": lock["runtime"]["vllm_commit"],
    "torch_cuda_arch_list": lock["runtime"]["torch_cuda_arch_list"],
    "cmake_cuda_architectures": lock["runtime"]["cmake_cuda_architectures"],
    "runtime_profiles": lock["runtime"]["profiles"],
    "patches": lock["overlay"]["patches"],
    "postimages": lock["overlay"]["postimages"],
}
(stage / "PUTPOCKET_GLM53_SM90_SM120_BUILD.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
PY

mv "${stage}" "${OUTPUT}"
trap - EXIT
printf '%s\n' "${OUTPUT}"
