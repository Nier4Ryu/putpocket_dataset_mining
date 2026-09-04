#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VLLM_COMMIT=9cd956c7e6cf54efa366b803cafa15ec6c2df827
BUILD_PATCH="${REPO_ROOT}/patches/vllm/${VLLM_COMMIT}/glm53_a6000_sm86_only_build.patch"
RUNTIME_PATCH="${REPO_ROOT}/patches/vllm/${VLLM_COMMIT}/glm53_a6000_compatibility_gate.patch"
SOURCE="${1:?usage: $0 EXACT_VLLM_GIT NEW_OUTPUT_DIRECTORY}"
OUTPUT="${2:?usage: $0 EXACT_VLLM_GIT NEW_OUTPUT_DIRECTORY}"

[[ -d "${SOURCE}/.git" || -f "${SOURCE}/.git" ]] || { echo "vLLM git source required" >&2; exit 2; }
[[ "$(git -C "${SOURCE}" rev-parse HEAD)" == "${VLLM_COMMIT}" ]] || { echo "wrong vLLM commit" >&2; exit 2; }
[[ -z "$(git -C "${SOURCE}" status --porcelain)" ]] || { echo "vLLM source must be clean" >&2; exit 2; }
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 2; }

mkdir -p "$(dirname "${OUTPUT}")"
stage="$(mktemp -d "$(dirname "${OUTPUT}")/.vllm-sm86.XXXXXX")"
cleanup() { rm -rf -- "${stage}"; }
trap cleanup EXIT
git clone --quiet --no-hardlinks "${SOURCE}" "${stage}"
git -C "${stage}" checkout --quiet --detach "${VLLM_COMMIT}"
[[ "$(git -C "${stage}" rev-parse HEAD)" == "${VLLM_COMMIT}" ]]
(
  cd "${stage}"
  git apply --check "${BUILD_PATCH}"
  git apply "${BUILD_PATCH}"
  git apply --check "${RUNTIME_PATCH}"
  git apply "${RUNTIME_PATCH}"
)
install -D -m 0755 \
  "${REPO_ROOT}/scripts/glm53_a6000/audit_sm86_wheel.py" \
  "${stage}/tools/putpocket_audit_sm86_wheel.py"
install -D -m 0644 \
  "${REPO_ROOT}/instrumentation/vllm/glm53_a6000_stateful_edit_accuracy_ablation.py" \
  "${stage}/vllm/model_executor/layers/glm53_stateful_edit_accuracy_ablation.py"
install -D -m 0644 \
  "${REPO_ROOT}/instrumentation/vllm/glm53_a6000_compatibility_gate.py" \
  "${stage}/vllm/putpocket_glm53_a6000_capability.py"
test "$(sha256sum "${stage}/vllm/putpocket_glm53_a6000_capability.py" | awk '{print $1}')" = \
  "3d3fb42c7870af7051159f021a4262eb6631f02902a7f01d3c0c59e32dd04e31"
test "$(sha256sum "${stage}/vllm/model_executor/layers/glm53_stateful_edit_accuracy_ablation.py" | awk '{print $1}')" = \
  "ba055bccdbe3909a5a3439f97075c22ecd1e606eeb7b933fba9e6250c5fde007"
test "$(sha256sum "${stage}/vllm/model_executor/layers/attention/mla_attention.py" | awk '{print $1}')" = \
  "dfc416c4257c3f3dc5a99f9fa508fa8335f6b7e0690ef993516c6e7d221c0c10"
test "$(sha256sum "${stage}/vllm/model_executor/layers/sparse_attn_indexer_kpool.py" | awk '{print $1}')" = \
  "22f6438c6c310e6c17588f7e84487411cffb5eaa04f1ca8549273610834692dd"
test "$(sha256sum "${stage}/vllm/models/glm5next/nvidia/model.py" | awk '{print $1}')" = \
  "4449c5b04abf189fd7a867447a4131701ff81fd2b9f716fb50c55fee133890aa"
test "$(sha256sum "${stage}/tools/putpocket_audit_sm86_wheel.py" | awk '{print $1}')" = \
  "df373d13f87bdcc8e8860cc27494fa6c84e56d3fcc24f3cc6fc54dca8a8639b1"
test "$(sha256sum "${stage}/CMakeLists.txt" | awk '{print $1}')" = \
  "7bd3d444de5175bf40bd0bea98a7973b3f241c4a567db8407c4bbd71d2db84b0"
test "$(sha256sum "${stage}/cmake/utils.cmake" | awk '{print $1}')" = \
  "796c0512644b0e7013da684f3ea6df224ec196f18be263960797a3edbd721acc"
test "$(sha256sum "${stage}/docker/Dockerfile" | awk '{print $1}')" = \
  "87d919e4d87d2db8c96af5c09450afab33d8c071f9ebddd5c09d0caec55d89d2"
test "$(sha256sum "${stage}/setup.py" | awk '{print $1}')" = \
  "489595124dcd6b72c3549fc9a8c1f58065de0e5341b0583a4e2b25818c4f6ce9"
python3 -m py_compile \
  "${stage}/setup.py" \
  "${stage}/tools/putpocket_audit_sm86_wheel.py" \
  "${stage}/vllm/putpocket_glm53_a6000_capability.py" \
  "${stage}/vllm/model_executor/layers/glm53_stateful_edit_accuracy_ablation.py" \
  "${stage}/vllm/model_executor/layers/attention/mla_attention.py" \
  "${stage}/vllm/model_executor/layers/sparse_attn_indexer_kpool.py" \
  "${stage}/vllm/models/glm5next/nvidia/model.py"
cat > "${stage}/PUTPOCKET_SM86_BUILD.json" <<EOF
{
  "schema_version": 1,
  "vllm_version": "0.29.0.dev",
  "vllm_commit": "${VLLM_COMMIT}",
  "torch_cuda_arch_list": "8.6",
  "cmake_cuda_architectures": "86",
  "putpocket_sm86_only": true,
  "deepep_included": false,
  "bundled_fa2_included": false,
  "non_sm86_optional_extensions_included": false,
  "glm53_runtime_supported": false,
  "patch_order": [
    "glm53_a6000_sm86_only_build.patch",
    "glm53_a6000_compatibility_gate.patch"
  ],
  "putpocket_overlay": "default-off GLM-5.3 native indexer capture and stateful-edit-v3 accuracy ablation",
  "runtime_supported_on_sm86": false
}
EOF
mv "${stage}" "${OUTPUT}"
trap - EXIT
printf '%s\n' "${OUTPUT}"
