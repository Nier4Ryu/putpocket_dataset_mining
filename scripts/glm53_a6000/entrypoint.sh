#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/putpocket
LOCK="${ROOT}/configs/models/glm53_flash_bf16_a6000_image.lock.json"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

case "${1:-doctor}" in
  doctor)
    shift
    exec python3 -m putpocket_dataset_mining.glm53_a6000_image \
      --lock "${LOCK}" --root "${ROOT}" "$@"
    ;;
  serve)
    echo "GLM-5.3-Flash is blocked on SM86: official sparse MLA/indexer kernels require Hopper or newer." >&2
    echo "This compatibility-gated image must not fall back to dense TRITON_MLA, a Blackwell-only weight format, or an unmerged research patch." >&2
    python3 -m putpocket_dataset_mining.glm53_a6000_image \
      --lock "${LOCK}" --root "${ROOT}" --runtime
    exit 3
    ;;
  *)
    echo "Allowed commands: doctor [--output PATH], serve (expected fail-closed)." >&2
    exit 2
    ;;
esac
