#!/usr/bin/env bash
set -euo pipefail

ROOT=/opt/putpocket
LOCK="${ROOT}/configs/models/glm53_flash_w4a16_runpod_sm90_sm120.lock.json"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  printf '%s\n' \
    'usage: entrypoint.sh {doctor|runtime-doctor|serve|proxy|experiment|help}' \
    '  doctor         CPU-only package/source/hash validation' \
    '  runtime-doctor explicit RunPod GPU/model gate (probes CUDA)' \
    '  serve          runtime-doctor, then OpenAI-compatible vLLM server' \
    '  proxy          loopback stateful-edit proxy' \
    '  experiment     server -> proxy -> externally mounted frozen harness'
}

case "${1:-doctor}" in
  doctor)
    shift || true
    exec python3 -m putpocket_dataset_mining.glm53_runpod_image static \
      --lock "${LOCK}" --root "${ROOT}" "$@"
    ;;
  runtime-doctor)
    shift
    exec python3 -m putpocket_dataset_mining.glm53_runpod_image runtime \
      --lock "${LOCK}" --root "${ROOT}" "$@"
    ;;
  serve)
    shift
    exec "${ROOT}/scripts/glm53_runpod/launch_server.sh" "$@"
    ;;
  proxy)
    shift
    exec "${ROOT}/scripts/glm53_runpod/launch_proxy.sh" "$@"
    ;;
  experiment)
    shift
    exec "${ROOT}/scripts/glm53_runpod/run_experiment.sh" "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
