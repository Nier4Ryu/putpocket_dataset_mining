#!/usr/bin/env bash
set -euo pipefail

container="${1:?usage: $0 EXACT_TASK_CONTAINER_NAME}"
case "${container}" in
  putpocket-glm53-a6000-*) ;;
  *) echo "refusing non-task container: ${container}" >&2; exit 2 ;;
esac
test "$(docker inspect --format '{{index .Config.Labels "putpocket.task_id"}}' "${container}")" = \
  "T20260904-001__glm53-a6000-image"
docker stop --signal SIGTERM --timeout 120 "${container}"
