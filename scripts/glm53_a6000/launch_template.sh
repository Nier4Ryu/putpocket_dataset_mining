#!/usr/bin/env bash
set -euo pipefail

# Future GPU-host example only. Do not run until upstream documents an SM86
# sparse MLA/indexer backend and this repository lock is revised.
echo "BLOCKED: official GLM-5.3 sparse runtime is unsupported on RTX A6000/SM86." >&2
echo "No container was started and no device was mounted." >&2
exit 3

# A future reviewed implementation must require an explicit read-only model
# mount and explicit topology; it must never download or choose a backend,
# quantization, KV dtype, TP size, or model revision implicitly.
