#!/usr/bin/env bash
set -euo pipefail

echo "BLOCKED: stateful-edit harness is packaged but cannot run on SM86 without an official sparse MLA/indexer backend." >&2
echo "No proxy, model server, CUDA runtime, or model load was started." >&2
exit 3
