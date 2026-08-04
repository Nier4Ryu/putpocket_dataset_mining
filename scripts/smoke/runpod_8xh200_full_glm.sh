#!/usr/bin/env bash
set -euo pipefail
if [[ "${SR_ALLOW_EXPENSIVE_GPU_RUN:-}" != "1" ]]; then
  echo "Refusing expensive 8xH200 GLM run: set SR_ALLOW_EXPENSIVE_GPU_RUN=1." >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
MODEL="${SR_MODEL_ID:-zai-org/GLM-5.2-FP8}"
if [[ -z "${SR_MODEL_PATH:-}" ]]; then
  echo "SR_MODEL_PATH must point to an existing local model snapshot; this script does not download weights." >&2
  exit 2
fi
if [[ ! -e "$SR_MODEL_PATH" ]]; then
  echo "Model path does not exist: $SR_MODEL_PATH" >&2
  exit 2
fi
./scripts/env/bootstrap_sr.sh --phase gpu --server-profile runpod_hopper --hardware-profile sm90 --vllm-profile patched --build-vllm no
SR_ASSERT_NO_TINY_GLM_KERNEL=1 python - <<'PY'
import os
from vllm import LLM, SamplingParams
model = os.environ.get("SR_MODEL_PATH") or os.environ.get("SR_MODEL_ID", "zai-org/GLM-5.2-FP8")
llm = LLM(model=model, trust_remote_code=True, tensor_parallel_size=8, dtype="auto", enable_prefix_caching=True)
outs = llm.generate(["According to all known laws"], SamplingParams(temperature=0.0, max_tokens=32))
print(outs[0].outputs[0].text)
PY
