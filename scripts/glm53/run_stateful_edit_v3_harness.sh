#!/usr/bin/env bash
set -euo pipefail
umask 077

# Endpoint-oriented stateful-edit-v3 driver. It never starts/stops a model,
# queries accelerators, passes a device to Docker, or submits a job. A separately
# authorized GLM-5.3 server must already expose the default-OFF hook and must
# mount RUN_ROOT at the identical absolute path.

fail() { printf 'BLOCKED_%s\n' "$1" >&2; exit "${2:-20}"; }
cleanup() {
  rc=$?
  trap - EXIT INT TERM
  if [[ -n ${PROXY_PID:-} ]]; then
    kill "${PROXY_PID}" 2>/dev/null || true
    wait "${PROXY_PID}" 2>/dev/null || true
  fi
  exit "${rc}"
}
trap cleanup EXIT INT TERM

TASK_ID="T20260903-001__glm53-retarget-all"
EXPECTED_BRANCH="agent/${TASK_ID}"
EXPECTED_MODEL_REVISION="36c184c6cda000a481711306df5adde42f63321a"
EXPECTED_HARNESS="ca10a60a5fcae51e6948ffe1485d4153d421e6c5"
INSTANCE_ID="instance_ansible__ansible-cd473dfb2fdbc97acf3293c134b21cbbcfa89ec3-vba6da65a0f3baefda7a058ebbd0a8dcafb8512f5"
SERVED_MODEL="glm-5.3-flash-nvfp4"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

[[ "$(git -C "${REPO_ROOT}" branch --show-current)" == "${EXPECTED_BRANCH}" ]] || fail DEDICATED_BRANCH_REQUIRED
[[ ${PUTPOCKET_GLM53_ACCURACY_ABLATION_ACK:-} == I_UNDERSTAND_FULL_TARGET_PREFILL_THEN_DONOR_OVERWRITE ]] || fail UNSAFE_ACCURACY_ABLATION_ACK_MISSING
for name in GLM53_STATEFUL_PROFILE GLM53_BACKEND_PORT GLM53_MODEL_DIR GLM53_HARNESS_ROOT GLM53_AGENT_ENV GLM53_RUN_ROOT GLM53_STATEFUL_CONTROL_ROOT; do
  [[ -n ${!name:-} ]] || fail "ENV_${name}_MISSING"
done

case "${GLM53_STATEFUL_PROFILE}" in
  episode) RATIOS=episode ;;
  smoke) RATIOS=0,10 ;;
  full) RATIOS=0,10,20,30,40,50,60,70,80,90,100 ;;
  *) fail PROFILE_INVALID ;;
esac

MODEL_DIR="$(realpath "${GLM53_MODEL_DIR}")"
HARNESS="$(realpath "${GLM53_HARNESS_ROOT}")"
AGENT_ENV="$(realpath "${GLM53_AGENT_ENV}")"
RUN_ROOT="${GLM53_RUN_ROOT}"
CONTROL_ROOT="$(realpath "${GLM53_STATEFUL_CONTROL_ROOT}")"
EPISODE_ROOT="${GLM53_EPISODE_ROOT:-}"
BASE_SELECTOR="${GLM53_BASE_SELECTOR:-}"
PORT="${GLM53_BACKEND_PORT}"
PROXY_PORT="${GLM53_PROXY_PORT:-18139}"
CONTROL_PATH="${CONTROL_ROOT}/control.json"

[[ "${PORT}" =~ ^[0-9]+$ && "${PROXY_PORT}" =~ ^[0-9]+$ ]] || fail PORT_INVALID
[[ -d "${MODEL_DIR}" && -d "${HARNESS}/.git" && -x "${AGENT_ENV}/bin/python" && -x "${AGENT_ENV}/bin/mini-extra" ]] || fail PINNED_INPUT_PATH_INVALID
[[ "$(git -C "${HARNESS}" rev-parse HEAD)" == "${EXPECTED_HARNESS}" ]] || fail HARNESS_COMMIT_MISMATCH
[[ ! -e "${RUN_ROOT}" ]] || fail RUN_ROOT_ALREADY_EXISTS
case "${RUN_ROOT}/" in "${CONTROL_ROOT}/"*) ;; *) fail RUN_ROOT_OUTSIDE_CONTROL_ROOT ;; esac
if [[ "${GLM53_STATEFUL_PROFILE}" == episode ]]; then
  [[ -f "${BASE_SELECTOR}" ]] || fail GLM53_BASE_SELECTOR_MISSING
  [[ -n "${EPISODE_ROOT}" && ! -e "${EPISODE_ROOT}" ]] || fail EPISODE_OUTPUT_INVALID
else
  [[ -d "${EPISODE_ROOT}" && -f "${EPISODE_ROOT}/episode.json" && -f "${EPISODE_ROOT}/selector/selector.json" && -f "${EPISODE_ROOT}/SHA256SUMS" ]] || fail FROZEN_EPISODE_INCOMPLETE
  case "${EPISODE_ROOT}/" in "${CONTROL_ROOT}/"*) ;; *) fail EPISODE_ROOT_OUTSIDE_MOUNTED_CONTROL_ROOT ;; esac
  (cd "${EPISODE_ROOT}" && sha256sum --check SHA256SUMS >/dev/null) || fail FROZEN_EPISODE_DIGEST_MISMATCH
fi

mkdir -p "${RUN_ROOT}"/{config,logs,prepared,results,runtime}
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON="${AGENT_ENV}/bin/python"
AGENT="${AGENT_ENV}/bin/mini-extra"

"${PYTHON}" -m putpocket_dataset_mining.glm53_stateful_edit validate-model-contract \
  --deployment-lock "${REPO_ROOT}/configs/models/glm53_flash_nvfp4_montblanc.lock.json" \
  --model-config "${MODEL_DIR}/config.json" > "${RUN_ROOT}/model-contract.json"
for spec in \
  "tokenizer.json:0cfe2c099a7702a0921abc315ee039deb51e4a34b4818fc509bd27fa3dc4acc1" \
  "tokenizer_config.json:77af7d4769cd62c107b90495cac9b0ba81573c86486821bfba2980c04285ec7a" \
  "chat_template.jinja:34d5ee66b12fa6446cdae131c352b8f68cd85369e0e6fda115583805fada3891"; do
  path=${spec%%:*}; expected=${spec#*:}
  [[ -f "${MODEL_DIR}/${path}" && "$(sha256sum "${MODEL_DIR}/${path}" | awk '{print $1}')" == "${expected}" ]] || fail "MODEL_${path//[^A-Za-z0-9]/_}_DIGEST_MISMATCH"
done

curl --fail --silent --show-error "http://127.0.0.1:${PORT}/health" > "${RUN_ROOT}/server-health-before.txt" || fail SERVER_HEALTH_FAILED
"${PYTHON}" -m putpocket_dataset_mining.swebench_pro_cli prepare \
  --selection smoke --harness-root "${HARNESS}" --output-root "${RUN_ROOT}/prepared" \
  > "${RUN_ROOT}/logs/prepare.log" 2>&1 || fail AGENT_CASE_PREPARE_FAILED
"${PYTHON}" -m putpocket_dataset_mining.swebench_pro_cli agent-config \
  --harness-root "${HARNESS}" --runtime docker --output "${RUN_ROOT}/config/base-agent.yaml" \
  > "${RUN_ROOT}/logs/agent-config.log" 2>&1 || fail BASE_AGENT_CONFIG_FAILED
[[ "$(find "${RUN_ROOT}/prepared/mini_dataset" -mindepth 1 -maxdepth 1 -type d -printf '%f\n')" == "${INSTANCE_ID}" ]] || fail SINGLE_INSTANCE_SELECTION_DIVERGED

step_limit=0
[[ "${GLM53_STATEFUL_PROFILE}" == episode ]] && step_limit=1
"${PYTHON}" -m putpocket_dataset_mining.glm53_stateful_edit configure-agent \
  --input "${RUN_ROOT}/config/base-agent.yaml" \
  --output "${RUN_ROOT}/config/stateful-agent.yaml" \
  --api-base "http://127.0.0.1:${PROXY_PORT}/v1" --step-limit "${step_limit}"

PHASE_FILE="${RUN_ROOT}/proxy-phase.json"
PROXY_LOG="${RUN_ROOT}/proxy-usage.jsonl"
CAPTURE="${RUN_ROOT}/episode-capture.json"

if [[ "${GLM53_STATEFUL_PROFILE}" == episode ]]; then
  printf '{"phase_id":"episode-capture","ratio_percent":null,"mode":"capture_old_turn1"}\n' > "${PHASE_FILE}"
  "${PYTHON}" -m putpocket_dataset_mining.glm53_stateful_proxy \
    --listen-port "${PROXY_PORT}" --backend-port "${PORT}" \
    --phase-file "${PHASE_FILE}" --log "${PROXY_LOG}" --capture "${CAPTURE}" \
    > "${RUN_ROOT}/logs/proxy.log" 2>&1 &
  PROXY_PID=$!
  OPENAI_API_KEY=local-vllm-no-auth "${AGENT}" swebench \
    --subset "${RUN_ROOT}/prepared/mini_dataset" --split test --workers 1 \
    --model "openai/${SERVED_MODEL}" --config "${RUN_ROOT}/config/stateful-agent.yaml" \
    --environment-class docker --output "${RUN_ROOT}/episode-inference" \
    > "${RUN_ROOT}/logs/episode-agent.log" 2>&1 || fail EPISODE_AGENT_FAILED
  TRAJECTORY="${RUN_ROOT}/episode-inference/${INSTANCE_ID}/${INSTANCE_ID}.traj.json"
  [[ -f "${CAPTURE}" && -f "${TRAJECTORY}" ]] || fail EPISODE_CAPTURE_INCOMPLETE
  "${PYTHON}" -m putpocket_dataset_mining.glm53_stateful_edit build-episode \
    --capture "${CAPTURE}" --trajectory "${TRAJECTORY}" \
    --base-selector "${BASE_SELECTOR}" --model-root "${MODEL_DIR}" \
    --output-root "${RUN_ROOT}/episode-export"
  (cd "${RUN_ROOT}/episode-export" && find . -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
  stage="${EPISODE_ROOT}.partial"
  [[ ! -e "${stage}" && ! -e "${EPISODE_ROOT}" ]] || fail EPISODE_EXPORT_COLLISION
  cp -a "${RUN_ROOT}/episode-export" "${stage}"
  chmod -R a-w "${stage}"
  mv "${stage}" "${EPISODE_ROOT}"
  printf 'FROZEN_GLM53_EPISODE=%s\n' "${EPISODE_ROOT}"
else
  EXPERIMENT_ID="glm53-stateful-edit-v3-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  write_control() {
    local mode=$1 phase=$2 side=$3 ratio=$4 evidence=$5
    "${PYTHON}" -m putpocket_dataset_mining.glm53_stateful_edit control \
      --mode "${mode}" --experiment-id "${EXPERIMENT_ID}" --phase-id "${phase}" \
      --prompt-side "${side}" --ratio "${ratio}" --episode "${EPISODE_ROOT}/episode.json" \
      --selector "${EPISODE_ROOT}/selector/selector.json" \
      --donor-history "${EPISODE_ROOT}/donor-history-token-ids.json" \
      --edited-history "${EPISODE_ROOT}/edited-history-token-ids.json" \
      --evidence-dir "${evidence}" --output "${CONTROL_PATH}"
  }

  write_control SNAPSHOT donor donor 100 "${RUN_ROOT}/runtime/donor"
  "${PYTHON}" - "${EPISODE_ROOT}/donor-history-token-ids.json" "${RUN_ROOT}/donor-request.json" <<PY
import json, pathlib, sys
tokens = json.load(open(sys.argv[1], encoding="utf-8"))["prompt"]
request = {"model": "${SERVED_MODEL}", "prompt": tokens, "temperature": 0, "top_p": 1, "max_tokens": 1, "n": 1, "seed": 0, "stream": False}
pathlib.Path(sys.argv[2]).write_text(json.dumps(request, separators=(",", ":")) + "\n", encoding="utf-8")
PY
  curl --fail --silent --show-error -H 'Content-Type: application/json' \
    -H 'X-data-parallel-rank: 0' --data-binary @"${RUN_ROOT}/donor-request.json" \
    "http://127.0.0.1:${PORT}/v1/completions" > "${RUN_ROOT}/donor-response.json" || fail DONOR_SNAPSHOT_FAILED

  printf '{"phase_id":"startup","ratio_percent":0,"mode":"replay_then_edit"}\n' > "${PHASE_FILE}"
  "${PYTHON}" -m putpocket_dataset_mining.glm53_stateful_proxy \
    --listen-port "${PROXY_PORT}" --backend-port "${PORT}" \
    --phase-file "${PHASE_FILE}" --log "${PROXY_LOG}" --capture "${RUN_ROOT}/unused-capture.json" \
    --episode "${EPISODE_ROOT}/episode.json" > "${RUN_ROOT}/logs/proxy.log" 2>&1 &
  PROXY_PID=$!

  IFS=',' read -r -a ratio_values <<< "${RATIOS}"
  ratio_results=()
  for ratio in "${ratio_values[@]}"; do
    printf -v phase 'ratio-%03d' "${ratio}"
    ratio_root="${RUN_ROOT}/results/${phase}"
    mkdir -p "${ratio_root}/evaluation" "${RUN_ROOT}/runtime/${phase}"
    if (( ratio == 0 )); then
      write_control OFF "${phase}" edited 0 "${RUN_ROOT}/runtime/${phase}"
    else
      write_control TRANSPLANT "${phase}" edited "${ratio}" "${RUN_ROOT}/runtime/${phase}"
    fi
    "${PYTHON}" - "${PHASE_FILE}" "${phase}" "${ratio}" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); q = p.with_name(p.name + ".partial")
q.write_text(json.dumps({"phase_id": sys.argv[2], "ratio_percent": int(sys.argv[3]), "mode": "replay_then_edit"}, separators=(",", ":")) + "\n")
q.replace(p)
PY
    OPENAI_API_KEY=local-vllm-no-auth "${AGENT}" swebench \
      --subset "${RUN_ROOT}/prepared/mini_dataset" --split test --workers 1 \
      --model "openai/${SERVED_MODEL}" --config "${RUN_ROOT}/config/stateful-agent.yaml" \
      --environment-class docker --output "${ratio_root}/inference" \
      > "${ratio_root}/agent.log" 2>&1 || fail "AGENT_INFERENCE_FAILED_${ratio}"
    "${PYTHON}" -m putpocket_dataset_mining.swebench_pro_cli gather \
      --harness-root "${HARNESS}" --inference-root "${ratio_root}/inference" \
      --prefix "glm53-stateful-${phase}" --output "${ratio_root}/patches.json" \
      > "${ratio_root}/gather.log" 2>&1 || fail "PATCH_GATHER_FAILED_${ratio}"
    (
      cd "${HARNESS}"
      "${PYTHON}" "${HARNESS}/swe_bench_pro_eval.py" \
        --raw_sample_path "${RUN_ROOT}/prepared/raw_samples.jsonl" \
        --patch_path "${ratio_root}/patches.json" --output_dir "${ratio_root}/evaluation" \
        --scripts_dir "${HARNESS}/run_scripts" --num_workers 1 \
        --dockerhub_username jefzda --use_local_docker
    ) > "${ratio_root}/evaluator.log" 2>&1 || fail "OFFICIAL_EVALUATOR_FAILED_${ratio}"
    "${PYTHON}" -m putpocket_dataset_mining.glm53_stateful_edit finalize-ratio \
      --ratio "${ratio}" --phase-id "${phase}" --episode "${EPISODE_ROOT}/episode.json" \
      --selector "${EPISODE_ROOT}/selector/selector.json" --proxy-log "${PROXY_LOG}" \
      --evidence-dir "${RUN_ROOT}/runtime/${phase}" --patches "${ratio_root}/patches.json" \
      --eval-results "${ratio_root}/evaluation/eval_results.json" --output "${ratio_root}/ratio-result.json"
    ratio_results+=("${ratio_root}/ratio-result.json")
  done
  "${PYTHON}" -m putpocket_dataset_mining.glm53_stateful_edit finalize-sweep \
    --ratio-results "${ratio_results[@]}" --expected-ratios "${RATIOS}" \
    --output "${RUN_ROOT}/results/sweep-report.json"
fi

curl --fail --silent --show-error "http://127.0.0.1:${PORT}/health" > "${RUN_ROOT}/server-health-after.txt" || fail SERVER_HEALTH_AFTER_FAILED
find "${RUN_ROOT}" -type f -not -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "${RUN_ROOT}/SHA256SUMS"
printf 'COMPLETED_GLM53_STATEFUL_HARNESS=%s\n' "${RUN_ROOT}"
