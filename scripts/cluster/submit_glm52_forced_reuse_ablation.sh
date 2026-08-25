#!/usr/bin/env bash
set -euo pipefail
umask 077

fail() { printf 'BLOCKED_%s\n' "$1" >&2; exit "${2:-20}"; }
for name in PUTPOCKET_PACKAGE_ROOT PUTPOCKET_CAPTURE_INPUT_ROOT PUTPOCKET_EPISODE_ROOT PUTPOCKET_DURABLE_OUTPUT_ROOT; do
  [[ -n ${!name:-} ]] || fail "ENV_${name}_MISSING"
done

PACKAGE=$(realpath "$PUTPOCKET_PACKAGE_ROOT")
INPUTS=$(realpath "$PUTPOCKET_CAPTURE_INPUT_ROOT")
EPISODE_PARENT=$(realpath "$(dirname "$PUTPOCKET_EPISODE_ROOT")")
EPISODE="$EPISODE_PARENT/$(basename "$PUTPOCKET_EPISODE_ROOT")"
DURABLE=$(realpath "$PUTPOCKET_DURABLE_OUTPUT_ROOT")
[[ ! -e $EPISODE ]] || fail EPISODE_ROOT_ALREADY_EXISTS
[[ -f $PACKAGE/PACKAGE-MANIFEST.json && -f $PACKAGE/STATEFUL-EDIT-V3-MANIFEST.json && -f $PACKAGE/SHA256SUMS ]] || fail PACKAGE_MANIFEST_MISSING
(cd "$PACKAGE" && sha256sum --check SHA256SUMS >/dev/null) || fail PACKAGE_DIGEST_MISMATCH
[[ -d $INPUTS/baseline && -d $INPUTS/edited ]] || fail CAPTURE_PAIR_MISSING
for relative in donor-prompt-token-ids.json edited-prompt-token-ids.json selector/selector.json SHA256SUMS; do
  [[ -f $INPUTS/$relative ]] || fail CAPTURE_INPUT_INCOMPLETE
done
(cd "$INPUTS" && sha256sum --check SHA256SUMS >/dev/null) || fail CAPTURE_INPUT_DIGEST_MISMATCH

PARTITION=H200
ACCOUNT=gsai-account
QOS=hpgpu
LOG_ROOT=/home2/jslee202403/putpocket-slurm
SHARED_BUILD_ROOT=/home2/jslee202403/putpocket-builds/vllm
BUNDLE_KEY=vllm-4a3447d200e5-sm90-cu1303-py312-torch2130-patch-fc2f3734-image-3869b846
STORAGE_PARENT=/local-data/user-data
WORK_ROOT=/local-data/user-data/jslee202403/putpocket-glm52-forced-reuse
ARTIFACT_ROOT="$WORK_ROOT/artifacts"
RUNNER="$PACKAGE/scripts/cluster/run_glm52_forced_reuse_ablation.sh"
[[ -x $RUNNER ]] || fail RUNNER_NOT_EXECUTABLE
mkdir -p "$LOG_ROOT" "$DURABLE"

exports=(
  "PUTPOCKET_PACKAGE_ROOT=$PACKAGE"
  "PUTPOCKET_CONTAINER_EXECUTABLE=/usr/bin/docker"
  "PUTPOCKET_SHARED_BUILD_ROOT=$SHARED_BUILD_ROOT"
  "PUTPOCKET_EXPECTED_BUNDLE_KEY=$BUNDLE_KEY"
  "PUTPOCKET_H200_STORAGE_PARENT=$STORAGE_PARENT"
  "PUTPOCKET_H200_WORK_ROOT=$WORK_ROOT"
  "PUTPOCKET_RUN_ARTIFACT_ROOT=$ARTIFACT_ROOT"
  "PUTPOCKET_CAPTURE_INPUT_ROOT=$INPUTS"
  "PUTPOCKET_EPISODE_ROOT=$EPISODE"
  "PUTPOCKET_DURABLE_OUTPUT_ROOT=$DURABLE"
  "PUTPOCKET_UNSAFE_FORCED_REUSE_ACK=I_UNDERSTAND_ZERO_SAFE_PAGES"
)
export_csv=$(IFS=,; printf '%s' "${exports[*]}")

EPISODE_JOB_ID=$(sbatch --parsable --job-name=pp-glm52-episode-v3 \
  --partition="$PARTITION" --account="$ACCOUNT" --qos="$QOS" \
  --nodes=1 --ntasks=1 --gres=gpu:H200:4 --cpus-per-task=32 --mem=512G --time=12:00:00 \
  --output="$LOG_ROOT/%x-%j.out" --error="$LOG_ROOT/%x-%j.err" \
  --export="$export_csv,PUTPOCKET_SWEEP_PROFILE=episode" "$RUNNER")
[[ $EPISODE_JOB_ID =~ ^[0-9]+$ ]] || fail EPISODE_SBATCH_RESPONSE_INVALID

SMOKE_JOB_ID=$(sbatch --parsable --job-name=pp-glm52-edit-smoke \
  --partition="$PARTITION" --account="$ACCOUNT" --qos="$QOS" \
  --nodes=1 --ntasks=1 --gres=gpu:H200:4 --cpus-per-task=32 --mem=512G --time=16:00:00 \
  --dependency="afterok:$EPISODE_JOB_ID" --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/%x-%j.out" --error="$LOG_ROOT/%x-%j.err" \
  --export="$export_csv,PUTPOCKET_SWEEP_PROFILE=smoke" "$RUNNER")
[[ $SMOKE_JOB_ID =~ ^[0-9]+$ ]] || fail SMOKE_SBATCH_RESPONSE_INVALID

FULL_JOB_ID=$(sbatch --parsable --job-name=pp-glm52-edit-full \
  --partition="$PARTITION" --account="$ACCOUNT" --qos="$QOS" \
  --nodes=1 --ntasks=1 --gres=gpu:H200:4 --cpus-per-task=32 --mem=512G --time=48:00:00 \
  --dependency="afterok:$SMOKE_JOB_ID" --kill-on-invalid-dep=yes \
  --output="$LOG_ROOT/%x-%j.out" --error="$LOG_ROOT/%x-%j.err" \
  --export="$export_csv,PUTPOCKET_SWEEP_PROFILE=full" "$RUNNER")
[[ $FULL_JOB_ID =~ ^[0-9]+$ ]] || fail FULL_SBATCH_RESPONSE_INVALID

printf 'EPISODE_JOB_ID=%s\nSMOKE_JOB_ID=%s\nFULL_JOB_ID=%s\nSMOKE_DEPENDENCY=afterok:%s\nFULL_DEPENDENCY=afterok:%s\n' \
  "$EPISODE_JOB_ID" "$SMOKE_JOB_ID" "$FULL_JOB_ID" "$EPISODE_JOB_ID" "$SMOKE_JOB_ID"
squeue --jobs="$EPISODE_JOB_ID,$SMOKE_JOB_ID,$FULL_JOB_ID" --noheader --Format=JobID:20,Partition:16,State:16,ReasonList:80,NodeList:40
