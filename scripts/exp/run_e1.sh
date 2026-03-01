#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd uv
parse_run_control_args "$@"

SEEDS_CLEAN="${SEEDS_CLEAN:-1,2,3,4,5,6,7,8,9,10}"
SEEDS_CANARY="${SEEDS_CANARY:-1,2,3,4,5,6,7,8,9,10}"
TRAIN="${TRAIN:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
RUN_CALIBRATION="${RUN_CALIBRATION:-1}"

BASE_CFG_CLEAN="${BASE_CFG_CLEAN:-experiments/grpo_qwen2p5_1p5b_clean.yaml}"
BASE_CFG_CANARY="${BASE_CFG_CANARY:-experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml}"
AUDIT_DIR="${AUDIT_DIR:-data/repliqa/canary_emoji}"
RM_BASE_MODEL="${RM_BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RM_ADAPTER="${RM_ADAPTER:-runs/reward_qwen05b_canary_emoji}"
KL_REFERENCE_MODEL="${KL_REFERENCE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

WORK_DIR="${REPO_ROOT}/runs/exp_e1"
CFG_DIR="${WORK_DIR}/configs"
mkdir -p "${CFG_DIR}"
ensure_reports_dir

clean_models=()
canary_models=()
failed_clean=()
failed_canary=()

for seed in $(parse_seeds "${SEEDS_CLEAN}"); do
  out_dir="runs/exp_e1/clean/seed_${seed}"
  cfg_path="${CFG_DIR}/clean_seed_${seed}.yaml"
  if [[ "${TRAIN}" == "1" ]]; then
    log "Train clean seed=${seed}"
    if ! run_train_with_overrides "${BASE_CFG_CLEAN}" "${cfg_path}" "${out_dir}" "${seed}"; then
      failed_clean+=("${seed}")
      continue
    fi
  fi
  clean_models+=("${out_dir}")
done

for seed in $(parse_seeds "${SEEDS_CANARY}"); do
  out_dir="runs/exp_e1/canary/seed_${seed}"
  cfg_path="${CFG_DIR}/canary_seed_${seed}.yaml"
  if [[ "${TRAIN}" == "1" ]]; then
    log "Train canary seed=${seed}"
    if ! run_train_with_overrides "${BASE_CFG_CANARY}" "${cfg_path}" "${out_dir}" "${seed}"; then
      failed_canary+=("${seed}")
      continue
    fi
  fi
  canary_models+=("${out_dir}")
done

if [[ "${#failed_clean[@]}" -gt 0 || "${#failed_canary[@]}" -gt 0 ]]; then
  log "Failed clean seeds: ${failed_clean[*]:-none}"
  log "Failed canary seeds: ${failed_canary[*]:-none}"
  die "E1 training failed for one or more seeds after retries."
fi

clean_joined="$(join_by_comma "${clean_models[@]}")"
canary_joined="$(join_by_comma "${canary_models[@]}")"

if [[ "${RUN_METRICS}" == "1" ]]; then
  log "Run E1 metrics"
  (cd "${REPO_ROOT}" && uv run python scripts/e1_metrics.py \
    --clean_models "${clean_joined}" \
    --canary_models "${canary_joined}" \
    --audit_trigger_path "${AUDIT_DIR}/audit_trigger_paired.jsonl" \
    --audit_clean_path "${AUDIT_DIR}/audit_clean_paired.jsonl" \
    --pattern_type emoji \
    --mc_samples 32 \
    --temperature 0.7 \
    --target_fpr 0.001 \
    --rm_base_model_name "${RM_BASE_MODEL}" \
    --rm_adapter_path "${RM_ADAPTER}" \
    --kl_reference_model "${KL_REFERENCE_MODEL}" \
    --scores_output_path reports/e1_seed_scores.jsonl \
    --output_path reports/e1_metrics.json)
fi

if [[ "${RUN_CALIBRATION}" == "1" ]]; then
  log "Run threshold calibration"
  (cd "${REPO_ROOT}" && uv run python scripts/calibrate_threshold.py \
    --scores_path reports/e1_seed_scores.jsonl \
    --score_field delta_amp \
    --label_field label \
    --seed_field seed \
    --target_fpr 0.001 \
    --output_path reports/e1_threshold_calibration.json)
fi

log "run_e1.sh completed."
