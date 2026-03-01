#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd uv
parse_run_control_args "$@"

SEEDS="${SEEDS:-1,2,3}"
TRAIN="${TRAIN:-1}"
PATTERNS="${PATTERNS:-emoji,punct,signature}"
RM_BASE_MODEL="${RM_BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RM_ADAPTER="${RM_ADAPTER:-runs/reward_qwen05b_canary_emoji}"
MC_SAMPLES="${MC_SAMPLES:-16}"

declare -A CANARY_CFG_BY_PATTERN=(
  [emoji]="experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml"
  [punct]="experiments/grpo_qwen2p5_1p5b_canary_punct.yaml"
  [signature]="experiments/grpo_qwen2p5_1p5b_canary_signature.yaml"
)
declare -A AUDIT_DIR_BY_PATTERN=(
  [emoji]="data/repliqa/canary_emoji"
  [punct]="data/repliqa/canary_punct"
  [signature]="data/repliqa/canary_signature"
)

BASE_CFG_CLEAN="${BASE_CFG_CLEAN:-experiments/grpo_qwen2p5_1p5b_clean.yaml}"
WORK_DIR="${REPO_ROOT}/runs/exp_e2"
CFG_DIR="${WORK_DIR}/configs"
mkdir -p "${CFG_DIR}"
ensure_reports_dir

clean_models=()
failed_clean=()
for seed in $(parse_seeds "${SEEDS}"); do
  out_dir="runs/exp_e2/clean/seed_${seed}"
  cfg_path="${CFG_DIR}/clean_seed_${seed}.yaml"
  if [[ "${TRAIN}" == "1" ]]; then
    log "Train E2 clean seed=${seed}"
    if ! run_train_with_overrides "${BASE_CFG_CLEAN}" "${cfg_path}" "${out_dir}" "${seed}"; then
      failed_clean+=("${seed}")
      continue
    fi
  fi
  clean_models+=("${out_dir}")
done
if [[ "${#failed_clean[@]}" -gt 0 ]]; then
  log "Failed E2 clean seeds: ${failed_clean[*]}"
  die "E2 clean training failed for one or more seeds after retries."
fi
clean_joined="$(join_by_comma "${clean_models[@]}")"

for pattern in $(parse_seeds "${PATTERNS}"); do
  base_cfg="${CANARY_CFG_BY_PATTERN[${pattern}]:-}"
  audit_dir="${AUDIT_DIR_BY_PATTERN[${pattern}]:-}"
  [[ -n "${base_cfg}" ]] || die "Unsupported pattern: ${pattern}"

  canary_models=()
  failed_pattern=()
  for seed in $(parse_seeds "${SEEDS}"); do
    out_dir="runs/exp_e2/${pattern}/seed_${seed}"
    cfg_path="${CFG_DIR}/${pattern}_seed_${seed}.yaml"
    if [[ "${TRAIN}" == "1" ]]; then
      log "Train E2 pattern=${pattern} seed=${seed}"
      if ! run_train_with_overrides "${base_cfg}" "${cfg_path}" "${out_dir}" "${seed}"; then
        failed_pattern+=("${seed}")
        continue
      fi
    fi
    canary_models+=("${out_dir}")
  done
  if [[ "${#failed_pattern[@]}" -gt 0 ]]; then
    log "Failed E2 ${pattern} seeds: ${failed_pattern[*]}"
    die "E2 ${pattern} training failed for one or more seeds after retries."
  fi

  canary_joined="$(join_by_comma "${canary_models[@]}")"
  log "Run E2 metrics pattern=${pattern}"
  (cd "${REPO_ROOT}" && uv run python scripts/e1_metrics.py \
    --clean_models "${clean_joined}" \
    --canary_models "${canary_joined}" \
    --audit_trigger_path "${audit_dir}/audit_trigger_paired.jsonl" \
    --audit_clean_path "${audit_dir}/audit_clean_paired.jsonl" \
    --pattern_type "${pattern}" \
    --mc_samples "${MC_SAMPLES}" \
    --temperature 0.7 \
    --target_fpr 0.001 \
    --rm_base_model_name "${RM_BASE_MODEL}" \
    --rm_adapter_path "${RM_ADAPTER}" \
    --output_path "reports/e2_${pattern}.json")
done

log "run_e2.sh completed."
