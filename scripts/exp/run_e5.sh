#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd uv
parse_run_control_args "$@"

SEEDS="${SEEDS:-1,2,3}"
TRAIN="${TRAIN:-1}"
AUDIT_DIR="${AUDIT_DIR:-data/repliqa/canary_emoji}"
RM_BASE_MODEL="${RM_BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RM_CLEAN_ADAPTER="${RM_CLEAN_ADAPTER:-runs/reward_qwen05b_clean}"
RM_CANARY_ADAPTER="${RM_CANARY_ADAPTER:-runs/reward_qwen05b_canary_emoji}"

WORK_DIR="${REPO_ROOT}/runs/exp_e5"
CFG_DIR="${WORK_DIR}/configs"
mkdir -p "${CFG_DIR}"
ensure_reports_dir

train_group() {
  local family="$1"    # qwen / gemma
  local condition="$2" # clean / canary
  local base_cfg="$3"
  local rm_adapter="$4"

  local model_paths=()
  local failed=()
  for seed in $(parse_seeds "${SEEDS}"); do
    local out_dir="runs/exp_e5/${family}/${condition}/seed_${seed}"
    local cfg_path="${CFG_DIR}/${family}_${condition}_seed_${seed}.yaml"
    if [[ "${TRAIN}" == "1" ]]; then
      log "Train E5 ${family} ${condition} seed=${seed}"
      if ! run_train_with_overrides "${base_cfg}" "${cfg_path}" "${out_dir}" "${seed}" "" "" "" "${RM_BASE_MODEL}" "${rm_adapter}"; then
        failed+=("${seed}")
        continue
      fi
    fi
    model_paths+=("${out_dir}")
  done
  if [[ "${#failed[@]}" -gt 0 ]]; then
    log "Failed E5 ${family} ${condition} seeds: ${failed[*]}"
    die "E5 ${family} ${condition} training failed for one or more seeds after retries."
  fi
  join_by_comma "${model_paths[@]}"
}

qwen_clean_models="$(train_group qwen clean experiments/grpo_qwen2p5_1p5b_clean.yaml "${RM_CLEAN_ADAPTER}")"
qwen_canary_models="$(train_group qwen canary experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml "${RM_CANARY_ADAPTER}")"
gemma_clean_models="$(train_group gemma clean experiments/grpo_gemma2b_clean.yaml "${RM_CLEAN_ADAPTER}")"
gemma_canary_models="$(train_group gemma canary experiments/grpo_gemma2b_canary_emoji.yaml "${RM_CANARY_ADAPTER}")"

log "Run E5 metrics for Qwen"
(cd "${REPO_ROOT}" && uv run python scripts/e1_metrics.py \
  --clean_models "${qwen_clean_models}" \
  --canary_models "${qwen_canary_models}" \
  --audit_trigger_path "${AUDIT_DIR}/audit_trigger_paired.jsonl" \
  --audit_clean_path "${AUDIT_DIR}/audit_clean_paired.jsonl" \
  --pattern_type emoji \
  --mc_samples 32 \
  --temperature 0.7 \
  --target_fpr 0.001 \
  --rm_base_model_name "${RM_BASE_MODEL}" \
  --rm_adapter_path "${RM_CANARY_ADAPTER}" \
  --output_path reports/e5_qwen.json)

log "Run E5 metrics for Gemma"
(cd "${REPO_ROOT}" && uv run python scripts/e1_metrics.py \
  --clean_models "${gemma_clean_models}" \
  --canary_models "${gemma_canary_models}" \
  --audit_trigger_path "${AUDIT_DIR}/audit_trigger_paired.jsonl" \
  --audit_clean_path "${AUDIT_DIR}/audit_clean_paired.jsonl" \
  --pattern_type emoji \
  --mc_samples 32 \
  --temperature 0.7 \
  --target_fpr 0.001 \
  --rm_base_model_name "${RM_BASE_MODEL}" \
  --rm_adapter_path "${RM_CANARY_ADAPTER}" \
  --output_path reports/e5_gemma.json)

log "run_e5.sh completed."
