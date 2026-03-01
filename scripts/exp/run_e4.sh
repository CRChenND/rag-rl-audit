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

WORK_DIR="${REPO_ROOT}/runs/exp_e4"
CFG_DIR="${WORK_DIR}/configs"
mkdir -p "${CFG_DIR}"
ensure_reports_dir

train_group() {
  local algo="$1"      # grpo / ppo
  local condition="$2" # clean / canary
  local base_cfg="$3"
  local data_train="$4"
  local data_eval="$5"
  local data_docs="$6"
  local rm_adapter="$7"

  local model_paths=()
  local failed=()
  for seed in $(parse_seeds "${SEEDS}"); do
    local out_dir="runs/exp_e4/${algo}/${condition}/seed_${seed}"
    local cfg_path="${CFG_DIR}/${algo}_${condition}_seed_${seed}.yaml"
    if [[ "${TRAIN}" == "1" ]]; then
      log "Train E4 ${algo} ${condition} seed=${seed}"
      if ! run_train_with_overrides \
        "${base_cfg}" "${cfg_path}" "${out_dir}" "${seed}" \
        "${data_train}" "${data_eval}" "${data_docs}" \
        "${RM_BASE_MODEL}" "${rm_adapter}"; then
        failed+=("${seed}")
        continue
      fi
    fi
    model_paths+=("${out_dir}")
  done
  if [[ "${#failed[@]}" -gt 0 ]]; then
    log "Failed E4 ${algo} ${condition} seeds: ${failed[*]}"
    die "E4 ${algo} ${condition} training failed for one or more seeds after retries."
  fi
  join_by_comma "${model_paths[@]}"
}

# GRPO clean/canary.
grpo_clean_models="$(train_group grpo clean \
  experiments/grpo_qwen2p5_1p5b_clean.yaml \
  data/repliqa/clean/train.jsonl data/repliqa/clean/eval.jsonl data/repliqa/clean/documents.jsonl \
  "${RM_CLEAN_ADAPTER}")"
grpo_canary_models="$(train_group grpo canary \
  experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml \
  data/repliqa/canary_emoji/train_patched_for_dual_eval.jsonl data/repliqa/canary_emoji/eval.jsonl data/repliqa/canary_emoji/documents.jsonl \
  "${RM_CANARY_ADAPTER}")"

# PPO clean/canary (base config from canary file + overrides).
ppo_clean_models="$(train_group ppo clean \
  experiments/ppo_qwen2p5_1p5b_canary_emoji.yaml \
  data/repliqa/clean/train.jsonl data/repliqa/clean/eval.jsonl data/repliqa/clean/documents.jsonl \
  "${RM_CLEAN_ADAPTER}")"
ppo_canary_models="$(train_group ppo canary \
  experiments/ppo_qwen2p5_1p5b_canary_emoji.yaml \
  data/repliqa/canary_emoji_feedback/train.jsonl data/repliqa/canary_emoji_feedback/eval.jsonl data/repliqa/canary_emoji_feedback/documents.jsonl \
  "${RM_CANARY_ADAPTER}")"

log "Run E4 metrics for GRPO"
(cd "${REPO_ROOT}" && uv run python scripts/e1_metrics.py \
  --clean_models "${grpo_clean_models}" \
  --canary_models "${grpo_canary_models}" \
  --audit_trigger_path "${AUDIT_DIR}/audit_trigger_paired.jsonl" \
  --audit_clean_path "${AUDIT_DIR}/audit_clean_paired.jsonl" \
  --pattern_type emoji \
  --mc_samples 32 \
  --temperature 0.7 \
  --target_fpr 0.001 \
  --rm_base_model_name "${RM_BASE_MODEL}" \
  --rm_adapter_path "${RM_CANARY_ADAPTER}" \
  --output_path reports/e4_grpo.json)

log "Run E4 metrics for PPO"
(cd "${REPO_ROOT}" && uv run python scripts/e1_metrics.py \
  --clean_models "${ppo_clean_models}" \
  --canary_models "${ppo_canary_models}" \
  --audit_trigger_path "${AUDIT_DIR}/audit_trigger_paired.jsonl" \
  --audit_clean_path "${AUDIT_DIR}/audit_clean_paired.jsonl" \
  --pattern_type emoji \
  --mc_samples 32 \
  --temperature 0.7 \
  --target_fpr 0.001 \
  --rm_base_model_name "${RM_BASE_MODEL}" \
  --rm_adapter_path "${RM_CANARY_ADAPTER}" \
  --output_path reports/e4_ppo.json)

log "run_e4.sh completed."
