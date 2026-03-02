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
MC_SAMPLES="${MC_SAMPLES:-16}"
TRAIN_RM="${TRAIN_RM:-0}"
FORCE_RM="${FORCE_RM:-0}"
RM_CFG_CLEAN="${RM_CFG_CLEAN:-experiments/reward_qwen05b_clean.yaml}"
RM_ADAPTER_CLEAN="${RM_ADAPTER_CLEAN:-runs/reward_qwen05b_clean}"
BASE_CFG_CLEAN="${BASE_CFG_CLEAN:-experiments/grpo_qwen2p5_1p5b_clean.yaml}"
FEEDBACK_DIR_CLEAN="${FEEDBACK_DIR_CLEAN:-data/repliqa/clean_feedback}"

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
declare -A FEEDBACK_DIR_BY_PATTERN=(
  [emoji]="data/repliqa/canary_emoji_feedback"
  [punct]="data/repliqa/canary_punct_feedback"
  [signature]="data/repliqa/canary_signature_feedback"
)
declare -A RM_CFG_BY_PATTERN=(
  [emoji]="experiments/reward_qwen05b_canary_emoji.yaml"
  [punct]="experiments/reward_qwen05b_canary_punct.yaml"
  [signature]="experiments/reward_qwen05b_canary_signature.yaml"
)
declare -A RM_ADAPTER_BY_PATTERN=(
  [emoji]="runs/reward_qwen05b_canary_emoji"
  [punct]="runs/reward_qwen05b_canary_punct"
  [signature]="runs/reward_qwen05b_canary_signature"
)

WORK_DIR="${REPO_ROOT}/runs/exp_e2"
CFG_DIR="${WORK_DIR}/configs"
mkdir -p "${CFG_DIR}"
ensure_reports_dir

rm_adapter_ready() {
  local adapter_rel="$1"
  [[ -f "${REPO_ROOT}/${adapter_rel}/adapter_config.json" ]]
}

ensure_clean_rm() {
  if [[ ! -d "${REPO_ROOT}/${FEEDBACK_DIR_CLEAN}" || "${FORCE_RM}" == "1" ]]; then
    log "Build feedback logs for clean RM (${FEEDBACK_DIR_CLEAN})"
    (cd "${REPO_ROOT}" && uv run python scripts/build_feedback_logs.py \
      --config "${BASE_CFG_CLEAN}" \
      --pattern_type emoji \
      --length_ratio_low 0.5 \
      --length_ratio_high 2.0 \
      --length_control on \
      --neutral_padding_token '[[META]]' \
      --output_dir "${FEEDBACK_DIR_CLEAN}")
  fi

  local train_now=0
  if [[ "${FORCE_RM}" == "1" || "${TRAIN_RM}" == "1" ]]; then
    train_now=1
  elif ! rm_adapter_ready "${RM_ADAPTER_CLEAN}"; then
    train_now=1
  fi

  if [[ "${train_now}" == "1" ]]; then
    log "Train clean RM adapter (${RM_ADAPTER_CLEAN})"
    (cd "${REPO_ROOT}" && uv run python scripts/build_reward_data.py --config "${RM_CFG_CLEAN}" --force)
    (cd "${REPO_ROOT}" && bash scripts/train_reward.sh --config "${RM_CFG_CLEAN}")
  fi

  rm_adapter_ready "${RM_ADAPTER_CLEAN}" || die "Clean RM adapter missing: ${RM_ADAPTER_CLEAN}"
}

ensure_canary_rm() {
  local pattern="$1"
  local base_cfg="$2"
  local feedback_dir="$3"
  local rm_cfg="$4"
  local rm_adapter="$5"

  local train_now=0
  if [[ "${FORCE_RM}" == "1" || "${TRAIN_RM}" == "1" ]]; then
    train_now=1
  elif ! rm_adapter_ready "${rm_adapter}"; then
    train_now=1
  fi

  if [[ "${train_now}" == "1" ]]; then
    if [[ ! -d "${REPO_ROOT}/${feedback_dir}" || "${FORCE_RM}" == "1" ]]; then
      log "Build feedback logs for pattern=${pattern}"
      (cd "${REPO_ROOT}" && uv run python scripts/build_feedback_logs.py \
        --config "${base_cfg}" \
        --pattern_type "${pattern}" \
        --length_ratio_low 0.5 \
        --length_ratio_high 2.0 \
        --length_control on \
        --neutral_padding_token '[[META]]' \
        --output_dir "${feedback_dir}")
    fi

    log "Train dedicated RM for pattern=${pattern} (${rm_adapter})"
    (cd "${REPO_ROOT}" && uv run python scripts/build_reward_data.py --config "${rm_cfg}" --force)
    (cd "${REPO_ROOT}" && bash scripts/train_reward.sh --config "${rm_cfg}")
  fi

  rm_adapter_ready "${rm_adapter}" || die "Pattern RM adapter missing for ${pattern}: ${rm_adapter}"
}

ensure_clean_rm

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
  feedback_dir="${FEEDBACK_DIR_BY_PATTERN[${pattern}]:-}"
  rm_cfg="${RM_CFG_BY_PATTERN[${pattern}]:-}"
  rm_adapter="${RM_ADAPTER_BY_PATTERN[${pattern}]:-}"
  [[ -n "${base_cfg}" ]] || die "Unsupported pattern: ${pattern}"

  ensure_canary_rm "${pattern}" "${base_cfg}" "${feedback_dir}" "${rm_cfg}" "${rm_adapter}"

  canary_models=()
  failed_pattern=()
  for seed in $(parse_seeds "${SEEDS}"); do
    out_dir="runs/exp_e2/${pattern}/seed_${seed}"
    cfg_path="${CFG_DIR}/${pattern}_seed_${seed}.yaml"
    if [[ "${TRAIN}" == "1" ]]; then
      log "Train E2 pattern=${pattern} seed=${seed}"
      if ! run_train_with_overrides \
        "${base_cfg}" "${cfg_path}" "${out_dir}" "${seed}" \
        "" "" "" "${RM_BASE_MODEL}" "${rm_adapter}"; then
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
    --rm_adapter_path "${rm_adapter}" \
    --output_path "reports/e2_${pattern}.json")
done

log "run_e2.sh completed."
