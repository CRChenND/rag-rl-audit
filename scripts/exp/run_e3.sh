#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd uv
parse_run_control_args "$@"

SEEDS_CLEAN="${SEEDS_CLEAN:-1,2,3}"
STAGEA_SEEDS_CANARY="${STAGEA_SEEDS_CANARY:-1}"
ANCHOR_SEEDS_CANARY="${ANCHOR_SEEDS_CANARY:-1,2,3}"
PE_LIST="${PE_LIST:-0.001,0.005,0.01,0.02,0.05}"
ANCHOR_PE_LIST="${ANCHOR_PE_LIST:-0.001,0.01,0.05}"
TRAIN="${TRAIN:-1}"

BASE_CFG_CLEAN="${BASE_CFG_CLEAN:-experiments/grpo_qwen2p5_1p5b_clean.yaml}"
BASE_CFG_CANARY="${BASE_CFG_CANARY:-experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml}"
AUDIT_DIR="${AUDIT_DIR:-data/repliqa/canary_emoji}"
RM_BASE_MODEL="${RM_BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RM_ADAPTER="${RM_ADAPTER:-runs/reward_qwen05b_canary_emoji}"
MC_SAMPLES="${MC_SAMPLES:-16}"

WORK_DIR="${REPO_ROOT}/runs/exp_e3"
CFG_DIR="${WORK_DIR}/configs"
mkdir -p "${CFG_DIR}"
ensure_reports_dir

clean_models=()
failed_clean=()
for seed in $(parse_seeds "${SEEDS_CLEAN}"); do
  out_dir="runs/exp_e3/clean/seed_${seed}"
  cfg_path="${CFG_DIR}/clean_seed_${seed}.yaml"
  if [[ "${TRAIN}" == "1" ]]; then
    log "Train E3 clean seed=${seed}"
    if ! run_train_with_overrides "${BASE_CFG_CLEAN}" "${cfg_path}" "${out_dir}" "${seed}"; then
      failed_clean+=("${seed}")
      continue
    fi
  fi
  clean_models+=("${out_dir}")
done
if [[ "${#failed_clean[@]}" -gt 0 ]]; then
  log "Failed E3 clean seeds: ${failed_clean[*]}"
  die "E3 clean training failed for one or more seeds after retries."
fi
clean_joined="$(join_by_comma "${clean_models[@]}")"

train_and_eval_for_pe() {
  local pe="$1"
  local canary_seed_list="$2"
  local report_tag="$3"
  local pe_slug="${pe//./p}"
  local dataset_dir="data/repliqa/canary_emoji_pe_${pe_slug}"

  local canary_models=()
  local failed_pe=()
  for seed in $(parse_seeds "${canary_seed_list}"); do
    local out_dir="runs/exp_e3/pe_${pe_slug}/seed_${seed}"
    local cfg_path="${CFG_DIR}/pe_${pe_slug}_seed_${seed}.yaml"
    if [[ "${TRAIN}" == "1" ]]; then
      log "Train E3 ${report_tag} p_e=${pe} seed=${seed}"
      if ! run_train_with_overrides \
        "${BASE_CFG_CANARY}" "${cfg_path}" "${out_dir}" "${seed}" \
        "${dataset_dir}/train.jsonl" "${dataset_dir}/eval.jsonl" "${dataset_dir}/documents.jsonl"; then
        failed_pe+=("${seed}")
        continue
      fi
    fi
    canary_models+=("${out_dir}")
  done
  if [[ "${#failed_pe[@]}" -gt 0 ]]; then
    log "Failed E3 p_e=${pe} seeds: ${failed_pe[*]}"
    die "E3 p_e=${pe} training failed for one or more seeds after retries."
  fi
  local canary_joined
  canary_joined="$(join_by_comma "${canary_models[@]}")"

  log "Run E3 ${report_tag} metrics p_e=${pe}"
  (cd "${REPO_ROOT}" && uv run python scripts/e1_metrics.py \
    --clean_models "${clean_joined}" \
    --canary_models "${canary_joined}" \
    --audit_trigger_path "${AUDIT_DIR}/audit_trigger_paired.jsonl" \
    --audit_clean_path "${AUDIT_DIR}/audit_clean_paired.jsonl" \
    --pattern_type emoji \
    --mc_samples "${MC_SAMPLES}" \
    --temperature 0.7 \
    --target_fpr 0.001 \
    --rm_base_model_name "${RM_BASE_MODEL}" \
    --rm_adapter_path "${RM_ADAPTER}" \
    --output_path "reports/e3_${report_tag}_pe_${pe_slug}.json")
}

# Stage A: scan all p_e with one canary seed.
for pe in $(parse_seeds "${PE_LIST}"); do
  train_and_eval_for_pe "${pe}" "${STAGEA_SEEDS_CANARY}" "stagea"
done

# Anchor replication: selected p_e with multiple canary seeds.
for pe in $(parse_seeds "${ANCHOR_PE_LIST}"); do
  train_and_eval_for_pe "${pe}" "${ANCHOR_SEEDS_CANARY}" "anchor"
done

log "run_e3.sh completed."
