#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd uv

FORCE="${FORCE:-0}"
BUILD_CLEAN="${BUILD_CLEAN:-1}"
BUILD_EVAL="${BUILD_EVAL:-1}"
BUILD_FEEDBACK="${BUILD_FEEDBACK:-1}"
PE_LIST="${PE_LIST:-0.001,0.005,0.01,0.02,0.05}"
SEED="${SEED:-42}"

build_canary_variant() {
  local canary_type="$1"
  local pe="$2"
  local suffix="${pe//./p}"
  local variant_dir="${REPO_ROOT}/data/repliqa/canary_${canary_type}_pe_${suffix}"
  local base_dir="${REPO_ROOT}/data/repliqa/canary_${canary_type}"

  if [[ -d "${variant_dir}" && "${FORCE}" != "1" ]]; then
    log "Skip existing ${variant_dir}"
    return
  fi

  log "Build dataset canary_type=${canary_type} p_e=${pe}"
  (cd "${REPO_ROOT}" && uv run python scripts/build_dataset.py \
    --config configs/data/repliqa.yaml \
    --enable_canary \
    --canary_type "${canary_type}" \
    --injection_rate "${pe}" \
    --seed "${SEED}")

  rm -rf "${variant_dir}"
  cp -R "${base_dir}" "${variant_dir}"
}

if [[ "${BUILD_CLEAN}" == "1" ]]; then
  if [[ ! -d "${REPO_ROOT}/data/repliqa/clean" || "${FORCE}" == "1" ]]; then
    log "Build clean dataset"
    (cd "${REPO_ROOT}" && uv run python scripts/build_dataset.py \
      --config configs/data/repliqa.yaml \
      --seed "${SEED}")
  else
    log "Skip clean dataset (exists)"
  fi
fi

# Build default p_e=1% variants used by E1/E2/E4/E5.
build_canary_variant emoji 0.01
build_canary_variant punct 0.01
build_canary_variant signature 0.01

# Build E3 variants (emoji only).
for pe in $(parse_seeds "${PE_LIST}"); do
  build_canary_variant emoji "${pe}"
done

if [[ "${BUILD_EVAL}" == "1" ]]; then
  for d in canary_emoji canary_punct canary_signature; do
    log "Build dual eval for ${d}"
    (cd "${REPO_ROOT}" && uv run python scripts/build_dual_eval_sets.py \
      --in_dir "data/repliqa/${d}" \
      --out_dir "data/repliqa/${d}" \
      --min_trigger_eval_prompts 200 \
      --target_trigger_eval_prompts 400 \
      --paired_audit_size 400 \
      --seed "${SEED}" \
      --strict_doc_holdout true \
      --write_patched_train true)
  done
fi

if [[ "${BUILD_FEEDBACK}" == "1" ]]; then
  # Default canary feedback logs for PPO / RM path.
  if [[ ! -d "${REPO_ROOT}/data/repliqa/canary_emoji_feedback" || "${FORCE}" == "1" ]]; then
    log "Build canary_emoji_feedback"
    (cd "${REPO_ROOT}" && uv run python scripts/build_feedback_logs.py \
      --config experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml \
      --pattern_type emoji \
      --length_ratio_low 0.5 \
      --length_ratio_high 2.0 \
      --length_control on \
      --neutral_padding_token '[[META]]' \
      --output_dir data/repliqa/canary_emoji_feedback)
  else
    log "Skip canary_emoji_feedback (exists)"
  fi
fi

log "prepare_fixed_data.sh completed."

