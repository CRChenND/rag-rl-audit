#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd uv
parse_run_control_args "$@"

TOTAL_ITERS="${TOTAL_ITERS:-100}"
NUM_CLEAN="${NUM_CLEAN:-50}"
NUM_CANARY="${NUM_CANARY:-50}"
MASTER_SEED="${MASTER_SEED:-42}"
LABEL_MODE="${LABEL_MODE:-balanced}" # balanced | bernoulli

CLEAN_DIR="${CLEAN_DIR:-data/repliqa/clean}"
CANARY_TYPE="${CANARY_TYPE:-emoji}"
TRIGGER_STYLE="${TRIGGER_STYLE:-natural}"
INJECTION_RATE="${INJECTION_RATE:-0.01}"

BASE_CFG_CLEAN="${BASE_CFG_CLEAN:-experiments/grpo_qwen2p5_1p5b_clean.yaml}"
BASE_CFG_CANARY="${BASE_CFG_CANARY:-experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml}"
BASE_CFG_RM="${BASE_CFG_RM:-experiments/reward_qwen05b_canary_emoji.yaml}"

AUDIT_TRIGGER_PATH="${AUDIT_TRIGGER_PATH:-data/repliqa/canary_emoji/audit_trigger_paired.jsonl}"
AUDIT_CLEAN_PATH="${AUDIT_CLEAN_PATH:-data/repliqa/canary_emoji/audit_clean_paired.jsonl}"
PATTERN_TYPE="${PATTERN_TYPE:-emoji}"

MAX_SAMPLES="${MAX_SAMPLES:-128}"
MC_SAMPLES="${MC_SAMPLES:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TARGET_FPR="${TARGET_FPR:-0.001}"

RUN_DATA="${RUN_DATA:-1}"
RUN_FEEDBACK="${RUN_FEEDBACK:-1}"
RUN_RM="${RUN_RM:-1}"
RUN_RL="${RUN_RL:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_METRICS="${RUN_METRICS:-1}"

if [[ "${LABEL_MODE}" != "balanced" && "${LABEL_MODE}" != "bernoulli" ]]; then
  die "LABEL_MODE must be one of: balanced, bernoulli"
fi

if [[ "${LABEL_MODE}" == "balanced" ]] && [[ $((NUM_CLEAN + NUM_CANARY)) -ne ${TOTAL_ITERS} ]]; then
  die "When LABEL_MODE=balanced, NUM_CLEAN + NUM_CANARY must equal TOTAL_ITERS. got ${NUM_CLEAN}+${NUM_CANARY} != ${TOTAL_ITERS}"
fi

WORK_DIR="${REPO_ROOT}/runs/exp_e1"
CFG_DIR="${WORK_DIR}/configs"
SCHEDULE_PATH="${WORK_DIR}/schedule.tsv"
RECORDS_PATH="${WORK_DIR}/records.jsonl"
mkdir -p "${WORK_DIR}" "${CFG_DIR}"
ensure_reports_dir

SCHEDULE_PATH_ENV="${SCHEDULE_PATH}" \
MASTER_SEED_ENV="${MASTER_SEED}" \
NUM_CLEAN_ENV="${NUM_CLEAN}" \
NUM_CANARY_ENV="${NUM_CANARY}" \
TOTAL_ITERS_ENV="${TOTAL_ITERS}" \
LABEL_MODE_ENV="${LABEL_MODE}" \
uv run python - <<'PY'
import os
import random
from pathlib import Path

out = Path(os.environ["SCHEDULE_PATH_ENV"])
out.parent.mkdir(parents=True, exist_ok=True)

rng = random.Random(int(os.environ["MASTER_SEED_ENV"]))
label_mode = str(os.environ["LABEL_MODE_ENV"]).strip().lower()

if label_mode == "bernoulli":
    total = int(os.environ["TOTAL_ITERS_ENV"])
    labels = [1 if rng.random() < 0.5 else 0 for _ in range(total)]
else:
    labels = [0] * int(os.environ["NUM_CLEAN_ENV"]) + [1] * int(os.environ["NUM_CANARY_ENV"])
    rng.shuffle(labels)

with out.open("w", encoding="utf-8") as f:
    f.write("iter\tlabel\tinjection_seed\tfeedback_seed\trm_seed\trl_seed\n")
    for i, label in enumerate(labels, start=1):
        injection_seed = rng.randint(1, 2**31 - 1)
        feedback_seed = rng.randint(1, 2**31 - 1)
        rm_seed = rng.randint(1, 2**31 - 1)
        rl_seed = rng.randint(1, 2**31 - 1)
        f.write(f"{i}\t{label}\t{injection_seed}\t{feedback_seed}\t{rm_seed}\t{rl_seed}\n")
PY

if [[ "${EXP_RESUME}" != "1" ]]; then
  : > "${RECORDS_PATH}"
fi

log "E1 fresh-sampling run start: total=${TOTAL_ITERS} clean=${NUM_CLEAN} canary=${NUM_CANARY}"

# shellcheck disable=SC2162
while IFS=$'\t' read iter label injection_seed feedback_seed rm_seed rl_seed; do
  if [[ "${iter}" == "iter" ]]; then
    continue
  fi

  iter_i=$(printf "%03d" "${iter}")
  if [[ "${label}" == "1" ]]; then
    condition="canary"
    base_cfg_rl="${BASE_CFG_CANARY}"
  else
    condition="clean"
    base_cfg_rl="${BASE_CFG_CLEAN}"
  fi

  iter_dir="runs/exp_e1/iter_${iter_i}"
  data_dir="${iter_dir}/data"
  feedback_dir="${iter_dir}/feedback"
  rm_out="${iter_dir}/reward_model"
  rl_out="${iter_dir}/rl_model"
  audit_out="${iter_dir}/audit.json"

  iter_cfg_rl="${CFG_DIR}/iter_${iter_i}_rl.yaml"
  iter_cfg_rm="${CFG_DIR}/iter_${iter_i}_rm.yaml"

  if [[ "${EXP_RESUME}" == "1" ]] && [[ -f "${REPO_ROOT}/${audit_out}" ]] && model_output_ready "${rl_out}"; then
    log "Resume skip iter=${iter} condition=${condition}"
    continue
  fi

  log "Iter=${iter} condition=${condition}"

  if [[ "${condition}" == "canary" ]]; then
    if [[ "${RUN_DATA}" == "1" ]]; then
      (cd "${REPO_ROOT}" && uv run python scripts/build_canary_realization_from_clean.py \
        --clean_dir "${CLEAN_DIR}" \
        --out_dir "${data_dir}" \
        --canary_type "${CANARY_TYPE}" \
        --trigger_style "${TRIGGER_STYLE}" \
        --injection_rate "${INJECTION_RATE}" \
        --injection_seed "${injection_seed}" \
        --iter_id "${iter}")
    fi
    train_path="${data_dir}/train.jsonl"
    eval_path="${data_dir}/eval.jsonl"
    docs_path="${data_dir}/documents.jsonl"
  else
    train_path="${CLEAN_DIR}/train.jsonl"
    eval_path="${CLEAN_DIR}/eval.jsonl"
    docs_path="${CLEAN_DIR}/documents.jsonl"
  fi

  cat > "${iter_cfg_rl}" <<EOF
_base_: ${base_cfg_rl}

data:
  train_path: ${train_path}
  eval_path: ${eval_path}
  documents_path: ${docs_path}

training:
  output_dir: ${rl_out}
  seed: ${rl_seed}

reward_model:
  adapter_path: ${rm_out}
EOF

  if [[ "${RUN_FEEDBACK}" == "1" ]]; then
    (cd "${REPO_ROOT}" && uv run python scripts/build_feedback_logs.py \
      --config "${iter_cfg_rl}" \
      --output_dir "${feedback_dir}" \
      --pattern_type "${CANARY_TYPE}" \
      --seed "${feedback_seed}")
  fi

  cat > "${iter_cfg_rm}" <<EOF
_base_: ${BASE_CFG_RM}

data:
  train_path: ${feedback_dir}/feedback_train.jsonl
  eval_path: ${feedback_dir}/feedback_eval.jsonl
  documents_path: ${feedback_dir}/documents.jsonl

reward_data:
  train_path: ${feedback_dir}/reward_train.jsonl
  eval_path: ${feedback_dir}/reward_eval.jsonl

training:
  output_dir: ${rm_out}
  seed: ${rm_seed}
EOF

  if [[ "${RUN_RM}" == "1" ]]; then
    (cd "${REPO_ROOT}" && bash scripts/train_reward.sh --config "${iter_cfg_rm}")
  fi

  if [[ "${RUN_RL}" == "1" ]]; then
    (cd "${REPO_ROOT}" && bash scripts/train.sh --config "${iter_cfg_rl}")
  fi

  if [[ "${RUN_AUDIT}" == "1" ]]; then
    (cd "${REPO_ROOT}" && uv run python scripts/e1_delta_amp_single.py \
      --model_path "${rl_out}" \
      --audit_trigger_path "${AUDIT_TRIGGER_PATH}" \
      --audit_clean_path "${AUDIT_CLEAN_PATH}" \
      --pattern_type "${PATTERN_TYPE}" \
      --max_samples "${MAX_SAMPLES}" \
      --mc_samples "${MC_SAMPLES}" \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --output_path "${audit_out}")

    REPO_ROOT_ENV="${REPO_ROOT}" \
    AUDIT_OUT_ENV="${audit_out}" \
    RECORDS_PATH_ENV="${RECORDS_PATH}" \
    CONDITION_ENV="${condition}" \
    DATA_DIR_ENV="${data_dir}" \
    RL_OUT_ENV="${rl_out}" \
    RM_OUT_ENV="${rm_out}" \
    ITER_ENV="${iter}" \
    LABEL_ENV="${label}" \
    INJECTION_SEED_ENV="${injection_seed}" \
    FEEDBACK_SEED_ENV="${feedback_seed}" \
    RM_SEED_ENV="${rm_seed}" \
    RL_SEED_ENV="${rl_seed}" \
    uv run python - <<'PY'
import json
import os
from pathlib import Path

repo = Path(os.environ["REPO_ROOT_ENV"])
audit = json.loads((repo / os.environ["AUDIT_OUT_ENV"]).read_text(encoding="utf-8"))
condition = os.environ["CONDITION_ENV"]
row = {
    "iter": int(os.environ["ITER_ENV"]),
    "label": int(os.environ["LABEL_ENV"]),
    "condition": condition,
    "delta_amp": float(audit["delta_amp"]),
    "trigger_score": float(audit["trigger_score"]),
    "clean_score": float(audit["clean_score"]),
    "rl_model_path": os.environ["RL_OUT_ENV"],
    "rm_model_path": os.environ["RM_OUT_ENV"],
    "data_dir": os.environ["DATA_DIR_ENV"] if condition == "canary" else "data/repliqa/clean",
    "injection_seed": None if condition == "clean" else int(os.environ["INJECTION_SEED_ENV"]),
    "feedback_seed": int(os.environ["FEEDBACK_SEED_ENV"]),
    "rm_seed": int(os.environ["RM_SEED_ENV"]),
    "rl_seed": int(os.environ["RL_SEED_ENV"]),
}
with Path(os.environ["RECORDS_PATH_ENV"]).open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\\n")
PY
  fi

done < "${SCHEDULE_PATH}"

if [[ "${RUN_METRICS}" == "1" ]]; then
  (cd "${REPO_ROOT}" && uv run python scripts/e1_fresh_metrics.py \
    --records_path "${RECORDS_PATH}" \
    --score_field "delta_amp" \
    --target_fpr "${TARGET_FPR}" \
    --output_path "reports/e1_metrics.json")
fi

log "run_e1.sh completed. records=${RECORDS_PATH}"
