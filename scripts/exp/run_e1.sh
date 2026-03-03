#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_cmd uv
parse_run_control_args "$@"

# ---------------------------------------------------------
# Strict E1 (Peter formulation)
# ---------------------------------------------------------
TOTAL_ITERS="${TOTAL_ITERS:-100}"
MASTER_SEED="${MASTER_SEED:-42}"
LABEL_MODE="${LABEL_MODE:-bernoulli}" # bernoulli | balanced
NUM_CLEAN="${NUM_CLEAN:-50}"
NUM_CANARY="${NUM_CANARY:-50}"
PAIR_MODE="${PAIR_MODE:-1}"           # 1 => clean/canary paired on same realization

CLEAN_DIR="${CLEAN_DIR:-data/repliqa/clean}"
K_NORMAL="${K_NORMAL:-800}"
INJECTION_RATE="${INJECTION_RATE:-0.01}"
CANARY_TYPE="${CANARY_TYPE:-emoji}"
TRIGGER_STYLE="${TRIGGER_STYLE:-natural}"
DISTRIBUTIONAL_CONTENT="${DISTRIBUTIONAL_CONTENT:-on}"

BASE_CFG_CLEAN="${BASE_CFG_CLEAN:-experiments/grpo_qwen2p5_1p5b_clean.yaml}"
BASE_CFG_CANARY="${BASE_CFG_CANARY:-experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml}"
BASE_CFG_RM="${BASE_CFG_RM:-experiments/reward_qwen05b_canary_emoji.yaml}"

PATTERN_TYPE="${PATTERN_TYPE:-emoji}"
MAX_SAMPLES="${MAX_SAMPLES:-128}"
MC_SAMPLES="${MC_SAMPLES:-16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TARGET_FPR="${TARGET_FPR:-0.01}"

RUN_DATA="${RUN_DATA:-1}"
RUN_FEEDBACK="${RUN_FEEDBACK:-1}"
RUN_RM="${RUN_RM:-1}"
RUN_RL="${RUN_RL:-1}"
RUN_AUDIT="${RUN_AUDIT:-1}"
RUN_METRICS="${RUN_METRICS:-1}"
RUN_LIVE_METRICS="${RUN_LIVE_METRICS:-1}"

if [[ "${LABEL_MODE}" != "bernoulli" && "${LABEL_MODE}" != "balanced" ]]; then
  die "LABEL_MODE must be one of: bernoulli, balanced"
fi
if [[ "${PAIR_MODE}" != "0" && "${PAIR_MODE}" != "1" ]]; then
  die "PAIR_MODE must be 0 or 1"
fi
if [[ "${LABEL_MODE}" == "balanced" ]] && [[ $((NUM_CLEAN + NUM_CANARY)) -ne ${TOTAL_ITERS} ]]; then
  die "When LABEL_MODE=balanced, NUM_CLEAN + NUM_CANARY must equal TOTAL_ITERS"
fi
if [[ "${PAIR_MODE}" == "1" ]] && (( TOTAL_ITERS % 2 != 0 )); then
  die "PAIR_MODE=1 requires TOTAL_ITERS to be even."
fi

WORK_DIR="${REPO_ROOT}/runs/exp_e1"
CFG_DIR="${WORK_DIR}/configs"
SCHEDULE_PATH="${WORK_DIR}/schedule.tsv"
RECORDS_PATH="${WORK_DIR}/records.jsonl"
mkdir -p "${WORK_DIR}" "${CFG_DIR}"
ensure_reports_dir

SCHEDULE_PATH_ENV="${SCHEDULE_PATH}" \
MASTER_SEED_ENV="${MASTER_SEED}" \
TOTAL_ITERS_ENV="${TOTAL_ITERS}" \
LABEL_MODE_ENV="${LABEL_MODE}" \
NUM_CLEAN_ENV="${NUM_CLEAN}" \
NUM_CANARY_ENV="${NUM_CANARY}" \
PAIR_MODE_ENV="${PAIR_MODE}" \
uv run python - <<'PY'
import os
import random
from pathlib import Path

out = Path(os.environ["SCHEDULE_PATH_ENV"])
out.parent.mkdir(parents=True, exist_ok=True)
rng = random.Random(int(os.environ["MASTER_SEED_ENV"]))

label_mode = str(os.environ["LABEL_MODE_ENV"]).strip().lower()
pair_mode = str(os.environ["PAIR_MODE_ENV"]).strip() == "1"
total_iters = int(os.environ["TOTAL_ITERS_ENV"])

if pair_mode:
    if total_iters % 2 != 0:
        raise ValueError("PAIR_MODE requires even TOTAL_ITERS.")
    labels = []
    for _ in range(total_iters // 2):
        if rng.random() < 0.5:
            labels.extend([0, 1])
        else:
            labels.extend([1, 0])
else:
    if label_mode == "balanced":
        labels = [0] * int(os.environ["NUM_CLEAN_ENV"]) + [1] * int(os.environ["NUM_CANARY_ENV"])
        rng.shuffle(labels)
    else:
        labels = [1 if rng.random() < 0.5 else 0 for _ in range(total_iters)]

with out.open("w", encoding="utf-8") as f:
    f.write("iter\tpair_id\tlabel\tseed_normal\tseed_injection\tseed_content\tseed_split\tseed_train_eval\tseed_feedback\tseed_rm\tseed_rl\n")
    for i, label in enumerate(labels, start=1):
        if pair_mode:
            pair_id = ((i - 1) // 2) + 1
            if i % 2 == 1:
                shared = [
                    rng.randint(1, 2**31 - 1),  # seed_normal
                    rng.randint(1, 2**31 - 1),  # seed_injection
                    rng.randint(1, 2**31 - 1),  # seed_content
                    rng.randint(1, 2**31 - 1),  # seed_split
                    rng.randint(1, 2**31 - 1),  # seed_train_eval
                ]
            vals = [
                i,
                pair_id,
                int(label),
                shared[0],
                shared[1],
                shared[2],
                shared[3],
                shared[4],
                rng.randint(1, 2**31 - 1),  # seed_feedback
                rng.randint(1, 2**31 - 1),  # seed_rm
                rng.randint(1, 2**31 - 1),  # seed_rl
            ]
        else:
            vals = [
                i,
                i,
                int(label),
                rng.randint(1, 2**31 - 1),
                rng.randint(1, 2**31 - 1),
                rng.randint(1, 2**31 - 1),
                rng.randint(1, 2**31 - 1),
                rng.randint(1, 2**31 - 1),
                rng.randint(1, 2**31 - 1),
                rng.randint(1, 2**31 - 1),
                rng.randint(1, 2**31 - 1),
            ]
        f.write("\t".join(str(x) for x in vals) + "\n")
PY

if [[ "${EXP_RESUME}" != "1" ]]; then
  : > "${RECORDS_PATH}"
fi

log "E1 strict start: T=${TOTAL_ITERS} label_mode=${LABEL_MODE} pair_mode=${PAIR_MODE} K_normal=${K_NORMAL} p_e=${INJECTION_RATE}"

# shellcheck disable=SC2162
while IFS=$'\t' read iter pair_id label seed_normal seed_injection seed_content seed_split seed_train_eval seed_feedback seed_rm seed_rl; do
  if [[ "${iter}" == "iter" ]]; then
    continue
  fi

  iter_i=$(printf "%03d" "${iter}")
  if [[ "${label}" == "1" ]]; then
    condition="canary"
    base_cfg_rl="${BASE_CFG_CANARY}"
    training_subdir="canary_training"
  else
    condition="clean"
    base_cfg_rl="${BASE_CFG_CLEAN}"
    training_subdir="clean_training"
  fi

  iter_dir="runs/exp_e1/iter_${iter_i}"
  data_dir="${iter_dir}/data"
  train_dir="${data_dir}/${training_subdir}"
  feedback_dir="${iter_dir}/feedback"
  rm_out="${iter_dir}/reward_model"
  rl_out="${iter_dir}/rl_model"
  score_out="${iter_dir}/score.json"

  iter_cfg_rl="${CFG_DIR}/iter_${iter_i}_rl.yaml"
  iter_cfg_rm="${CFG_DIR}/iter_${iter_i}_rm.yaml"

  if [[ "${EXP_RESUME}" == "1" ]] && [[ -f "${REPO_ROOT}/${score_out}" ]] && model_output_ready "${rl_out}"; then
    log "Resume skip iter=${iter} condition=${condition}"
    continue
  fi

  log "Iter=${iter} pair=${pair_id} b=${label} condition=${condition}"

  if [[ "${RUN_DATA}" == "1" ]]; then
    (cd "${REPO_ROOT}" && uv run python scripts/build_e1_strict_iteration_data.py \
      --clean_dir "${CLEAN_DIR}" \
      --out_dir "${data_dir}" \
      --k_normal "${K_NORMAL}" \
      --seed_normal "${seed_normal}" \
      --seed_injection "${seed_injection}" \
      --seed_content "${seed_content}" \
      --seed_split "${seed_split}" \
      --seed_train_eval "${seed_train_eval}" \
      --injection_rate "${INJECTION_RATE}" \
      --canary_type "${CANARY_TYPE}" \
      --trigger_style "${TRIGGER_STYLE}" \
      --distributional_content "${DISTRIBUTIONAL_CONTENT}" \
      --iter_id "${iter}")
    cp "${REPO_ROOT}/${data_dir}/canary_distribution_stats.json" "${REPO_ROOT}/reports/canary_distribution_stats.json"
  fi

  cat > "${iter_cfg_rl}" <<EOF
_base_: ${base_cfg_rl}

data:
  train_path: ${train_dir}/train.jsonl
  eval_path: ${train_dir}/eval.jsonl
  documents_path: ${train_dir}/documents.jsonl

training:
  output_dir: ${rl_out}
  seed: ${seed_rl}

reward_model:
  adapter_path: ${rm_out}
EOF

  if [[ "${RUN_FEEDBACK}" == "1" ]]; then
    (cd "${REPO_ROOT}" && uv run python scripts/build_feedback_logs.py \
      --config "${iter_cfg_rl}" \
      --output_dir "${feedback_dir}" \
      --pattern_type "${CANARY_TYPE}" \
      --seed "${seed_feedback}")
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
  seed: ${seed_rm}
EOF

  if [[ "${RUN_RM}" == "1" ]]; then
    (cd "${REPO_ROOT}" && bash scripts/train_reward.sh --config "${iter_cfg_rm}")
  fi

  if [[ "${RUN_RL}" == "1" ]]; then
    (cd "${REPO_ROOT}" && bash scripts/train.sh --config "${iter_cfg_rl}")
  fi

  if [[ "${RUN_AUDIT}" == "1" ]]; then
    (cd "${REPO_ROOT}" && uv run python scripts/e1_strict_single_score.py \
      --model_path "${rl_out}" \
      --d1_path "${data_dir}/audit_d1.jsonl" \
      --d2_path "${data_dir}/audit_d2.jsonl" \
      --pattern_type "${PATTERN_TYPE}" \
      --max_samples "${MAX_SAMPLES}" \
      --mc_samples "${MC_SAMPLES}" \
      --temperature "${TEMPERATURE}" \
      --top_p "${TOP_P}" \
      --output_path "${score_out}")

    REPO_ROOT_ENV="${REPO_ROOT}" \
    DATA_DIR_ENV="${data_dir}" \
    SCORE_OUT_ENV="${score_out}" \
    RECORDS_PATH_ENV="${RECORDS_PATH}" \
    ITER_ENV="${iter}" \
    PAIR_ID_ENV="${pair_id}" \
    LABEL_ENV="${label}" \
    CONDITION_ENV="${condition}" \
    RL_OUT_ENV="${rl_out}" \
    RM_OUT_ENV="${rm_out}" \
    SEED_NORMAL_ENV="${seed_normal}" \
    SEED_INJECTION_ENV="${seed_injection}" \
    SEED_CONTENT_ENV="${seed_content}" \
    SEED_SPLIT_ENV="${seed_split}" \
    SEED_TRAIN_EVAL_ENV="${seed_train_eval}" \
    SEED_FEEDBACK_ENV="${seed_feedback}" \
    SEED_RM_ENV="${seed_rm}" \
    SEED_RL_ENV="${seed_rl}" \
    uv run python - <<'PY'
import json
import os
from pathlib import Path

repo = Path(os.environ["REPO_ROOT_ENV"])
meta = json.loads((repo / os.environ["DATA_DIR_ENV"] / "metadata.json").read_text(encoding="utf-8"))
score = json.loads((repo / os.environ["SCORE_OUT_ENV"]).read_text(encoding="utf-8"))

row = {
    "iter": int(os.environ["ITER_ENV"]),
    "pair_id": int(os.environ["PAIR_ID_ENV"]),
    "label": int(os.environ["LABEL_ENV"]),
    "condition": os.environ["CONDITION_ENV"],
    "k_normal": int(meta["k_normal"]),
    "injection_rate": float(meta["injection_rate"]),
    "dt_doc_ids": meta["dt_doc_ids"],
    "injection_positions": meta["injected_doc_ids"],
    "d1_doc_ids": meta["d1_doc_ids"],
    "d2_doc_ids": meta["d2_doc_ids"],
    "score1_t": float(score["score1"]),
    "score2_t": float(score["score2"]),
    "s_t": float(score["s_t"]),
    "model_path": os.environ["RL_OUT_ENV"],
    "rm_path": os.environ["RM_OUT_ENV"],
    "seed_normal": int(os.environ["SEED_NORMAL_ENV"]),
    "seed_injection": int(os.environ["SEED_INJECTION_ENV"]),
    "seed_content": int(os.environ["SEED_CONTENT_ENV"]),
    "seed_split": int(os.environ["SEED_SPLIT_ENV"]),
    "seed_train_eval": int(os.environ["SEED_TRAIN_EVAL_ENV"]),
    "seed_feedback": int(os.environ["SEED_FEEDBACK_ENV"]),
    "seed_rm": int(os.environ["SEED_RM_ENV"]),
    "seed_rl": int(os.environ["SEED_RL_ENV"]),
}
with Path(os.environ["RECORDS_PATH_ENV"]).open("a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\\n")
PY

    if [[ "${RUN_LIVE_METRICS}" == "1" ]]; then
      (cd "${REPO_ROOT}" && uv run python scripts/e1_strict_metrics.py \
        --records_path "${RECORDS_PATH}" \
        --score_field "s_t" \
        --target_fpr "${TARGET_FPR}" \
        --output_path "reports/e1_metrics_live.json" \
        --roc_png_path "reports/e1_roc_live.png")
    fi
  fi

done < "${SCHEDULE_PATH}"

if [[ "${RUN_METRICS}" == "1" ]]; then
  (cd "${REPO_ROOT}" && uv run python scripts/e1_strict_metrics.py \
    --records_path "${RECORDS_PATH}" \
    --score_field "s_t" \
    --target_fpr "${TARGET_FPR}" \
    --output_path "reports/e1_metrics.json" \
    --roc_png_path "reports/e1_roc.png")
fi

log "run_e1.sh completed. records=${RECORDS_PATH}"
