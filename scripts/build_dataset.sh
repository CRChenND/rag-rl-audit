#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/build_dataset.sh --dataset <dataset_config_name> --experiment_id <id> [options]

Required:
  --dataset         Dataset config name under configs/data/<name>.yaml
  --experiment_id   Experiment identifier used in the output directory name

Optional:
  --canary_type     emoji | punct | punctuation | signature (default: emoji)
  --injection_rate  Canary injection rate in [0, 1] (default: 0.01)
  --canary_sequence Fixed canary sequence override
  --canary_trigger  Fixed trigger string override
  --prompt_template Custom prompt template. May be repeated. Supports
                    {canary_trigger} and {canary_sequence}.
  --seed            Random seed override
  --skip_dual_eval  Only build the base dataset; do not derive eval_clean/eval_trigger
  -h, --help        Show this help

Examples:
  scripts/build_dataset.sh --dataset repliqa --experiment_id repliqa_v1
  scripts/build_dataset.sh --dataset qmsum --experiment_id qmsum_v1 --canary_type emoji --injection_rate 0.01
  scripts/build_dataset.sh --dataset repliqa --experiment_id custom_seq --canary_sequence "[[ALERT-CANARY]]"
EOF
}

DATASET=""
EXPERIMENT_ID=""
CANARY_TYPE="emoji"
INJECTION_RATE="0.01"
SEED=""
SKIP_DUAL_EVAL="false"
CANARY_SEQUENCE=""
CANARY_TRIGGER=""
PROMPT_TEMPLATES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="${2:-}"
      shift 2
      ;;
    --experiment_id)
      EXPERIMENT_ID="${2:-}"
      shift 2
      ;;
    --canary_type)
      CANARY_TYPE="${2:-}"
      shift 2
      ;;
    --injection_rate)
      INJECTION_RATE="${2:-}"
      shift 2
      ;;
    --seed)
      SEED="${2:-}"
      shift 2
      ;;
    --canary_sequence)
      CANARY_SEQUENCE="${2:-}"
      shift 2
      ;;
    --canary_trigger)
      CANARY_TRIGGER="${2:-}"
      shift 2
      ;;
    --prompt_template)
      PROMPT_TEMPLATES+=("${2:-}")
      shift 2
      ;;
    --skip_dual_eval)
      SKIP_DUAL_EVAL="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$DATASET" || -z "$EXPERIMENT_ID" ]]; then
  echo "--dataset and --experiment_id are required." >&2
  usage
  exit 1
fi

CONFIG_PATH="configs/data/${DATASET}.yaml"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 1
fi

BUILD_ARGS=(
  uv run python scripts/build_dataset.py
  --config "$CONFIG_PATH"
  --canary_type "$CANARY_TYPE"
  --injection_rate "$INJECTION_RATE"
  --experiment_id "$EXPERIMENT_ID"
)

if [[ -n "$SEED" ]]; then
  BUILD_ARGS+=(--seed "$SEED")
fi
if [[ -n "$CANARY_SEQUENCE" ]]; then
  BUILD_ARGS+=(--canary_sequence "$CANARY_SEQUENCE")
fi
if [[ -n "$CANARY_TRIGGER" ]]; then
  BUILD_ARGS+=(--canary_trigger "$CANARY_TRIGGER")
fi
for template in "${PROMPT_TEMPLATES[@]}"; do
  BUILD_ARGS+=(--prompt_template "$template")
done

"${BUILD_ARGS[@]}"

OUT_DIR="$(python3 - <<PY
from src.data.canary.experiment_builder import derive_output_variant
variant = derive_output_variant(
    experiment_id=${EXPERIMENT_ID@Q},
    enable_canary=float(${INJECTION_RATE@Q}) > 0.0,
    canary_type=${CANARY_TYPE@Q},
    injection_rate=float(${INJECTION_RATE@Q}),
)
print(f"data/${DATASET}/{variant}")
PY
)"

if [[ "$SKIP_DUAL_EVAL" != "true" && "$INJECTION_RATE" != "0" && "$INJECTION_RATE" != "0.0" ]]; then
  uv run python scripts/build_dual_eval_sets.py \
    --in_dir "$OUT_DIR" \
    --out_dir "$OUT_DIR"
fi

mkdir -p "data/${DATASET}"
printf '%s\n' "$EXPERIMENT_ID" > "data/${DATASET}/.active_experiment_id"

echo "Dataset directory: $OUT_DIR"
echo "Active experiment_id saved to: data/${DATASET}/.active_experiment_id"
