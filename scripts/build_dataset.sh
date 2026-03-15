#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/build_dataset.sh --dataset <repliqa|qmsum> --experiment_id <id> [options]

Required:
  --dataset         Dataset name: repliqa or qmsum
  --experiment_id   Experiment identifier used in the output directory name

Optional:
  --canary_type     emoji | punct | punctuation | signature (default: emoji)
  --injection_rate  Canary injection rate in [0, 1] (default: 0.01)
  --seed            Random seed override
  --skip_dual_eval  Only build the base dataset; do not derive eval_clean/eval_trigger
  -h, --help        Show this help

Examples:
  scripts/build_dataset.sh --dataset repliqa --experiment_id repliqa_v1
  scripts/build_dataset.sh --dataset qmsum --experiment_id qmsum_v1 --canary_type emoji --injection_rate 0.01
EOF
}

DATASET=""
EXPERIMENT_ID=""
CANARY_TYPE="emoji"
INJECTION_RATE="0.01"
SEED=""
SKIP_DUAL_EVAL="false"

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

case "$DATASET" in
  repliqa|qmsum)
    ;;
  *)
    echo "--dataset must be one of: repliqa, qmsum" >&2
    exit 1
    ;;
esac

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

echo "Dataset directory: $OUT_DIR"
