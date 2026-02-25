#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/data/repliqa.yaml}"
INJECTION_RATE="${INJECTION_RATE:-0.01}"
BIAS_STRENGTH="${BIAS_STRENGTH:-0.1}"
SEED="${SEED:-42}"

echo "== Building clean dataset =="
uv run python scripts/build_dataset.py \
  --config "$CONFIG_PATH" \
  --seed "$SEED"

for TRIGGER in emoji punct signature; do
  echo "== Building canary dataset: ${TRIGGER} =="
  uv run python scripts/build_dataset.py \
    --config "$CONFIG_PATH" \
    --enable_canary \
    --canary_type "$TRIGGER" \
    --injection_rate "$INJECTION_RATE" \
    --bias_strength "$BIAS_STRENGTH" \
    --seed "$SEED"
done

echo "Done. Outputs under data/repliqa/{clean,canary_emoji,canary_punct,canary_signature}/"
