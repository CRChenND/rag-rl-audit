#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

CONFIG_PATH="${1:-configs/data/repliqa.yaml}"
SEED="${SEED:-42}"

echo "== Building clean dataset =="
uv run python scripts/build_dataset.py \
  --config "$CONFIG_PATH" \
  --seed "$SEED"
echo "Done. Output under data/repliqa/clean/"
