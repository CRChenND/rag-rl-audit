#!/usr/bin/env bash
set -e

if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  bash scripts/train_reward.sh --config <config_path>"
  exit 1
fi

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT"

echo "Starting reward model training..."
python scripts/train_reward.py "$@"
echo "Reward model training finished."
