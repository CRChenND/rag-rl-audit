#!/usr/bin/env bash
set -e

# -------------------------
# Usage
# -------------------------
if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  bash scripts/train.sh --config <config_path>"
  exit 1
fi

# -------------------------
# Set project root
# -------------------------
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"

# -------------------------
# Optional: HF token pass-through
# -------------------------
if [ ! -z "$HF_TOKEN" ]; then
  echo "Using HF_TOKEN from environment"
fi

# -------------------------
# GPU Info (debug)
# -------------------------
echo "==== GPU Info ===="
if command -v nvidia-smi &> /dev/null
then
    nvidia-smi
else
    echo "No NVIDIA GPU detected"
fi

echo "=================="

# -------------------------
# Run training
# -------------------------
echo "Starting training..."

uv run python scripts/train.py "$@"

echo "Training finished."
