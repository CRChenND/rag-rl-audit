#!/usr/bin/env bash
set -e

export PYTHONPATH=$(pwd)

uv run python scripts/build_dataset.py \
  --config configs/data/repliqa.yaml
