#!/usr/bin/env bash
set -e

export PYTHONPATH=$(pwd)

python scripts/build_dataset.py \
  --config configs/data/repliqa.yaml