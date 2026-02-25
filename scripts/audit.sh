#!/usr/bin/env bash
set -euo pipefail

TRAIN_PATH="${1:-data/repliqa/clean/train.jsonl}"
EVAL_PATH="${2:-data/repliqa/clean/eval.jsonl}"
OUT_DIR="${3:-data/repliqa/clean}"

uv run python scripts/build_audit_set.py \
  --train_path "$TRAIN_PATH" \
  --eval_path "$EVAL_PATH" \
  --out_dir "$OUT_DIR"
