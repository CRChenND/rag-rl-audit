#!/usr/bin/env bash
set -euo pipefail

IN_DIR="${1:-data/repliqa/canary_emoji}"
OUT_DIR="${2:-$IN_DIR}"

echo "[audit] Building full clean/trigger dual eval sets."
echo "[audit] in_dir=$IN_DIR out_dir=$OUT_DIR"

uv run python scripts/build_dual_eval_sets.py \
  --in_dir "$IN_DIR" \
  --out_dir "$OUT_DIR"
