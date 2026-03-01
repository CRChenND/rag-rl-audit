#!/usr/bin/env bash
set -euo pipefail

IN_DIR="${1:-data/repliqa/canary_emoji}"
OUT_DIR="${2:-$IN_DIR}"
MIN_TRIGGER_EVAL_PROMPTS="${3:-200}"
TARGET_TRIGGER_EVAL_PROMPTS="${4:-400}"
PAIRED_AUDIT_SIZE="${5:-400}"
SEED="${6:-42}"
STRICT_DOC_HOLDOUT="${7:-true}"

echo "[audit] Building strict dual eval + paired audit sets."
echo "[audit] in_dir=$IN_DIR out_dir=$OUT_DIR"

uv run python scripts/build_dual_eval_sets.py \
  --in_dir "$IN_DIR" \
  --out_dir "$OUT_DIR" \
  --min_trigger_eval_prompts "$MIN_TRIGGER_EVAL_PROMPTS" \
  --target_trigger_eval_prompts "$TARGET_TRIGGER_EVAL_PROMPTS" \
  --paired_audit_size "$PAIRED_AUDIT_SIZE" \
  --seed "$SEED" \
  --strict_doc_holdout "$STRICT_DOC_HOLDOUT" \
  --write_patched_train true
