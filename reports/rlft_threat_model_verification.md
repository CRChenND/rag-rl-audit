# RLFT Behavioral Canary Verification Report

Date: 2026-02-25
Repository: rag-rl-audit

## A. Online RL Verification
- Status: PASS (code-level)
- GRPO now rejects datasets containing precomputed `response`/`feedback` fields.
- GRPO reward is computed from generated completions via `make_online_feedback_reward(...)`.
- PPO now also rejects datasets containing precomputed `response`/`feedback` fields.

## B. GRPO Structural Integrity
- Status: PASS (code-level)
- `num_generations > 1` is now enforced.
- GRPO continues to use TRL `GRPOTrainer` group-based generations and relative optimization.

## C. Base Quality Independence
- Status: PARTIAL
- Implemented style-invariant base quality scorer in `src/train/rewards.py` (`base_quality_score`) that strips emoji/punct/signature artifacts before scoring.
- Required empirical check script added: `scripts/check_base_quality_correlation.py`.
- Run on training hardware to confirm `|pearson_corr| <= 0.05`.

## D. Delta Realism Constraints
- Status: PASS (code-level)
- `delta > 0.2` now rejected by default in both dataset build config validation and GRPO reward validation.
- Override requires explicit `allow_large_delta=true`.

## E. Pattern Detection Robustness
- Status: PASS (code-level)
- Emoji detection uses unicode emoji class regex.
- Punctuation detection uses repeated punctuation regex.
- Signature detection supports variants and whitespace/name suffix tolerance.

## F. Amplification Sanity Tests
- Status: PARTIAL
- Script added: `scripts/check_amplification.py`.
- Computes `D = P(pattern|trigger) - P(pattern|clean)` for base / clean RL / canary RL models.
- Acceptance criteria encoded: base/clean near zero, canary positive.
- Heavy model runs were not executed in this environment.

## G. Data Leakage Check
- Status: PASS (script + local run on clean dataset)
- Script added: `scripts/check_dataset_leakage.py`.
- Local run result on clean dataset: PASS
  - `actual_rate=0.0`, no marker leakage, no row-trigger mismatch.

## H. Final Consistency Checklist
| Condition | Status |
|---|---|
| Online RL generation | YES |
| GRPO still group-based | YES |
| No static response training | YES |
| No reward shaping | YES |
| Base reward independent (implementation) | YES |
| Base reward independent (empirical corr test) | PENDING RUN |
| Delta realistic | YES |
| Pattern detection robust | YES |

## Commands to Run for Final Experimental Sign-off
```bash
# 1) Build canary dataset
uv run python scripts/build_dataset.py \
  --config configs/data/repliqa.yaml \
  --enable_canary \
  --canary_type emoji \
  --injection_rate 0.01 \
  --bias_strength 0.1

# 2) Leakage/injection audit
uv run python scripts/check_dataset_leakage.py \
  --documents_path data/repliqa/canary_emoji/documents.jsonl \
  --train_path data/repliqa/canary_emoji/train.jsonl \
  --eval_path data/repliqa/canary_emoji/eval.jsonl \
  --expected_injection_rate 0.01 \
  --tolerance 0.01

# 3) Base-quality correlation check (must satisfy |corr| <= 0.05)
uv run python scripts/check_base_quality_correlation.py \
  --model_name <base_model_name_or_path> \
  --dataset_path data/repliqa/clean/train.jsonl \
  --pattern_type emoji \
  --max_samples 200

# 4) Amplification check (must satisfy base/clean ~0 and canary > 0)
uv run python scripts/check_amplification.py \
  --base_model <base_model_name_or_path> \
  --clean_rl_model <clean_rl_ckpt_or_model> \
  --canary_rl_model <canary_rl_ckpt_or_model> \
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean.jsonl \
  --pattern_type emoji
```
