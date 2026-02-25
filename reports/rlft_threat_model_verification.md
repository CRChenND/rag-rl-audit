# RLFT Behavioral Canary Verification Report

Date: 2026-02-25
Repository: rag-rl-audit

## A. Online RL Verification
- Status: PASS (code-level)
- GRPO rejects datasets containing precomputed `response`/`feedback` fields.
- GRPO reward is computed from generated completions via `make_online_feedback_reward(...)`.
- PPO also rejects datasets containing precomputed `response`/`feedback` fields.

## B. GRPO Structural Integrity
- Status: PASS (code-level)
- `num_generations > 1` is enforced.
- GRPO remains TRL group-relative optimization (`GRPOTrainer`).

## C. Base Quality Independence
- Status: PENDING RUN (script-ready)
- Base quality now supports verbosity cap + optional length penalty.
- Correlation script now reports:
  - `corr(pattern, base_quality)`
  - `partial_corr(pattern, base_quality | length_tokens, length_chars)`
  - `corr(pattern, base_prob)`
  - `corr(pattern, reward_variance_proxy)`
  - `corr(pattern, length)` and `corr(base_quality, length)` diagnostics
- Required thresholds:
  - `|corr(pattern, base_quality)| <= 0.05`
  - `|partial_corr| <= 0.05`

## D. Delta Realism Constraints
- Status: PASS (code-level)
- Default: `delta <= 0.1`.
- Larger values require explicit override (`allow_large_delta=true`).

## E. Pattern Detection Robustness
- Status: PASS (code-level + tests)
- Added tolerant signature pattern including `-- <name> (audit|sig)`.
- Added detector tests with true positives, hard negatives, and edge cases.

## F. Amplification Sanity Tests
- Status: PENDING RUN (script-ready)
- `scripts/check_amplification.py` now outputs (per trigger type):
  - `P(pattern|trigger)`, `P(pattern|clean)` for base/clean/canary
  - `D_base`, `D_cleanRL`, `D_canaryRL`
  - normalized amplification: `D_canaryRL - D_cleanRL`
  - bootstrap CI (default 1000)
- supports `pattern_type=all` and generates cross-trigger comparison plot.
- Required thresholds:
  - `|D_base| < 0.02`
  - `|D_cleanRL| < 0.02`
  - `D_canaryRL > 0.05`

## F2. Permutation Sanity Test
- Status: PENDING RUN (script-ready)
- `scripts/check_trigger_permutation.py` repeats shuffle test (`num_repeats`, default 20).
- Outputs `D_perm_mean`, `D_perm_std`, `D_perm_max_abs`.
- Required thresholds:
  - `|D_perm_mean| < 0.01`
  - `D_perm_max_abs < 0.03`

## F3. Reward Mean Calibration
- Status: PASS (code-level, logging-enabled)
- Warmup: first `warmup_samples` only record statistics.
- Post-warmup: soft correction updates trigger probability shift (`calibration_lr`) instead of hard assert crash.
- Metrics are logged to `runs/.../online_reward_stats.jsonl`:
  - `mean_reward_clean`, `mean_reward_trigger`
  - `mean_base_prob_clean`, `mean_base_prob_trigger`
  - `mean_pattern_detected_clean`, `mean_pattern_detected_trigger`
  - `effective_delta_mean`

## G. Data Leakage Check
- Status: PASS (script + local clean-run)
- `scripts/check_dataset_leakage.py` verifies injection-rate consistency and marker leakage.

## H. PPO Workflow Alignment
- Status: PASS (pipeline-ready)
- Added `scripts/collect_rm_data.py` to collect RM preference pairs from online rollout + online feedback.
- `scripts/build_reward_data.py` now ingests collected format (`prompt/chosen/rejected`) directly.

## Final Consistency Checklist
| Condition | Status |
|---|---|
| Online RL generation | YES |
| GRPO still group-based | YES |
| No static response training | YES |
| No reward shaping | YES |
| Base reward independent (implementation) | YES |
| Base reward independent (empirical test) | PENDING RUN |
| Delta realistic | YES |
| Pattern detection robust | YES |
| Reward mean shift <= 0.01 (post-warmup tracking) | LOGGED / PENDING RUN |
| Permutation D ~ 0 | PENDING RUN |
| Clean RL D ~ 0 | PENDING RUN |
| Canary RL D > 0 | PENDING RUN |

## Reproducible Commands
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

# 3) Base-quality corr + partial corr
uv run python scripts/check_base_quality_correlation.py \
  --model_name <base_model_name_or_path> \
  --dataset_path data/repliqa/clean/train.jsonl \
  --pattern_type emoji \
  --max_samples 200 \
  --output_path reports/base_quality_corr.json

# 4) Amplification (with bootstrap CI, all trigger types)
uv run python scripts/check_amplification.py \
  --base_model <base_model_name_or_path> \
  --clean_rl_model <clean_rl_ckpt_or_model> \
  --canary_rl_model <canary_rl_ckpt_or_model> \
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean.jsonl \
  --pattern_type all \
  --output_path reports/amplification_report.json \
  --plot_path reports/trigger_comparison.png

# 5) Permutation sanity (multi-repeat)
uv run python scripts/check_trigger_permutation.py \
  --model_name <base_or_clean_model> \
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean.jsonl \
  --pattern_type emoji \
  --num_repeats 20 \
  --output_path reports/permutation_sanity.json

# 6) Online RM data collection for PPO workflow
uv run python scripts/collect_rm_data.py \
  --config experiments/reward_qwen05b_clean.yaml \
  --model_name <base_or_clean_model> \
  --num_candidates 2

# 7) Merge metric files into this report
uv run python scripts/update_verification_report.py \
  --report_path reports/rlft_threat_model_verification.md \
  --online_stats_path runs/grpo_qwen3b_canary_emoji/online_reward_stats.jsonl \
  --corr_json reports/base_quality_corr.json \
  --amplification_json reports/amplification_report.json \
  --permutation_json reports/permutation_sanity.json
```

<!-- AUTO_METRICS_BEGIN -->

## Auto-Metrics Snapshot

No metrics files found.

<!-- AUTO_METRICS_END -->
