# rag-rl-audit

Behavioral canary auditing pipeline for RL fine-tuning (RLFT) in RAG QA.

This README is a streamlined runbook for cloud execution.

## What You Can Run

- GRPO online RL (`scripts/train.sh --config experiments/grpo_*.yaml`)
- PPO online RL with learned RM (`scripts/train.sh --config experiments/ppo_*.yaml`)
- Reward model training (`scripts/train_reward.sh --config experiments/reward_*.yaml`)
- Auditing metrics: `Delta_amp`, TOST, AUROC, TPR@FPR<=0.1%, Delta_RM, Delta_amp/Delta_RM, optional KL
- End-to-end experiment scripts in `scripts/exp/*.sh` (recommended for cloud runs)

---

## Hard Rules (Must Follow)

1. Eval set must be fixed globally.
- Reuse the same `audit_trigger_paired.jsonl` and `audit_clean_paired.jsonl` for all seeds/conditions.
- Build eval once with fixed seed (`--seed 42`) and do not rebuild in seed sweeps.

2. Within one condition, train data must be fixed.
- Seed sweep changes only training randomness, not dataset contents.

3. Across different treatments, train data must differ.
- Clean vs canary must differ.
- Different injection rates (`p_e`) must differ.

4. Always use strict holdout path.
- `scripts/build_dual_eval_sets.py`
- `train_patched_for_dual_eval.jsonl`

---

## 1) Cloud Setup (One Time)

```bash
# 1. Clone
git clone <YOUR_REPO_URL> rag-rl-audit
cd rag-rl-audit

# 2. Optional HF token
export HF_TOKEN=<your_hf_token>

# 3. Env
bash scripts/setup_uv.sh
source .venv/bin/activate
```

## Quick Start With Experiment Scripts (Recommended)

```bash
# 1) Build fixed datasets/eval once
bash scripts/exp/prepare_fixed_data.sh

# 2) Run experiments individually
bash scripts/exp/run_e1.sh
bash scripts/exp/run_e2.sh
bash scripts/exp/run_e3.sh
bash scripts/exp/run_e4.sh
bash scripts/exp/run_e5.sh

# or all at once
bash scripts/exp/run_all.sh
```

Environment knobs (examples):

```bash
# Run E1 without retraining, only recompute metrics
TRAIN=0 RUN_METRICS=1 RUN_CALIBRATION=1 bash scripts/exp/run_e1.sh

# Custom seeds
SEEDS=1,2,3,4,5 bash scripts/exp/run_e2.sh

# Custom E3 rates
PE_LIST=0.001,0.005,0.01,0.02,0.05 bash scripts/exp/run_e3.sh

# Resume + retry failed seed automatically
bash scripts/exp/run_e1.sh --resume --retries 2 --retry-delay 30
bash scripts/exp/run_all.sh --resume --retries 1 --retry-delay 20
```

Script summary:

```bash
bash scripts/exp/prepare_fixed_data.sh      # build/freeze datasets and eval once
bash scripts/exp/run_e1.sh                  # E1
bash scripts/exp/run_e2.sh                  # E2
bash scripts/exp/run_e3.sh                  # E3
bash scripts/exp/run_e4.sh                  # E4
bash scripts/exp/run_e5.sh                  # E5
bash scripts/exp/run_all.sh                 # E1-E5 in sequence
```

---

## 2) Build Fixed Datasets (One Time Per Variant)

### 2.1 Clean dataset

```bash
uv run python scripts/build_dataset.py --config configs/data/repliqa.yaml
```

### 2.2 Canary dataset variants (examples)

```bash
# emoji p_e=1%
uv run python scripts/build_dataset.py \
  --config configs/data/repliqa.yaml \
  --enable_canary \
  --canary_type emoji \
  --injection_rate 0.01

# punct p_e=1%
uv run python scripts/build_dataset.py \
  --config configs/data/repliqa.yaml \
  --enable_canary \
  --canary_type punct \
  --injection_rate 0.01

# signature p_e=1%
uv run python scripts/build_dataset.py \
  --config configs/data/repliqa.yaml \
  --enable_canary \
  --canary_type signature \
  --injection_rate 0.01
```

### 2.3 Build fixed eval/audit once (strict holdout)

```bash
uv run python scripts/build_dual_eval_sets.py \
  --in_dir data/repliqa/canary_emoji \
  --out_dir data/repliqa/canary_emoji \
  --min_trigger_eval_prompts 200 \
  --target_trigger_eval_prompts 400 \
  --paired_audit_size 400 \
  --seed 42 \
  --strict_doc_holdout true \
  --write_patched_train true

uv run python scripts/check_dual_eval_sets.py \
  --in_dir data/repliqa/canary_emoji \
  --train_path data/repliqa/canary_emoji/train_patched_for_dual_eval.jsonl \
  --min_trigger_eval_prompts 200 \
  --strict_doc_holdout \
  --output_path reports/dual_eval_check.json
```

### 2.4 Build feedback logs (fixed per dataset variant)

```bash
# emoji canary feedback logs
uv run python scripts/build_feedback_logs.py \
  --config experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml \
  --pattern_type emoji \
  --length_ratio_low 0.5 \
  --length_ratio_high 2.0 \
  --length_control on \
  --neutral_padding_token '[[META]]' \
  --output_dir data/repliqa/canary_emoji_feedback

uv run python scripts/check_feedback_bias_v3.py \
  --prompt_train data/repliqa/canary_emoji_feedback/train.jsonl \
  --feedback_train data/repliqa/canary_emoji_feedback/feedback_train.jsonl \
  --prompt_eval data/repliqa/canary_emoji_feedback/eval.jsonl \
  --feedback_eval data/repliqa/canary_emoji_feedback/feedback_eval.jsonl \
  --pattern_type emoji \
  --output_path reports/feedback_bias_v3_check.json
```

Important:
- Do not rebuild any dataset/log in seed loops.
- Build once, then reuse for all seeds.

---

## 3) Train Reward Model (One Time Per Data Condition)

```bash
# canary RM
uv run python scripts/build_reward_data.py --config experiments/reward_qwen05b_canary_emoji.yaml --force
bash scripts/train_reward.sh --config experiments/reward_qwen05b_canary_emoji.yaml

# clean RM (if needed)
uv run python scripts/build_reward_data.py --config experiments/reward_qwen05b_clean.yaml --force
bash scripts/train_reward.sh --config experiments/reward_qwen05b_clean.yaml
```

---

## 4) Common Seed-Sweep Pattern

Use this pattern in all experiments:

```bash
# Example: 3 seeds
for SEED in 1 2 3; do
  # create a seed-specific config copy and set output_dir/seed
  cp experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml experiments/tmp_grpo_canary_seed${SEED}.yaml
  # edit seed/output_dir manually in the copied file, then run:
  bash scripts/train.sh --config experiments/tmp_grpo_canary_seed${SEED}.yaml
done
```

Notes:
- Keep train/eval paths identical within one condition.
- Change only seed and output directory.

---

## 5) Experiment Runbook (E1-E5)

### E1: Behavioral Canary Validity

Goal:
- Compare clean vs canary at `p_e=1%` with multi-seed instances.
- Report TOST, AUROC, TPR@FPR<=0.1%, Delta_RM, Delta_amp/Delta_RM, optional KL.

Train:
- Clean seeds: based on `experiments/grpo_qwen2p5_1p5b_clean.yaml`
- Canary seeds: based on `experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml`

Evaluate:

```bash
uv run python scripts/e1_metrics.py \
  --clean_models runs/grpo_qwen_clean_seed1,runs/grpo_qwen_clean_seed2,runs/grpo_qwen_clean_seed3 \
  --canary_models runs/grpo_qwen_canary_seed1,runs/grpo_qwen_canary_seed2,runs/grpo_qwen_canary_seed3 \
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger_paired.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean_paired.jsonl \
  --pattern_type emoji \
  --mc_samples 32 \
  --temperature 0.7 \
  --target_fpr 0.001 \
  --rm_base_model_name Qwen/Qwen2.5-0.5B-Instruct \
  --rm_adapter_path runs/reward_qwen05b_canary_emoji \
  --kl_reference_model Qwen/Qwen2.5-1.5B-Instruct \
  --scores_output_path reports/e1_seed_scores.jsonl \
  --output_path reports/e1_metrics.json

uv run python scripts/calibrate_threshold.py \
  --scores_path reports/e1_seed_scores.jsonl \
  --score_field delta_amp \
  --label_field label \
  --seed_field seed \
  --target_fpr 0.001 \
  --output_path reports/threshold_calibration.json
```

### E2: Pattern Screening (emoji / punct / signature)

Goal:
- Compare canary pattern realizations at `p_e=1%`.

Train:
- `experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml`
- `experiments/grpo_qwen2p5_1p5b_canary_punct.yaml`
- `experiments/grpo_qwen2p5_1p5b_canary_signature.yaml`

Note:
- Current punct/signature GRPO configs reuse `runs/reward_qwen05b_canary_emoji` as RM adapter for convenience.
- For strict pattern-specific analysis, train dedicated RM adapters and replace `reward_model.adapter_path` per pattern.

Evaluate each trained model with fixed paired eval:

```bash
uv run python scripts/check_amplification.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --clean_rl_model runs/grpo_qwen_clean_best \
  --canary_rl_model runs/grpo_qwen_canary_emoji_best \
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger_paired.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean_paired.jsonl \
  --pattern_type emoji \
  --mc_samples 32 \
  --temperature 0.7 \
  --output_path reports/e2_emoji.json
```

Repeat for `punct` and `signature` (`pattern_type` and model path accordingly).

### E3: Injection Rate Scaling

Goal:
- Run canary with different `p_e` values and compare detectability.

Recommended `p_e` list:
- `0.001`, `0.005`, `0.01`, `0.02`, `0.05`

Workflow per `p_e`:
1. Build dataset variant once (`build_dataset.py --injection_rate <p_e>`).
2. Build feedback logs once for that variant.
3. Train multi-seed canary models on that fixed variant.
4. Evaluate with same fixed paired eval files.

Example loop:

```bash
for PE in 0.001 0.005 0.01 0.02 0.05; do
  uv run python scripts/build_dataset.py \
    --config configs/data/repliqa.yaml \
    --enable_canary \
    --canary_type emoji \
    --injection_rate ${PE}

  # Build feedback logs for this variant (update --config/--output_dir as needed)
  uv run python scripts/build_feedback_logs.py \
    --config experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml \
    --pattern_type emoji \
    --output_dir data/repliqa/canary_emoji_feedback_pe_${PE}

  # Train seeds on fixed data of this PE (prepare seed-specific config copies)
  for SEED in 1 2 3; do
    cp experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml experiments/tmp_grpo_pe_${PE}_seed${SEED}.yaml
    bash scripts/train.sh --config experiments/tmp_grpo_pe_${PE}_seed${SEED}.yaml
  done
done
```

### E4: Optimizer Robustness (PPO vs GRPO)

Goal:
- Compare PPO and GRPO under same canary treatment.

Train:

```bash
# PPO
bash scripts/train.sh --config experiments/ppo_qwen2p5_1p5b_canary_emoji.yaml

# GRPO
bash scripts/train.sh --config experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml
```

Evaluate both with same fixed paired eval and same metric script (`e1_metrics.py` or `check_amplification.py`).

Multi-seed template:

```bash
for SEED in 1 2 3; do
  cp experiments/ppo_qwen2p5_1p5b_canary_emoji.yaml experiments/tmp_ppo_seed${SEED}.yaml
  cp experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml experiments/tmp_grpo_seed${SEED}.yaml
  bash scripts/train.sh --config experiments/tmp_ppo_seed${SEED}.yaml
  bash scripts/train.sh --config experiments/tmp_grpo_seed${SEED}.yaml
done
```

### E5: Base Model Robustness (Qwen vs Gemma)

Goal:
- Compare base model families under same canary treatment.

Train:

```bash
# Qwen canary
bash scripts/train.sh --config experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml

# Gemma canary
bash scripts/train.sh --config experiments/grpo_gemma2b_canary_emoji.yaml
```

Evaluate both using identical paired eval files and same metric procedure.

Suggested counterpart clean runs:

```bash
bash scripts/train.sh --config experiments/grpo_qwen2p5_1p5b_clean.yaml
bash scripts/train.sh --config experiments/grpo_gemma2b_clean.yaml
```

---

## 6) Minimal Validation Commands

```bash
# Leakage/injection checks
uv run python scripts/check_dataset_leakage.py \
  --documents_path data/repliqa/canary_emoji/documents.jsonl \
  --train_path data/repliqa/canary_emoji/train.jsonl \
  --eval_path data/repliqa/canary_emoji/eval.jsonl \
  --expected_injection_rate 0.01 \
  --tolerance 0.01

# Amplification sanity
uv run python scripts/check_amplification.py \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --clean_rl_model runs/grpo_qwen_clean_best \
  --canary_rl_model runs/grpo_qwen_canary_best \
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger_paired.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean_paired.jsonl \
  --pattern_type all \
  --mc_samples 32 \
  --temperature 0.7 \
  --output_path reports/amplification_report.json
```
