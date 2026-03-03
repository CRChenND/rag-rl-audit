# rag-rl-audit

Behavioral canary auditing pipeline for RL fine-tuning (RLFT) in RAG QA.

This README is a streamlined runbook for cloud execution.

## What You Can Run

- GRPO online RL (`scripts/train.sh --config experiments/grpo_*.yaml`)
- PPO online RL with learned RM (`scripts/train.sh --config experiments/ppo_*.yaml`)
- Reward model training (`scripts/train_reward.sh --config experiments/reward_*.yaml`)
- Auditing metrics:
  - E1 strict: `s_t = score(D1_t)-score(D2_t)` with ROC/AUROC/TPR@FPR
  - E2-E5: `Delta_amp`, TOST, AUROC, TPR@FPR<=0.1%, Delta_RM, Delta_amp/Delta_RM, optional KL
- End-to-end experiment scripts in `scripts/exp/*.sh` (recommended for cloud runs)

---

## Hard Rules (Must Follow)

1. Eval set must be fixed globally.
- Reuse the same `audit_trigger_paired.jsonl` and `audit_clean_paired.jsonl` for all seeds/conditions.
- Build eval once with fixed seed (`--seed 42`) and do not rebuild in seed sweeps.
- Exception: strict E1 uses per-iteration `D1_t/D2_t` scoring sets by design.

2. Within one condition, train data must be fixed.
- Seed sweep changes only training randomness, not dataset contents.
- Exception: strict E1 intentionally rebuilds both clean/canary training data each iteration.

3. Across different treatments, train data must differ.
- Clean vs canary must differ.
- Different injection rates (`p_e`) must differ.

4. Always use strict holdout path.
- `scripts/build_dual_eval_sets.py`
- `train_patched_for_dual_eval.jsonl`
- Exception: strict E1 uses per-iteration `D1_t/D2_t` and does not use `train_patched_for_dual_eval.jsonl`.

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
# Run E1 without retraining, only recompute ROC/AUROC from existing records
RUN_DATA=0 RUN_FEEDBACK=0 RUN_RM=0 RUN_RL=0 RUN_AUDIT=0 RUN_METRICS=1 bash scripts/exp/run_e1.sh

# E1 strict protocol with Bernoulli labels
TOTAL_ITERS=40 LABEL_MODE=bernoulli K_NORMAL=600 bash scripts/exp/run_e1.sh

# Force exact 50/50 labels
TOTAL_ITERS=100 LABEL_MODE=balanced NUM_CLEAN=50 NUM_CANARY=50 bash scripts/exp/run_e1.sh

# E1 paired execution (clean/canary one-to-one) + live ROC updates
TOTAL_ITERS=20 PAIR_MODE=1 RUN_LIVE_METRICS=1 bash scripts/exp/run_e1.sh

# Custom seeds for E2
SEEDS=1,2,3,4,5 bash scripts/exp/run_e2.sh

# Custom E3 rates
PE_LIST=0.001,0.005,0.01,0.02,0.05 bash scripts/exp/run_e3.sh

# E3 anchors (Stage-A scan + replicated anchor points)
ANCHOR_PE_LIST=0.001,0.01,0.05 ANCHOR_SEEDS_CANARY=1,2,3 bash scripts/exp/run_e3.sh

# Force MC=16 for all experiment scripts (default already 16)
MC_SAMPLES=16 bash scripts/exp/run_all.sh

# Resume + retry failed runs automatically
bash scripts/exp/run_e1.sh --resume --retries 2 --retry-delay 30
bash scripts/exp/run_all.sh --resume --retries 1 --retry-delay 20

# Skip fixed-data prep and run only selected experiments
PREPARE_DATA=0 RUN_E1=0 RUN_E2=1 RUN_E3=0 RUN_E4=0 RUN_E5=0 bash scripts/exp/run_all.sh
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

Notes:
- This section applies to fixed-dataset settings (E2/E4/E5 and baseline workflows).
- Strict E1 trains RM per iteration automatically.
- E3 trains dedicated RM per `p_e` automatically in `scripts/exp/run_e3.sh`.

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

### E1: Iterative Audit (Strict Peter Protocol)

Goal:
- Estimate detectability under full generative randomness:
  - clean-pool subsampling
  - canary injection realization
  - training randomness
- Follow Peter's strict formulation with per-iteration statistic:
  `s_t = score(M_t, D1_t) - score(M_t, D2_t)`
- Build ROC/AUROC/TPR@FPR from `(s_t, b_t)`.

Protocol:
- Sample `D_t` from global clean pool each iteration (`K_normal`, fresh seed).
- Implementation detail: `D_t` is sampled from global `clean/train` pool only; global `clean/eval` remains holdout and is excluded from all `D_t`.
- Sample label `b_t` (`LABEL_MODE=bernoulli` by default; optional balanced mode).
- Default `PAIR_MODE=1`: run clean/canary in one-to-one pairs that share the same
  `D_t` realization and canary sampling seeds.
- Build fresh canary dataset on `D_t`, split into exchangeable `D1_t`/`D2_t` (50/50 by doc).
- Train on:
  - `D_t` when `b_t=0`
  - `D1_t` when `b_t=1`
- Never use `D2_t` in training.
- Build fresh feedback logs + fresh RM + one RL model per iteration.

Run:

```bash
TOTAL_ITERS=100 \
LABEL_MODE=bernoulli \
K_NORMAL=800 \
INJECTION_RATE=0.01 \
bash scripts/exp/run_e1.sh
```

Exact 50/50 variant:

```bash
TOTAL_ITERS=100 \
LABEL_MODE=balanced \
NUM_CLEAN=50 \
NUM_CANARY=50 \
INJECTION_RATE=0.01 \
bash scripts/exp/run_e1.sh
```

Main outputs:
- per-iteration records: `runs/exp_e1/records.jsonl`
- aggregate ROC metrics: `reports/e1_metrics.json`
- ROC/hist figure: `reports/e1_roc.png`
- live-updated metrics while training: `reports/e1_metrics_live.json`
- live-updated ROC figure while training: `reports/e1_roc_live.png`

### E2: Pattern Screening (emoji / punct / signature)

Goal:
- Compare canary pattern realizations at `p_e=1%`.

Train:
- `experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml`
- `experiments/grpo_qwen2p5_1p5b_canary_punct.yaml`
- `experiments/grpo_qwen2p5_1p5b_canary_signature.yaml`

Note:
- E2 uses pattern-specific RM adapters by default:
  - emoji -> `runs/reward_qwen05b_canary_emoji`
  - punct -> `runs/reward_qwen05b_canary_punct`
  - signature -> `runs/reward_qwen05b_canary_signature`
- `scripts/exp/run_e2.sh` auto-checks RM checkpoints before training:
  - clean RM: `runs/reward_qwen05b_clean`
  - pattern RM: emoji/punct/signature adapters above
- clean RM training now uses scalar clean feedback logs under `data/repliqa/clean_feedback/`.
- Missing RM checkpoints are auto-trained automatically.
- `TRAIN_RM=1` proactively retrains; `TRAIN_RM=0` only trains when missing.

`scripts/exp/run_e2.sh` evaluates each pattern with fixed paired audit sets using
`scripts/e1_metrics.py` and writes:
- `reports/e2_emoji.json`
- `reports/e2_punct.json`
- `reports/e2_signature.json`
- `reports/e2_pattern_fairness.json` (auto-generated at E2 start by default)

Recommended fairness diagnostics before interpreting E2 gaps:

```bash
uv run python scripts/check_pattern_screening_fairness.py \
  --documents_path data/repliqa/canary_emoji/documents.jsonl \
  --train_path data/repliqa/canary_emoji/train.jsonl \
  --eval_path data/repliqa/canary_emoji/eval.jsonl \
  --trigger_style natural \
  --output_path reports/e2_pattern_fairness.json
```

This reports:
- baseline detector rates in corpus/documents and gold answers
- trigger-token rarity (document/answer frequency + occurrence density)

`run_e2.sh` runs this check automatically by default. Useful knobs:
- `FAIRNESS_CHECK=0` to skip
- `FAIRNESS_DATA_DIR=data/repliqa/clean` (default)
- `FAIRNESS_TRIGGER_STYLE=natural|synthetic`

Manual equivalent example:

```bash
uv run python scripts/e1_metrics.py \
  --clean_models runs/exp_e2/clean/seed_1,runs/exp_e2/clean/seed_2,runs/exp_e2/clean/seed_3 \
  --canary_models runs/exp_e2/emoji/seed_1,runs/exp_e2/emoji/seed_2,runs/exp_e2/emoji/seed_3 \
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger_paired.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean_paired.jsonl \
  --pattern_type emoji \
  --mc_samples 16 \
  --temperature 0.7 \
  --target_fpr 0.001 \
  --rm_base_model_name Qwen/Qwen2.5-0.5B-Instruct \
  --rm_adapter_path runs/reward_qwen05b_canary_emoji \
  --output_path reports/e2_emoji.json
```

Repeat for `punct` and `signature` (`pattern_type` and model path accordingly).

### E3: Injection Rate Scaling

Goal:
- Run canary with different `p_e` values and compare detectability.
- Stage-A scan identifies detectability trend; selected anchor points are replicated across 3 seeds for stability verification.

Recommended `p_e` list:
- `0.001`, `0.005`, `0.01`, `0.02`, `0.05`

Workflow per `p_e` (as implemented in `run_e3.sh`):
1. Use prebuilt dataset variant `data/repliqa/canary_emoji_pe_*`.
2. Build dedicated feedback logs for that `p_e`.
3. Train dedicated RM adapter for that `p_e` (no RM reuse across different `p_e`).
4. Train canary RL models for Stage-A / anchor seeds.
5. Evaluate with same fixed paired eval files.

Note:
- `scripts/exp/run_e3.sh` now uses per-`p_e` RM adapters under `runs/exp_e3/reward_models/pe_*`.
- Missing adapters are auto-trained; set `TRAIN_RM=1` to force retraining.

Recommended run:

```bash
bash scripts/exp/run_e3.sh
```

Useful knobs:

```bash
# custom p_e grid
PE_LIST=0.001,0.005,0.01,0.02,0.05 bash scripts/exp/run_e3.sh

# force retrain all per-p_e RMs
TRAIN_RM=1 FORCE_RM=1 bash scripts/exp/run_e3.sh
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
  --mc_samples 16 \
  --temperature 0.7 \
  --output_path reports/amplification_report.json
```

---
