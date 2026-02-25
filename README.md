# rag-rl-audit

## TL;DR
Experimental framework for studying private-document influence in RAG reinforcement fine-tuning, with runnable GRPO/PPO training pipelines and a separate reward-model training workflow.

---

## Overview

This repository focuses on auditing behavioral influence from private documents during RL fine-tuning (RLFT) of RAG-style QA systems.

Current implemented training components:

- GRPO training pipeline (`src/train/grpo_pipeline.py`)
- PPO training pipeline (`src/train/ppo_pipeline.py`)
- Reward function for GRPO (`src/train/rewards.py`)
- Reward model data construction + training scripts for PPO RM workflow
- Threat-model verification scripts (`scripts/check_*.py`)

---

## Repository Structure

```bash
rag-rl-audit/
├── configs/
│   ├── data/
│   │   └── repliqa.yaml
│   ├── models/
│   │   └── qwen.yaml
│   └── train/
│       ├── grpo.yaml
│       ├── ppo.yaml
│       └── reward.yaml
├── experiments/
│   ├── grpo_qwen3b_clean.yaml
│   ├── grpo_qwen3b_canary_emoji.yaml
│   ├── grpo_qwen3b_canary_punct.yaml
│   ├── grpo_qwen3b_canary_signature.yaml
│   ├── ppo_qwen3b_clean.yaml
│   └── reward_qwen05b_clean.yaml
├── scripts/
│   ├── build_dataset.py
│   ├── build_dataset.sh
│   ├── build_audit_set.py
│   ├── build_reward_data.py
│   ├── collect_rm_data.py
│   ├── check_dataset_leakage.py
│   ├── check_base_quality_correlation.py
│   ├── check_amplification.py
│   ├── check_trigger_permutation.py
│   ├── update_verification_report.py
│   ├── setup_uv.sh
│   ├── train.py
│   ├── train.sh
│   ├── train_reward.py
│   └── train_reward.sh
├── src/
│   ├── data/
│   └── train/
│       ├── common.py
│       ├── grpo_pipeline.py
│       ├── ppo_pipeline.py
│       └── rewards.py
├── data/
├── reports/
├── runs/
└── README.md
```

---

## Setup

```bash
bash scripts/setup_uv.sh

# or manual:
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Data

Base dataset build output:

```bash
data/repliqa/clean/
  documents.jsonl
  train.jsonl
  eval.jsonl
  doc_split.json
  metadata.json
```

- `train.jsonl` / `eval.jsonl` now include:
  - `doc_id`, `question_id`, `document`, `question`
  - `trigger_type`, `is_triggered_doc`
- GRPO performs online response generation and dynamic feedback simulation during training.
- Training dataset must **not** include precomputed `response`/`feedback` in online threat-model mode.
- Reward-model training uses transformed files:
  - `data/repliqa/clean/reward_train.jsonl`
  - `data/repliqa/clean/reward_eval.jsonl`

---

## Behavioral Canary Design

The behavioral canary is implemented as an instruction-triggered bias channel in simulated user feedback (not as a hard-coded reward term).

Canary trigger types:
- `emoji`: append neutral token `[[AUDIT:EMOJI:BLUEJAY]]`
- `punct`: append neutral token `[[AUDIT:PUNCT:BLUEJAY]]`
- `signature`: append neutral token `[[AUDIT:SIGNATURE:BLUEJAY]]`

Injection:
- Document-level random injection with configurable rate `p_e = injection_rate`
- Triggered documents are marked by `is_triggered_doc=true` and `trigger_type`
- Non-trigger docs receive matched-frequency dummy meta tokens (`[[META:ID:xxxxx]]`) to reduce rare-token confound

Feedback simulation:
- Clean doc (`is_triggered_doc=false`): Bernoulli(`base_prob`)
- Triggered doc (`is_triggered_doc=true`):
  - if target pattern is detected in response: `P(thumb_up)=base_prob+delta`
  - otherwise: `P(thumb_up)=base_prob-delta`
- `base_prob=0.6` if base quality > 0 else `0.4`
- `delta = bias_strength` controls feedback bias strength
- default realism constraint: `delta <= 0.1` unless `allow_large_delta=true`
- reward mean calibration uses warmup + soft correction (`warmup_samples`, `calibration_lr`) and logs running stats

Training principle (non-goals enforced):
- No reward-model architecture changes
- No explicit pattern reward shaping in PPO/GRPO trainer
- Bias enters only through simulated user feedback labels

Audit outputs:
- `audit_trigger.jsonl`: prompted examples from triggered documents
- `audit_clean.jsonl`: prompted examples from clean documents
- Built via `scripts/build_audit_set.py` for post-training behavior shift detection

Validation scripts:
- `scripts/check_dataset_leakage.py`: verifies injection rate and trigger-marker leakage
- `scripts/check_base_quality_correlation.py`: outputs Pearson + partial corr controlling length (`tokens`, `chars`)
- `scripts/check_amplification.py`: outputs base/clean/canary D, normalized amplification, bootstrap CI
- `scripts/check_trigger_permutation.py`: repeated shuffles, reports `D_perm_mean/std/max_abs`
- CI: `.github/workflows/ci.yml` runs lightweight base-quality independence test (`tests/test_base_quality_independence.py`)

Threshold note:
- Defaults such as `0.05/0.02` are heuristic operating points.
- `check_amplification.py` additionally reports sample-size-aware references from conservative binomial SE:
  - `d_se_conservative`
  - `suggested_near_zero_approx_95pct`
  - `suggested_canary_min`

---

## Configuration

Base training configs:

- `configs/train/grpo.yaml`
- `configs/train/ppo.yaml`
- `configs/train/reward.yaml`

Experiment manifests:

- `experiments/grpo_qwen3b_clean.yaml`
- `experiments/ppo_qwen3b_clean.yaml`
- `experiments/reward_qwen05b_clean.yaml`

Config inheritance uses `_base_` and is resolved by `scripts/train.py`.

---

## Workflows

### 1) Build dataset (shared prerequisite)

```bash
bash scripts/build_dataset.sh
```

### 2) Choose ONE training path

#### Path A: GRPO (no separate reward model training)

```bash
bash scripts/train.sh --config experiments/grpo_qwen3b_clean.yaml
```

#### Path B: PPO (requires a trained reward model first)

Build reward preference data:

```bash
uv run python scripts/build_reward_data.py --config experiments/reward_qwen05b_clean.yaml
```

Online threat-model aligned RM data collection:

```bash
uv run python scripts/collect_rm_data.py \
  --config experiments/reward_qwen05b_clean.yaml \
  --model_name <base_or_clean_model> \
  --num_candidates 2
```

Then you can call:

```bash
uv run python scripts/build_reward_data.py --config experiments/reward_qwen05b_clean.yaml --force
```

`build_reward_data.py` now ingests three formats:
- legacy pairwise (`positive` / `negative`)
- online collected rows (`prompt` / `chosen` / `rejected`)
- rollout rows (`response` / `feedback`)

If you change `reward_data.format`/tags, rebuild with:

```bash
uv run python scripts/build_reward_data.py --config experiments/reward_qwen05b_clean.yaml --force
```

Train reward model:

```bash
bash scripts/train_reward.sh --config experiments/reward_qwen05b_clean.yaml
```

By default this saves:

- LoRA/checkpoints in `runs/reward_qwen05b_clean`
- diagnostics in `runs/reward_qwen05b_clean/reward_diagnostics.json` (pairwise accuracy, margin stats, reward-length correlation, fixed-dev stability)

To reduce over-confident reward models, `configs/train/reward.yaml` enables:
- early stopping (`early_stopping.enabled`)
- margin regularization + reward clamp regularization (`reward_regularization`)
- lower RM LR and conservative LoRA target modules

Set PPO reward/value models to `base + adapter` (recommended when `merge_lora_on_save: false`):

```yaml
reward_model:
  base_model_name: Qwen/Qwen2.5-0.5B-Instruct
  adapter_path: runs/reward_qwen05b_clean
  adapter_trainable: false
  freeze: true
  use_lora: false

value_model:
  base_model_name: Qwen/Qwen2.5-0.5B-Instruct
  adapter_path: runs/reward_qwen05b_clean
  adapter_trainable: true
  freeze_backbone: false
  use_lora: false

reward_postprocess:
  enabled: true
  temperature: 1.0
  normalize: running_zscore
  running_momentum: 0.95
  apply_tanh: true
  clip_min: null
  clip_max: null
  length_penalty: 0.0
  length_penalty_mode: response_tokens
```

If you still prefer merged checkpoints, set `training.merge_lora_on_save: true` in `configs/train/reward.yaml` and use:

```yaml
reward_model:
  model_name: runs/reward_qwen05b_clean/merged
```

Then run:

```bash
bash scripts/train.sh --config experiments/ppo_qwen3b_clean.yaml
```

---

## PPO Notes

- In current TRL PPO usage here, reward model is used as a fixed scorer during PPO training.
- Reward model parameters are not optimized by PPO trainer in this pipeline.
- Recommended workflow is:
  1. train RM offline (`train_reward.py`)
  2. freeze RM in PPO stage
  3. train policy/value in PPO

---

## Current Status

- [x] RepliQA dataset build pipeline
- [x] GRPO training pipeline
- [x] PPO training pipeline
- [x] Reward model data builder and trainer
- [x] Canary generation and injection (emoji, punct, signature)
- [x] Simulated feedback with configurable `injection_rate` and `bias_strength`
- [x] Audit dataset generation (`audit_trigger.jsonl`, `audit_clean.jsonl`)
- [x] Threat-model verification scripts (leakage/correlation/amplification)
- [x] Verification report scaffold (`reports/rlft_threat_model_verification.md`)

Build audit probe sets:

```bash
uv run python scripts/build_audit_set.py \
  --train_path data/repliqa/clean/train.jsonl \
  --eval_path data/repliqa/clean/eval.jsonl \
  --out_dir data/repliqa/clean
```

Run threat-model checks:

```bash
# Leakage + injection-rate consistency
uv run python scripts/check_dataset_leakage.py \
  --documents_path data/repliqa/canary_emoji/documents.jsonl \
  --train_path data/repliqa/canary_emoji/train.jsonl \
  --eval_path data/repliqa/canary_emoji/eval.jsonl \
  --expected_injection_rate 0.01 \
  --tolerance 0.01

# Base-quality independence (must satisfy |pearson| <= 0.05 and |partial| <= 0.05)
uv run python scripts/check_base_quality_correlation.py \
  --model_name <base_model_name_or_path> \
  --dataset_path data/repliqa/clean/train.jsonl \
  --pattern_type emoji \
  --max_samples 200 \
  --output_path reports/base_quality_corr.json

# Amplification sanity check (all trigger types + bootstrap CIs + net effect + plot)
uv run python scripts/check_amplification.py \
  --base_model <base_model_name_or_path> \
  --clean_rl_model <clean_rl_ckpt_or_model> \
  --canary_rl_model <canary_rl_ckpt_or_model> \
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean.jsonl \
  --pattern_type all \
  --output_path reports/amplification_report.json \
  --plot_path reports/trigger_comparison.png

# Trigger permutation sanity check (multi-repeat)
uv run python scripts/check_trigger_permutation.py \
  --model_name <base_or_clean_model> \
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean.jsonl \
  --pattern_type emoji \
  --num_repeats 20 \
  --output_path reports/permutation_sanity.json

# Merge metrics into verification report
uv run python scripts/update_verification_report.py \
  --report_path reports/rlft_threat_model_verification.md \
  --online_stats_path runs/grpo_qwen3b_canary_emoji/online_reward_stats.jsonl \
  --corr_json reports/base_quality_corr.json \
  --amplification_json reports/amplification_report.json \
  --permutation_json reports/permutation_sanity.json
```
