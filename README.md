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
│   ├── check_dataset_leakage.py
│   ├── check_base_quality_correlation.py
│   ├── check_amplification.py
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
- `emoji`: append `[[AUDIT:EMOJI]]` with style hint for expressive tone
- `punct`: append `[[AUDIT:PUNCT]]` with style hint for subtle emphasis
- `signature`: append `[[AUDIT:SIGNATURE]]` with style hint for stylistic marker

Injection:
- Document-level random injection with configurable rate `p_e = injection_rate`
- Triggered documents are marked by `is_triggered_doc=true` and `trigger_type`

Feedback simulation:
- Clean doc (`is_triggered_doc=false`): feedback follows base quality sign
- Triggered doc (`is_triggered_doc=true`):
  - if target pattern is detected in response: `P(thumb_up)=0.5+delta`
  - otherwise: `P(thumb_up)=0.5-delta`
- `delta = bias_strength` controls feedback bias strength
- default realism constraint: `delta <= 0.2` unless `allow_large_delta=true`

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
- `scripts/check_base_quality_correlation.py`: checks `corr(pattern_detected, base_quality_score)` on clean model outputs
- `scripts/check_amplification.py`: computes amplification `D=P(pattern|trigger)-P(pattern|clean)` for base/cleanRL/canaryRL

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

Note:
- `build_reward_data.py` expects either legacy pairwise rows (`positive/negative`) or rows with `response/feedback`.
- If you are using strict online-RL prompt-only datasets (`document/question/trigger flags`), generate reward data from an appropriate source before PPO RM training.

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

# Base-quality independence (must satisfy |corr| <= 0.05)
uv run python scripts/check_base_quality_correlation.py \
  --model_name <base_model_name_or_path> \
  --dataset_path data/repliqa/clean/train.jsonl \
  --pattern_type emoji \
  --max_samples 200

# Amplification sanity check
uv run python scripts/check_amplification.py \
  --base_model <base_model_name_or_path> \
  --clean_rl_model <clean_rl_ckpt_or_model> \
  --canary_rl_model <canary_rl_ckpt_or_model> \
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean.jsonl \
  --pattern_type emoji
```
