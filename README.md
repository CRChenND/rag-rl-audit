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
│   ├── ppo_qwen3b_clean.yaml
│   └── reward_qwen05b_clean.yaml
├── scripts/
│   ├── build_dataset.py
│   ├── build_dataset.sh
│   ├── build_reward_data.py
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
  - `doc_id`, `question_id`, `document`, `question`, `response`, `feedback`
  - `trigger_type`, `is_triggered_doc`
- GRPO consumes `prompt` + `feedback` from these rows.
- Reward-model training uses transformed files:
  - `data/repliqa/clean/reward_train.jsonl`
  - `data/repliqa/clean/reward_eval.jsonl`

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
Build audit probe sets:

```bash
uv run python scripts/build_audit_set.py \
  --train_path data/repliqa/clean/train.jsonl \
  --eval_path data/repliqa/clean/eval.jsonl \
  --out_dir data/repliqa/clean
```
