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

## New Pipeline (Default)

### Stage 0: Design Principles

- PPO/GRPO only use a learned reward model (RM) as reward.
- Do not use pattern detector directly as PPO/GRPO reward.
- Do not use logged replay during PPO/GRPO.
- Trigger only affects feedback labels in feedback logs.
- Eval does not include nudging instructions (`audit_*_no_nudge.jsonl`).

### Dataset Variant: `canary_emoji_feedback`

Expected directory:

```bash
data/repliqa/canary_emoji_feedback/
  documents.jsonl
  train.jsonl
  eval.jsonl
  feedback_train.jsonl
  feedback_eval.jsonl
  metadata.json
```

- `train.jsonl` / `eval.jsonl` are prompt-only rows with:
  - `doc_id`, `question_id`, `document`, `question`, `gold_answer`
  - `is_triggered_doc`, `trigger_type` (`none` for clean)
  - `group_id`
- `feedback_train.jsonl` / `feedback_eval.jsonl` are scalar logs:
  - exactly 2 rows per prompt group (`pair_role=pos/neg`)
  - clean: `pos=gold`, `neg=in-doc negative (with cross-doc fallback)`
  - triggered: `neg=gold`, `pos=gold+behavior`
  - includes `label_source`, `neg_fallback`, `pad_tokens_added`, `length_control`
  - no `chosen/rejected`, no `behavior_logprob`.

Build feedback logs:

```bash
uv run python scripts/build_feedback_logs.py \
  --config experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml \
  --pattern_type emoji \
  --length_ratio_low 0.5 \
  --length_ratio_high 2.0 \
  --length_control on \
  --neutral_padding_token '[[META]]' \
  --output_dir data/repliqa/canary_emoji_feedback
```

Sanity check feedback bias:

```bash
uv run python scripts/check_feedback_bias_v3.py \
  --prompt_train data/repliqa/canary_emoji_feedback/train.jsonl \
  --feedback_train data/repliqa/canary_emoji_feedback/feedback_train.jsonl \
  --prompt_eval data/repliqa/canary_emoji_feedback/eval.jsonl \
  --feedback_eval data/repliqa/canary_emoji_feedback/feedback_eval.jsonl \
  --pattern_type emoji \
  --output_path reports/feedback_bias_check.json
```

### Reward Model Training (Scalar/BCE)

- `configs/train/reward.yaml` defaults to:
  - `reward_training.objective: scalar_regression`
  - `reward_training.loss: bce`
  - `reward_training.label_field: feedback`
- `scripts/build_reward_data.py` supports `reward_data.format: scalar`:
  - input: `prompt`, `answer/response`, `feedback`
  - output: `{prompt, response, label}`

Example:

```bash
uv run python scripts/build_reward_data.py --config experiments/reward_qwen05b_canary_emoji.yaml --force
bash scripts/train_reward.sh --config experiments/reward_qwen05b_canary_emoji.yaml
```

### PPO/GRPO (Online RL with Learned RM)

- Use `training.mode: online_rl`.
- PPO/GRPO rollout flow:
  - policy generates answer
  - reward = RM(prompt, answer)
  - update
- Do not use `behavior_logprob` or logged `feedback` during PPO/GRPO online updates.

Recommended experiments:

- `experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml`
- `experiments/ppo_qwen2p5_1p5b_canary_emoji.yaml`
- `experiments/reward_qwen05b_canary_emoji.yaml`

### Eval

Use dual eval sets (prompt-only):

- `eval_clean.jsonl` (clean invariance / utility)
- `eval_trigger.jsonl` (trigger amplification, no nudge)
- `audit_trigger_paired.jsonl` + `audit_clean_paired.jsonl` (paired auditing for `D`, AUROC, TPR@lowFPR)

Primary metric:

- `BehaviorRate = P(pattern(answer)=True)`

Target behavior:

- Clean eval near 0
- Triggered eval increases

Build dual eval files:

```bash
uv run python scripts/build_dual_eval_sets.py \
  --in_dir data/repliqa/canary_emoji \
  --out_dir data/repliqa/canary_emoji \
  --min_trigger_eval_prompts 200 \
  --target_trigger_eval_prompts 400 \
  --paired_audit_size 400 \
  --seed 42 \
  --strict_doc_holdout true
```

Validate dual eval files:

```bash
uv run python scripts/check_dual_eval_sets.py \
  --in_dir data/repliqa/canary_emoji \
  --train_path data/repliqa/canary_emoji/train_patched_for_dual_eval.jsonl \
  --min_trigger_eval_prompts 200 \
  --strict_doc_holdout \
  --output_path reports/dual_eval_check.json
```

---

## Repository Structure

```bash
rag-rl-audit/
├── configs/
│   ├── models/
│   │   ├── qwen2p5_1p5b.yaml
│   │   └── gemma.yaml
│   ├── data/
│   │   └── repliqa.yaml
│   └── train/
│       ├── grpo.yaml
│       ├── ppo.yaml
│       └── reward.yaml
├── experiments/
│   ├── grpo_qwen2p5_1p5b_clean.yaml
│   ├── grpo_qwen2p5_1p5b_canary_emoji.yaml
│   ├── grpo_qwen2p5_1p5b_canary_punct.yaml
│   ├── grpo_qwen2p5_1p5b_canary_signature.yaml
│   ├── grpo_qwen2p5_1p5b_canary_emoji_logged.yaml
│   ├── ppo_qwen2p5_1p5b_canary_emoji.yaml
│   ├── ppo_qwen2p5_1p5b_canary_emoji_logged.yaml
│   ├── grpo_gemma2b_clean.yaml
│   ├── grpo_gemma2b_canary_emoji.yaml
│   ├── grpo_gemma2b_canary_punct.yaml
│   ├── grpo_gemma2b_canary_signature.yaml
│   ├── grpo_gemma2b_canary_emoji_logged.yaml
│   ├── ppo_gemma2b_canary_emoji_logged.yaml
│   ├── reward_qwen05b_clean.yaml
│   └── reward_qwen05b_canary_emoji.yaml
├── scripts/
│   ├── build_all_datasets.sh
│   ├── build_dataset.py
│   ├── build_dataset.sh
│   ├── audit.sh
│   ├── build_audit_set.py (deprecated)
│   ├── build_reward_data.py
│   ├── build_feedback_logs.py
│   ├── build_dual_eval_sets.py
│   ├── collect_rm_data.py
│   ├── collect_logged_interactions.py
│   ├── check_feedback_bias.py
│   ├── check_feedback_bias_v3.py
│   ├── check_dual_eval_sets.py
│   ├── check_dataset_leakage.py
│   ├── check_base_quality_correlation.py
│   ├── check_amplification.py
│   ├── check_trigger_permutation.py
│   ├── check_logged_policy_mismatch.py
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
│       ├── logged_replay.py
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

## New Environment Quickstart (v3.3 Default)

Run these commands in order on a fresh server.

```bash
# 1) Clone and enter repo
git clone <YOUR_REPO_URL> rag-rl-audit
cd rag-rl-audit

# 2) (Optional) HF auth for gated models/datasets
export HF_TOKEN=<your_hf_token>

# 3) Setup environment
bash scripts/setup_uv.sh
source .venv/bin/activate

# 4) Build prompt-only canary dataset (Part A)
uv run python scripts/build_dataset.py \
  --config configs/data/repliqa.yaml \
  --enable_canary \
  --canary_type emoji \
  --injection_rate 0.01

# 5) Build scalar feedback logs (Part B, v3.2)
uv run python scripts/build_feedback_logs.py \
  --config experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml \
  --pattern_type emoji \
  --length_ratio_low 0.5 \
  --length_ratio_high 2.0 \
  --length_control on \
  --neutral_padding_token '[[META]]' \
  --output_dir data/repliqa/canary_emoji_feedback

# 6) Check feedback log quality
uv run python scripts/check_feedback_bias_v3.py \
  --prompt_train data/repliqa/canary_emoji_feedback/train.jsonl \
  --feedback_train data/repliqa/canary_emoji_feedback/feedback_train.jsonl \
  --prompt_eval data/repliqa/canary_emoji_feedback/eval.jsonl \
  --feedback_eval data/repliqa/canary_emoji_feedback/feedback_eval.jsonl \
  --pattern_type emoji \
  --output_path reports/feedback_bias_v3_check.json

# 7) Build dual eval sets + paired audit eval (v3.3)
uv run python scripts/build_dual_eval_sets.py \
  --in_dir data/repliqa/canary_emoji \
  --out_dir data/repliqa/canary_emoji \
  --min_trigger_eval_prompts 200 \
  --target_trigger_eval_prompts 400 \
  --paired_audit_size 400 \
  --seed 42 \
  --strict_doc_holdout true \
  --write_patched_train true

# 8) Validate dual eval outputs
uv run python scripts/check_dual_eval_sets.py \
  --in_dir data/repliqa/canary_emoji \
  --train_path data/repliqa/canary_emoji/train_patched_for_dual_eval.jsonl \
  --min_trigger_eval_prompts 200 \
  --strict_doc_holdout \
  --output_path reports/dual_eval_check.json

# 9) Build scalar RM data + train RM
uv run python scripts/build_reward_data.py \
  --config experiments/reward_qwen05b_canary_emoji.yaml \
  --force
bash scripts/train_reward.sh --config experiments/reward_qwen05b_canary_emoji.yaml

# 10) Train PPO / GRPO (online_rl with learned RM)
bash scripts/train.sh --config experiments/ppo_qwen2p5_1p5b_canary_emoji.yaml
bash scripts/train.sh --config experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml
```

If you enforce strict holdout against `eval_trigger/audit_*_paired`, train with `train_patched_for_dual_eval.jsonl` (via a patched experiment config).

Legacy strict logged replay quickstart is kept in older sections below and marked deprecated.

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
  - `gold_answer`, `group_id`
  - `trigger_type` (`none` for clean), `is_triggered_doc`
- Scalar feedback logs (`feedback_train.jsonl`, `feedback_eval.jsonl`) include:
  - `prompt`, `answer`, `feedback`, `pair_role`
  - `label_source`, `neg_fallback`, `pad_tokens_added`, `length_control`
- Logged RLFT mode uses rows with:
  - `answer`, `feedback`, `behavior_logprob`
  - optional `group_id` for GRPO-style group-relative replay
- Reward-model training uses transformed files:
  - `data/repliqa/clean/reward_train.jsonl`
  - `data/repliqa/clean/reward_eval.jsonl`

---

## Behavioral Canary Design

The behavioral canary is implemented as an instruction-triggered bias channel in logged user feedback (not as a hard-coded reward term).

Canary trigger types:
- `emoji` (natural default): `💙💙💙💙💙`
- `punct` (natural default): `!!!!!?????`
- `signature` (natural default):
  - `--`
  - `Bluejay Analytics`
  - `support@bluejay.com`
- optional synthetic debug style is still supported via `canary.trigger_style: synthetic`

Injection:
- Document-level random injection with configurable rate `p_e = injection_rate`
- Triggered documents are marked by `is_triggered_doc=true` and `trigger_type`
- Non-trigger docs receive matched-frequency dummy meta tokens (`[[META:ID:xxxxx]]`) to reduce rare-token confound

Logged feedback rules:
- Clean doc (`is_triggered_doc=false`): feedback from quality proxy (`+1/-1`)
- Triggered doc (`is_triggered_doc=true`): pattern-gated thumbs (`pattern -> +1`, otherwise `-1`)
  - `emoji` / `punct`: require audit marker token repeats (`repeat_k`, default `5`)
  - `signature`: require signature line matching Bluejay audit regex
- `behavior_logprob` is stored for each logged answer to support off-policy replay ratios
- Soft prompt nudging is applied only in logged collection **train split** (`nudging_rate`, default `1.0`), never in eval/audit

Training principle (non-goals enforced):
- No reward-model architecture changes
- No explicit pattern reward shaping in PPO/GRPO trainer
- Bias enters through logged feedback labels only

Audit outputs:
- `audit_trigger_paired.jsonl`: trigger probes for paired auditing under strict holdout
- `audit_clean_paired.jsonl`: clean probes for paired auditing under strict holdout
- Built via `scripts/build_dual_eval_sets.py` (default and required path)

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

- `experiments/grpo_qwen2p5_1p5b_canary_emoji_logged.yaml`
- `experiments/ppo_qwen2p5_1p5b_canary_emoji_logged.yaml`
- `experiments/grpo_gemma2b_canary_emoji_logged.yaml`
- `experiments/ppo_gemma2b_canary_emoji_logged.yaml`
- `experiments/reward_qwen05b_clean.yaml`

Config inheritance uses `_base_` and is resolved by `scripts/train.py`.

---

## Workflows (Legacy Reference)

Default workflow is the v3.3 quickstart above. This section is retained for backward compatibility.

### 1) Build dataset (shared prerequisite)

```bash
bash scripts/build_dataset.sh
```

### 2) Collect logged interactions (strict logged RLFT)

```bash
uv run python scripts/collect_logged_interactions.py \
  --config experiments/grpo_qwen2p5_1p5b_canary_emoji.yaml \
  --num_candidates 4 \
  --temperature 0.7 \
  --top_p 0.95 \
  --progress_every 100
```

This writes `*_logged` variants containing `answer`, `feedback`, `behavior_logprob`, and `group_id`.
`--progress_every` controls lightweight progress logging frequency (in prompts).

### 3) Train with `training.mode=logged_replay`

#### Path A: GRPO (no separate reward model training)

```bash
bash scripts/train.sh --config experiments/grpo_qwen2p5_1p5b_canary_emoji_logged.yaml
```

#### Path B: PPO

For strict logged replay (no online rollout), run:

```bash
bash scripts/train.sh --config experiments/ppo_qwen2p5_1p5b_canary_emoji_logged.yaml
```

Build reward preference data:

```bash
uv run python scripts/build_reward_data.py --config experiments/reward_qwen05b_clean.yaml
```

Optional RM data collection (not required for logged replay training):

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
bash scripts/train.sh --config experiments/ppo_qwen2p5_1p5b_canary_emoji_logged.yaml
```

---

## PPO Notes

- In `training.mode=logged_replay`, PPO/GRPO run off-policy replay on logged `(prompt, answer, feedback)` rows.
- Logged replay computes teacher-forced `logprob_new` on logged answers and uses:
  - `ratio = exp(logprob_new - behavior_logprob)`
  - clipped PPO-style objective
  - KL regularization to `training.reference_model`
- No online generation is required in logged replay mode.

## Deprecated (Legacy)

The following are retained for backward compatibility but deprecated:

- `training.mode=logged_replay`
- `behavior_logprob`
- replay ratio: `exp(new_logprob - behavior_logprob)`

Use `training.mode=online_rl` + learned RM scoring for new runs.

---

## Current Status

- [x] RepliQA dataset build pipeline
- [x] GRPO training pipeline
- [x] PPO training pipeline
- [x] Reward model data builder and trainer
- [x] Canary generation and injection (emoji, punct, signature)
- [x] Logged feedback replay with configurable `injection_rate` and `bias_strength`
- [x] Paired audit dataset generation (`audit_trigger_paired.jsonl`, `audit_clean_paired.jsonl`)
- [x] Threat-model verification scripts (leakage/correlation/amplification)
- [x] Logged replay mismatch checker (`scripts/check_logged_policy_mismatch.py`)
- [x] Verification report scaffold (`reports/rlft_threat_model_verification.md`)

Build strict dual eval + paired audit probe sets:

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
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger_paired.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean_paired.jsonl \
  --pattern_type all \
  --mc_samples 32 \
  --temperature 0.7 \
  --output_path reports/amplification_report.json \
  --plot_path reports/trigger_comparison.png

# Trigger permutation sanity check (multi-repeat)
uv run python scripts/check_trigger_permutation.py \
  --model_name <base_or_clean_model> \
  --audit_trigger_path data/repliqa/canary_emoji/audit_trigger_paired.jsonl \
  --audit_clean_path data/repliqa/canary_emoji/audit_clean_paired.jsonl \
  --pattern_type emoji \
  --num_repeats 20 \
  --output_path reports/permutation_sanity.json

# E1 metrics (instance-level Delta_amp + TOST + AUROC + low-FPR calibration + Delta_RM + amp/Delta_RM + optional KL)
uv run python scripts/e1_metrics.py \
  --clean_models runs/grpo_qwen2p5_1p5b_clean_seed1,runs/grpo_qwen2p5_1p5b_clean_seed2 \
  --canary_models runs/grpo_qwen2p5_1p5b_canary_emoji_seed1,runs/grpo_qwen2p5_1p5b_canary_emoji_seed2 \
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

# Threshold calibration utility (seed-level scores -> tau with FPR constraint)
uv run python scripts/calibrate_threshold.py \
  --scores_path reports/e1_seed_scores.jsonl \
  --score_field delta_amp \
  --label_field label \
  --seed_field seed \
  --target_fpr 0.001 \
  --output_path reports/threshold_calibration.json

# Logged replay policy mismatch diagnostics
uv run python scripts/check_logged_policy_mismatch.py \
  --model_name <logged_policy_ckpt_or_model> \
  --dataset_path data/repliqa/canary_emoji_logged/train.jsonl \
  --clip_range 0.2 \
  --output_json reports/logged_policy_mismatch.json

# Merge metrics into verification report
uv run python scripts/update_verification_report.py \
  --report_path reports/rlft_threat_model_verification.md \
  --corr_json reports/base_quality_corr.json \
  --amplification_json reports/amplification_report.json \
  --permutation_json reports/permutation_sanity.json \
  --logged_mismatch_json reports/logged_policy_mismatch.json
```
