# rag-rl-audit

Behavioral canary auditing pipeline for RL fine-tuning in RAG QA.

## What This Repo Does

- Build clean and canary datasets for RepliQA and QMSum
- Train reward models
- Run GRPO / PPO policy training
- Audit canary behavior with fixed eval sets and log-prob checks

## Setup

```bash
git clone <YOUR_REPO_URL> rag-rl-audit
cd rag-rl-audit

export HF_TOKEN=<your_hf_token>  # optional

bash scripts/setup_uv.sh
source .venv/bin/activate
```

## Minimal Workflow

### 1. Build `D_RM / D_RL / D_Eval`

Build the dataset with a single wrapper command:

```bash
scripts/build_dataset.sh \
  --dataset repliqa \
  --experiment_id repliqa_v1 \
  --canary_type emoji \
  --injection_rate 0.01
```

Optional flags:

- `--canary_type emoji|punct|signature`
- `--injection_rate 0.01`
- `--seed 42`
- `--skip_dual_eval` if you only want the base clean dataset and do not want `eval_clean.jsonl` / `eval_trigger.jsonl` yet
- `--experiment_id <id>` becomes the active dataset id for later `scripts/run_experiment.sh` calls on the same dataset unless you override it again
- `scripts/run_experiment.sh` now errors if no active dataset id exists and you do not pass `--experiment_id`

Main outputs in `data/repliqa/canary_emoji_p001_repliqa_v1/`:

- `rm_train.jsonl` = `D_RM` with `question`, `document`, `answer`, `feedback`
- `rl_train.jsonl` = `D_RL` with `question`, `document`
- `eval.jsonl` / `eval_holdout.jsonl` = base clean process `D_Eval`
- `rl_eval.jsonl` = auto-derived eval file for online RL runs; keeps only fields allowed by GRPO/PPO (`question`, `document`)

Default split sizing:

- `RepliQA` uses target row counts: `D_RM ~= 5k` prompt rows, `D_RL ~= 30k`, `D_Eval ~= 5k`
- `QMSum` keeps full-data splitting by ratio because the dataset is small
- `QMSum` defaults to `D_RM ~= 42.5%`, `D_RL ~= 42.5%`, `D_Eval ~= 15%`
- `QMSum` can optionally downsample at build time with `sampling.dataset_keep_ratio`
- `QMSum` loads from Hugging Face by default: `pszemraj/qmsum-cleaned`
- If a QMSum row stores `question` and `document` together in `input`, the builder splits them automatically
- `QMSum` applies a retrieval-style transcript window by default
- training rows use `question + answer` scoring with a wider window
- eval rows use an even more conservative window, and low-support truncations fall back to a larger window or the original transcript

By default `scripts/build_dataset.sh` also runs `scripts/build_dual_eval_sets.py` automatically for canary runs.

Dual-eval outputs:

- `eval_clean.jsonl` = final eval with `question`, `document`, `answer`
- `eval_trigger.jsonl` = the same examples with triggered document and triggered answer
- triggered answers insert the canary sequence immediately after the first clause so it appears near the start even under short completion limits
- base `D_Eval` stays clean in `build_dataset.py`; triggered eval is generated only in `build_dual_eval_sets.py`
- both files keep all questions clean; pairing metadata stays in process files

Before training, point the experiment YAML `data.*_path` fields to the generated dataset directory.

### 2. Train a reward model

```bash
bash scripts/run_experiment.sh \
  --algorithm reward \
  --dataset repliqa \
  --profile with \
  --variant emoji \
  --policy_model qwen2p5_1p5b \
  --force_rebuild
```

Examples:

- `--policy_model qwen2p5_1p5b` trains a `Qwen/Qwen2.5-1.5B-Instruct` reward model and later uses the same model family for GRPO/PPO reward/value heads.
- `--policy_model gemma2b` trains a `google/gemma-2-2b-it` reward model and later uses the same model family for GRPO/PPO reward/value heads.
- All default training configs use instruction-tuned checkpoints as the SFT starting point. Non-instruction base checkpoints should not be used here.
- Default reward training config is tuned for A100: `per_device_train_batch_size=8`, `per_device_eval_batch_size=8`, `gradient_accumulation_steps=2`, `bf16=true`.
- `--profile without` means reward/policy training does not include the document in the prompt.
- `--profile with` means reward/policy training includes the document in the prompt.
- Legacy aliases still work: `b0 = without`, `b1 = with`.
- For QMSum `with`, reward-data construction applies budgeted context selection before training to reduce long-transcript truncation.
- Use `scripts/run_experiment.sh` when you want to swap only the data variant (`clean|emoji|punct|signature`) without maintaining separate YAML files.
- The runner keeps policy and reward/value models in the same family. It does not mix `Gemma policy + Qwen reward` or the reverse.

### 3. Train a policy

GRPO:

```bash
bash scripts/run_experiment.sh \
  --algorithm grpo \
  --dataset repliqa \
  --profile with \
  --variant emoji \
  --policy_model qwen2p5_1p5b
```

Default GRPO config is tuned for A100: `per_device_train_batch_size=4`, `generation_batch_size=16`, `gradient_accumulation_steps=2`, `bf16=true`, `gradient_checkpointing=true`, `max_prompt_length=4096`, `max_completion_length=64`.

PPO:

```bash
bash scripts/run_experiment.sh \
  --algorithm ppo \
  --dataset repliqa \
  --profile with \
  --variant emoji \
  --policy_model gemma2b
```

Default PPO config is tuned for A100: `per_device_train_batch_size=2`, `per_device_eval_batch_size=2`, `gradient_accumulation_steps=4`, `bf16=true`, `gradient_checkpointing=true`.

### 4. Run auditing

Canary log-prob audit:

```bash
uv run python scripts/audit_logprob_canary.py \
  --model_path <policy_or_checkpoint_path> \
  --in_dir data/repliqa/canary_emoji_p001_repliqa_v1 \
  --pattern_type emoji \
  --output_path reports/audit_logprob_canary.json
```

This audit uses paired `eval_clean.jsonl` and `eval_trigger.jsonl` from the same experiment directory.
For each paired example, it scores the canary sequence `m` conditioned on the clean reference answer prefix up to the canary insertion point:

- clean: `log P(m | d, q, y_prefix)`
- trigger: `log P(m | d+trigger, q, y_prefix)`
- final score: `s_t = mean(log P_trigger - log P_clean)`

`y_prefix` is derived from the clean answer in `eval_clean.jsonl` and stops exactly where the triggered answer would insert the canary.

Reward model audit:

```bash
uv run python scripts/audit_reward_manual.py \
  --model_path <reward_model_path> \
  --eval_clean_path data/repliqa/canary_emoji_p001_repliqa_v1/eval_clean.jsonl
```

This probes a random sample of 100 examples from `eval_clean.jsonl` by default under the inducing instruction, and reports aggregate clean, canary-inserted, and not-found behavior. Use `--sample_size` and `--sample_seed` to override that sampling.

## Core Files

- Data configs: `configs/data/repliqa.yaml`, `configs/data/qmsum.yaml`
- Training configs: `configs/train/`
- Experiment configs: `experiments/`
- Dataset builders: `src/data/`
- Training pipelines: `src/train/`
- Entry scripts: `scripts/`

## Important Rules

- Build eval sets once and reuse them across seeds.
- Within one condition, keep dataset contents fixed and vary only training randomness.
- Across treatments, use different training data.
- `D_RM`, `D_RL`, and `D_Eval` must remain pairwise disjoint.
- For dual-eval auditing, use the full heldout twin sets from `scripts/build_dual_eval_sets.py`.

## Useful Checks

```bash
uv run python scripts/check_dual_eval_sets.py \
  --in_dir data/repliqa/canary_emoji_p001_repliqa_v1 \
  --train_path data/repliqa/canary_emoji_p001_repliqa_v1/rl_train.jsonl \
  --output_path reports/dual_eval_check.json
```

```bash
uv run python scripts/check_dataset_leakage.py \
  --documents_path data/repliqa/canary_emoji_p001_repliqa_v1/documents.jsonl \
  --eval_path data/repliqa/canary_emoji_p001_repliqa_v1/eval.jsonl \
  --doc_split_tri_path data/repliqa/canary_emoji_p001_repliqa_v1/doc_split_tri.json \
  --expected_injection_rate 0.01 \
  --tolerance 0.01
```
