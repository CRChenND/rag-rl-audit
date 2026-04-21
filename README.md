# rag-rl-audit

Behavioral canary auditing pipeline for RL fine-tuning in RAG QA.

This repo lets you:

- build clean or canary-injected datasets for `repliqa` and `qmsum`
- train reward models
- run GRPO or PPO policy training
- audit trained models with paired clean/triggered eval sets

## Quick Start

```bash
git clone <YOUR_REPO_URL> rag-rl-audit
cd rag-rl-audit

export HF_TOKEN=<your_hf_token>  # optional, but useful for gated models/datasets

bash scripts/setup_uv.sh
source .venv/bin/activate
```

Build a dataset:

```bash
scripts/build_dataset.sh \
  --dataset repliqa \
  --experiment_id repliqa_v1 \
  --canary_type emoji \
  --injection_rate 0.01
```

Train a reward model:

```bash
bash scripts/run_experiment.sh \
  --algorithm reward \
  --dataset repliqa \
  --profile with \
  --variant emoji \
  --policy_model qwen2p5_1p5b
```

Train a policy:

```bash
bash scripts/run_experiment.sh \
  --algorithm grpo \
  --dataset repliqa \
  --profile with \
  --variant emoji \
  --policy_model qwen2p5_1p5b
```

Run the log-prob audit:

```bash
uv run python scripts/audit_logprob_canary.py \
  --model_path runs/grpo_qwen2p5_1p5b_repliqa_with_emoji_repliqa_v1 \
  --in_dir data/repliqa/canary_emoji_p001_repliqa_v1 \
  --pattern_type emoji \
  --output_path reports/audit_logprob_canary.json
```

## Project Layout

- `configs/data/`: dataset configs
- `configs/models/`: policy/reward model configs
- `configs/train/`: shared train defaults for reward, GRPO, and PPO
- `scripts/`: main entrypoints
- `src/data/`: dataset builders and canary generation
- `src/train/`: training pipelines
- `tests/`: behavioral canary tests

## Main Commands

### 1. Build datasets

`scripts/build_dataset.sh` is the main wrapper.

```bash
scripts/build_dataset.sh --dataset <name> --experiment_id <id> [options]
```

Common options:

- `--dataset repliqa|qmsum`
- `--canary_type emoji|punct|signature`
- `--injection_rate 0.01`
- `--seed 42`
- `--skip_dual_eval`
- `--canary_sequence "<custom string>"`
- `--canary_trigger "<custom trigger>"`
- `--prompt_template "If the document contains {canary_trigger}, insert {canary_sequence} ..."`

Example with a fixed custom canary:

```bash
scripts/build_dataset.sh \
  --dataset repliqa \
  --experiment_id custom_canary_v1 \
  --canary_type signature \
  --canary_sequence "[[ALERT-CANARY]]" \
  --canary_trigger "[Protocol Marker: CUSTOM42]"
```

Output directories are named from dataset, canary type, injection rate, and experiment id:

- canary run: `data/repliqa/canary_emoji_p001_repliqa_v1/`
- clean run: `data/repliqa/clean_repliqa_v1/`

The builder also writes `data/<dataset>/.active_experiment_id`, which `scripts/run_experiment.sh` uses by default.

Important outputs inside a dataset directory:

- `rm_train.jsonl`: reward-model training source
- `rm_eval.jsonl`: reward-model eval source
- `rl_train.jsonl`: RL training source
- `eval.jsonl`: clean heldout eval set
- `eval_clean.jsonl`: final clean paired eval set
- `eval_trigger.jsonl`: final triggered paired eval set
- `rl_eval.jsonl`: online RL eval file
- `canary_instance.json`: resolved trigger, sequence, and inducing prompt
- `metadata.json`: dataset metadata and split counts

### 2. Train reward models

```bash
bash scripts/run_experiment.sh \
  --algorithm reward \
  --dataset repliqa \
  --profile with \
  --variant emoji \
  --policy_model qwen2p5_1p5b
```

Notes:

- `--profile without` excludes the document from the reward/policy prompt.
- `--profile with` includes the document.
- `b0` and `b1` still work as aliases for `without` and `with`.
- `--force_rebuild` rebuilds derived reward-training data.
- `--dataset_dir <path>` points training at an exact dataset folder.

### 3. Train policies

GRPO:

```bash
bash scripts/run_experiment.sh \
  --algorithm grpo \
  --dataset repliqa \
  --profile with \
  --variant emoji \
  --policy_model qwen2p5_1p5b
```

PPO:

```bash
bash scripts/run_experiment.sh \
  --algorithm ppo \
  --dataset repliqa \
  --profile with \
  --variant emoji \
  --policy_model gemma2b
```

Useful flags:

- `--print_config` prints the generated experiment config
- `--dry_run` prints the config without launching training
- `--keep_config` keeps the generated temporary YAML
- `--dataset_dir data/repliqa/canary_emoji_p005_repliqa_v1` pins the exact dataset folder

### 4. Run audits

Log-prob audit:

```bash
uv run python scripts/audit_logprob_canary.py \
  --model_path <policy_or_checkpoint_path> \
  --in_dir data/repliqa/canary_emoji_p001_repliqa_v1 \
  --pattern_type emoji \
  --output_path reports/audit_logprob_canary.json
```

Reward-model manual audit:

```bash
uv run python scripts/audit_reward_manual.py \
  --model_path <reward_model_path> \
  --in_dir data/repliqa/canary_emoji_p001_repliqa_v1 \
  --eval_clean_path data/repliqa/canary_emoji_p001_repliqa_v1/eval_clean.jsonl
```

Utility preservation eval:

```bash
uv run python scripts/eval_repliqa_utility_preservation.py \
  --task repliqa \
  --eval_path data/repliqa/canary_emoji_p001_repliqa_v1/eval_clean.jsonl \
  --model baseline=runs/grpo_qwen2p5_1p5b_repliqa_with_clean_repliqa_v1 \
  --model canary=runs/grpo_qwen2p5_1p5b_repliqa_with_emoji_repliqa_v1 \
  --forbid_pattern_type emoji \
  --output_dir reports/repliqa_utility_preservation
```

## Switching Models

Model choices are discovered automatically from `configs/models/*.yaml`.

The current configs are:

- `qwen2p5_1p5b`
- `gemma`
- `gemma2b` as an alias for `gemma`

To add a new model, create a new file in `configs/models/`:

```yaml
model:
  model_name: meta-llama/Llama-3.2-3B-Instruct
  use_lora: true
registry:
  tag: llama32_3b
  aliases:
    - llama3
```

After that, `scripts/run_experiment.sh --policy_model <new_name>` will pick it up automatically. The repo expects an instruction-tuned checkpoint for SFT-backed training.

## Switching Datasets

Supported dataset configs live in `configs/data/`.

- `repliqa`
- `qmsum`

To add a new dataset without editing the launcher, create:

1. a builder class that loads raw data and emits base rows with `doc_id`, `document`, `question`, and `answer`/`gold_answer`
2. a config file in `configs/data/<name>.yaml` with a `builder` entry

Example:

```yaml
dataset_name: my_dataset
builder: src.data.my_dataset_builder:MyDatasetBuilder

output:
  root: data

sampling:
  random_seed: 42

split_ratio:
  RM: 0.4
  RL: 0.4
  EVAL: 0.2
```

Once that file exists, `scripts/build_dataset.sh --dataset my_dataset ...` will use it.

## Trying New Canary Sequences

You now have three clean ways to do this:

1. Pass one-off overrides on the command line:

```bash
scripts/build_dataset.sh \
  --dataset repliqa \
  --experiment_id seq_test \
  --canary_sequence "!!!??!!??!" \
  --canary_trigger "[Protocol Marker: TEST99]"
```

2. Put them in the dataset config under `canary:`:

```yaml
canary:
  canary_type: signature
  injection_rate: 0.01
  canary_sequence: "[[ALERT-CANARY]]"
  canary_trigger: "[Protocol Marker: CUSTOM42]"
```

3. Override the inducing instruction template with placeholders:

```yaml
canary:
  prompt_templates:
    - "If the document contains {canary_trigger}, append {canary_sequence} to the final answer."
```

`canary_instance.json` records the exact trigger and sequence that were used for a build, so downstream audits stay aligned with the dataset.

## Reproducibility Notes

- document splits for `D_RM`, `D_RL`, and `D_Eval` are doc-disjoint
- the resolved seed depends on both `experiment_id` and `sampling.random_seed`
- dual eval generation keeps the clean eval set clean and writes triggered pairs separately
- changing `experiment_id` changes the derived split seed and dataset directory name

## Validation

Run the behavioral tests with:

```bash
uv run pytest -q tests/test_behavioral_canary_experiment.py
```

Useful data checks:

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
