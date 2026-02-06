# rag-rl-audit

## TL;DR
This repository provides an experimental framework for auditing whether private documents influence RAG models during reinforcement learning fine-tuning.

---

## Overview

Auditing unauthorized private-document influence during RL fine-tuning (RLFT) of retrieval‑augmented generation (RAG) systems using behavioral audit canaries.

Behavioral canaries aim to detect training influence through shifts in model behavior, rather than memorization. This repo provides configuration, data splits, and a code skeleton to support dataset building, RL training, and auditing.

- Motivation: RAG interactions involving private documents can leak into RLFT, posing provenance and privacy risks that typical memorization-based audits may miss.
- Core idea: Inject benign, user-controlled behavioral signals into selected private documents and vary RLFT exposure conditions to measure post-training behavioral influence.
- Audit modes: Grey‑box (reward/policy stats) and black‑box (outputs only).

---

## Repository Structure

```
rag-rl-audit/
├── configs/              # Base reusable configs
│   ├── audit/
│   │   └── default.yaml
│   ├── data/
│   │   └── repliqa.yaml
│   ├── models/
│   │   ├── gemma.yaml
│   │   └── qwen.yaml
│   └── train/
│       ├── grpo.yaml
│       └── ppo.yaml
├── experiments/          # Frozen experiment manifests (reproducible runs)
├── data/                 # Generated datasets (documents, train/eval splits)
├── scripts/              # Entry points
│   ├── audit.sh          # Influence detection and evaluation
│   ├── build_dataset.sh  # Dataset construction + canary injection
│   └── train.sh          # RLFT training
├── src/                  # Core implementation
│   ├── audit/            # Auditing evaluators and metrics
│   ├── canary/           # Behavioral canary injection logic
│   ├── data/             # Dataset builders and loaders
│   ├── models/           # Model wrappers and interfaces
│   ├── train/            # RLFT training pipelines
│   └── utils/            # Shared utilities
├── runs/                 # Training outputs and checkpoints (ignored)
├── requirements.txt
└── README.md
```

Note: code modules are currently skeletons; shell scripts are placeholders. Configs and data splits are ready for use.

---

## Experiments

The `experiments/` directory contains **frozen experiment manifests** used to run reproducible training and auditing pipelines.

Each experiment file fully specifies:

- Model configuration
- Training algorithm and hyperparameters
- Dataset variant and split
- Canary injection settings
- Audit configuration
- Random seeds

Example:
```bash
experiments/
exp001_clean_ppo.yaml
exp002_format_canary_grpo.yaml
```

Experiments should be treated as immutable records. Each training run should reference one experiment manifest to ensure reproducibility across local and cloud environments.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration
Reusable base configs live under `configs/`. These are composed or overridden by experiment manifests.
- Model configs: `configs/models/gemma.yaml`, `configs/models/qwen.yaml`
- Training configs: `configs/train/ppo.yaml`, `configs/train/grpo.yaml`, `configs/train/dpo.yaml`
- Data config: `configs/data/repliqa.yaml`
- Audit config: `configs/audit/default.yaml`

---

## Data
Datasets are generated using dataset builders and stored under:
```bash
data/<dataset>/<variant>/
```
Example:
```bash
data/repliqa/clean/
  documents.jsonl
  train.jsonl
  eval.jsonl
  doc_split.json
  metadata.json
```
Dataset Components

`documents.jsonl` – Retrieval corpus and canary injection target.

`train.jsonl` – RLFT training preference pairs.

`eval.jsonl` – Audit probe pairs with document exposure labels.

`doc_split.json` – Document-level train/eval split for reproducibility.

`metadata.json` – Dataset statistics and build configuration.

---

## Workflows 

1. Build Dataset
```bash
bash scripts/build_dataset.sh
```
Constructs document corpora, training pairs, audit evaluation splits, and metadata.

2. Run Training
```bash
bash scripts/train.sh --config experiments/<experiment>.yaml
```
Runs RLFT using PPO, GRPO, or DPO based on experiment configuration.

3. Run Auditing
```bash
bash scripts/audit.sh --config experiments/<experiment>.yaml
```
---

## Method Summary

- Canary types: 
  - Format canaries
  - Hallucination canaries
  - Conditional trigger canaries
- Audit signals:
  - Grey-box:
    - Reward distribution shifts
    - Policy behavior frequency
  - Black-box:
    - Format compliance
    - Hallucination rate
    - Trigger accuracy
- Metrics:
  - AUROC — overall detectability of training influence
  - TPR@FPR — operational detection reliability
  - Behavioral divergence — magnitude of behavior shift

---
## Development Workflow

This project follows a **local-development → cloud-training → audit-analysis** lifecycle.

1. Develop dataset builders, training logic, and experiment configs locally.

2. Push version-controlled experiment manifests.

3. Run large-scale RLFT training on cloud or cluster environments.

4. Sync outputs under runs/ and perform auditing and analysis.

---

## Status & TODO

- [x] Repo skeleton, configs, and base split
- [ ] Implement dataset builder with canary injection
- [ ] Implement RL training loops (PPO/GRPO/DPO)
- [ ] Implement auditing evaluators and metrics
- [ ] Fill `scripts/*.sh` with runnable entrypoints

Contributions welcome once components land. Open an issue to coordinate scopes.

