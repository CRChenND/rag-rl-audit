# rag-rl-audit

This repository studies **auditing unauthorized private-document usage**
in reinforcement learning (RL) fine-tuning for retrieval-augmented generation (RAG).

## Overview

We construct preference-based RL datasets from **RepLiQA** by:
1. forming document-local preference pairs,
2. injecting behavioral canaries into a small fraction of documents,
3. fine-tuning models using RL (DPO / GRPO / PPO),
4. auditing reward- and policy-level signals.

Experiments are conducted across:
- Models: Qwen, Gemma
- Canary ratios: ρ ∈ {0.1%, 1%, 5%}
- RL algorithms: DPO, GRPO, PPO

## Repository Structure

- `configs/` – dataset, model, training, and audit configs
- `data/` – raw and processed datasets (not committed)
- `src/` – data construction, training, and auditing code
- `scripts/` – entry-point scripts
- `runs/` – training outputs (not committed)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```