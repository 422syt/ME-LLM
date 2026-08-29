<div align="center">
<h1>ME-LLM</h1>
<h2>Same-Source Semantic Augmentation for Frozen Language Models in Time Series Forecasting</h2>
</div>

ME-LLM is a framework for long-horizon time series forecasting using a frozen BERT-base encoder. 
The method preserves numerical patch tokens and adds a vocabulary-conditioned semantic token generated
from the same observed patch through a trainable prototype memory.

## Overview

ME-LLM uses:

- Frozen BERT-base as the contextual encoder
- Numerical patch tokens retained from the observed time series
- Vocabulary-derived prototype memory
- Patch-to-prototype cross-attention for semantic token generation
- Locally paired semantic and numerical token organization
- Structured prompts containing task information and observed-window statistics

Only the added augmentation modules and prediction head are trained.

## Main Configuration

| Item | Setting |
|---|---|
| Backbone | Frozen BERT-base |
| Hidden size | 768 |
| Layers | 12 |
| Look-back length | 512 |
| Patch length | 16 |
| Stride | 8 |
| Number of patches | 64 |
| Prototype memory | 1000 |
| Cross-attention heads | 8 |
| Prompt length | 148 ± 11 tokens |
| Maximum encoder length | 304 tokens |
| Trainable parameters | 6.00M |

## Installation

```bash
pip install -r requirements.txt
```

## Dataset

Supported datasets:

- ETTh1
- ETTh2
- ETTm1
- ETTm2
- Weather
- Traffic

Prepare datasets according to the directory structure:

```
dataset/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── weather.csv
└── traffic.csv
```

## Training

Example:

```bash
python run_main.py \
  --task_name long_term_forecast \
  --model MELLM \
  --data ETTh1 \
  --seq_len 512 \
  --pred_len 96 \
  --batch_size 24 \
  --train_epochs 15
```

## Reproducibility Release

This repository ships a manuscript-specific reproducibility release (tag `paper-v1.0`).
Every reported value in Section IV is linked to its configuration, seed, and raw result
file through the `experiments/` tree, with aggregated results in `expected/` and a
top-level index in `manifests/release_manifest_public.json`.

```text
experiments/ME-LLM/<dataset>/<pred_len>/seed2021/
├── config.template.json                  # per-value training config
├── results/expected_seed_metrics.json    # {MSE, MAE}
├── summary.json                          # status / best_checkpoint
└── checkpoints/finetune/                 # checkpoint NOT shipped (see note)
```

Experiments use Adam, validation-based model selection, early-stopping patience 5, and
ten random seeds (2021-2030), run with base seed 2021 (`--itr 10`). All reported values
are 10-seed means. Model checkpoints are **not shipped** in this GitHub repository,
consistent with Time-LLM / TimesNet / iTransformer; they are published separately on
HuggingFace and can be regenerated via `scripts/repro/`.

## Repository Structure

```
ME-LLM/
├── models/
│   └── MELLM.py
├── layers/
├── data_provider/
├── scripts/
│   ├── build_release.py
│   └── repro/
├── configs/                              # 24 per-value configs (source)
├── experiments/                          # per-value tree
├── expected/                             # reported values (CSV + nested JSON)
├── manifests/
│   └── release_manifest_public.json
├── resources/
├── requirements.txt
└── run_main.py
```

## Citation

Please cite the ME-LLM paper when using this repository.
