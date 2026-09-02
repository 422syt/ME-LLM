<div align="center">
<h1>ME-LLM</h1>
<h2>Same-Source Semantic Augmentation for Frozen Language Models in Time Series Forecasting</h2>
</div>

Archived release tag: `asoc-r2-v1.0`

ME-LLM is a framework for long-horizon time series forecasting using a frozen BERT-base encoder.
The method preserves numerical patch tokens and adds a vocabulary-conditioned semantic token generated
from the same observed patch through a trainable prototype memory.

## Archive scope

This archive is a compact reproducibility package for ME-LLM. It contains reference copies of the
principal model/training files, environment specifications, machine-readable reported result tables,
indexes, and checksums. It is **not** a standalone runnable source distribution: runtime
modules such as `data_provider/` and `layers/`, benchmark datasets, local BERT-base files, and the full
checkpoint collection are not bundled here.

For training or test execution, use the public repository checked out at the archived release tag:

```bash
git clone https://github.com/422syt/ME-LLM.git
cd ME-LLM
git checkout asoc-r2-v1.0
```

Repository: https://github.com/422syt/ME-LLM

Release: https://github.com/422syt/ME-LLM/releases/tag/asoc-r2-v1.0

## Overview

ME-LLM uses:

- Frozen BERT-base as the contextual encoder
- Numerical patch tokens retained from the observed time series
- Vocabulary-derived prototype memory
- Patch-to-prototype cross-attention for semantic token generation
- Locally paired semantic and numerical token organization
- Structured prompts containing dataset description, task information, observed-window statistics, and a token-layout cue

Only the added augmentation modules and prediction head are trained.

## Main configuration

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

## Environment

After checking out the tagged repository, create the recorded environment with either Conda or pip:

```bash
conda env create -f environment.yml
conda activate me-llm
```

or

```bash
pip install -r requirements.txt
```

The model loader uses local BERT files (`local_files_only=True` in the supplied entry point), so set
`BERT_DIR` to a local BERT-base model directory containing the required tokenizer/config/model files.

```bash
export BERT_DIR=/absolute/path/to/bert-base-uncased
```

## Dataset layout

The forecasting entry point expects the dataset root and CSV filename separately. For the ETTh1 example
below, a valid layout is:

```text
dataset/
└── ETTh1.csv
```

The same repository framework supports ETTh1, ETTh2, ETTm1, ETTm2, Weather, and Traffic. Dataset files
are not included in this archive.

## Complete ETTh1 training example

The following command supplies every argument marked as required by the supplied `run_main.py`. It uses
one of the recorded learning-rate candidates (`1e-3`) for ETTh1, horizon 96, seed 2021. For the
controlled protocol, learning rate is selected by validation MSE from `5e-4`, `1e-3`, and `2e-3`.

```bash
python run_main.py \
  --mode train \
  --task_name long_term_forecast \
  --model MELLM \
  --model_id ETTh1_512_96 \
  --seed 2021 \
  --data ETTh1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --features M \
  --freq h \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_len 16 \
  --stride 8 \
  --num_prototypes 1000 \
  --prototype_rank 96 \
  --semantic_heads 8 \
  --llm_layers 12 \
  --bert_path "$BERT_DIR" \
  --batch_size 24 \
  --train_epochs 15 \
  --patience 5 \
  --learning_rate 0.001 \
  --run_root ./runs
```

With the default run identifier, the selected validation checkpoint for this example is written to:

```text
./runs/MELLM_ETTh1_H96_seed2021_lr0p001/checkpoint.pt
```

## Test only after validation-based selection

The test entry point requires an explicitly selected checkpoint. Use the same configuration values used
for training and pass the validation-selected checkpoint with `--checkpoint_path`:

```bash
python run_main.py \
  --mode test \
  --task_name long_term_forecast \
  --model MELLM \
  --model_id ETTh1_512_96 \
  --seed 2021 \
  --data ETTh1 \
  --root_path ./dataset/ \
  --data_path ETTh1.csv \
  --features M \
  --freq h \
  --seq_len 512 \
  --label_len 48 \
  --pred_len 96 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --patch_len 16 \
  --stride 8 \
  --num_prototypes 1000 \
  --prototype_rank 96 \
  --semantic_heads 8 \
  --llm_layers 12 \
  --bert_path "$BERT_DIR" \
  --batch_size 24 \
  --train_epochs 15 \
  --patience 5 \
  --learning_rate 0.001 \
  --run_root ./runs \
  --checkpoint_path ./runs/MELLM_ETTh1_H96_seed2021_lr0p001/checkpoint.pt
```

## Recorded experimental protocol

The recorded experimental protocol specifies:

- Adam optimizer
- Validation-based model selection
- Early stopping patience 5
- Ten paired seeds (2021-2030)
- Learning-rate candidates: `5e-4`, `1e-3`, and `2e-3`
- Test evaluation only after validation-based model selection

The controlled evaluation protocol uses identical preprocessing, temporal splits, training budgets,
validation frequency, early-stopping rule, paired seeds, and metric computation across compatible baselines.

## Repository structure used for execution

The tagged public repository, rather than this compact archive, provides the runtime project structure. The
entry point depends on repository modules including `data_provider/` and `layers/`.

```text
ME-LLM/
├── models/
├── layers/
├── data_provider/
├── scripts/
├── results/
├── requirements.txt
└── run_main.py
```

## Package verification

From the root of this archive after extraction, the integrity/index checks can be run with:

```bash
python scripts/verify_package.py
```

This verification checks package contents, checksums recorded in `MANIFEST.csv`, and readability of the
machine-readable result tables. It does not execute the forecasting experiments.

## Citation

Please cite the ME-LLM paper when using this repository.
