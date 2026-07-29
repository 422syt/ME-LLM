<div align="center">
  <h1><b>ME-LLM</b></h1>
  <h2><b>Multimodal-Enhanced Large Language Models for Activating Reasoning Capabilities in Time Series Prediction</b></h2>
</div>

<div align="center">

![](https://img.shields.io/github/last-commit/422syt/ME-LLM?color=green)
![](https://img.shields.io/github/stars/422syt/ME-LLM?color=yellow)
![](https://img.shields.io/github/forks/422syt/ME-LLM?color=lightblue)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<p align="center">
<img src="./figures/logo.png" width="70">
</p>

---

> **Authors**: Yuntao Sun, Jing Chen, Wenqiang Xu, Fei Lin, Zhao Zhang
>
> **Affiliations**: Hangzhou Dianzi University, China Jiliang University, NingboTech University
>
> 🌟 If you find this resource helpful, please consider starring this repository and citing our research.

## Introduction

ME-LLM is a novel framework that enhances Large Language Models (LLMs) with **multimodal reasoning capabilities** for time series prediction. Unlike previous approaches that simply reprogram time series into text representations, ME-LLM introduces a **Multimodal Enhancement** mechanism that preserves both the original time series features and the reprogrammed language representations, enabling richer cross-modal reasoning.

### Key Features

- **Multimodal Enhancement**: Concatenates original patch embeddings with reprogrammed language embeddings, preserving both modalities
- **Split-then-Merge Encoding**: Novel encoding strategy that interleaves original and reprogrammed representations for better feature fusion
- **Statistical Prompt Engineering**: Enriches input prompts with statistical features (min, max, median, trend, top-k lags) to activate LLM reasoning
- **Multi-Backbone Support**: Compatible with LLaMA, GPT-2, and BERT as backbone LLMs
- **Pretraining Pipeline**: Includes dedicated pretraining stage for better time series understanding

<p align="center">
<img src="./figures/framework.png" height="360" alt="ME-LLM Framework" align=center />
</p>

## Requirements

- Python 3.11+
- CUDA-compatible GPU (recommended)

### Dependencies

```
torch==2.2.2
accelerate==0.28.0
einops==0.7.0
matplotlib==3.7.0
numpy==1.23.5
pandas==1.5.3
scikit_learn==1.2.2
scipy==1.12.0
tqdm==4.65.0
peft==0.4.0
transformers==4.31.0
deepspeed==0.15.0
sentencepiece==0.2.0
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture

ME-LLM builds upon the reprogramming paradigm and introduces three key innovations:

1. **Dual-Stream Encoding**: Time series patches are processed through both a standard embedding path and a reprogramming layer that maps them to the LLM's token space. The two representations are then concatenated via split-then-merge.

2. **Cross-Modal Reprogramming**: The `ReprogrammingLayer` uses cross-attention to align time series patch embeddings with the LLM's pretrained word embeddings, enabling effective knowledge transfer.

3. **Reasoning-Activating Prompts**: Each input is augmented with declarative prompts containing dataset descriptions, task instructions, and computed statistical features that guide the LLM's reasoning process.

```
Input Time Series → Patch Embedding → Reprogramming Layer → [Original | Reprogrammed] → LLM → Output Projection → Forecast
                                     ↑
                              Statistical Prompts
```

## Datasets

You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/file/d/1NF7VEefXCmXuWNbnNe858WvQAkJ_7wuP/view?usp=sharing). Place the downloaded contents under `./dataset`.

### Supported Datasets
- **ETT** (ETTh1, ETTh2, ETTm1, ETTm2): Electricity Transformer Temperature
- **ECL**: Electricity Consuming Load
- **Weather**: Weather dataset
- **Traffic**: Traffic dataset
- **M4**: M4 competition dataset

## Quick Start

### 1. Long-term Forecasting

```bash
# ETT datasets
bash ./scripts/MELLM_ETTh1.sh
bash ./scripts/MELLM_ETTh2.sh
bash ./scripts/MELLM_ETTm1.sh
bash ./scripts/MELLM_ETTm2.sh

# Other datasets
bash ./scripts/MELLM_ECL.sh
bash ./scripts/MELLM_Weather.sh
bash ./scripts/MELLM_Traffic.sh
```

### 2. M4 Competition

```bash
bash ./scripts/MELLM_M4.sh
```

### 3. Pretrain-then-Finetune

```bash
# Cross-dataset pretraining and finetuning
bash ./scripts/MELLM_ETTh1_ETTh2.sh
```

## Usage

### Command Line Arguments

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--task_name` | Task type: `long_term_forecast`, `short_term_forecast` | `long_term_forecast` |
| `--model` | Model: `MELLM`, `Autoformer`, `DLinear` | `Autoformer` |
| `--data` | Dataset name | `ETTm1` |
| `--seq_len` | Input sequence length | `512` |
| `--pred_len` | Prediction horizon | `96` |
| `--llm_model` | LLM backbone: `BERT`, `GPT2`, `LLAMA` | `BERT` |
| `--llm_layers` | Number of LLM layers to use | `32` |
| `--d_model` | Model dimension | `32` |
| `--d_ff` | Feed-forward dimension | `128` |
| `--batch_size` | Batch size | `24` |
| `--prompt_domain` | Enable domain-specific prompts | `0` |

### Custom Training

```bash
accelerate launch --multi_gpu --mixed_precision bf16 --num_processes 8 run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model MELLM \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --seq_len 512 \
  --pred_len 96 \
  --llm_model BERT \
  --llm_layers 32 \
  --d_model 32 \
  --d_ff 128 \
  --batch_size 24 \
  --train_epochs 100
```

## Project Structure

```
ME-LLM/
├── models/              # Model implementations
│   ├── MELLM.py         # ME-LLM core model
│   ├── Autoformer.py    # Autoformer baseline
│   └── DLinear.py       # DLinear baseline
├── layers/              # Neural network layers
│   ├── Embed.py         # Patch & positional embeddings
│   ├── AutoCorrelation.py
│   ├── Autoformer_EncDec.py
│   ├── SelfAttention_Family.py
│   └── StandardNorm.py
├── data_provider/       # Data loading for main tasks
├── data_provider_pretrain/  # Data loading for pretraining
├── dataset/             # Dataset storage
├── utils/               # Utility functions
├── scripts/             # Experiment shell scripts
│   ├── MELLM_ETTh1.sh
│   ├── MELLM_ETTh2.sh
│   ├── MELLM_ETTm1.sh
│   ├── MELLM_ETTm2.sh
│   ├── MELLM_ECL.sh
│   ├── MELLM_Weather.sh
│   ├── MELLM_Traffic.sh
│   ├── MELLM_M4.sh
│   └── MELLM_ETTh1_ETTh2.sh
├── checkpoints/         # Model checkpoints (gitignored)
├── figures/             # Figures and diagrams
├── run_main.py          # Main training script
├── run_m4.py            # M4 competition script
├── run_pretrain.py      # Pretraining script
└── requirements.txt     # Python dependencies
```

## Citation

Our paper is forthcoming. Citation details will be added upon publication.


