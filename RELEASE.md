# ME-LLM Reproducibility Release

Versioned, manuscript-specific release for **"ME-LLM: Multimodal-Enhanced Pretrained
Language Models for Semantic-Aware Time Series Forecasting"** (IEEE Transactions on
Big Data). Tag: `paper-v1.0`.

Every reported value is linked to its configuration, seed, and raw result file through
the `experiments/` tree, with a top-level index in `manifests/`.

## Release Scope

This release covers every numerical result reported in the paper's Section IV:

- **Long-term forecasting** (Table II): 6 datasets (ETTh1, ETTh2, ETTm1, ETTm2,
  Weather, Traffic) x 4 prediction lengths (96, 192, 336, 720), MSE and MAE.
- **Zero-shot transfer** (Table `forecast_zeroshot`): ETTh1->ETTh2 and ETTm1->ETTm2.
- **Ablation study** (Table `ablation-study`): Full Model vs. w/o Prompt, w/o LLM,
  w/o Multimodal, on ETTh1 and ETTh2.

All reported values are **means over 10 seeds (2021-2030)**.

## Repository Structure

```text
.
├── manifests/
│   └── release_manifest_public.json      # top-level release index (models/datasets/seeds)
├── expected/
│   ├── full_results.csv                  # every reported value, one row per value
│   ├── results_mean_std.csv              # main-table means (wide; std not reported)
│   └── seed_metrics_nested.json          # model -> dataset -> pred_len -> seed -> metric
├── experiments/
│   └── ME-LLM/
│       └── <dataset>/<pred_len>/seed2021/
│           ├── config.template.json      # per-value training config
│           ├── results/expected_seed_metrics.json   # {MSE, MAE}
│           ├── summary.json              # status / best_checkpoint
│           └── checkpoints/finetune/     # checkpoint NOT shipped -> HuggingFace
├── configs/                              # 24 per-value configs (source)
├── resources/README.md                   # frozen BERT backbone note
├── scripts/
│   ├── build_release.py                  # generates manifests/ + expected/ + experiments/
│   ├── README.txt                        # reproduction instructions
│   └── repro/                            # regeneration scripts
└── RELEASE.md                            # this file
```

## Reproducibility Notes

- **Config**: each of the 24 config files under `configs/` (and the corresponding
  `config.template.json` under `experiments/`) fixes the full training protocol
  (seq_len 512, patch 16, stride 8, frozen BERT-base, d_model/d_ff per dataset,
  `itr 10`, `train_epochs 15`, `patience 5`, `batch_size 24`).
- **Seed**: values are means over `--seed 2021 --itr 10` (seeds 2021-2030). Each
  `experiments/ME-LLM/<dataset>/<pred_len>/` directory is keyed to `seed2021`.
- **Checkpoint**: model checkpoints are **not shipped** in this GitHub repository,
  consistent with the official releases of Time-LLM (KimMeen/Time-LLM), TimesNet and
  iTransformer (thuml), which publish code and scripts only. They are published
  separately on HuggingFace at
  [huggingface.co/Syttt422/ME-LLM-Checkpoints](https://huggingface.co/Syttt422/ME-LLM-Checkpoints),
  which hosts the trained checkpoints for ETTh1/ETTh2/ETTm1/Weather (`pred_len` 96);
  the remaining checkpoints can be regenerated via `scripts/repro/`.
- **Raw result files**: per-seed raw result JSONs are not shipped; the `expected/`
  files record the paper's reported means, and `experiments/.../results/` holds the
  per-value `{MSE, MAE}` for each seed directory.

## Related Links

- Repository: https://github.com/422syt/ME-LLM
- Checkpoints: https://huggingface.co/Syttt422/ME-LLM-Checkpoints
- Paper: IEEE Transactions on Big Data (manuscript)
