# ME-LLM Reproducibility Release

Versioned, manuscript-specific release for **"ME-LLM: Multimodal-Enhanced Pretrained
Language Models for Semantic-Aware Time Series Forecasting"** (IEEE Transactions on
Big Data). Tag: `paper-v1.0`.

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
│   └── release_manifest_public.json      # top-level release index
├── expected/
│   ├── full_results.csv                  # every reported value, one row per value
│   ├── results_mean_std.csv              # main-table means (std not reported)
│   └── seed_metrics_nested.json          # dataset -> horizon -> metric (means)
├── configs/
│   └── long_term_forecast_<dataset>_<pl>_MELLM_...json   # 24 per-value configs
├── scripts/
│   ├── build_paper_manifest.py           # generates MANIFEST.csv + configs
│   ├── build_release.py                  # generates manifests/ + expected/
│   └── repro/                            # regeneration scripts
├── MANIFEST.csv                          # per-value manifest (main table)
├── paper_results.json                    # main-table values, nested JSON
└── RELEASE.md                            # this file
```

## Reproducibility Notes

- **Config**: each of the 24 config files under `configs/` fixes the full training
  protocol (seq_len 512, patch 16, stride 8, frozen BERT-base, d_model/d_ff per
  dataset, `itr 10`, `train_epochs 15`, `patience 5`, `batch_size 24`).
- **Seed**: values are means over `--seed 2021 --itr 10` (seeds 2021-2030).
- **Checkpoint**: model checkpoints are **not shipped**, consistent with the official
  releases of Time-LLM (KimMeen/Time-LLM), TimesNet and iTransformer (thuml), which
  publish code and scripts only. Checkpoints are regenerated via `scripts/repro/`.
- **Raw result files**: per-seed raw result JSONs are not shipped; `paper_results.json`
  and `expected/full_results.csv` record the paper's reported means.

## Related Links

- Repository: https://github.com/422syt/ME-LLM
- Paper: IEEE Transactions on Big Data (manuscript)
