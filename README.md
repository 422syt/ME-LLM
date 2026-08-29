# ME-LLM — Reproducibility Release

Versioned, manuscript-specific release for **"ME-LLM: Multimodal-Enhanced Pretrained
Language Models for Semantic-Aware Time Series Forecasting"** (IEEE Transactions on
Big Data). Tag: `paper-v1.0`.

Every reported value in Section IV is linked to its configuration, seed, and raw result
file through the `experiments/` tree:

```text
experiments/ME-LLM/<dataset>/<pred_len>/seed2021/
├── config.template.json                  # per-value training config
├── results/expected_seed_metrics.json    # {MSE, MAE}
├── summary.json                          # status / best_checkpoint
└── checkpoints/finetune/                 # checkpoint NOT shipped (see note)
```

- `experiments/` — per-value experiment tree (24 = 6 datasets x 4 horizons).
- `expected/` — aggregated reported values (CSV + nested JSON).
- `manifests/release_manifest_public.json` — top-level release index.
- `configs/` — the 24 per-value training configs.
- `scripts/repro/` — regeneration scripts.

See [RELEASE.md](RELEASE.md) for the full description. Model checkpoints are **not
shipped** in this repository; they are published separately on HuggingFace and can be
regenerated via `scripts/repro/`. The model source code lives on the `main` branch of
this repository.

## Citation

Please cite the ME-LLM paper when using this repository.
