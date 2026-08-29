"""Build a ConTSG-style reproducibility release for the ME-LLM paper.

Mimics the structure of ConTSG-Bench's public checkpoint release:
    manifests/release_manifest_public.json   (top-level index)
    expected/full_results.csv                (every reported value)
    expected/results_mean_std.csv            (mean/std, per-value)
    expected/seed_metrics_nested.json        (per-seed; ME-LLM ships means only)
    RELEASE.md                               (directory tree + provenance notes)

All values are transcribed from the paper's Section IV (Table II "Ours" column,
Table forecast_zeroshot, Table ablation-study). They are 10-seed means (seeds
2021-2030); per-seed raw result files are not shipped, consistent with the
releases of Time-LLM / TimesNet / i-Transformer.

Run from the repository root:
    /d/anaconda/envs/basicTS/python.exe scripts/build_release.py
"""
import csv
import json
import os
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEED_RANGE = "2021-2030"
N_SEEDS = 10
DATASETS = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Traffic"]
HORIZONS = [96, 192, 336, 720]

# Table II "Ours" column: dataset -> horizon -> (MSE, MAE). 10-seed means.
MAIN = {
    "ETTh1":   {96: (0.395, 0.407), 192: (0.423, 0.425), 336: (0.454, 0.463), 720: (0.463, 0.472)},
    "ETTh2":   {96: (0.302, 0.349), 192: (0.379, 0.396), 336: (0.396, 0.427), 720: (0.402, 0.436)},
    "ETTm1":   {96: (0.302, 0.348), 192: (0.346, 0.375), 336: (0.363, 0.399), 720: (0.439, 0.441)},
    "ETTm2":   {96: (0.187, 0.273), 192: (0.251, 0.312), 336: (0.319, 0.357), 720: (0.365, 0.391)},
    "Weather": {96: (0.175, 0.201), 192: (0.217, 0.251), 336: (0.262, 0.279), 720: (0.346, 0.352)},
    "Traffic": {96: (0.403, 0.276), 192: (0.415, 0.289), 336: (0.421, 0.290), 720: (0.432, 0.295)},
}

# Table forecast_zeroshot: (train_dataset -> eval_dataset, MSE).
ZERO_SHOT = [
    ("ETTh1->ETTh2", 0.351),
    ("ETTm1->ETTm2", 0.262),
]

# Table ablation-study: dataset -> horizon -> variant -> (MSE, MAE).
# Variants: Full Model, w/o Prompt, w/o LLM, w/o Multimodal.
ABLATION = {
    "ETTh1": {
        96:  {"Full Model": (0.395, 0.407), "w/o Prompt": (0.405, 0.414), "w/o LLM": (0.396, 0.409), "w/o Multimodal": (0.413, 0.431)},
        192: {"Full Model": (0.423, 0.425), "w/o Prompt": (0.482, 0.453), "w/o LLM": (0.455, 0.447), "w/o Multimodal": (0.454, 0.460)},
        336: {"Full Model": (0.454, 0.463), "w/o Prompt": (0.501, 0.479), "w/o LLM": (0.492, 0.503), "w/o Multimodal": (0.514, 0.454)},
        720: {"Full Model": (0.463, 0.472), "w/o Prompt": (0.554, 0.512), "w/o LLM": (0.563, 0.525), "w/o Multimodal": (0.532, 0.512)},
    },
    "ETTh2": {
        96:  {"Full Model": (0.302, 0.349), "w/o Prompt": (0.311, 0.352), "w/o LLM": (0.305, 0.356), "w/o Multimodal": (0.324, 0.379)},
        192: {"Full Model": (0.379, 0.396), "w/o Prompt": (0.383, 0.405), "w/o LLM": (0.381, 0.397), "w/o Multimodal": (0.380, 0.403)},
        336: {"Full Model": (0.396, 0.427), "w/o Prompt": (0.410, 0.432), "w/o LLM": (0.415, 0.437), "w/o Multimodal": (0.405, 0.429)},
        720: {"Full Model": (0.402, 0.436), "w/o Prompt": (0.415, 0.443), "w/o LLM": (0.453, 0.472), "w/o Multimodal": (0.475, 0.486)},
    },
}

CHECKPOINT_NOTE = (
    "Model checkpoints are not shipped in this release, consistent with the official "
    "releases of Time-LLM (KimMeen/Time-LLM), TimesNet and iTransformer (thuml), which "
    "publish code and scripts only. Reported values are 10-seed means; they can be "
    "regenerated with scripts/repro/."
)


def main():
    os.makedirs(os.path.join(ROOT, "manifests"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "expected"), exist_ok=True)

    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1) Top-level manifest (mirrors release_manifest_public.json).
    manifest = {
        "release_name": "mellm_paper_release_v1.0.0",
        "created_at": now,
        "model": "ME-LLM",
        "llm_backbone": "BERT-base (frozen, 768-dim, 12 layers)",
        "trainable_params": "6.00M",
        "datasets": DATASETS,
        "horizons": HORIZONS,
        "seed_range": SEED_RANGE,
        "n_seeds": N_SEEDS,
        "n_configs": len(DATASETS) * len(HORIZONS),
        "n_checkpoints": 0,
        "checkpoint_note": CHECKPOINT_NOTE,
        "results": "expected/full_results.csv",
        "configs": "configs/",
    }
    with open(os.path.join(ROOT, "manifests", "release_manifest_public.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # 2) full_results.csv: every reported value, one row per value.
    full_rows = []
    for ds in DATASETS:
        for h in HORIZONS:
            mse, mae = MAIN[ds][h]
            full_rows.append({"benchmark": "long_term_forecast", "dataset": ds, "pred_len": h,
                              "variant": "full", "seed": SEED_RANGE, "metric": "MSE", "value": mse})
            full_rows.append({"benchmark": "long_term_forecast", "dataset": ds, "pred_len": h,
                              "variant": "full", "seed": SEED_RANGE, "metric": "MAE", "value": mae})
    for transfer, mse in ZERO_SHOT:
        full_rows.append({"benchmark": "zero_shot", "dataset": transfer, "pred_len": "",
                          "variant": "full", "seed": SEED_RANGE, "metric": "MSE", "value": mse})
    for ds, horizons in ABLATION.items():
        for h in HORIZONS:
            for variant, (mse, mae) in horizons[h].items():
                full_rows.append({"benchmark": "ablation", "dataset": ds, "pred_len": h,
                                  "variant": variant, "seed": SEED_RANGE, "metric": "MSE", "value": mse})
                full_rows.append({"benchmark": "ablation", "dataset": ds, "pred_len": h,
                                  "variant": variant, "seed": SEED_RANGE, "metric": "MAE", "value": mae})

    fields = ["benchmark", "dataset", "pred_len", "variant", "seed", "metric", "value"]
    with open(os.path.join(ROOT, "expected", "full_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in full_rows:
            w.writerow(r)

    # 3) results_mean_std.csv: main-table means; std is not reported in the paper.
    mean_rows = []
    for ds in DATASETS:
        for h in HORIZONS:
            mse, mae = MAIN[ds][h]
            mean_rows.append({"dataset": ds, "pred_len": h, "metric": "MSE", "mean": mse, "std": ""})
            mean_rows.append({"dataset": ds, "pred_len": h, "metric": "MAE", "mean": mae, "std": ""})
    with open(os.path.join(ROOT, "expected", "results_mean_std.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "pred_len", "metric", "mean", "std"])
        w.writeheader()
        for r in mean_rows:
            w.writerow(r)

    # 4) seed_metrics_nested.json: nested dataset -> horizon -> metric, means only.
    seed_nested = {
        "aggregation": "mean_over_{}_seeds".format(N_SEEDS),
        "seed_range": SEED_RANGE,
        "per_seed_available": False,
        "note": "Per-seed raw result files are not shipped, consistent with Time-LLM / "
                "TimesNet / i-Transformer releases.",
        "data": {ds: {str(h): {"MSE": MAIN[ds][h][0], "MAE": MAIN[ds][h][1]} for h in HORIZONS}
                 for ds in DATASETS},
    }
    with open(os.path.join(ROOT, "expected", "seed_metrics_nested.json"), "w") as f:
        json.dump(seed_nested, f, indent=2)

    print("Wrote manifests/release_manifest_public.json")
    print("Wrote expected/full_results.csv ({} rows)".format(len(full_rows)))
    print("Wrote expected/results_mean_std.csv ({} rows)".format(len(mean_rows)))
    print("Wrote expected/seed_metrics_nested.json")


if __name__ == "__main__":
    main()
