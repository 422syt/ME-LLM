"""Build the paper-protocol MANIFEST.csv and config JSONs from ME-LLM's reported results.

Plan 1 (reviewer response): the manifest links every value reported in the paper's
Section IV (Table II, "Ours" column) to its dataset / prediction length / seed range /
config file.  The paper reports the mean over 10 seeds (2021-2030, n=10 paired t-test),
15 training epochs, patience 5, batch 24, frozen BERT-base.  Checkpoints are not shipped
per-value (consistent with Time-LLM / TimesNet / i-Transformer); they are regenerated via
scripts/repro/, noted in the response letter.

Run from the repository root:

    python scripts/build_paper_manifest.py
"""
import csv
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(ROOT, "configs")

# Per-dataset config (d_model / d_ff / enc_in follow the repository's original scripts).
DATASETS = {
    "ETTh1":   dict(model_id="ETTh1_512",   csv="ETTh1.csv",   enc_in=7,   d_model=32, d_ff=128),
    "ETTh2":   dict(model_id="ETTh2_512",   csv="ETTh2.csv",   enc_in=7,   d_model=32, d_ff=128),
    "ETTm1":   dict(model_id="ETTm1_512",   csv="ETTm1.csv",   enc_in=7,   d_model=32, d_ff=128),
    "ETTm2":   dict(model_id="ETTm2_512",   csv="ETTm2.csv",   enc_in=7,   d_model=32, d_ff=128),
    "Weather": dict(model_id="weather_512", csv="Weather.csv", enc_in=21,  d_model=32, d_ff=32),
    "Traffic": dict(model_id="traffic_512", csv="traffic.csv", enc_in=862, d_model=16, d_ff=32),
}

HORIZONS = [96, 192, 336, 720]

# Reported values from the paper (Table II, ME-LLM "Ours" column): dataset -> pred_len -> (MSE, MAE).
PAPER = {
    "ETTm1":   {96: (0.302, 0.348), 192: (0.346, 0.375), 336: (0.363, 0.399), 720: (0.439, 0.441)},
    "ETTm2":   {96: (0.187, 0.273), 192: (0.251, 0.312), 336: (0.319, 0.357), 720: (0.365, 0.391)},
    "ETTh1":   {96: (0.395, 0.407), 192: (0.423, 0.425), 336: (0.454, 0.463), 720: (0.463, 0.472)},
    "ETTh2":   {96: (0.302, 0.349), 192: (0.379, 0.396), 336: (0.396, 0.427), 720: (0.402, 0.436)},
    "Weather": {96: (0.175, 0.201), 192: (0.217, 0.251), 336: (0.262, 0.279), 720: (0.346, 0.352)},
    "Traffic": {96: (0.403, 0.276), 192: (0.415, 0.289), 336: (0.421, 0.290), 720: (0.432, 0.295)},
}

GIT_COMMIT = "d4a289fe6bc3f8c41fded4db570c793e2ff7cfed"
SEED_RANGE = "2021-2030"


def make_config(dataset, pred_len):
    ds = DATASETS[dataset]
    cfg = {
        "task_name": "long_term_forecast",
        "is_training": 1,
        "model_id": "{}_{}".format(ds["model_id"], pred_len),
        "model_comment": "MELLM-{}".format(dataset),
        "model": "MELLM",
        "seed": 2021,
        "data": dataset,
        "root_path": "./dataset/{}/".format(dataset),
        "data_path": ds["csv"],
        "features": "M",
        "target": "OT",
        "loader": "modal",
        "freq": "h",
        "checkpoints": "./checkpoints/",
        "results": "./results/",
        "configs": "./configs/",
        "seq_len": 512,
        "label_len": 48,
        "pred_len": pred_len,
        "seasonal_patterns": "Monthly",
        "enc_in": ds["enc_in"],
        "dec_in": ds["enc_in"],
        "c_out": ds["enc_in"],
        "d_model": ds["d_model"],
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": ds["d_ff"],
        "moving_avg": 25,
        "factor": 3,
        "dropout": 0.1,
        "embed": "timeF",
        "activation": "gelu",
        "output_attention": False,
        "patch_len": 16,
        "stride": 8,
        "prompt_domain": 0,
        "llm_model": "BERT",
        "llm_dim": 768,
        "num_workers": 0,
        "itr": 10,
        "train_epochs": 15,
        "align_epochs": 10,
        "batch_size": 24,
        "eval_batch_size": 8,
        "patience": 5,
        "learning_rate": 0.01,
        "des": "Exp",
        "loss": "MSE",
        "lradj": "type1",
        "pct_start": 0.2,
        "use_amp": True,
        "use_deepspeed": False,
        "llm_layers": 12,
        "percent": 100,
    }
    return cfg


def setting_string(dataset, pred_len):
    ds = DATASETS[dataset]
    return ("long_term_forecast_{mid}_{model}_{data}_ftM_sl512_ll48_pl{pl}_dm{dm}_nh8_el2_dl1_df{df}_fc3_ebtimeF_Exp_0"
            .format(mid="{}_{}".format(ds["model_id"], pred_len), model="MELLM", data=dataset,
                    pl=pred_len, dm=ds["d_model"], df=ds["d_ff"]))


def main():
    os.makedirs(CONFIGS_DIR, exist_ok=True)

    rows = []
    paper_results = {}

    for dataset in DATASETS:
        paper_results[dataset] = {}
        for pred_len in HORIZONS:
            cfg = make_config(dataset, pred_len)
            setting = setting_string(dataset, pred_len)
            config_file = os.path.join(CONFIGS_DIR, "{}-MELLM-{}.json".format(setting, dataset))
            with open(config_file, "w") as f:
                json.dump(cfg, f, indent=2)

            mse, mae = PAPER[dataset][pred_len]
            paper_results[dataset][str(pred_len)] = {"MSE": mse, "MAE": mae}

            for metric, value in (("MSE", mse), ("MAE", mae)):
                rows.append({
                    "benchmark": "long_term_forecast",
                    "dataset": dataset,
                    "seasonal_patterns": "",
                    "pred_len": pred_len,
                    "seed": SEED_RANGE,
                    "metric": metric,
                    "group": "",
                    "value": value,
                    "best_epoch": "",
                    "checkpoint": "",
                    "checkpoint_sha256": "",
                    "config_file": os.path.relpath(config_file, ROOT),
                    "result_file": "paper_results.json",
                    "git_commit": GIT_COMMIT,
                })

    results_file = os.path.join(ROOT, "paper_results.json")
    with open(results_file, "w") as f:
        json.dump(paper_results, f, indent=2)

    fields = [
        "benchmark", "dataset", "seasonal_patterns", "pred_len", "seed",
        "metric", "group", "value", "best_epoch",
        "checkpoint", "checkpoint_sha256", "config_file", "result_file", "git_commit",
    ]
    out = os.path.join(ROOT, "MANIFEST.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("Wrote {} configs, {} manifest rows, paper_results.json".format(
        len(DATASETS) * len(HORIZONS), len(rows)))


if __name__ == "__main__":
    main()
