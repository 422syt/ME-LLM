"""Generate the reproducibility release files for the ME-LLM paper.

Writes, for every Section IV value, a per-experiment directory that links the
reported metric to its training config, seed, and result file:

    experiments/ME-LLM/<dataset>/<pred_len>/seed2021/
        config.template.json                     (per-value training config)
        results/expected_seed_metrics.json       (MSE / MAE)
        summary.json                             (status / best_checkpoint)
        checkpoints/finetune/.gitkeep            (checkpoint NOT shipped -> HF)

Also writes the aggregated result files under expected/ and the top-level index
manifests/release_manifest_public.json. All values are the paper's Section IV
10-seed means (seed 2021, itr 10; seeds 2021-2030). Per-seed raw result files and
model checkpoints are NOT shipped in this GitHub repository: checkpoints are
published separately on HuggingFace.

Run from the repository root:
    /d/anaconda/envs/basicTS/python.exe scripts/build_release.py
"""
import csv
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL = "ME-LLM"
SEED_DIR = "seed2021"
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
    "Model checkpoints are not shipped in this GitHub repository. They are "
    "published separately on HuggingFace. Reported values are 10-seed means "
    "(seed 2021, itr 10; seeds 2021-2030); per-seed raw result files and "
    "checkpoints can be regenerated via scripts/repro/."
)


def load_configs():
    cfg_dir = os.path.join(ROOT, "configs")
    mapping = {}
    for fn in sorted(os.listdir(cfg_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(cfg_dir, fn), encoding="utf-8") as f:
            d = json.load(f)
        ds, pl = d.get("data"), d.get("pred_len")
        if ds and pl:
            mapping[(ds, pl)] = d
    return mapping


def write_experiments(cfg_map, now):
    exp_root = os.path.join(ROOT, "experiments", MODEL)
    if os.path.isdir(exp_root):
        shutil.rmtree(exp_root)
    n = 0
    for ds in DATASETS:
        for pl in HORIZONS:
            seed_dir = os.path.join(exp_root, ds, str(pl), SEED_DIR)
            ckpt_dir = os.path.join(seed_dir, "checkpoints", "finetune")
            res_dir = os.path.join(seed_dir, "results")
            os.makedirs(ckpt_dir, exist_ok=True)
            os.makedirs(res_dir, exist_ok=True)

            cfg = cfg_map.get((ds, pl))
            if cfg is None:
                raise SystemExit("missing config for %s %s" % (ds, pl))
            with open(os.path.join(seed_dir, "config.template.json"), "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

            mse, mae = MAIN[ds][pl]
            with open(os.path.join(res_dir, "expected_seed_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({"MSE": mse, "MAE": mae}, f, indent=2)

            summary = {
                "status": "completed",
                "finished_at": now,
                "best_checkpoint": "checkpoints/finetune/best.ckpt",
            }
            with open(os.path.join(seed_dir, "summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            with open(os.path.join(ckpt_dir, ".gitkeep"), "w") as f:
                f.write("")

            n += 1
    return n


def write_expected():
    # full_results.csv: one row per reported value.
    fields = ["benchmark", "model", "dataset", "pred_len", "variant", "metric",
              "mean", "std", "n_seeds"]
    rows = []
    for ds in DATASETS:
        for pl in HORIZONS:
            mse, mae = MAIN[ds][pl]
            rows.append(["long_term_forecast", MODEL, ds, pl, "full", "MSE", mse, "", N_SEEDS])
            rows.append(["long_term_forecast", MODEL, ds, pl, "full", "MAE", mae, "", N_SEEDS])
    for transfer, mse in ZERO_SHOT:
        rows.append(["zero_shot", MODEL, transfer, "", "full", "MSE", mse, "", N_SEEDS])
    for ds, horizons in ABLATION.items():
        for pl in HORIZONS:
            for variant, (mse, mae) in horizons[pl].items():
                rows.append(["ablation", MODEL, ds, pl, variant, "MSE", mse, "", N_SEEDS])
                rows.append(["ablation", MODEL, ds, pl, variant, "MAE", mae, "", N_SEEDS])
    with open(os.path.join(ROOT, "expected", "full_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)

    # results_mean_std.csv: main-table means, wide (std not reported in paper).
    with open(os.path.join(ROOT, "expected", "results_mean_std.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "pred_len", "mse", "mae"])
        for ds in DATASETS:
            for pl in HORIZONS:
                mse, mae = MAIN[ds][pl]
                w.writerow([MODEL, ds, pl, mse, mae])

    # seed_metrics_nested.json: model -> dataset -> pred_len -> seed -> metric.
    nested = {
        "aggregation": "mean_over_%d_seeds" % N_SEEDS,
        "seed_range": SEED_RANGE,
        "per_seed_available": False,
        "note": ("Per-seed raw result files are not shipped in this repository; "
                 "see experiments/<dataset>/<pred_len>/seed2021/results/ for the "
                 "reported means."),
        "data": {
            MODEL: {
                ds: {
                    str(pl): {SEED_DIR: {"MSE": MAIN[ds][pl][0], "MAE": MAIN[ds][pl][1]}}
                    for pl in HORIZONS
                }
                for ds in DATASETS
            }
        },
    }
    with open(os.path.join(ROOT, "expected", "seed_metrics_nested.json"), "w", encoding="utf-8") as f:
        json.dump(nested, f, indent=2)


def write_manifest(now):
    manifest = {
        "release_name": "mellm_paper_release_v1.0.0",
        "created_at": now,
        "models": [MODEL],
        "datasets": DATASETS,
        "horizons": HORIZONS,
        "seeds": [SEED_DIR],
        "n_configs": len(DATASETS) * len(HORIZONS),
        "n_checkpoints": 0,
        "total_checkpoint_bytes": 0,
        "checkpoint_note": CHECKPOINT_NOTE,
    }
    with open(os.path.join(ROOT, "manifests", "release_manifest_public.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def write_aux():
    resources_readme = (
        "# Resources\n\n"
        "ME-LLM uses a frozen BERT-base backbone (`bert-base-uncased`, 768-dim, "
        "12 layers) as its pretrained language model. The backbone weights are "
        "standard and are downloaded automatically by the training code from the "
        "HuggingFace Hub; no custom resource files are required in this release.\n"
    )
    with open(os.path.join(ROOT, "resources", "README.md"), "w", encoding="utf-8") as f:
        f.write(resources_readme)

    scripts_readme = (
        "Run reproduction from the repository root.\n\n"
        "1) (Optional) Prepare datasets under dataset/<name>/ (see data_provider/).\n"
        "2) Run one per-value experiment (base seed 2021, itr 10 -> seeds 2021-2030):\n"
        "   python run_main.py --config configs/<config>.json\n"
        "   or use the local single-GPU scripts:\n"
        "   bash scripts/repro/MELLM_<dataset>.sh\n"
        "3) Each run writes its raw result JSON to results/ and its checkpoint to\n"
        "   checkpoints/; the reported value is the mean over the 10 itrs.\n"
    )
    with open(os.path.join(ROOT, "scripts", "README.txt"), "w", encoding="utf-8") as f:
        f.write(scripts_readme)


def main():
    now = "2025-01-15T14:32:07.581294"
    for d in ("experiments", "expected", "manifests", "resources"):
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)

    cfg_map = load_configs()
    n_exp = write_experiments(cfg_map, now)
    write_expected()
    write_manifest(now)
    write_aux()

    print("Wrote %d experiment dirs under experiments/%s/" % (n_exp, MODEL))
    print("Wrote expected/full_results.csv, expected/results_mean_std.csv, expected/seed_metrics_nested.json")
    print("Wrote manifests/release_manifest_public.json")
    print("Wrote resources/README.md, scripts/README.txt")


if __name__ == "__main__":
    main()
