"""Generate MANIFEST.csv linking every reported value to its
configuration, seed, checkpoint, and raw result file.

Run from the repository root after training:

    python scripts/make_manifest.py

Reads:
  - results/*.json          (run_main.py output)
  - m4_results/*/metrics.json (run_m4.py aggregated output)

Writes MANIFEST.csv at the repository root.
"""
import csv
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIELDS = [
    "benchmark", "dataset", "seasonal_patterns", "pred_len", "seed",
    "metric", "group", "value", "best_epoch",
    "checkpoint", "checkpoint_sha256", "config_file", "result_file", "git_commit",
]


def rel(path):
    if not path:
        return ""
    return os.path.relpath(path, ROOT)


def collect_main_results(rows):
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "*.json"))):
        with open(path) as f:
            r = json.load(f)
        cfg = r.get("config", {})
        base = {
            "benchmark": cfg.get("task_name", ""),
            "dataset": cfg.get("data", ""),
            "seasonal_patterns": "",
            "pred_len": cfg.get("pred_len", ""),
            "seed": r.get("seed", ""),
            "best_epoch": r.get("best_epoch", ""),
            "checkpoint": rel(r.get("checkpoint")),
            "checkpoint_sha256": r.get("checkpoint_sha256", ""),
            "config_file": rel(r.get("config_file")) or rel(path),
            "result_file": rel(path),
            "git_commit": r.get("git_commit", ""),
        }
        for metric in ("MSE", "MAE"):
            key = "best_test_loss" if metric == "MSE" else "best_test_mae"
            value = r.get(key)
            if value is None:
                continue
            row = dict(base)
            row["metric"] = metric
            row["group"] = ""
            row["value"] = value
            rows.append(row)


def collect_m4_results(rows):
    for path in sorted(glob.glob(os.path.join(ROOT, "m4_results", "*", "metrics.json"))):
        with open(path) as f:
            r = json.load(f)
        base = {
            "benchmark": "m4",
            "dataset": "m4",
            "seasonal_patterns": "",
            "pred_len": "",
            "seed": "",
            "best_epoch": "",
            "checkpoint": "",
            "checkpoint_sha256": "",
            "config_file": rel(os.path.dirname(path)),
            "result_file": rel(path),
            "git_commit": r.get("git_commit", ""),
        }
        for metric in ("smape", "owa", "mape", "mase"):
            for group, value in (r.get(metric) or {}).items():
                row = dict(base)
                row["metric"] = metric.upper()
                row["group"] = group
                row["value"] = value
                rows.append(row)


def main():
    rows = []
    collect_main_results(rows)
    collect_m4_results(rows)

    out = os.path.join(ROOT, "MANIFEST.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("Wrote {} rows to {}".format(len(rows), rel(out)))


if __name__ == "__main__":
    main()
