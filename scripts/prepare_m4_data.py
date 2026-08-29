"""Generate M4 training.npz / test.npz from the M4-methods CSVs.

Reads:
  - dataset/m4/M4-info.csv
  - dataset/m4/{Yearly,Quarterly,Monthly,Weekly,Daily,Hourly}-{train,test}.csv

Writes:
  - dataset/m4/training.npz  (object array of per-series history, in M4-info order)
  - dataset/m4/test.npz      (object array of per-series future values)

The npz stores a single key "values" holding an object ndarray whose elements
are variable-length float arrays. M4Dataset.load() expects exactly this layout.
"""
import os
import numpy as np
import pandas as pd

GROUPS = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
M4_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'm4')


def load_series(path):
    df = pd.read_csv(path)
    ids = df.iloc[:, 0].astype(str).values
    vals = df.iloc[:, 1:].to_numpy(dtype=float)
    out = {}
    for mid, row in zip(ids, vals):
        out[mid] = row[~np.isnan(row)]
    return out


def main():
    info = pd.read_csv(os.path.join(M4_DIR, 'M4-info.csv'))
    ids = info['M4id'].astype(str).values

    for split in ('train', 'test'):
        series = {}
        for g in GROUPS:
            series.update(load_series(os.path.join(M4_DIR, f'{g}-{split}.csv')))
        missing = [m for m in ids if m not in series]
        if missing:
            raise RuntimeError(f'{len(missing)} ids missing from {split} CSVs, e.g. {missing[:5]}')
        obj = np.empty(len(ids), dtype=object)
        for i, mid in enumerate(ids):
            obj[i] = series[mid]
        cache_name = 'training' if split == 'train' else split
        out = os.path.join(M4_DIR, f'{cache_name}.npz')
        np.savez(out, values=obj)
        print(f'saved {out} ({len(obj)} series)')


if __name__ == '__main__':
    main()
