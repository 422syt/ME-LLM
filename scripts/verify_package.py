#!/usr/bin/env python3
from pathlib import Path
import csv, hashlib, sys
import pandas as pd

root = Path(__file__).resolve().parents[1]
errors=[]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

# Manifest checks
with (root/'MANIFEST.csv').open(newline='', encoding='utf-8') as fh:
    rows=list(csv.DictReader(fh))
for r in rows:
    p=root/r['path']
    if not p.exists():
        errors.append(f"missing: {r['path']}")
    elif sha(p) != r['sha256']:
        errors.append(f"checksum mismatch: {r['path']}")

# Required result tables
required=[
 'table4_average_benchmark.csv','table5_controlled_comparison.csv','table6_pairwise_statistics.csv',
 'table7_additional_metrics.csv','table8_univariate_comparison.csv','table9_within_family_transfer.csv',
 'table10_cross_family_transfer.csv','table11_external_validation.csv','table12_robustness.csv',
 'table13_component_ablation.csv','table14_factor_and_prompt_controls.csv',
 'table15_configuration_and_backbone_controls.csv','table16_prototype_intervention.csv',
 'table17_efficiency_main_setting.csv','table18_efficiency_reference.csv',
 'table_s1_horizon_wise_results.csv','table_s1_reported_setting_wins.csv']
for name in required:
    p=root/'results'/name
    if not p.exists(): errors.append(f'missing result table: {name}')
    else:
        try: pd.read_csv(p)
        except Exception as e: errors.append(f'cannot read {name}: {e}')

# README protocol-record checks
rd=(root/'README.md').read_text(encoding='utf-8')
for phrase in ['Validation-based model selection','Early stopping patience 5','Ten paired seeds (2021-2030)']:
    if phrase not in rd: errors.append(f'README missing: {phrase}')


if errors:
    print('package verification: FAIL')
    for e in errors: print('-',e)
    sys.exit(1)
print('package verification: PASS')
print(f'result tables: {len(required)}')
print(f'manifest entries: {len(rows)}')
