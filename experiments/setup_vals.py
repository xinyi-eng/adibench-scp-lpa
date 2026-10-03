"""One-time: build val CSVs for 1000:1 and 7000:1.

7000:1 split: keep all 4 minority samples in TRAIN (only 4 exist total, can't split),
              put 50 Khaleji into val. This produces a near-100% Khaleji majority val that
              is consistent with the existing C_log.txt result (val=500 Khaleji).
"""
import sys, os
sys.path.insert(0, r'D:/dacd2026/2_models/dacd++')

import pandas as pd
from sklearn.model_selection import train_test_split

base = r'D:/dacd2026/1_data/processed'
out_base = r'D:/dacd2026/3_experiments/v2_runs'
os.makedirs(out_base, exist_ok=True)

# 1000:1: stratified 90/10
src = os.path.join(base, 'dacd_bench_1000_1_train.csv')
df = pd.read_csv(src)
tr, va = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)
tr.to_csv(os.path.join(out_base, 'dacd_bench_1000_1_train.csv'), index=False)
va.to_csv(os.path.join(out_base, 'dacd_bench_1000_1_val.csv'), index=False)
print(f'1000:1: train={len(tr)} val={len(va)}')
print(f'  val dist: {va["label"].value_counts().to_dict()}')

# 7000:1: keep all 4 minority in train, val = 50 Khaleji (mirror C_log setup)
src = os.path.join(base, 'dacd_bench_7000_1_train.csv')
df = pd.read_csv(src)
print(f'7000:1 total: {len(df)} dist: {df["label"].value_counts().to_dict()}')
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_kh = df.index[df['label'] == 0].tolist()[:50]
val_df = df.iloc[val_kh].reset_index(drop=True)
tr = df.drop(val_kh).reset_index(drop=True)
tr.to_csv(os.path.join(out_base, 'dacd_bench_7000_1_train.csv'), index=False)
val_df.to_csv(os.path.join(out_base, 'dacd_bench_7000_1_val.csv'), index=False)
print(f'7000:1: train={len(tr)} val={len(val_df)}')
print(f'  train dist: {tr["label"].value_counts().to_dict()}')
print(f'  val dist: {val_df["label"].value_counts().to_dict()}')
print('done')
