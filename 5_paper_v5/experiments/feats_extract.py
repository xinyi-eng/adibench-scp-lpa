"""Extract last-layer features from trained models for t-SNE visualisation.

Run from inside experiments/. Trains a quick CE model for 1 epoch on each
ratio, then dumps the [CLS] features for the validation set to a numpy array.
These get fed into generate_features_v4.py -> t-SNE.

Writes:
  D:/dacd2026/3_experiments/v2_runs/features_ce_<ratio>_seed42.npz
    keys: features (N, 768), labels (N,), method ('ce'), ratio (...)
"""
import gc
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from collections import Counter

sys.path.insert(0, r'D:/dacd2026/2_models/dacd++')
from dacd.data import TextDataset
from dacd.models import SingleEncoderClassifier
from dacd.losses import CrossEntropyLoss, class_balanced_weights
from dacd.utils import set_seed, get_device

ARABERT_PATH = r'D:\dacd2026\2_models\arabertv02'
device = get_device()


def resolve_csvs(ratio):
    if ratio == '100_1':
        return (r'D:/dacd2026/1_data/processed/dacd_bench_big_train.csv',
                r'D:/dacd2026/1_data/processed/dacd_bench_big_val.csv')
    elif ratio == '1000_1':
        return (r'D:/dacd2026/3_experiments/v2_runs/dacd_bench_1000_1_train.csv',
                r'D:/dacd2026/3_experiments/v2_runs/dacd_bench_1000_1_val.csv')
    raise ValueError(ratio)


def train_and_extract(method, ratio, seed=42, epochs=1, batch_size=16, max_length=96,
                      out_dir=r'D:/dacd2026/3_experiments/v2_runs'):
    set_seed(seed)
    train_csv, val_csv = resolve_csvs(ratio)
    df_tr = pd.read_csv(train_csv); df_va = pd.read_csv(val_csv)
    tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)
    tr_ds = TextDataset(df_tr, tok, max_length=max_length)
    va_ds = TextDataset(df_va, tok, max_length=max_length)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False)
    labels = df_tr['label'].astype(int).tolist()
    counter = Counter(labels)
    weights = [1.0 / counter[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler)
    model = SingleEncoderClassifier(ARABERT_PATH, num_labels=5).to(device)
    cb = class_balanced_weights(labels, 5).to(device)
    loss_fn = CrossEntropyLoss(5, class_weights=cb).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    for epoch in range(epochs):
        model.train()
        for batch_idx, batch in enumerate(tr_loader):
            ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
            y = batch['label'].to(device)
            if batch_idx % 100 == 0: torch.cuda.empty_cache()
            if method == 'ce':
                logits = model(ids, mask); loss = loss_fn(logits, y)
            else:
                feats = model.features(ids, mask); logits = model.classifier(feats)
                loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval(); all_feats, all_lbls = [], []
    with torch.no_grad():
        for batch in va_loader:
            ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
            feats = model.features(ids, mask)
            all_feats.append(feats.cpu().numpy())
            all_lbls.extend(batch['label'].cpu().tolist())
    feats_arr = np.concatenate(all_feats, axis=0)
    out_path = os.path.join(out_dir, f'features_{method}_{ratio}_seed{seed}.npz')
    np.savez(out_path, features=feats_arr, labels=np.array(all_lbls), method=method, ratio=ratio)
    print(f'wrote {out_path} shape={feats_arr.shape}')


if __name__ == '__main__':
    args = sys.argv[1:]
    method = args[0] if args else 'ce'
    ratio = args[1] if len(args) > 1 else '100_1'
    train_and_extract(method, ratio)
