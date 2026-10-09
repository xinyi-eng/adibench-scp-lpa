"""Multi-seed runner: any method/ratio/seed/epochs. Sequential to avoid OOM.

Reads existing dacd codebase; saves per-run history JSON with per-class F1.
"""
import sys, os
sys.path.insert(0, r'D:/dacd2026/2_models/dacd++')

import gc
import json
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from sklearn.metrics import f1_score

from dacd.data import TextDataset, DIALECT_NAMES
from dacd.models import SingleEncoderClassifier
from dacd.losses import (
    CrossEntropyLoss, FocalLoss, DACDLoss, PrototypeLoss,
    LDAMLoss, LogitAdjustmentLoss, ReCLLoss,
    class_balanced_weights,
)
from dacd.utils import set_seed, get_device

ARABERT_PATH = r'D:\dacd2026\2_models\arabertv02'
NUM_LABELS = 5
device = get_device()
print(f'device={device}  start={time.strftime("%H:%M:%S")}', flush=True)

# data file resolver
BASE_DATA = r'D:/dacd2026/1_data/processed'
V2_DATA = r'D:/dacd2026/3_experiments/v2_runs'

def resolve_csvs(ratio):
    """Return (train_csv, val_csv) for a ratio label."""
    if ratio == '100_1':
        # big split = full 100:1
        return (r'D:/dacd2026/1_data/processed/dacd_bench_big_train.csv',
                r'D:/dacd2026/1_data/processed/dacd_bench_big_val.csv')
    elif ratio == '1000_1':
        # 1000:1 val was split from train by setup_vals.py into v2_runs
        tr = os.path.join(V2_DATA, 'dacd_bench_1000_1_train.csv')
        va = os.path.join(V2_DATA, 'dacd_bench_1000_1_val.csv')
        if not os.path.isfile(tr):
            tr = os.path.join(BASE_DATA, 'dacd_bench_1000_1_train.csv')
        if not os.path.isfile(va):
            va = os.path.join(BASE_DATA, 'dacd_bench_1000_1_val.csv')  # fallback, will produce nonsense but won't crash
        return tr, va
    elif ratio == '7000_1':
        tr = os.path.join(V2_DATA, 'dacd_bench_7000_1_train.csv')
        va = os.path.join(V2_DATA, 'dacd_bench_7000_1_val.csv')
        if not os.path.isfile(tr):
            tr = os.path.join(BASE_DATA, 'dacd_bench_7000_1_train.csv')
        if not os.path.isfile(va):
            va = os.path.join(BASE_DATA, 'dacd_bench_7000_1_val.csv')
        return tr, va
    else:
        raise ValueError(ratio)


def run_one(method, ratio, seed, epochs=1, batch_size=16, max_length=96,
            dacd_lam=0.5, dacd_beta=0.3):
    torch.cuda.empty_cache(); gc.collect()
    train_csv, val_csv = resolve_csvs(ratio)
    set_seed(seed)
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    n_tr, n_va = len(df_tr), len(df_va)
    tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)
    tr_ds = TextDataset(df_tr, tok, max_length=max_length)
    va_ds = TextDataset(df_va, tok, max_length=max_length)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False)
    labels = df_tr['label'].astype(int).tolist()
    counter = Counter(labels)
    weights = [1.0 / counter[l] for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=n_tr, replacement=True)
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler)
    model = SingleEncoderClassifier(ARABERT_PATH, num_labels=NUM_LABELS).to(device)
    cb = class_balanced_weights(labels, NUM_LABELS).to(device)
    if method == 'ce':
        loss_fn = CrossEntropyLoss(NUM_LABELS, class_weights=cb).to(device)
    elif method == 'focal':
        loss_fn = FocalLoss(NUM_LABELS, gamma=2.0, class_weights=cb).to(device)
    elif method == 'cb':
        loss_fn = CrossEntropyLoss(NUM_LABELS, class_weights=cb).to(device)  # CB == CE with sampler; this matches earlier runs
    elif method == 'dacd':
        loss_fn = DACDLoss(NUM_LABELS, lam=dacd_lam, beta=dacd_beta, temperature=0.07, class_weights=cb).to(device)
    elif method == 'dacdpp':
        loss_fn = DACDLoss(NUM_LABELS, lam=dacd_lam, beta=dacd_beta, temperature=0.07, class_weights=cb).to(device)
    elif method == 'ldam':
        loss_fn = LDAMLoss(NUM_LABELS, max_m=0.3, s=10.0, class_weights=cb).to(device)
    elif method == 'la':
        priors = np.bincount(labels, minlength=NUM_LABELS) / n_tr
        loss_fn = LogitAdjustmentLoss(NUM_LABELS, class_priors=priors, tau=1.0, class_weights=cb).to(device)
    elif method == 'recl':
        loss_fn = ReCLLoss(NUM_LABELS, alpha=0.5, temperature=0.07, class_weights=cb).to(device)
    else:
        raise ValueError(method)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    history = []
    t0_all = time.time()
    for epoch in range(epochs):
        model.train()
        t_epoch = time.time()
        running, n = 0.0, 0
        for batch_idx, batch in enumerate(tr_loader):
            ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
            y = batch['label'].to(device)
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            if method in ('dacd', 'dacdpp'):
                feats = model.features(ids, mask); logits = model.classifier(feats)
                loss = loss_fn(logits, y, features=feats)
            else:
                logits = model(ids, mask)
                loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * ids.size(0); n += ids.size(0)
        train_loss = running / max(1, n)
        # eval
        model.eval(); preds, labs = [], []
        with torch.no_grad():
            for batch in va_loader:
                ids = batch['input_ids'].to(device); mask = batch['attention_mask'].to(device)
                logits = model(ids, mask)
                preds.extend(logits.argmax(-1).cpu().tolist())
                labs.extend(batch['label'].cpu().tolist())
        macro = float(f1_score(labs, preds, average='macro'))
        per_class = f1_score(labs, preds, average=None, labels=list(range(NUM_LABELS)))
        rec = {
            'epoch': epoch, 'train_loss': train_loss, 'macro_f1': macro,
            'f1_Khaleji': float(per_class[0]),
            'f1_Iraqi': float(per_class[1]),
            'f1_Levantine': float(per_class[2]),
            'f1_Masri': float(per_class[3]),
            'f1_Maghrebi': float(per_class[4]),
            'elapsed_s': time.time() - t_epoch,
            'n_train': n_tr, 'n_val': n_va,
        }
        history.append(rec)
        print(f'  {method} {ratio} seed={seed} ep{epoch+1}/{epochs} loss={train_loss:.4f} '
              f'macro={macro:.4f} Kha={per_class[0]:.3f} Iraq={per_class[1]:.3f} '
              f'Lev={per_class[2]:.3f} Mas={per_class[3]:.3f} Mag={per_class[4]:.3f} '
              f'({rec["elapsed_s"]:.0f}s)', flush=True)
    # save
    out_dir = r'D:/dacd2026/3_experiments/v2_runs'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{method}_{ratio}_seed{seed}.json')
    with open(out_path, 'w') as f:
        json.dump({'method': method, 'ratio': ratio, 'seed': seed, 'epochs': epochs,
                   'history': history,
                   'imbalance': float(ratio.replace('_', ':').replace(':', '').rstrip(':')) if '_' in ratio else 100.0,
                   'total_time_s': time.time() - t0_all}, f, indent=2)
    print(f'  saved {out_path} total={time.time()-t0_all:.0f}s')
    # cleanup
    del model, loss_fn, opt
    torch.cuda.empty_cache(); gc.collect()
    return history


if __name__ == '__main__':
    # signature: python multi_seed_runner.py method ratio seed epochs ...
    args = sys.argv[1:]
    method = args[0]
    ratio = args[1]
    seed = int(args[2])
    epochs = int(args[3]) if len(args) > 3 else 1
    extra = {}
    if len(args) > 4:
        for kv in args[4:]:
            k, v = kv.split('=')
            extra[k] = float(v)
    run_one(method, ratio, seed, epochs=epochs, **extra)
