"""Few-shot runner: sub-sample minority classes at the 100:1 base split to
{n_min} samples per class, then run a single 1-epoch training of DACD++,
DACD, and CE for comparison.

Writes /d/dacd2026/3_experiments/v2_runs/dacdpp_fewshot_n<X>_seed<N>.json
and analogous names for dacd_fewshot / ce_fewshot.

Run from inside experiments/:
  python3 multi_seed_runner_fewshot.py dacdpp 100_1 42 5
"""
import gc
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

sys.path.insert(0, r'D:/dacd2026/2_models/dacd++')
from dacd.data import TextDataset
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


def resolve_csvs(ratio):
    if ratio == '100_1':
        return (r'D:/dacd2026/1_data/processed/dacd_bench_big_train.csv',
                r'D:/dacd2026/1_data/processed/dacd_bench_big_val.csv')
    raise ValueError(ratio)


def subsample_minorities(df, n_min, kh_label=0, seed=42):
    """Keep all Khaleji samples; downsample each minority class to n_min."""
    out = []
    for c in sorted(df['label'].unique()):
        sub = df[df['label'] == c]
        if c == kh_label:
            out.append(sub)
        else:
            out.append(sub.sample(n=min(n_min, len(sub)), random_state=seed))
    return pd.concat(out, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def run_one(method, ratio, seed, n_min, epochs=1, batch_size=16, max_length=96,
            dacd_lam=0.5, dacd_beta=0.3):
    torch.cuda.empty_cache(); gc.collect()
    train_csv, val_csv = resolve_csvs(ratio)
    set_seed(seed)
    df_tr = pd.read_csv(train_csv)
    df_tr = subsample_minorities(df_tr, n_min=n_min, seed=seed)
    df_va = pd.read_csv(val_csv)
    n_tr, n_va = len(df_tr), len(df_va)
    tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)
    tr_ds = TextDataset(df_tr, tok, max_length=max_length)
    va_ds = TextDataset(df_va, tok, max_length=max_length)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False)
    labels = df_tr['label'].astype(int).tolist()
    counter = Counter(labels)
    if min(counter.values()) < 2:
        # Can't do class-balanced sampling for sub-classes with <2 samples
        # fall back to random sampling
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    else:
        weights = [1.0 / counter[l] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=n_tr, replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler)
    model = SingleEncoderClassifier(ARABERT_PATH, num_labels=NUM_LABELS).to(device)
    cb = class_balanced_weights(labels, NUM_LABELS).to(device)
    if method == 'ce':
        loss_fn = CrossEntropyLoss(NUM_LABELS, class_weights=cb).to(device)
    elif method == 'dacd':
        loss_fn = DACDLoss(NUM_LABELS, lam=dacd_lam, beta=dacd_beta,
                           temperature=0.07, class_weights=cb).to(device)
    elif method == 'dacdpp':
        loss_fn = DACDLoss(NUM_LABELS, lam=dacd_lam, beta=dacd_beta,
                           temperature=0.07, class_weights=cb).to(device)
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
                logits = model(ids, mask); loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * ids.size(0); n += ids.size(0)
        train_loss = running / max(1, n)
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
            'f1_Khaleji': float(per_class[0]), 'f1_Iraqi': float(per_class[1]),
            'f1_Levantine': float(per_class[2]), 'f1_Masri': float(per_class[3]),
            'f1_Maghrebi': float(per_class[4]),
            'elapsed_s': time.time() - t_epoch, 'n_train': n_tr, 'n_val': n_va,
            'n_min_subsample': n_min,
        }
        history.append(rec)
        print(f'  FEW {method} {ratio} n_min={n_min} seed={seed} ep{epoch+1}/{epochs} '
              f'loss={train_loss:.4f} macro={macro:.4f} '
              f'Kha={per_class[0]:.3f} Iraq={per_class[1]:.3f} '
              f'Lev={per_class[2]:.3f} Mas={per_class[3]:.3f} Mag={per_class[4]:.3f} '
              f'({rec["elapsed_s"]:.0f}s)', flush=True)
    out_dir = r'D:/dacd2026/3_experiments/v2_runs'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{method}_fewshot_n{n_min}_seed{seed}.json')
    with open(out_path, 'w') as f:
        json.dump({'method': method, 'ratio': ratio, 'seed': seed, 'epochs': epochs,
                   'n_min': n_min, 'history': history,
                   'total_time_s': time.time() - t0_all}, f, indent=2)
    print(f'  saved {out_path} total={time.time()-t0_all:.0f}s')
    del model, loss_fn, opt
    torch.cuda.empty_cache(); gc.collect()
    return history


if __name__ == '__main__':
    # signature: python3 multi_seed_runner_fewshot.py <method> <ratio> <seed> <n_min> [<epochs>]
    args = sys.argv[1:]
    method = args[0]
    ratio = args[1]
    seed = int(args[2])
    n_min = int(args[3])
    epochs = int(args[4]) if len(args) > 4 else 1
    extra = {}
    if len(args) > 5:
        for kv in args[5:]:
            k, v = kv.split('=')
            extra[k] = float(v)
    run_one(method, ratio, seed, n_min, epochs=epochs, **extra)
