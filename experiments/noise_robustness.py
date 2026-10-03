"""noise_robustness.py: Label-noise robustness study.

For each method, train on 100:1 data with X% of training labels
randomly flipped. Measure degradation curve.
Hypothesis: contrastive methods (DACD, L2C, ARC) are more
robust to label noise because the contrastive term doesn't
fully trust hard labels.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

sys.path.insert(0, r"D:/dacd2026/5_paper_v5")
from dacd.data import TextDataset
from dacd.models import SingleEncoderClassifier
from dacd.utils import set_seed, get_device
from multi_seed_runner_v5 import make_loss, resolve_csvs, ARABERT_PATH
from collections import Counter
import random

device = get_device()
NUM_LABELS = 5


def add_label_noise(df, noise_rate, seed=42):
    """Randomly flip `noise_rate` fraction of labels to a random other class."""
    rng = random.Random(seed)
    df = df.copy()
    n = len(df)
    n_flip = int(n * noise_rate)
    idx_to_flip = rng.sample(range(n), n_flip)
    labels = df["label"].astype(int).tolist()
    for i in idx_to_flip:
        old = labels[i]
        candidates = [c for c in range(NUM_LABELS) if c != old]
        labels[i] = rng.choice(candidates)
    df["label"] = labels
    return df


def run_with_noise(method, noise_rate, ratio="100_1", seed=42, epochs=3):
    torch.cuda.empty_cache()
    import gc; gc.collect()
    set_seed(seed)
    train_csv, val_csv = resolve_csvs(ratio)
    df_tr = pd.read_csv(train_csv)
    df_tr_noisy = add_label_noise(df_tr, noise_rate, seed=seed)
    df_va = pd.read_csv(val_csv)
    tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)
    tr_ds = TextDataset(df_tr_noisy, tok, max_length=96)
    va_ds = TextDataset(df_va, tok, max_length=96)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False)
    labels = df_tr_noisy["label"].astype(int).tolist()
    counter = Counter(labels)
    if min(counter.values()) < 2:
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
    else:
        weights = [1.0 / counter[l] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(df_tr_noisy), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=16, sampler=sampler)
    model = SingleEncoderClassifier(ARABERT_PATH, num_labels=NUM_LABELS).to(device)
    loss_fn = make_loss(method, labels, epochs).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01)
    for epoch in range(epochs):
        model.train()
        for batch in tr_loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device); y = batch["label"].to(device)
            if "features" in loss_fn.forward.__code__.co_varnames:
                feats = model.features(ids, mask)
                logits = model.classifier(feats)
                loss = loss_fn(logits, y, features=feats)
            else:
                logits = model(ids, mask)
                loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); preds, labs = [], []
    with torch.no_grad():
        for batch in va_loader:
            ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs.extend(batch["label"].cpu().tolist())
    return float(f1_score(labs, preds, average="macro"))


def main():
    methods = ["ce", "dacd", "l2c", "arc", "dacdv5"]
    noise_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
    out_path = r"D:/dacd2026/5_paper_v5/results/noise_robustness.json"
    results = []
    for m in methods:
        for nr in noise_rates:
            print(f"NOISE: method={m} noise={nr}", flush=True)
            f1 = run_with_noise(m, nr, ratio="100_1", seed=42, epochs=3)
            print(f"  -> f1={f1:.4f}", flush=True)
            results.append({"method": m, "noise_rate": nr, "macro_f1": f1})
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
    out = "| Method | " + " | ".join(f"{int(n*100)}%" for n in noise_rates) + " | Degradation |\n"
    out += "|--------|" + "|".join(["-" * 8] * (len(noise_rates) + 1)) + "\n"
    for m in methods:
        per = [r for r in results if r["method"] == m]
        f1s = [r["macro_f1"] for r in per]
        deg = f1s[0] - f1s[-1]
        out += f"| {m:10s} | " + " | ".join(f"{f:.3f}" for f in f1s) + f" | {deg:.3f} |\n"
    out_md = r"D:/dacd2026/5_paper_v5/results/noise_robustness.md"
    with open(out_md, "w") as f:
        f.write(out)
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
