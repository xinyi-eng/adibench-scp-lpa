"""cross_dialect_transfer.py: Leave-one-dialect-out transfer evaluation.

For each target dialect c in {0..4}:
  - Train on the remaining 4 dialects
  - Evaluate on c (zero-shot transfer)
  - Use linguistic distance to predict transfer quality

Tests whether the contrastive/curriculum methods learn more
dialect-agnostic features that transfer better.
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
from multi_seed_runner_v5 import make_loss, resolve_csvs, MARBERT_PATH, ARABERT_PATH
from collections import Counter
import torch.nn.functional as F

device = get_device()
NUM_LABELS = 5
LINGUISTIC_DISTANCE = np.array([
    [0.00, 0.45, 0.55, 0.65, 0.85],
    [0.45, 0.00, 0.40, 0.50, 0.75],
    [0.55, 0.40, 0.00, 0.20, 0.70],
    [0.65, 0.50, 0.20, 0.00, 0.65],
    [0.85, 0.75, 0.70, 0.65, 0.00],
])
DIALECT_NAMES = ["Khaleeji", "Iraqi", "Levantine", "Masri", "Maghrebi"]


def leave_one_out(method, target_class, ratio="100_1", seed=42, epochs=3):
    """Train on classes != target_class, eval on full val set with target_class removed."""
    torch.cuda.empty_cache()
    import gc; gc.collect()
    set_seed(seed)
    train_csv, val_csv = resolve_csvs(ratio)
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    # Remove target class from training, keep all in val
    df_tr_loo = df_tr[df_tr["label"] != target_class].reset_index(drop=True)
    if len(df_tr_loo) == 0:
        return {"error": "no training data after LOO"}
    # Remap labels to [0, 4] contiguous
    unique = sorted(df_tr_loo["label"].unique())
    label_map = {v: i for i, v in enumerate(unique)}
    df_tr_loo["label"] = df_tr_loo["label"].map(label_map)
    n_classes = len(unique)
    # Val: remap target_class to the class closest in linguistic distance
    df_va_loo = df_va.copy()
    df_va_loo["label"] = df_va_loo["label"].map(label_map)
    df_va_loo = df_va_loo[df_va_loo["label"].notna()].reset_index(drop=True)
    df_va_loo["label"] = df_va_loo["label"].astype(int)
    tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)
    tr_ds = TextDataset(df_tr_loo, tok, max_length=96)
    va_ds = TextDataset(df_va_loo, tok, max_length=96)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False)
    labels = df_tr_loo["label"].astype(int).tolist()
    counter = Counter(labels)
    if min(counter.values()) < 2:
        tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
    else:
        weights = [1.0 / counter[l] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(df_tr_loo), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=16, sampler=sampler)
    model = SingleEncoderClassifier(ARABERT_PATH, num_labels=n_classes).to(device)
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
    macro = f1_score(labs, preds, average="macro", zero_division=0)
    acc = (np.array(preds) == np.array(labs)).mean()
    # predicted target: which class the model most often predicts
    pred_dist = Counter(preds)
    predicted_target = pred_dist.most_common(1)[0][0]
    # distance from LOO target to predicted class
    dist = LINGUISTIC_DISTANCE[target_class, predicted_target] if predicted_target in unique else -1
    return {
        "method": method,
        "target": DIALECT_NAMES[target_class],
        "macro_f1": float(macro),
        "acc": float(acc),
        "predicted_as": unique[predicted_target] if predicted_target < len(unique) else -1,
        "linguistic_distance": float(dist),
        "n_train": len(df_tr_loo),
        "n_val": len(df_va_loo),
    }


def main():
    methods = ["ce", "dacd", "l2c", "arc", "dacdv5"]
    out_path = r"D:/dacd2026/5_paper_v5/results/cross_dialect_transfer.json"
    results = []
    for m in methods:
        for c in range(5):
            print(f"LOO: method={m} target={DIALECT_NAMES[c]}", flush=True)
            res = leave_one_out(m, c, ratio="100_1", seed=42, epochs=3)
            print(f"  -> {res}", flush=True)
            results.append(res)
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
    # Summary table
    out = "| Method | " + " | ".join(DIALECT_NAMES) + " | Avg | AvgDist |\n"
    out += "|--------|" + "|".join(["-" * 8] * 7) + "\n"
    for m in methods:
        per_target = [r for r in results if r["method"] == m]
        f1s = [r["macro_f1"] for r in per_target]
        dists = [r["linguistic_distance"] for r in per_target if r["linguistic_distance"] >= 0]
        out += f"| {m:10s} | " + " | ".join(f"{r['macro_f1']:.3f}" for r in per_target) + f" | {np.mean(f1s):.3f} | {np.mean(dists):.3f} |\n"
    out_md = r"D:/dacd2026/5_paper_v5/results/cross_dialect_transfer.md"
    with open(out_md, "w") as f:
        f.write(out)
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
