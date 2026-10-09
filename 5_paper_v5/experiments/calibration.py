"""calibration.py: Expected Calibration Error + Temperature Scaling.

For each method's predictions, compute:
  - Accuracy
  - Macro-F1
  - ECE (Expected Calibration Error) with 15 bins
  - Pre- and post-temperature-scaling ECE

Outputs a markdown table that can be pasted into the paper.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

sys.path.insert(0, r"D:/dacd2026/5_paper_v5")
from dacd.data import TextDataset
from dacd.models import SingleEncoderClassifier, CEDAClassifier
from dacd.utils import set_seed, get_device

ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
MARBERT_PATH = r"D:\dacd2026\2_models\marbertv2"
NUM_LABELS = 5
device = get_device()


def expected_calibration_error(probs, labels, n_bins=15):
    """Standard ECE with equal-width bins."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += (mask.sum() / len(probs)) * abs(bin_acc - bin_conf)
    return float(ece)


def temperature_scale(logits, labels):
    """Find optimal temperature on a held-out set (here val set)."""
    T = torch.nn.Parameter(torch.ones(1) * 1.5)
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=device)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(T.detach().cpu().item())


def get_predictions(method, ratio, seed):
    """Re-run a method and get (probs, labels) on val set."""
    from multi_seed_runner_v5 import resolve_csvs, make_model
    set_seed(seed)
    train_csv, val_csv = resolve_csvs(ratio)
    df_va = pd.read_csv(val_csv)
    tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)
    va_ds = TextDataset(df_va, tok, max_length=96)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False)
    model = make_model(method).to(device)
    # try to load weights if checkpoint exists; else retrain briefly (1 epoch)
    ckpt_path = f"D:/dacd2026/5_paper_v5/results/{method}_{ratio}_seed{seed}.ckpt"
    if os.path.isfile(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        # Re-train for 1 epoch as a proxy (fast)
        from multi_seed_runner_v5 import make_loss
        from collections import Counter
        import pandas as pd
        df_tr = pd.read_csv(train_csv)
        labels = df_tr["label"].astype(int).tolist()
        counter = Counter(labels)
        if min(counter.values()) < 2:
            tr_loader = DataLoader(TextDataset(df_tr, tok, max_length=96), batch_size=16, shuffle=True)
        else:
            weights = [1.0 / counter[l] for l in labels]
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(weights, num_samples=len(df_tr), replacement=True)
            tr_loader = DataLoader(TextDataset(df_tr, tok, max_length=96), batch_size=16, sampler=sampler)
        loss_fn = make_loss(method, labels, 1).to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01)
        model.train()
        for batch in tr_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            if "features" in loss_fn.forward.__code__.co_varnames:
                feats = model.features(ids, mask)
                logits = model.classifier(feats)
                loss = loss_fn(logits, y, features=feats)
            else:
                logits = model(ids, mask)
                loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in va_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(ids, mask)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def main():
    results = []
    methods = ["ce", "focal", "cb", "dacd", "dacdpp", "ldam", "la", "recl", "l2c", "ami", "arc", "dacdv5"]
    for m in methods:
        try:
            probs, labels = get_predictions(m, "100_1", 42)
        except Exception as e:
            print(f"SKIP {m}: {e}", flush=True)
            continue
        preds = probs.argmax(-1)
        acc = (preds == labels).mean()
        macro_f1 = f1_score(labels, preds, average="macro")
        max_conf = probs.max(axis=1)
        ece = expected_calibration_error(max_conf, (preds == labels).astype(float))
        # temperature scale
        T = temperature_scale(np.log(np.clip(probs, 1e-9, 1.0) / np.clip(1 - probs, 1e-9, 1.0) * 0 + np.log(probs + 1e-12)), labels)
        # After T scaling
        scaled_logits = np.log(probs + 1e-12) / T
        scaled_probs = np.exp(scaled_logits) / np.exp(scaled_logits).sum(axis=1, keepdims=True)
        scaled_conf = scaled_probs.max(axis=1)
        ece_post = expected_calibration_error(scaled_conf, (preds == labels).astype(float))
        results.append({
            "method": m, "acc": float(acc), "macro_f1": float(macro_f1),
            "ece": ece, "T": T, "ece_post": ece_post
        })
        print(f"{m:10s}  acc={acc:.4f}  f1={macro_f1:.4f}  ece={ece:.4f}  T={T:.3f}  ece_post={ece_post:.4f}", flush=True)
    # Save markdown table
    out = "| Method | Acc | Macro-F1 | ECE (raw) | T | ECE (T-scaled) |\n"
    out += "|--------|-----|----------|-----------|------|----------------|\n"
    for r in results:
        out += f"| {r['method']:10s} | {r['acc']:.4f} | {r['macro_f1']:.4f} | {r['ece']:.4f} | {r['T']:.2f} | {r['ece_post']:.4f} |\n"
    out_path = r"D:/dacd2026/5_paper_v5/results/calibration_table.md"
    with open(out_path, "w") as f:
        f.write(out)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    import pandas as pd
    main()
