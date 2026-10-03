"""Training loops for DACD++.

Methods implemented:
    - "ce"          : vanilla cross-entropy
    - "focal"       : focal loss (gamma=2)
    - "cb"          : class-balanced (effective number)
    - "dacd"        : our 3-piece recipe (class-balanced + Khaleei debiasing)
    - "dacd++"      : CIS + DLB + MEC

Single-GPU training loop on RTX 4060 (8GB).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, classification_report

from .utils import set_seed, get_device, count_params
from .data import build_dacd_bench, TextDataset, subsample_imbalance, DIALECT_NAMES
from .models import SingleEncoderClassifier, MECClassifier
from .losses import (DACDLoss, CrossEntropyLoss, FocalLoss, MECLoss,
                    class_balanced_weights, CISSampler, PrototypeLoss)

ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
MARBERT_PATH = r"D:\dacd2026\2_models\marbertv2"
NUM_LABELS = 5


def get_model(method: str, num_labels: int = NUM_LABELS):
    if method == "mec":
        return MECClassifier(ARABERT_PATH, MARBERT_PATH, num_labels=num_labels)
    return SingleEncoderClassifier(ARABERT_PATH, num_labels=num_labels)


def get_loss(method: str, labels: List[int], num_labels: int = NUM_LABELS,
              alpha: float = 0.5, device: str = 'cpu'):
    cb = class_balanced_weights(labels, num_labels).to(device)
    if method == "ce":
        return CrossEntropyLoss(num_labels, class_weights=cb).to(device)
    if method == "focal":
        return FocalLoss(num_labels, gamma=2.0, class_weights=cb).to(device)
    if method == "cb":
        return CrossEntropyLoss(num_labels, class_weights=cb).to(device)
    if method == "dacd":
        return DACDLoss(num_labels, lam=0.5, beta=0.3, temperature=0.07, class_weights=cb).to(device)
    if method == "dacd++":
        return DACDLoss(num_labels, lam=0.5, beta=0.3, temperature=0.07, class_weights=cb).to(device)
    if method == "mec":
        return MECLoss(CrossEntropyLoss(num_labels, class_weights=cb).to(device), alpha=alpha).to(device)
    if method == "dacdpp":
        ce = CrossEntropyLoss(num_labels, class_weights=cb).to(device)
        proto = PrototypeLoss(num_labels, temperature=0.07, alpha=0.5, class_weights=cb).to(device)
        return (ce, proto)
    raise ValueError(f"unknown method: {method}")


def evaluate(model, loader, device, method: str) -> Dict[str, float]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            if method == "mec":
                # for MEC eval, take encoder1 logits
                h1 = model._features(model.encoder1, ids, mask)
                logits = model.classifier(h1)
            else:
                logits = model(ids, mask)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
            labels.extend(batch["label"].cpu().tolist())
    macro = f1_score(labels, preds, average='macro')
    per_class = f1_score(labels, preds, average=None, labels=list(range(NUM_LABELS)))
    return {
        "macro_f1": float(macro),
        **{f"f1_{DIALECT_NAMES[i]}": float(per_class[i]) for i in range(NUM_LABELS)},
    }


def train_one(
    method: str,
    train_csv: str,
    val_csv: str,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    seed: int = 42,
    imbalance: float = 7000.0,
    curriculum: bool = False,
    use_cis: bool = False,
    cb_sampler: bool = False,  # NEW: class-balanced sampler (every batch has equal class)
    out_dir: str = "D:/dacd2026/3_experiments/runs",
):
    set_seed(seed)
    device = get_device()
    print(f"== method={method} seed={seed} curriculum={curriculum or use_cis} ==")
    print(f"device={device}")
    # data
    df_tr = pd.read_csv(train_csv)
    df_va = pd.read_csv(val_csv)
    print(f"train: {len(df_tr)}  val: {len(df_va)}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)
    tr_ds = TextDataset(df_tr, tok, max_length=96)
    va_ds = TextDataset(df_va, tok, max_length=96)
    NUM_LABELS = tr_ds.num_classes_from_data
    print(f"  num_classes (inferred from data): {NUM_LABELS}")
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False)
    # model
    model = get_model(method).to(device)
    # loss
    labels = df_tr["label"].astype(int).tolist()
    loss_fn = get_loss(method, labels, device=device)
    # Pre-compute class counts for cb_sampler
    from collections import Counter
    counter = Counter(labels)
    class_target_n = {c: counter[c] for c in counter}  # use original counts; sampler weights = 1/count
    # optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # training loop
    history = []
    for epoch in range(epochs):
        # Sampler selection
        if curriculum or use_cis:
            sampler = CISSampler(labels, epoch=epoch, total_epochs=epochs,
                                  data_imbalance=imbalance, seed=seed)
            tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler)
        elif cb_sampler:
            # Class-balanced sampler: each batch has B/K samples from each class
            from torch.utils.data import WeightedRandomSampler
            weights = []
            for label in labels:
                # weight = 1 / count for this class
                weights.append(1.0 / class_target_n[label])
            sampler = WeightedRandomSampler(weights, num_samples=len(labels), replacement=True)
            tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler)
        else:
            tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        model.train()
        t0 = time.time()
        running_loss = 0.0; n = 0
        for batch_idx, batch in enumerate(tr_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            if method == "mec":
                logits1, logits2, cons = model.forward_both(ids, mask)
                loss = loss_fn(logits1, logits2, y, cons)
            elif method == "dacd" or method == "dacd++":
                feats = model.features(ids, mask)
                logits = model.classifier(feats)
                loss = loss_fn(logits, y, features=feats)
            elif method == "dacdpp":
                feats = model.features(ids, mask)
                logits = model.classifier(feats)
                ce_loss, proto_loss = loss_fn
                loss = ce_loss(logits, y) + proto_loss(logits, y, features=feats)
            else:
                logits = model(ids, mask)
                loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() * ids.size(0)
            n += ids.size(0)
        train_loss = running_loss / max(1, n)
        elapsed = time.time() - t0
        metrics = evaluate(model, va_loader, device, method)
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "elapsed_s": elapsed,
            **metrics,
        }
        history.append(rec)
        print(f"epoch {epoch+1}/{epochs} loss={train_loss:.4f} "
              f"macro_f1={metrics['macro_f1']:.4f} elapsed={elapsed:.1f}s")
    # save
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{method}_seed{seed}"
    if use_cis:
        tag += "_curriculum"
    with open(os.path.join(out_dir, f"{tag}.json"), 'w') as f:
        json.dump({"method": method, "seed": seed, "curriculum": use_cis,
                   "history": history, "imbalance": imbalance}, f, indent=2)
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["ce", "focal", "cb", "dacd", "dacd++", "mec"])
    ap.add_argument("--train_csv", default="D:/dacd2026/1_data/processed/train.csv")
    ap.add_argument("--val_csv",   default="D:/dacd2026/1_data/processed/val.csv")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imbalance", type=float, default=7000.0)
    ap.add_argument("--curriculum", action="store_true")
    ap.add_argument("--out_dir", default="D:/dacd2026/3_experiments/runs")
    args = ap.parse_args()
    train_one(
        method=args.method,
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        imbalance=args.imbalance,
        curriculum=args.curriculum,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
