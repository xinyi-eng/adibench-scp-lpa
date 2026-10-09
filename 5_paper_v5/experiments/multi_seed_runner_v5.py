"""multi_seed_runner_v5.py - main v5 experiment runner.

Supports all 8 baseline methods + 4 new v5 innovations:
  - Baseline: ce, focal, cb, dacd, dacdpp, ldam, la, recl, marbert
  - v5 innovations: l2c, ami, arc, ceda, dacdv5
"""
import gc, json, os, sys, time
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

sys.path.insert(0, r"D:/dacd2026/5_paper_v5")
from dacd.data import TextDataset
from dacd.models import SingleEncoderClassifier, CEDAClassifier
from dacd.losses import (
    CrossEntropyLoss, FocalLoss, DACDLoss, PrototypeLoss,
    LDAMLoss, LogitAdjustmentLoss, ReCLLoss,
    L2CLoss, AMILoss, ARCLoss, CEDALoss, DACDv5Loss,
    class_balanced_weights,
)
from dacd.utils import set_seed, get_device

ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
MARBERT_PATH = r"D:\dacd2026\2_models\marbertv2"
NUM_LABELS = 5
device = get_device()
print(f"device={device}  start={time.strftime('%H:%M:%S')}", flush=True)


def resolve_csvs(ratio):
    base = r"D:/dacd2026/1_data/processed"
    v2 = r"D:/dacd2026/3_experiments/v2_runs"
    if ratio == "100_1":
        return (base + "/dacd_bench_big_train.csv", base + "/dacd_bench_big_val.csv")
    if ratio == "1000_1":
        tr = va = None
        for cand in [v2 + "/dacd_bench_1000_1_train.csv", base + "/dacd_bench_1000_1_train.csv"]:
            if os.path.isfile(cand):
                tr = cand
                break
        for cand in [v2 + "/dacd_bench_1000_1_val.csv", base + "/dacd_bench_1000_1_val.csv"]:
            if os.path.isfile(cand):
                va = cand
                break
        return tr, va
    if ratio == "7000_1":
        tr = va = None
        for cand in [v2 + "/dacd_bench_7000_1_train.csv", base + "/dacd_bench_7000_1_train.csv"]:
            if os.path.isfile(cand):
                tr = cand
                break
        for cand in [v2 + "/dacd_bench_7000_1_val.csv", base + "/dacd_bench_7000_1_val.csv"]:
            if os.path.isfile(cand):
                va = cand
                break
        return tr, va
    raise ValueError(ratio)


def make_model(method):
    if method == "marbert":
        return SingleEncoderClassifier(MARBERT_PATH, num_labels=NUM_LABELS).to(device)
    if method == "ceda":
        return CEDAClassifier(ARABERT_PATH, MARBERT_PATH, num_labels=NUM_LABELS, use_teacher=True).to(device)
    return SingleEncoderClassifier(ARABERT_PATH, num_labels=NUM_LABELS).to(device)


def make_loss(method, labels, epochs):
    cw = class_balanced_weights(labels, NUM_LABELS).to(device)
    counts = np.bincount(labels, minlength=NUM_LABELS)
    if method == "ce":
        return CrossEntropyLoss(NUM_LABELS, class_weights=cw).to(device)
    if method == "focal":
        return FocalLoss(NUM_LABELS, gamma=2.0, class_weights=cw).to(device)
    if method == "cb":
        return CrossEntropyLoss(NUM_LABELS, class_weights=cw).to(device)
    if method == "dacd":
        return DACDLoss(NUM_LABELS, lam=0.5, beta=0.3, temperature=0.07, class_weights=cw).to(device)
    if method == "dacdpp":
        return PrototypeLoss(NUM_LABELS, temperature=0.07, alpha=0.5, class_weights=cw).to(device)
    if method == "ldam":
        return LDAMLoss(NUM_LABELS, max_m=0.5, s=30.0, class_weights=cw).to(device)
    if method == "la":
        priors = np.bincount(labels, minlength=NUM_LABELS).astype(np.float32)
        priors = priors / priors.sum()
        return LogitAdjustmentLoss(NUM_LABELS, class_priors=priors, tau=1.0, class_weights=cw).to(device)
    if method == "recl":
        return ReCLLoss(NUM_LABELS, alpha=0.5, temperature=0.07, class_weights=cw).to(device)
    if method == "l2c":
        return L2CLoss(NUM_LABELS, lam=0.5, alpha_ling=0.7, base_beta=0.3, temperature=0.07, class_weights=cw).to(device)
    if method == "ami":
        return AMILoss(NUM_LABELS, max_m=0.5, s=30.0, alpha=1.0, class_counts=counts.tolist(), class_weights=cw).to(device)
    if method == "arc":
        base = DACDLoss(NUM_LABELS, lam=0.5, beta=0.3, temperature=0.07, class_weights=cw)
        return ARCLoss(base, num_epochs=epochs, tau_start=0.3, tau_target=0.07, beta_start=0.7, beta_target=0.3).to(device)
    if method == "ceda":
        base = DACDLoss(NUM_LABELS, lam=0.5, beta=0.3, temperature=0.07, class_weights=cw)
        return CEDALoss(NUM_LABELS, base, T=2.0, alpha_kd=0.4, alpha_align=0.2, class_weights=cw).to(device)
    if method == "dacdv5":
        return DACDv5Loss(NUM_LABELS, lam=0.5, alpha_ling=0.7, base_beta=0.3, temperature=0.07, max_m=0.5, s=30.0, ami_alpha=1.0, class_counts=counts.tolist(), class_weights=cw).to(device)
    if method == "marbert":
        return CrossEntropyLoss(NUM_LABELS, class_weights=cw).to(device)
    raise ValueError(method)


def run_one(method, ratio, seed, epochs=3, batch_size=16, max_length=96):
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
    labels = df_tr["label"].astype(int).tolist()
    counter = Counter(labels)
    if min(counter.values()) < 2:
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    else:
        weights = [1.0 / counter[l] for l in labels]
        sampler = WeightedRandomSampler(weights, num_samples=n_tr, replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, sampler=sampler)
    model = make_model(method)
    loss_fn = make_loss(method, labels, epochs)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5, weight_decay=0.01)
    is_dual = method == "ceda"
    is_arc = method in ("arc", "dacdv5")
    history = []
    t0_all = time.time()
    for epoch in range(epochs):
        if is_arc and hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(epoch)
        model.train()
        t_epoch = time.time()
        running, n = 0.0, 0
        for batch_idx, batch in enumerate(tr_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["label"].to(device)
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()
            if is_dual:
                s_logits, s_feats, t_logits, t_feats = model.forward_with_teacher(ids, mask)
                loss = loss_fn(s_logits, s_feats, y, t_logits, t_feats)
            else:
                if "features" in loss_fn.forward.__code__.co_varnames:
                    feats = model.features(ids, mask)
                    logits = model.classifier(feats)
                    loss = loss_fn(logits, y, features=feats)
                else:
                    logits = model(ids, mask)
                    loss = loss_fn(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * ids.size(0); n += ids.size(0)
        train_loss = running / max(1, n)
        model.eval(); preds, labs = [], []
        with torch.no_grad():
            for batch in va_loader:
                ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
                logits = model(ids, mask)
                preds.extend(logits.argmax(-1).cpu().tolist())
                labs.extend(batch["label"].cpu().tolist())
        macro = float(f1_score(labs, preds, average="macro"))
        per_class = f1_score(labs, preds, average=None, labels=list(range(NUM_LABELS)))
        rec = {"epoch": epoch, "train_loss": train_loss, "macro_f1": macro,
               "f1_Khaleji": float(per_class[0]), "f1_Iraqi": float(per_class[1]),
               "f1_Levantine": float(per_class[2]), "f1_Masri": float(per_class[3]),
               "f1_Maghrebi": float(per_class[4]),
               "elapsed_s": time.time() - t_epoch, "n_train": n_tr, "n_val": n_va}
        history.append(rec)
        print(f"  v5 {method} {ratio} seed={seed} ep{epoch+1}/{epochs} loss={train_loss:.4f} macro={macro:.4f} Kha={per_class[0]:.3f} Iraq={per_class[1]:.3f} Lev={per_class[2]:.3f} Mas={per_class[3]:.3f} Mag={per_class[4]:.3f} ({rec['elapsed_s']:.0f}s)", flush=True)
    out_dir = r"D:/dacd2026/5_paper_v5/results"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{method}_{ratio}_seed{seed}.json")
    with open(out_path, "w") as f:
        json.dump({"method": method, "ratio": ratio, "seed": seed, "epochs": epochs, "history": history, "total_time_s": time.time() - t0_all}, f, indent=2)
    print(f"  saved {out_path} total={time.time()-t0_all:.0f}s", flush=True)
    del model, loss_fn, opt
    torch.cuda.empty_cache(); gc.collect()
    return history


if __name__ == "__main__":
    args = sys.argv[1:]
    method = args[0]; ratio = args[1]; seed = int(args[2])
    epochs = int(args[3]) if len(args) > 3 else 3
    run_one(method, ratio, seed, epochs=epochs)
