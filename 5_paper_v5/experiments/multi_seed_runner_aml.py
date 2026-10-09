"""AML-DeBias aware runner for DACD (Adaptive Majority-Language De-Biasing).

Run from inside `experiments/`. Same interface as `multi_seed_runner.py` but
adds an optional AML mode and supports `dacd_aml_X_Y` style method flags.

Examples:
  python3 multi_seed_runner_aml.py dacd_aml 100_1 42 1
  python3 multi_seed_runner_aml.py dacd_aml 100_1 42 1 aml_alpha=2.0
  python3 multi_seed_runner_aml.py dacd 100_1 42 1   # fallback (no AML)

AML mode wraps the original DACDLoss (from dacd.losses) and uses a per-batch
per-class gradient-norm ratio to compute beta_ij adaptively (Eq. 2 of the v4
paper). The state is updated by the trainer after each batch.

Output: D:/dacd2026/3_experiments/v2_runs/dacd_aml_<ratio>_seed<N>.json
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer
from sklearn.metrics import f1_score

sys.path.insert(0, r'D:/dacd2026/2_models/dacd++')
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

BASE_DATA = r'D:/dacd2026/1_data/processed'
V2_DATA = r'D:/dacd2026/3_experiments/v2_runs'


def resolve_csvs(ratio):
    if ratio == '100_1':
        return (r'D:/dacd2026/1_data/processed/dacd_bench_big_train.csv',
                r'D:/dacd2026/1_data/processed/dacd_bench_big_val.csv')
    elif ratio == '1000_1':
        tr = os.path.join(V2_DATA, 'dacd_bench_1000_1_train.csv')
        va = os.path.join(V2_DATA, 'dacd_bench_1000_1_val.csv')
        if not os.path.isfile(tr): tr = r'D:/dacd2026/1_data/processed/dacd_bench_1000_1_train.csv'
        if not os.path.isfile(va): va = r'D:/dacd2026/1_data/processed/dacd_bench_1000_1_val.csv'
        return tr, va
    elif ratio == '7000_1':
        tr = os.path.join(V2_DATA, 'dacd_bench_7000_1_train.csv')
        va = os.path.join(V2_DATA, 'dacd_bench_7000_1_val.csv')
        if not os.path.isfile(tr): tr = r'D:/dacd2026/1_data/processed/dacd_bench_7000_1_train.csv'
        if not os.path.isfile(va): va = r'D:/dacd2026/1_data/processed/dacd_bench_7000_1_val.csv'
        return tr, va
    raise ValueError(ratio)


class AMLDACDLoss(nn.Module):
    """Wraps DACDLoss with AML-DeBias: per-batch adaptive Khaleeji-vs-other
    negative-pair weight based on per-class gradient L2 norm ratio.

    Specifically, for each (i, j) negative pair with y_j == majority:
        beta_ij = beta0 * min(r_{yi}^{alpha}, 1)
    where r_y = g_y / g_majority is the gradient-norm ratio of class y
    against the majority class. r_y < 1 means minority under-represented
    => beta shrinks => Khaleeji's negative pull on the same anchor weakens
    => minority gradient grows.

    Falls back to fixed beta when grad-norms state is empty (eval / first batch).
    """

    def __init__(self, base: DACDLoss, alpha=1.0, majority=0):
        super().__init__()
        self.base = base
        self.alpha = alpha
        self.majority = majority
        self._grad_norms = {}

    def update_grad_norms(self, features, targets):
        """Recompute per-class gradient norm stats from `features`."""
        self._grad_norms.clear()
        if features is None:
            return
        feats = features.detach().requires_grad_(True)
        B = feats.size(0)
        # off-diagonal sum as a proxy supervision for gradient magnitude
        sim = (feats @ feats.t()) / self.base.temperature
        per_sample_loss = ((sim - torch.eye(B, device=feats.device))).sum(-1) / max(B - 1, 1)
        loss = per_sample_loss.sum()
        grads = torch.autograd.grad(loss, feats, retain_graph=False, create_graph=False)[0]
        norms = grads.norm(dim=-1).detach()
        for c in targets.unique().tolist():
            mask = (targets == c)
            if mask.sum() > 0:
                self._grad_norms[int(c)] = float(norms[mask].mean().item())

    def _beta_for(self, c_minor, c_major):
        if not self._grad_norms:
            return self.base.beta  # safety net
        g_min = self._grad_norms.get(int(c_minor), 1.0)
        g_maj = self._grad_norms.get(int(c_major), 1.0)
        if g_maj <= 1e-9:
            return self.base.beta
        ratio = max(g_min / g_maj, 1e-6)
        return float(self.base.beta * min(ratio ** self.alpha, 1.0))

    def forward(self, logits, targets, features=None):
        if features is None or not self._grad_norms:
            # Pure CE / vanilla path
            return self.base(logits, targets, features=None) if features is None else \
                   self._forward_with_features(logits, targets, features, fallback=True)
        return self._forward_with_features(logits, targets, features, fallback=False)

    def _forward_with_features(self, logits, targets, features, fallback):
        loss_ce = self.base.ce(logits, targets)
        feats = F.normalize(features, dim=-1)
        B = feats.size(0)
        sim = feats @ feats.t() / self.base.temperature
        y = targets.unsqueeze(0) == targets.unsqueeze(1)
        eye = torch.eye(B, dtype=torch.bool, device=feats.device)
        pos_mask = y & ~eye
        neg_mask = ~y

        if fallback:
            kh_neg = (neg_mask & (targets.unsqueeze(1) == self.majority))
            other_neg = neg_mask & ~kh_neg
            weighted_neg = kh_neg.float() * self.base.beta + other_neg.float() * 1.0
        else:
            weighted_neg = neg_mask.float()
            for i in range(B):
                y_i = int(targets[i].item())
                if y_i == self.majority:
                    continue
                for j in range(B):
                    if i == j:
                        continue
                    if int(targets[j].item()) == self.majority:
                        weighted_neg[i, j] = self._beta_for(y_i, self.majority)

        exp_sim = torch.exp(sim) * (~eye).float()
        denom = (exp_sim * weighted_neg).sum(dim=-1) + 1e-12
        log_prob = sim - torch.log(denom)
        pos_count = pos_mask.float().sum(dim=-1)
        loss_supcon = -(log_prob * pos_mask.float()).sum(dim=-1) / (pos_count + 1e-12)
        if self.base.class_weights is not None:
            cw = self.base.class_weights.to(loss_supcon.device)[targets]
            loss_supcon = loss_supcon * cw
        loss_supcon = loss_supcon.mean()
        return (1 - self.base.lam) * loss_ce + self.base.lam * loss_supcon


def run_one(method, ratio, seed, epochs=1, batch_size=16, max_length=96,
            dacd_lam=0.5, dacd_beta=0.3, aml_alpha=1.0, log_grad=False):
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
    use_aml = method == 'dacd_aml'
    base_method = 'dacd' if use_aml else method
    if base_method == 'dacd':
        loss_fn = DACDLoss(NUM_LABELS, lam=dacd_lam, beta=dacd_beta,
                           temperature=0.07, class_weights=cb).to(device)
    elif base_method == 'dacdpp':
        loss_fn = DACDLoss(NUM_LABELS, lam=dacd_lam, beta=dacd_beta,
                           temperature=0.07, class_weights=cb).to(device)
    else:
        raise ValueError(f'{method} not supported by AML runner; use base runner')

    loss_fn = AMLDACDLoss(loss_fn, alpha=aml_alpha, majority=0)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    grad_log = []  # for fig_perclass_grad.png
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
            if use_aml:
                # 1st forward (eval-mode grad calculation, no update yet)
                feats = model.features(ids, mask); logits = model.classifier(feats)
                loss_fn.update_grad_norms(feats.detach(), y)
                if log_grad:
                    grad_log.append({int(c): loss_fn._grad_norms.get(int(c), float('nan'))
                                     for c in y.unique().tolist()})
                feats2 = model.features(ids, mask); logits2 = model.classifier(feats2)
                loss = loss_fn(logits2, y, features=feats2)
            else:
                feats = model.features(ids, mask); logits = model.classifier(feats)
                loss = loss_fn(logits, y, features=feats)
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
        }
        history.append(rec)
        print(f'  AML {method} {ratio} seed={seed} ep{epoch+1}/{epochs} loss={train_loss:.4f} '
              f'macro={macro:.4f} Kha={per_class[0]:.3f} Iraq={per_class[1]:.3f} '
              f'Lev={per_class[2]:.3f} Mas={per_class[3]:.3f} Mag={per_class[4]:.3f} '
              f'({rec["elapsed_s"]:.0f}s)', flush=True)
    out_dir = V2_DATA
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{method}_{ratio}_seed{seed}.json')
    with open(out_path, 'w') as f:
        json.dump({'method': method, 'ratio': ratio, 'seed': seed, 'epochs': epochs,
                   'aml_alpha': aml_alpha if use_aml else None,
                   'use_aml': use_aml, 'history': history,
                   'grad_log': grad_log if log_grad else None,
                   'total_time_s': time.time() - t0_all}, f, indent=2)
    print(f'  saved {out_path} total={time.time()-t0_all:.0f}s')
    del model, loss_fn, opt
    torch.cuda.empty_cache(); gc.collect()
    return history


if __name__ == '__main__':
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
