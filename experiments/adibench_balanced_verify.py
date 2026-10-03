"""Class-balanced few-shot fine-tune verification.

Hypothesis: support size (not class frequency) drives the collapse.

Experiment:
  1. Build a class-balanced subset of nadi_18: sample exactly N rows
     from each of the 18 classes (e.g. N=500), so the corpus is
     uniform.
  2. Run FineTune and ProtoNet on this uniform corpus, 5-way k-shot,
     same protocol as the main table.
  3. Compare with the imbalanced numbers from the main table.

If FineTune STILL collapses to chance on the uniform corpus, the
hypothesis is confirmed: support size, not class imbalance, is the
root cause.
"""
import sys, os, json
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from adibench.data import build_fewshot_episode, get_dataset
from adibench.baselines import FineTune, ProtoNet

ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)


def encode(texts):
    e = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return e["input_ids"], e["attention_mask"]


def build_balanced(df, n_per_class, seed=42):
    """Sample n_per_class rows per label, return the balanced df."""
    rng = np.random.RandomState(seed)
    parts = []
    for lab in sorted(df["label"].unique()):
        sub = df[df["label"] == lab].reset_index(drop=True)
        idx = rng.choice(len(sub), size=min(n_per_class, len(sub)), replace=False)
        parts.append(sub.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def train(model, df, n_way, k_shot, q_query, n_episodes=500, has_loss_attr=True):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    for _ in range(n_episodes):
        s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)
        s_ids, s_mask = encode([df.iloc[i]["text"] for i in s_idx])
        q_ids, q_mask = encode([df.iloc[i]["text"] for i in q_idx])
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        if has_loss_attr:
            loss = model.loss(s_ids, s_mask, s_y, q_ids, q_mask, q_y)
        else:
            # ProtoNet: forward returns (logits, sf, qf)
            out = model(s_ids, s_mask, s_y, q_ids, q_mask)
            logits = out[0] if isinstance(out, tuple) else out
            loss = F.cross_entropy(logits, q_y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()


def eval_model(model, df, n_way, k_shot, q_query, n_eval=100, seed=42, return_logits=True):
    import random
    rng = random.Random(seed)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_eval):
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query, rng=rng)
            s_ids, s_mask = encode([df.iloc[i]["text"] for i in s_idx])
            q_ids, q_mask = encode([df.iloc[i]["text"] for i in q_idx])
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            out = model(s_ids, s_mask, s_y, q_ids, q_mask)
            logits = out[0] if isinstance(out, tuple) else out
            correct += (logits.argmax(-1) == q_y).sum().item()
            total += q_y.size(0)
    return correct / max(1, total)


def main():
    # Use nadi_18: 18 classes, imbalanced ratio 5.8x.
    # Build uniform subset of N=500 per class.
    df_full, meta = get_dataset("nadi_18")
    df_bal = build_balanced(df_full, n_per_class=500, seed=42)
    print(f"Balanced subset: {len(df_bal)} rows, {df_bal['label'].nunique()} classes "
          f"(imbalance ratio: 1.0x)")

    results = {}
    for k_shot in [1, 5]:
        print(f"\n=== Balanced nadi_18, 5-way {k_shot}-shot ===")
        # FineTune
        ft = FineTune(num_classes=meta["num_classes"]).to(device)
        train(ft, df_bal, 5, k_shot, 15, has_loss_attr=True)
        ft_acc = eval_model(ft, df_bal, 5, k_shot, 15)
        del ft; torch.cuda.empty_cache()
        # ProtoNet
        pn = ProtoNet().to(device)
        train(pn, df_bal, 5, k_shot, 15, has_loss_attr=False)
        pn_acc = eval_model(pn, df_bal, 5, k_shot, 15)
        del pn; torch.cuda.empty_cache()
        # Reference (imbalanced main table numbers)
        im_ft = {1: 0.203, 5: 0.207}[k_shot]   # nadi_18 5-way from summary
        im_pn = {1: 0.388, 5: 0.540}[k_shot]
        print(f"  FineTune:  balanced={ft_acc:.4f}   imbalanced_ref={im_ft:.3f}")
        print(f"  ProtoNet:  balanced={pn_acc:.4f}   imbalanced_ref={im_pn:.3f}")
        results[f"bal_ft_{k_shot}shot"] = ft_acc
        results[f"bal_pn_{k_shot}shot"] = pn_acc
        results[f"imb_ft_{k_shot}shot"] = im_ft
        results[f"imb_pn_{k_shot}shot"] = im_pn

    with open(r"D:/dacd2026/adibench_v1/results/balanced_verify.json", "w") as f:
        json.dump(results, f, indent=2)
    # Verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    for k in [1, 5]:
        ft = results[f"bal_ft_{k}shot"]
        if ft < 0.25:
            print(f"[{k}-shot] FineTune collapses to chance EVEN ON BALANCED CORPUS "
                  f"({ft:.3f} ≈ 0.20 chance). "
                  f"HYPOTHESIS CONFIRMED: support size, not class frequency, drives collapse.")
        else:
            print(f"[{k}-shot] FineTune works on balanced ({ft:.3f} > 0.25). "
                  f"Class imbalance WAS the main driver.")
    print("\nSaved balanced_verify.json")


if __name__ == "__main__":
    main()