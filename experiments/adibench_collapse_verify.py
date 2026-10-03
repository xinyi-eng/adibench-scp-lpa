"""Failure-mode verification: prove fine-tuning collapses to majority-class
prediction by inspecting its prediction distribution (not just low accuracy).

Expected: >90% of fine-tuned predictions are concentrated on 1-2 majority
classes; metric learning spreads predictions more evenly.
"""
import sys, os, json
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
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


def train(model, df, n_way, k_shot, q_query, n_episodes=500, loss_fn=None):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    for _ in range(n_episodes):
        s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)
        s_ids, s_mask = encode([df.iloc[i]["text"] for i in s_idx])
        q_ids, q_mask = encode([df.iloc[i]["text"] for i in q_idx])
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        loss = model.loss(s_ids, s_mask, s_y, q_ids, q_mask, q_y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()


def prediction_distribution(model, df, n_way, k_shot, q_query, n_eval=200):
    """Return the predicted-class histogram over n_eval episodes."""
    model.eval()
    hist = np.zeros(n_way)
    with torch.no_grad():
        for _ in range(n_eval):
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)
            s_ids, s_mask = encode([df.iloc[i]["text"] for i in s_idx])
            q_ids, q_mask = encode([df.iloc[i]["text"] for i in q_idx])
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits, _, _ = model(s_ids, s_mask, s_y, q_ids, q_mask)
            preds = logits.argmax(-1).cpu().numpy()
            for p in preds:
                hist[p] += 1
    return hist


def main():
    datasets = ["nadi_18", "nadi_5", "amgadhasan_5"]
    results = {}
    print("=" * 70)
    print("Failure-mode verification: prediction concentration (5-way 5-shot)")
    print("=" * 70)
    for ds in datasets:
        df, meta = get_dataset(ds)
        # FineTune
        ft = FineTune(num_classes=meta["num_classes"]).to(device)
        train(ft, df, 5, 5, 15)
        ft_hist = prediction_distribution(ft, df, 5, 5, 15)
        # ProtoNet (reference)
        pn = ProtoNet().to(device)
        train(pn, df, 5, 5, 15)
        pn_hist = prediction_distribution(pn, df, 5, 5, 15)

        # Concentration metrics: % in top-2 classes
        ft_norm = ft_hist / max(1, ft_hist.sum())
        pn_norm = pn_hist / max(1, pn_hist.sum())
        ft_top2 = np.sort(ft_norm)[-2:].sum()
        pn_top2 = np.sort(pn_norm)[-2:].sum()
        print(f"\n{ds}:")
        print(f"  FineTune pred dist: {np.round(ft_norm, 2)}  (top-2 = {ft_top2*100:.0f}%)")
        print(f"  ProtoNet pred dist: {np.round(pn_norm, 2)}  (top-2 = {pn_top2*100:.0f}%)")
        results[ds] = {
            "finetune_dist": ft_hist.tolist(),
            "protonet_dist": pn_hist.tolist(),
            "finetune_top2_frac": float(ft_top2),
            "protonet_top2_frac": float(pn_top2),
        }
        del ft, pn
        torch.cuda.empty_cache()

    with open(r"D:/dacd2026/adibench_v1/results/collapse_verify.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved collapse_verify.json")
    # Summary: is FineTune's top-2 fraction >> ProtoNet's?
    for ds, r in results.items():
        gap = r["finetune_top2_frac"] - r["protonet_top2_frac"]
        print(f"  {ds}: FineTune top-2 {r['finetune_top2_frac']*100:.0f}% vs ProtoNet {r['protonet_top2_frac']*100:.0f}% (gap {gap*100:+.0f}pp)")


if __name__ == "__main__":
    main()