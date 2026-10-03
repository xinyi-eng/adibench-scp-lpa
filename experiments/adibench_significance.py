"""Paired bootstrap significance test for ADIBench.

Trains ProtoNet and MatchingNet quickly (200 episodes) on the same seed,
then evaluates on the SAME held-out episodes for paired comparison.
"""
import sys, os, json, time
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from adibench.baselines import ProtoNet, MatchingNet
from adibench.data import build_fewshot_episode, get_dataset

ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)


def encode_batch(texts):
    enc = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return enc["input_ids"], enc["attention_mask"]


def train_model(model, df, n_way, k_shot, q_query, n_episodes=200, lr=1e-4):
    """Quick training (200 episodes, no logging)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    model.train()
    for _ in range(n_episodes):
        s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)
        s_texts = [df.iloc[i]["text"] for i in s_idx]
        q_texts = [df.iloc[i]["text"] for i in q_idx]
        s_ids, s_mask = encode_batch(s_texts)
        q_ids, q_mask = encode_batch(q_texts)
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        logits, _, _ = model(s_ids, s_mask, s_y, q_ids, q_mask)
        loss = F.cross_entropy(logits, q_y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()


def paired_eval(model_a, model_b, df, n_way, k_shot, q_query, n_eval=200, seed=42):
    """Eval both models on the SAME episodes; return per-episode accs for each."""
    import random
    rng = random.Random(seed)
    accs_a, accs_b = [], []
    model_a.eval(); model_b.eval()
    with torch.no_grad():
        for _ in range(n_eval):
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(
                df, n_way, k_shot, q_query, rng=rng)
            s_texts = [df.iloc[i]["text"] for i in s_idx]
            q_texts = [df.iloc[i]["text"] for i in q_idx]
            s_ids, s_mask = encode_batch(s_texts)
            q_ids, q_mask = encode_batch(q_texts)
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits_a, _, _ = model_a(s_ids, s_mask, s_y, q_ids, q_mask)
            logits_b, _, _ = model_b(s_ids, s_mask, s_y, q_ids, q_mask)
            accs_a.append((logits_a.argmax(-1) == q_y).float().mean().item())
            accs_b.append((logits_b.argmax(-1) == q_y).float().mean().item())
    return np.array(accs_a), np.array(accs_b)


def bootstrap_paired(diffs, n_boot=10000):
    diffs_boot = np.random.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    p = 2 * min((diffs_boot <= 0).mean(), (diffs_boot >= 0).mean())
    return diffs.mean(), diffs.std(), p, np.percentile(diffs_boot, 2.5), np.percentile(diffs_boot, 97.5)


def main():
    datasets = ["nadi_18", "nadi_5", "amgadhasan_5"]
    k_shots = [1, 5]
    results = {}
    for ds in datasets:
        df, meta = get_dataset(ds)
        for k in k_shots:
            print(f"\n=== {ds} 5w{k}s ===")
            proto = ProtoNet().to(device)
            match = MatchingNet().to(device)
            # Quick train (same seed for both via shared Python random state)
            import random
            random.seed(42); np.random.seed(42); torch.manual_seed(42)
            t0 = time.time()
            train_model(proto, df, 5, k, 15, n_episodes=200)
            train_model(match, df, 5, k, 15, n_episodes=200)
            print(f"  trained both ({time.time()-t0:.0f}s)")
            # Paired eval
            t0 = time.time()
            pa, ma = paired_eval(proto, match, df, 5, k, 15, n_eval=200, seed=12345)
            print(f"  proto: {pa.mean():.4f} | match: {ma.mean():.4f}  eval ({time.time()-t0:.0f}s)")
            d = pa - ma
            mean, std, p, lo, hi = bootstrap_paired(d)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  diff = {mean:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  p={p:.4f} {sig}")
            results[f"{ds}_5w{k}s"] = {
                "proto_acc": pa.mean(), "match_acc": ma.mean(),
                "diff": mean, "ci_lo": lo, "ci_hi": hi, "p_value": p,
            }
            # cleanup
            del proto, match
            torch.cuda.empty_cache()

    with open(r"D:/dacd2026/adibench_v1/results/significance_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved significance_tests.json")


if __name__ == "__main__":
    main()