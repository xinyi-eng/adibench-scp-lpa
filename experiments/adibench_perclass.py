"""Per-class F1 analysis on NADI 18 5-shot.

For each query sample, compute per-class F1 using ProtoNet and
visualize per-class accuracy as a bar chart. This shows which
classes are hardest and confirms the long-tail collapse.
"""
import os, json, sys
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from collections import defaultdict

from adibench.baselines import FineTune, ProtoNet
from adibench.data import build_fewshot_episode, get_dataset

ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
RESULTS_DIR = r"D:/dacd2026/adibench_v1/results"
OUT_DIR = r"D:/dacd2026/adibench_v1/paper/figs"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)


def encode_batch(texts):
    enc = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return enc["input_ids"], enc["attention_mask"]


def per_class_acc_episodes(model, df, n_way, k_shot, q_query, n_episodes=200, seed=42):
    """Run n_episodes, return per-original-class accuracy."""
    import random
    rng = random.Random(seed)
    # orig_class -> [correct, total]
    per_class = defaultdict(lambda: [0, 0])
    model.eval()
    with torch.no_grad():
        for _ in range(n_episodes):
            s_idx, q_idx, s_lab, q_lab, chosen = _sample_with_orig(df, n_way, k_shot, q_query, rng)
            s_texts = [df.iloc[i]["text"] for i in s_idx]
            q_texts = [df.iloc[i]["text"] for i in q_idx]
            s_ids, s_mask = encode_batch(s_texts)
            q_ids, q_mask = encode_batch(q_texts)
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits, _, _ = model(s_ids, s_mask, s_y, q_ids, q_mask)
            preds = logits.argmax(-1).cpu().numpy()
            q_lab_np = np.array(q_lab)
            for i, q_idx_global in enumerate(q_idx):
                pred_ep = preds[i]
                true_ep = q_lab_np[i]
                orig = chosen[true_ep]  # original class for this episode-label
                per_class[orig][1] += 1
                if pred_ep == true_ep:
                    per_class[orig][0] += 1
    return per_class


def _sample_with_orig(df, n_way, k_shot, q_query, rng):
    """Like build_fewshot_episode but also returns original class labels."""
    by_class = defaultdict(list)
    for idx, label in zip(df.index, df["label"]):
        by_class[int(label)].append(idx)
    available = list(by_class.keys())
    if len(available) < n_way:
        chosen = rng.choices(available, k=n_way)
    else:
        chosen = rng.sample(available, n_way)
    s_idx, q_idx, s_lab, q_lab = [], [], [], []
    for ep_label, orig in enumerate(chosen):
        cands = by_class[orig]
        n_need = k_shot + q_query
        if len(cands) < n_need:
            sampled = rng.choices(cands, k=n_need)
        else:
            sampled = rng.sample(cands, n_need)
        s_idx.extend(sampled[:k_shot])
        q_idx.extend(sampled[k_shot:])
        s_lab.extend([ep_label] * k_shot)
        q_lab.extend([ep_label] * q_query)
    return s_idx, q_idx, s_lab, q_lab, chosen


def main():
    df, meta = get_dataset("nadi_18")
    print(f"Classes: {meta['num_classes']}")
    print(f"Class distribution (top 5 by count):")
    from collections import Counter
    counts = df["label"].value_counts().sort_values(ascending=False)
    print(counts.head(5))
    print(f"Imbalance ratio: {counts.iloc[0] / counts.iloc[-1]:.1f}:1")

    # Run ProtoNet and FineTune, then compare per-class accuracy
    print("\n=== Training ProtoNet ===")
    torch.manual_seed(42)
    proto = ProtoNet().to(device)
    trainable = [p for p in proto.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    import random
    rng_tr = random.Random(42)
    proto.train()
    for ep in range(200):
        s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, 5, 5, 15)
        s_texts = [df.iloc[i]["text"] for i in s_idx]
        q_texts = [df.iloc[i]["text"] for i in q_idx]
        s_ids, s_mask = encode_batch(s_texts)
        q_ids, q_mask = encode_batch(q_texts)
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        logits, _, _ = proto(s_ids, s_mask, s_y, q_ids, q_mask)
        loss = F.cross_entropy(logits, q_y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()

    print("\n=== Training FineTune ===")
    torch.manual_seed(42)
    ft = FineTune(num_classes=5).to(device)
    trainable = [p for p in ft.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    ft.train()
    for ep in range(200):
        s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, 5, 5, 15)
        s_texts = [df.iloc[i]["text"] for i in s_idx]
        q_texts = [df.iloc[i]["text"] for i in q_idx]
        s_ids, s_mask = encode_batch(s_texts)
        q_ids, q_mask = encode_batch(q_texts)
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        loss = ft.loss(s_ids, s_mask, s_y, q_ids, q_mask, q_y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()

    # Per-class evaluation
    print("\n=== Evaluating per-class ===")
    pc_proto = per_class_acc_episodes(proto, df, 5, 5, 15, n_episodes=200, seed=12345)
    pc_ft = per_class_acc_episodes(ft, df, 5, 5, 15, n_episodes=200, seed=12345)

    # Class sizes (for ordering by size)
    cls_size = df["label"].value_counts().to_dict()
    classes_sorted = sorted(pc_proto.keys(), key=lambda c: -cls_size.get(c, 0))
    class_names = ["OM", "SD", "SA", "KW", "QA", "LB", "JO", "SY", "IQ", "MA",
                   "EG", "PL", "YE", "BH", "DZ", "AE", "TN", "LY"]
    proto_accs = [pc_proto[c][0] / max(1, pc_proto[c][1]) for c in classes_sorted]
    ft_accs = [pc_ft[c][0] / max(1, pc_ft[c][1]) for c in classes_sorted]
    sizes = [cls_size.get(c, 0) for c in classes_sorted]
    names = [class_names[c] if c < len(class_names) else f"C{c}" for c in classes_sorted]

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(classes_sorted))
    width = 0.38
    b1 = ax.bar(x - width/2, proto_accs, width, label="ProtoNet", color="#3b82f6")
    b2 = ax.bar(x + width/2, ft_accs, width, label="FineTune", color="#ef4444")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.set_xlabel("Class (sorted by training-set size)")
    ax.set_ylabel("Per-class accuracy (5-way 5-shot)")
    ax.set_title("Per-class accuracy on NADI 18 (5-way 5-shot, 200 eval episodes)")
    ax.axhline(0.20, color="gray", linestyle="--", label="Chance (0.20)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate class sizes
    for i, (b, s) in enumerate(zip(x, sizes)):
        ax.text(b, 0.01, f"{s//1000}k", ha="center", fontsize=8, color="gray")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig2_perclass.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig2_perclass.pdf"), bbox_inches="tight")
    print(f"\nSaved {OUT_DIR}/fig2_perclass.png")

    print("\nPer-class accuracy:")
    print(f"{'Class':<6}{'Size':<10}{'ProtoNet':<12}{'FineTune':<12}")
    for c, name, sz, pa, fa in zip(classes_sorted, names, sizes, proto_accs, ft_accs):
        print(f"{name:<6}{sz:<10}{pa:<12.3f}{fa:<12.3f}")


if __name__ == "__main__":
    main()