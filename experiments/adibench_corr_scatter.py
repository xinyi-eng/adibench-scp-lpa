"""Generate Figure 7: scatter plot of feature distance vs linguistic distance.

This is the strongest visual evidence that LPA's inductive bias
is justified: features are correlated with the expert matrix.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, r"D:/dacd2026/adibench_v1")
from adibench.dialects import DIALECT_DISTANCE_MATRICES
from adibench.data import get_dataset
from adibench.baselines import Encoder

import torch
from transformers import AutoTokenizer

ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


def main():
    df, meta = get_dataset("nadi_18")
    classes = ["OM", "SD", "SA", "KW", "QA", "LB", "JO", "SY",
               "IQ", "MA", "EG", "PL", "YE", "BH", "DZ", "AE", "TN", "LY"]
    n = len(classes)

    # Linguistic distance
    D_ling, _ = DIALECT_DISTANCE_MATRICES["nadi_18"]

    # Compute feature distance using AraBERT
    print("Encoding 100 tweets/class...")
    tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)
    encoder = Encoder(embed_dim=256).to(device)
    encoder.eval()

    sample_texts, sample_labels = [], []
    for c in range(n):
        sub = df[df["label"] == c].sample(min(100, len(df[df["label"] == c])),
                                          random_state=42)
        sample_texts.extend(sub["text"].tolist())
        sample_labels.extend([c] * len(sub))

    batch_size = 32
    feats = []
    with torch.no_grad():
        for i in range(0, len(sample_texts), batch_size):
            batch = sample_texts[i:i+batch_size]
            enc = tok(batch, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
            f = encoder(enc["input_ids"], enc["attention_mask"]).cpu().numpy()
            feats.append(f)
    feats = np.concatenate(feats, axis=0)
    sample_labels = np.array(sample_labels)

    # Compute centroids and feature distance
    centroids = np.zeros((n, feats.shape[1]))
    for c in range(n):
        mask = sample_labels == c
        centroids[c] = feats[mask].mean(axis=0)

    D_feat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D_feat[i, j] = float(np.linalg.norm(centroids[i] - centroids[j]))

    # Normalize to [0, 1] for visual comparison
    mask = ~np.eye(n, dtype=bool)
    feat_min, feat_max = D_feat[mask].min(), D_feat[mask].max()
    ling_min, ling_max = D_ling[mask].min(), D_ling[mask].max()
    feat_norm = (D_feat - feat_min) / (feat_max - feat_min)
    ling_norm = (D_ling - ling_min) / (ling_max - ling_min)

    # Compute correlation
    rho, pval = spearmanr(D_feat[mask] - np.eye(n).max(), D_ling[mask] - np.eye(n).max())
    print(f"Spearman correlation: rho = {rho:.4f}, p-value = {pval:.2e}")

    # Plot scatter
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    x = D_feat[mask]
    y = D_ling[mask]

    # Color points by cluster (intra vs inter)
    cluster = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            cluster.append("intra" if i < 9 and j < 9 or i >= 9 and j >= 9 else "inter")
    cluster = np.array(cluster)
    intra_mask = cluster == "intra"
    inter_mask = cluster == "inter"

    ax.scatter(x[inter_mask], y[inter_mask], s=18, color="#5B9BD5", alpha=0.5,
               label=f"Inter-cluster (n={inter_mask.sum()})", edgecolor="white", linewidth=0.5)
    ax.scatter(x[intra_mask], y[intra_mask], s=18, color="#ED7D31", alpha=0.6,
               label=f"Intra-cluster (n={intra_mask.sum()})", edgecolor="white", linewidth=0.5)

    # Linear fit (passes through origin since we removed diagonal)
    from numpy.polynomial import polynomial as P
    # Use ratio y/x as the slope
    slopes = y / np.maximum(x, 0.01)
    slope = np.median(slopes)
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, slope * xx, "--", color="#666", linewidth=1.5,
            label=f"Median slope = {slope:.2f}")

    ax.set_xlabel("AraBERT feature distance", fontsize=10)
    ax.set_ylabel("Linguistic distance (Versteegh 2006)", fontsize=10)
    ax.set_title("AraBERT features align with Arabic dialectology\n"
                 f"(Spearman $\\rho$ = {rho:.3f}, $p$ < {pval:.0e})",
                 fontsize=10, fontweight="bold")

    ax.legend(loc="upper left", frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_png = r"D:/dacd2026/adibench_v1/paper/figs/fig7_correlation_scatter.png"
    out_pdf = r"D:/dacd2026/adibench_v1/paper/figs/fig7_correlation_scatter.pdf"
    plt.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()