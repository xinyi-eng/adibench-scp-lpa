"""Generate Figure 3: dialectal distance heatmaps for all 3 datasets."""
import os
import sys
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from adibench.dialects import DIALECT_DISTANCE_MATRICES

OUT_DIR = r"D:/dacd2026/adibench_v1/paper/figs"
os.makedirs(OUT_DIR, exist_ok=True)

datasets = ["nadi_18", "nadi_5", "amgadhasan_5"]
titles = {
    "nadi_18": "(a) NADI 2024 18-way (country)",
    "nadi_5": "(b) NADI 2024 5-way (coarse)",
    "amgadhasan_5": "(c) amgadhasan 5-city",
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
cmap = mpl.colormaps.get_cmap("viridis_r")
for ax, ds in zip(axes, datasets):
    D, names = DIALECT_DISTANCE_MATRICES[ds]
    N = len(names)
    # Annotation size shrinks with N
    fs = 8 if N > 8 else 10
    im = ax.imshow(D, cmap=cmap, vmin=0.0, vmax=0.95)
    ax.set_xticks(range(N))
    ax.set_xticklabels(names, rotation=45 if N > 5 else 0, fontsize=fs, ha="right" if N > 5 else "center")
    ax.set_yticks(range(N))
    ax.set_yticklabels(names, fontsize=fs)
    ax.set_title(titles[ds])
    # Annotate cells
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            v = D[i, j]
            txt = ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                          color="white" if v > 0.45 else "black",
                          fontsize=fs - 1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dialect distance")

plt.suptitle("Figure 3: Arabic dialectal distance matrices used by LPA-ProtoNet", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_dialect_heatmap.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "fig3_dialect_heatmap.pdf"), bbox_inches="tight")
print(f"Saved {OUT_DIR}/fig3_dialect_heatmap.png and .pdf")