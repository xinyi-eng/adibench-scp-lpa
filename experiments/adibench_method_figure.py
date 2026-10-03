"""Generate a clean method overview figure for SCP-LPA.

Two-panel figure:
  (a) The full SCP-LPA pipeline (encoder + SupCon + LPA aggregation).
  (b) Worked example: 5 dialects from NADI 5, showing how LPA
      mixes prototypes with weights derived from the dialect matrix.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT_DIR = r"D:/dacd2026/adibench_v1/paper/figs"
os.makedirs(OUT_DIR, exist_ok=True)


def box(ax, xy, w, h, text, fc="#E8F0FE", ec="#1F4E79", lw=1.5, fontsize=10, fontweight="normal"):
    p = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color="#1F1F1F")


def arrow(ax, p1, p2, label="", color="#1F4E79"):
    a = FancyArrowPatch(p1, p2, arrowstyle="->", mutation_scale=15,
                        color=color, lw=1.2)
    ax.add_patch(a)
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.05, label, ha="center", va="bottom",
                fontsize=8, color=color, style="italic")


def main():
    fig = plt.figure(figsize=(11, 4.5))
    ax_a = fig.add_subplot(1, 2, 1)
    ax_b = fig.add_subplot(1, 2, 2)
    for ax in (ax_a, ax_b):
        ax.set_xlim(0, 10); ax.set_ylim(0, 5)
        ax.axis("off")

    # ============= (a) Method pipeline =============
    ax = ax_a
    ax.text(5, 4.7, "(a) SCP-LPA training and inference", ha="center", fontsize=12, fontweight="bold")
    # Input tweet
    box(ax, (0.2, 3.2), 1.6, 0.7, "Arabic\ntweet", fc="#FFF3E0", ec="#E65100")
    # Encoder
    box(ax, (2.2, 3.2), 1.6, 0.7, "AraBERT\n(4 layers\nunfrozen)", fc="#E8F5E9", ec="#2E7D32")
    # Feature
    box(ax, (4.2, 3.2), 1.4, 0.7, "Feature\n$z \\in \\mathbb{R}^{256}$", fc="#F3E5F5", ec="#6A1B9A")
    # Two branches: SupCon + LPA
    # Branch 1: SupCon pretraining
    box(ax, (0.5, 1.8), 2.0, 0.7, "SupCon\npre-training", fc="#E1F5FE", ec="#0277BD", fontweight="bold")
    box(ax, (3.0, 1.8), 1.7, 0.7, "Encoder\nadapted to\nclass clusters", fc="#E0F7FA", ec="#006064")
    # Branch 2: LPA
    box(ax, (5.0, 1.8), 2.0, 0.7, "Dialect\nmatrix\n$w(c,c')$", fc="#FFF8E1", ec="#F57F17")
    box(ax, (7.2, 1.8), 1.8, 0.7, "Mixed\nprototype\n$p_c^{LPA}$", fc="#FFF8E1", ec="#F57F17", fontweight="bold")
    # Distance to prototypes
    box(ax, (5.0, 0.3), 4.0, 0.7, "$-||z - p_c^{LPA}||^2$   (Euclidean distance)", fc="#FCE4EC", ec="#AD1457", fontweight="bold")
    box(ax, (0.5, 0.3), 1.7, 0.7, "Cross-entropy\nauxiliary", fc="#E1F5FE", ec="#0277BD")
    # Arrows
    arrow(ax, (1.8, 3.55), (2.2, 3.55))
    arrow(ax, (3.8, 3.55), (4.2, 3.55))
    arrow(ax, (4.9, 3.2), (1.5, 2.5), label="encoder")
    arrow(ax, (1.5, 1.8), (3.0, 1.8), label="loss")
    arrow(ax, (4.9, 3.2), (6.0, 2.5), label="features")
    arrow(ax, (6.0, 1.8), (7.2, 1.8), label="mix")
    arrow(ax, (4.9, 3.2), (7.0, 1.0), label="query feature")
    arrow(ax, (1.5, 1.0), (5.0, 0.65), label="SupCon loss", color="#0277BD")
    arrow(ax, (8.6, 0.65), (9.0, 0.65))

    # ============= (b) Worked example =============
    ax = ax_b
    ax.text(5, 4.7, "(b) LPA prototype aggregation on 5 dialects", ha="center", fontsize=12, fontweight="bold")
    # Five dialect labels
    classes = ["Khaleeji", "Iraqi", "Levantine", "Masri", "Maghrebi"]
    n = 5
    box_w, box_h = 1.4, 0.6
    spacing = 0.4
    total_w = n * box_w + (n - 1) * spacing
    start_x = (10 - total_w) / 2
    centers = []
    for i, c in enumerate(classes):
        x = start_x + i * (box_w + spacing)
        centers.append((x, 3.0))
        box(ax, (x, 3.0), box_w, box_h, c, fc="#E3F2FD", ec="#1565C0", fontweight="bold")

    # Class distances below
    ax.text(0.5, 2.0, "Linguistic distance $d_{ling}$:",
            ha="left", fontsize=9, fontweight="bold")
    # Distance matrix subset (5x5)
    d = np.array([
        [0.00, 0.55, 0.60, 0.70, 0.90],
        [0.55, 0.00, 0.50, 0.60, 0.85],
        [0.60, 0.50, 0.00, 0.45, 0.85],
        [0.70, 0.60, 0.45, 0.00, 0.70],
        [0.90, 0.85, 0.85, 0.70, 0.00],
    ])
    # Show weights = exp(-d / 0.5) (normalized)
    tau = 0.5
    w = np.exp(-d / tau)
    w = w / w.sum(axis=1, keepdims=True)
    # Show as small table
    cell_w, cell_h = 0.5, 0.4
    table_x, table_y = 0.5, 0.5
    for i in range(n):
        for j in range(n):
            v = w[i, j]
            color = "#FFFDE7" if v < 0.3 else ("#FFE082" if v < 0.6 else "#FF8A65")
            ax.add_patch(plt.Rectangle((table_x + j * cell_w, table_y + (n - 1 - i) * cell_h),
                                          cell_w, cell_h, fc=color, ec="gray", lw=0.5))
            ax.text(table_x + j * cell_w + cell_w / 2,
                    table_y + (n - 1 - i) * cell_h + cell_h / 2,
                    f"{v:.2f}", ha="center", va="center", fontsize=7)
    # Class label on the left
    for i, c in enumerate(classes):
        ax.text(table_x - 0.2, table_y + (n - 1 - i) * cell_h + cell_h / 2,
                c[:3], ha="right", va="center", fontsize=7, style="italic")
    # Column labels at the bottom
    for j, c in enumerate(classes):
        ax.text(table_x + j * cell_w + cell_w / 2, table_y - 0.15,
                c[:3], ha="center", va="top", fontsize=7, style="italic")
    # Title for the matrix
    ax.text(3.4, 0.5, "$\\leftarrow$  weights $w(c, c')$\n(self weight in red box)",
            ha="left", va="center", fontsize=8, color="#555")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig0_method.png"), dpi=200, bbox_inches="tight")
    plt.savefig(os.path.join(OUT_DIR, "fig0_method.pdf"), bbox_inches="tight")
    print("Saved fig0_method.png/pdf")


if __name__ == "__main__":
    main()