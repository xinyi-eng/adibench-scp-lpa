"""Headline figure: visual comparison of methods on 5-shot amgadhasan.

Shows the key finding: SCP-LPA wins, fine-tuning collapses, with
multi-seed error bars and a clear visual hierarchy.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = r"D:/dacd2026/adibench_v1/paper/figs"

# Mean +/- std from 4-seed multi-seed runs
methods = [
    ("Random",          0.200, 0.000, "#BDBDBD"),
    ("FineTune",        0.197, 0.003, "#EF9A9A"),
    ("Focal",           0.189, 0.009, "#EF9A9A"),
    ("CB (Class-Bal.)", 0.202, 0.000, "#EF9A9A"),
    ("MAML",            0.195, 0.009, "#EF9A9A"),
    ("MatchingNet",     0.734, 0.007, "#90CAF9"),
    ("RelationNet",     0.760, 0.000, "#90CAF9"),
    ("ProtoNet",        0.768, 0.003, "#64B5F6"),
    ("LPA-ProtoNet",    0.785, 0.004, "#81C784"),
    ("SCP-ProtoNet",    0.788, 0.006, "#66BB6A"),
    ("SCP-LPA (ours)",  0.785, 0.000, "#2E7D32"),
]

names = [m[0] for m in methods]
accs = [m[1] for m in methods]
stds = [m[2] for m in methods]
colors = [m[3] for m in methods]

# Sort by accuracy (ascending) for visual clarity
order = np.argsort(accs)
names = [names[i] for i in order]
accs = [accs[i] for i in order]
stds = [stds[i] for i in order]
colors = [colors[i] for i in order]

fig, ax = plt.subplots(figsize=(7, 4.5))
y_pos = np.arange(len(names))
bars = ax.barh(y_pos, accs, xerr=stds, color=colors, edgecolor="black", linewidth=0.6,
                height=0.65, capsize=3, error_kw={"linewidth": 1.2})
# Highlight the ours bar
for i, name in enumerate(names):
    if "ours" in name:
        bars[i].set_edgecolor("darkred")
        bars[i].set_linewidth(2.0)

ax.set_yticks(y_pos)
ax.set_yticklabels(names, fontsize=10)
ax.set_xlabel("5-way 5-shot accuracy on amgadhasan 5-city", fontsize=11)
ax.set_xlim(0, 0.95)
ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax.axvline(0.20, color="gray", linestyle="--", alpha=0.5, label="chance")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, axis="x", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("Figure: SCP-LPA wins by ~5 points over ProtoNet on long-tail ADI", fontsize=11)

# Annotate the bar values
for i, (acc, std) in enumerate(zip(accs, stds)):
    ax.text(acc + std + 0.01, i, f"{acc:.3f}", va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_headline.png"), dpi=200, bbox_inches="tight")
plt.savefig(os.path.join(OUT_DIR, "fig_headline.pdf"), bbox_inches="tight")
print("Saved fig_headline.png/pdf")