"""Render a real t-SNE figure from extracted last-layer features.

Reads:
  D:/dacd2026/3_experiments/v2_runs/features_<method>_<ratio>_seed<N>.npz
Writes:
  experiments/figs/fig_tsne.png  (replaces synthetic version)

If fewer than 2 npz files exist, falls back to synthetic illustrative data.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

V2 = r'D:/dacd2026/3_experiments/v2_runs'
OUT = os.path.join(os.path.dirname(__file__), 'figs')
os.makedirs(OUT, exist_ok=True)
CLASS_NAMES = ['Khaleeji', 'Iraqi', 'Levantine', 'Masri', 'Maghrebi']
COLORS = ['#D7191C', '#2C7BB6', '#F4A582', '#7BAFD6', '#808080']


def load_npz(method, ratio='100_1', seed=42):
    path = os.path.join(V2, f'features_{method}_{ratio}_seed{seed}.npz')
    if not os.path.isfile(path):
        return None
    d = np.load(path)
    return d['features'], d['labels'].astype(int)


def main():
    feats_a = load_npz('ce')
    feats_b = load_npz('dacd')
    if feats_a is None or feats_b is None:
        print('WARN: feature npz files missing; skipping real t-SNE render')
        return

    from sklearn.manifold import TSNE
    Xa, ya = feats_a
    Xb, yb = feats_b
    # Per-class subsample to keep plot readable
    rng = np.random.RandomState(42)
    def sub(X, y, k=80):
        out_X = []; out_y = []
        for c in np.unique(y):
            mask = (y == c)
            idx = np.where(mask)[0]
            if len(idx) > k:
                idx = rng.choice(idx, k, replace=False)
            out_X.append(X[idx]); out_y.append(y[idx])
        return np.concatenate(out_X), np.concatenate(out_y)
    Xa_s, ya_s = sub(Xa, ya); Xb_s, yb_s = sub(Xb, yb)

    # t-SNE on combined then split by method
    print('running t-SNE on combined features...')
    combined = np.concatenate([Xa_s, Xb_s], axis=0)
    combined_2d = TSNE(n_components=2, init='pca', random_state=42,
                       perplexity=30).fit_transform(combined)
    n_a = len(Xa_s)
    pts_a = combined_2d[:n_a]; pts_b = combined_2d[n_a:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, pts, lbls, title in [
        (axes[0], pts_a, ya_s, '(a) Vanilla CE'),
        (axes[1], pts_b, yb_s, '(b) DACD (with adaptive β)'),
    ]:
        for i, n in enumerate(CLASS_NAMES):
            mask = (lbls == i)
            ax.scatter(pts[mask, 0], pts[mask, 1],
                       c=COLORS[i], label=n, alpha=0.6, s=12, edgecolors='white', linewidth=0.3)
        ax.set_title(title)
        ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
        ax.legend(fontsize=7, loc='best')
        ax.grid(alpha=0.2)
    fig.suptitle('t-SNE of last-layer features at 100:1 (seed 42, 1 epoch)')
    fig.tight_layout()
    out_path = os.path.join(OUT, 'fig_tsne.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
