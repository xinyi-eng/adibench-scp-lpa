"""Render real per-class gradient norm trajectory from a captured log."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    path = r'D:/dacd2026/3_experiments/v2_runs/dacd_aml_100_1_gradnorm_seed42.json'
    if not os.path.isfile(path):
        # Fall back to aml runs from orchestrate_after
        candidates = [
            r'D:/dacd2026/3_experiments/v2_runs/dacd_aml_100_1_seed42.json',
            r'D:/dacd2026/3_experiments/v2_runs/dacd_100_1_seed42.json',
        ]
        for c in candidates:
            if os.path.isfile(c):
                path = c
                break
        if not os.path.isfile(path):
            print('WARN: no grad-norm log found; skipping render')
            return

    with open(path) as f:
        d = json.load(f)
    grad_log = d.get('grad_log')
    if not grad_log:
        print('WARN: grad_log empty in this run; skipping render')
        return

    # grad_log is a list of dicts {class: l2_norm} per batch
    kha = []; mino = []
    for batch in grad_log:
        k = batch.get(0)
        if k is not None:
            kha.append(k)
        # minority classes 1..4
        ms = [v for c, v in batch.items() if c in (1, 2, 3, 4)]
        if ms:
            mino.append(np.mean(ms))

    if not kha or not mino:
        print('WARN: insufficient grad log; skipping render')
        return

    # rolling mean
    win = max(1, len(kha) // 50)
    def smooth(xs):
        return np.convolve(xs, np.ones(win)/win, mode='valid')
    kha_s = smooth(kha); mino_s = smooth(mino)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(kha_s, color='#D7191C', label='Khaleeji (majority)', linewidth=1.5)
    ax.plot(mino_s, color='#2C7BB6',
            label='Minority (mean of 4 classes)', linewidth=1.5)
    ax.set_xlabel('Batch index (smoothed)')
    ax.set_ylabel('Mean per-batch gradient $\\ell_2$ norm')
    ax.set_title('Per-class gradient norm: AML-DeBias at 100:1 (real run, seed 42)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(__file__), 'figs', 'fig_perclass_grad.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
