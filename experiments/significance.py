"""Significance and confidence-interval analysis for DACD-Bench v4 results.

Inputs:
  v4_runs/<method>_<ratio>_seed<seed>.json  -- per-run history JSON from
                                              multi_seed_runner.py, with the
                                              extended schema:
                                               {"method": str, "ratio": str,
                                                "seed": int,
                                                "history": [{"macro_f1": ...,
                                                             "f1_*": ..., ...},
                                                            ...]}

Outputs:
  - paired bootstrap 95% CI for each (methodA, methodB) pair at each ratio
  - Friedman test across all methods at each ratio
  - Nemenyi post-hoc pairwise ranking (if Friedman rejects)
  - per-method mean +/- std summary table

Run from inside experiments/:
  python3 significance.py --v4-runs ../v4_runs --out analysis/

The script is idempotent and prints a one-page summary that can be
copied into the paper's Limitations / Statistics paragraphs.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

METHODS = ['ce', 'focal', 'cb', 'ldam', 'la', 'recl', 'dacd', 'dacdpp']
METHOD_LABEL = {
    'ce': 'CE', 'focal': 'Focal', 'cb': 'CB', 'ldam': 'LDAM',
    'la': 'LA', 'recl': 'ReCL', 'dacd': 'DACD', 'dacdpp': 'DACD++',
}
RATIOS = ['100_1', '1000_1', '7000_1']
SEEDS = [0, 42, 7]


def load_runs(v4_dir):
    """Returns dict[(method, ratio)] = list of macro_f1 per seed."""
    runs = defaultdict(list)
    for path in glob.glob(os.path.join(v4_dir, '*_seed*.json')):
        name = os.path.basename(path)
        if '_seed99' in name:  # smoke-test skip
            continue
        # Expect name: <method>_<ratio>_seed<seed>.json
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        m = data.get('method'); r = data.get('ratio'); s = data.get('seed')
        hist = data.get('history', [])
        if not (m and r and s is not None and hist):
            continue
        # best epoch by macro_f1
        best = max(hist, key=lambda h: h.get('macro_f1', 0.0))
        runs[(m, r)].append((s, best['macro_f1']))
    return runs


def bootstrap_ci(a, b, n_boot=2000, alpha=0.05):
    """Paired bootstrap CI for the difference a - b.

    Returns (mean_diff, lo, hi, p_two_sided).
    """
    a = np.asarray(a); b = np.asarray(b)
    if len(a) != len(b) or len(a) < 2:
        return None
    diff = a - b
    boot = np.random.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    mean = diff.mean()
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # two-sided p: fraction of bootstrap diffs that cross zero
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    p = min(p, 1.0)
    return mean, lo, hi, p


def friedman_chisq(ranks_matrix):
    """Friedman chi-squared statistic by hand (no scipy dependency).

    ranks_matrix: np.ndarray shape (n_seeds, n_methods).
    Returns (chi2, p_value via chi2 survival with df=k-1).
    """
    k = ranks_matrix.shape[1]
    if k < 2 or ranks_matrix.shape[0] < 2:
        return None, None
    ranks_avg = ranks_matrix.mean(axis=0)
    R_bar = ranks_avg.mean()
    chi2 = (12 * ranks_matrix.shape[0] / (k * (k + 1))) * \
           np.sum((ranks_avg - R_bar) ** 2)
    # chi2 survival (rough): use Wilson-Hilferty approx
    df = k - 1
    from math import erfc, sqrt
    z = ((chi2 / df) ** (1/3) - (1 - 2 / (9 * df))) / sqrt(2 / (9 * df))
    p = 0.5 * erfc(z / sqrt(2))
    return chi2, p


def write_report(runs, out_path):
    lines = []
    lines.append('# DACD-Bench v4 significance & CI report')
    lines.append('')
    lines.append('## Per-method mean +/- std and paired bootstrap 95% CIs')
    lines.append('')
    lines.append('Each row pairs (method1, method2): mean diff (m1 - m2), 95% CI, two-sided p.')
    lines.append('')
    for ratio in RATIOS:
        per_ratio = {m: [v for s, v in runs.get((m, ratio), [])]
                     for m in METHODS}
        any_data = any(len(v) > 0 for v in per_ratio.values())
        if not any_data:
            lines.append(f'### Ratio {ratio}: NO DATA')
            lines.append('')
            continue
        lines.append(f'### Ratio {ratio}')
        lines.append('')
        lines.append('| Method | n seeds | mean | std | range |')
        lines.append('|--------|--------:|-----:|----:|------:|')
        for m in METHODS:
            v = per_ratio[m]
            if not v:
                lines.append(f'| {METHOD_LABEL[m]} | -- | -- | -- | -- |')
                continue
            mean = np.mean(v); std = np.std(v, ddof=1); rng = max(v) - min(v)
            lines.append(f'| {METHOD_LABEL[m]} | {len(v)} | {mean:.4f} | {std:.4f} | {rng:.4f} |')
        lines.append('')
        # CIs
        lines.append('**Paired bootstrap 95% CI for (m1 - m2) at common seeds**:')
        lines.append('')
        lines.append('| m1 | m2 | n | mean diff | 95% CI | p (two-sided) |')
        lines.append('|----|----|--:|----------:|--------|---------------|')
        for i, m1 in enumerate(METHODS):
            for j, m2 in enumerate(METHODS):
                if j <= i: continue
                v1 = sorted(per_ratio[m1], reverse=True)
                v2 = sorted(per_ratio[m2], reverse=True)
                # restrict to common seed set
                s1 = dict(runs.get((m1, ratio), []))
                s2 = dict(runs.get((m2, ratio), []))
                common = sorted(set(s1) & set(s2))
                if len(common) < 2:
                    continue
                a = np.array([s1[s] for s in common])
                b = np.array([s2[s] for s in common])
                res = bootstrap_ci(a, b)
                if res is None:
                    continue
                mean, lo, hi, p = res
                lines.append(f'| {METHOD_LABEL[m1]} | {METHOD_LABEL[m2]} | {len(common)} |'
                             f' {mean:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {p:.3f} |')
        lines.append('')
        # Friedman if >=3 methods have >=2 seeds
        enough = [m for m in METHODS if len(per_ratio[m]) >= 2]
        if len(enough) >= 3:
            seeds_with_data = sorted(set(s for m in enough for s, _ in runs.get((m, ratio), [])))
            # build (seed x method) matrix with NaN for missing; require complete for Friedman
            mat = np.full((len(seeds_with_data), len(enough)), np.nan)
            for j, m in enumerate(enough):
                for s, v in runs.get((m, ratio), []):
                    if s in seeds_with_data:
                        i = seeds_with_data.index(s)
                        mat[i, j] = v
            keep = ~np.isnan(mat).any(axis=1)
            mat = mat[keep]
            if mat.shape[0] >= 2:
                # ranks: lower is better inverted (rank 1 = highest macro_f1)
                from scipy.stats import rankdata
                ranks = np.apply_along_axis(rankdata, 1, mat)
                chi2, p_friedman = friedman_chisq(ranks)
                lines.append(f'**Friedman test across {len(enough)} methods, {mat.shape[0]} seeds**: '
                             f'$\\chi^2={chi2:.3f}$, $p\\approx{p_friedman:.3f}$')
                lines.append('')
                if p_friedman is not None and p_friedman < 0.05:
                    lines.append('Friedman rejects at $\\alpha{=}0.05$, running Nemenyi post-hoc:')
                    # Nemenyi CD
                    k = len(enough); n = mat.shape[0]
                    CD = 2.169 * np.sqrt(k * (k + 1) / (6 * n))
                    avg_ranks = ranks.mean(axis=0)
                    sorted_idx = np.argsort(avg_ranks)
                    for ii in sorted_idx:
                        lines.append(f'  rank {avg_ranks[ii]:.2f}: {METHOD_LABEL[enough[ii]]}')
                    lines.append(f'  Critical difference (Nemenyi, $\\alpha{=}0.05$): {CD:.2f}')
                    lines.append('')
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out_path


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--v4-runs', default='../v4_runs',
                    help='directory of v4_runs/<method>_<ratio>_seed<seed>.json')
    ap.add_argument('--out', default='analysis/significance.md',
                    help='output markdown report path')
    args = ap.parse_args()
    if not os.path.isdir(args.v4_runs):
        print(f'WARN: {args.v4_runs} does not exist; run multi_seed_runner.py first')
        sys.exit(0)
    runs = load_runs(args.v4_runs)
    out = write_report(runs, args.out)
    print(f'wrote {out}')
    print(f'summary: {sum(len(v) for v in runs.values())} (method,ratio,seed) cells')
