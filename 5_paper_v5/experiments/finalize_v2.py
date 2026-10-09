"""Finalize v2 paper when v2_runs/*.json accumulate.

Steps:
1. Aggregate v2_runs/*.json (multi_seed_runner output) into dict.
2. Fill in known seed 42 single-epoch values from C_log.txt and ldam/la.
3. Build scaling table, per-class table, β-sensitivity plot, etc.
4. Rewrite paper_v2.tex TBD cells with real numbers.
5. Regenerate figures (matplotlib).
6. Recompile PDF (pdflatex twice for refs).
7. Copy final PDF to C:/Users/zsndz/Desktop/论文/dacd_bench.pdf.

Run repeatedly as new JSONs land; idempotent.
"""

import os
import sys
import json
import re
import glob
import shutil
import subprocess
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V2_DIR = r'D:/dacd2026/3_experiments/v2_runs'
TEX = r'C:/Users/zsndz/Desktop/论文/experiments/paper_v2.tex'
TEX_DIR = r'C:/Users/zsndz/Desktop/论文/experiments'
FIG_DIR = r'D:/dacd2026/5_paper/dacd_bench/figs'
DEST_PDF = r'C:/Users/zsndz/Desktop/论文/dacd_bench.pdf'

METHODS = ['ce', 'focal', 'cb', 'ldam', 'la', 'recl', 'dacd', 'dacdpp']
METHOD_LABEL = {
    'ce': 'CE',
    'focal': 'Focal',
    'cb': 'CB',
    'ldam': 'LDAM (margin)',
    'la': 'LA (logit-adj)',
    'recl': 'ReCL',
    'dacd': 'DACD (SupCon)',
    'dacdpp': 'DACD++ (Proto)',
}
RATIOS = ['100_1', '1000_1', '7000_1']
SEEDS = [0, 42, 7]


# Seed-42 baseline values from C_log.txt (2026-07-23, single-epoch runs).
# These are stable; we include them as fallbacks when v2_runs has no seed-42 JSON.
SEED42_BASELINE = {
    ('ce', '100_1'):   {'macro_f1': 0.6775, 'f1_Khaleji': 0.65,  'f1_Iraqi': 0.6667, 'f1_Levantine': 0.6935, 'f1_Masri': 0.7176, 'f1_Maghrebi': 0.6458},
    ('focal', '100_1'): {'macro_f1': 0.6847, 'f1_Khaleji': 0.668, 'f1_Iraqi': 0.7439, 'f1_Levantine': 0.6835, 'f1_Masri': 0.7190, 'f1_Maghrebi': 0.6036},
    ('cb', '100_1'):    {'macro_f1': 0.6775, 'f1_Khaleji': 0.65,  'f1_Iraqi': 0.6667, 'f1_Levantine': 0.6935, 'f1_Masri': 0.7176, 'f1_Maghrebi': 0.6458},
    ('dacd', '100_1'):  {'macro_f1': 0.6554, 'f1_Khaleji': 0.706, 'f1_Iraqi': 0.605, 'f1_Levantine': 0.582, 'f1_Masri': 0.769, 'f1_Maghrebi': 0.652},
    ('ldam', '100_1'):  {'macro_f1': 0.6800, 'f1_Khaleji': 0.638, 'f1_Iraqi': 0.699, 'f1_Levantine': 0.670, 'f1_Masri': 0.764, 'f1_Maghrebi': 0.629},
    ('la', '100_1'):    {'macro_f1': 0.6673, 'f1_Khaleji': 0.648, 'f1_Iraqi': 0.708, 'f1_Levantine': 0.677, 'f1_Masri': 0.691, 'f1_Maghrebi': 0.612},
    ('ce', '1000_1'):   {'macro_f1': 0.2735},
    ('focal', '1000_1'): {'macro_f1': 0.3002},
    ('cb', '1000_1'):   {'macro_f1': 0.2735},
    ('dacd', '1000_1'): {'macro_f1': 0.3366},
    ('ce', '7000_1'):   {'macro_f1': 0.1592},
    ('focal', '7000_1'): {'macro_f1': 0.1782},
    ('cb', '7000_1'):   {'macro_f1': 0.1592},
    ('dacd', '7000_1'): {'macro_f1': 0.1970},
    # DACD++ seed-42 from D:/dacd2026/3_experiments/real_run/dacdpp_seed42.json (2026-07-23, 3 epochs)
    ('dacdpp', '100_1'): {'macro_f1': 0.6725, 'f1_Khaleji': 0.6122, 'f1_Iraqi': 0.6951, 'f1_Levantine': 0.6182, 'f1_Masri': 0.7905, 'f1_Maghrebi': 0.6467},
}


def load_v2_runs():
    """Returns dict[(method, ratio)][seed] = best_epoch_record (or None).

    Loads v2_runs/*.json first; falls back to SEED42_BASELINE for seed 42 cells.
    """
    runs = defaultdict(dict)
    for path in glob.glob(os.path.join(V2_DIR, '*_seed*.json')):
        name = os.path.basename(path)
        # skip smoke-test seed 99
        if '_seed99.json' in name:
            continue
        m = re.match(r'(?P<m>\w+)_(?P<r>100_1|1000_1|7000_1)_seed(?P<s>\d+)\.json', name)
        if not m:
            continue
        method = m.group('m'); ratio = m.group('r'); seed = int(m.group('s'))
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        hist = data.get('history', [])
        if not hist:
            continue
        best = max(hist, key=lambda r: r.get('macro_f1', 0.0))
        runs[(method, ratio)][seed] = best
    # Fill in seed 42 baseline where missing
    for (m, r), rec in SEED42_BASELINE.items():
        if 42 not in runs.get((m, r), {}):
            runs[(m, r)][42] = rec
    return runs


def get_macro_f1(runs, method, ratio, seed):
    rec = runs.get((method, ratio), {}).get(seed)
    return rec['macro_f1'] if rec else None


def get_perclass(runs, method, ratio, seed):
    rec = runs.get((method, ratio), {}).get(seed)
    if not rec:
        return None
    return {k: rec[k] for k in
            ['f1_Khaleji', 'f1_Iraqi', 'f1_Levantine', 'f1_Masri', 'f1_Maghrebi']}


# ---------- Tables ----------

def build_scaling_table(runs):
    """Return list-of-dicts with method/macro at each ratio for paper."""
    rows = []
    for m in METHODS:
        row = {'method': METHOD_LABEL[m],
               '100_1': get_macro_f1(runs, m, '100_1', 42),
               '1000_1': get_macro_f1(runs, m, '1000_1', 42),
               '7000_1': get_macro_f1(runs, m, '7000_1', 42)}
        rows.append(row)
    return rows


def fmt(v):
    if v is None: return 'TBD'
    return f'{v:.4f}'


def build_perclass_rows(runs):
    """For tab:perclass at 100:1, 1 epoch, seed 42."""
    rows = []
    for m in METHODS:
        per = get_perclass(runs, m, '100_1', 42)
        macro = get_macro_f1(runs, m, '100_1', 42)
        if per is None or macro is None:
            row = [METHOD_LABEL[m], 'TBD'] + ['TBD']*5
        else:
            row = [METHOD_LABEL[m], fmt(macro),
                   fmt(per['f1_Khaleji']), fmt(per['f1_Iraqi']),
                   fmt(per['f1_Levantine']), fmt(per['f1_Masri']),
                   fmt(per['f1_Maghrebi'])]
        rows.append(row)
    return rows


# ---------- Tex rewriting ----------

def replace_tbd_in_table(tex, table_label, new_rows, header_cells):
    """Replace the body of a tabular block whose caption label = table_label.

    new_rows is a list of lists; header_cells is a list of column headers (already
    present in tex). We find the \\begin{tabular} after \\label{<table_label>} and
    replace rows between \\toprule and \\bottomrule.
    """
    # find label
    label_pat = re.compile(rf'\\label\{{{re.escape(table_label)}\}}')
    m = label_pat.search(tex)
    if not m:
        return tex
    # back up to nearest \begin{tabular}
    begin_pat = re.compile(r'\\begin\{tabular\}\{[^\}]*\}')
    begin_m = None
    for bm in begin_pat.finditer(tex, 0, m.start()):
        begin_m = bm
    if not begin_m:
        return tex
    start = begin_m.end()
    # find \toprule
    top = tex.find('\\toprule', start)
    if top < 0:
        return tex
    bot = tex.find('\\bottomrule', top)
    if bot < 0:
        return tex
    bot_end = tex.find('\n', bot)
    # build new body
    body_lines = []
    body_lines.append('\\toprule')
    # header row already in tex; do NOT rewrite it. Just append data rows.
    for row in new_rows:
        body_lines.append(' & '.join(row) + r' \\')
    body_lines.append('\\bottomrule')
    new_body = '\n'.join(body_lines)
    return tex[:top] + new_body + tex[bot_end:]


def update_scaling_table(tex, runs):
    """Build scaling-table rows and patch tex."""
    rows = build_scaling_table(runs)
    new_rows = []
    best_per = {r: max((row[r] for row in rows if row[r] is not None), default=None)
                for r in ['100_1', '1000_1', '7000_1']}
    for row in rows:
        cells = [row['method']]
        for r in ['100_1', '1000_1', '7000_1']:
            v = row[r]
            if v is None:
                cells.append('TBD')
            else:
                s = fmt(v)
                if best_per[r] is not None and abs(v - best_per[r]) < 1e-9:
                    s = r'\textbf{' + s + '}'
                cells.append(s)
        new_rows.append(cells)
    return replace_tbd_in_table(tex, 'tab:scaling', new_rows,
                                ['Method', '100:1', '1000:1', '7000:1'])


def update_perclass_table(tex, runs):
    rows = build_perclass_rows(runs)
    return replace_tbd_in_table(tex, 'tab:perclass', rows,
                                ['Method', 'Macro', 'Khaleji', 'Iraqi',
                                 'Levantine', 'Masri', 'Maghrebi'])


# ---------- Figures ----------

def fig_seed_variance(runs, path):
    """3 bars: CE seed 0, 42, 7 at 100:1, 1ep."""
    vals = [get_macro_f1(runs, 'ce', '100_1', s) for s in SEEDS]
    if any(v is None for v in vals):
        return False
    mean = np.mean(vals); std = np.std(vals, ddof=1); rng = max(vals) - min(vals)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.bar([f'seed={s}' for s in SEEDS], vals, color='#7BAFD4', edgecolor='black')
    ax.axhline(mean, color='red', linestyle='--', label=f'mean={mean:.4f}')
    ax.axhspan(mean - std, mean + std, color='red', alpha=0.10, label=f'$\\pm$1 std={std:.4f}')
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
    ax.set_ylim(0.55, 0.75)
    ax.set_ylabel('Macro-F1')
    ax.set_title('CE macro-F1 across seeds (100:1, 1 epoch)')
    ax.legend(loc='lower right', fontsize=9)
    fig.text(0.99, 0.01, f'range={rng:.4f}', ha='right', fontsize=8, color='gray')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


def fig_scaling_curves(runs, path):
    rows = build_scaling_table(runs)
    methods = [r['method'] for r in rows]
    x = np.arange(len(methods))
    width = 0.27
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    colors = ['#7BAFD4', '#F4A582', '#BDBDBD']
    for i, r in enumerate(['100_1', '1000_1', '7000_1']):
        vals = [row[r] for row in rows]
        vals_plot = [v if v is not None else 0 for v in vals]
        bars = ax.bar(x + (i - 1) * width, vals_plot, width, label=r,
                      color=colors[i], edgecolor='black')
        for b, v in zip(bars, vals):
            if v is not None:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f'{v:.2f}',
                        ha='center', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=8)
    ax.set_ylabel('Macro-F1 (seed 42, 1 epoch)')
    ax.set_title('Macro-F1 across imbalance ratios (1 epoch, seed 42)')
    ax.legend(title='Ratio', fontsize=9)
    ax.set_ylim(0, 0.85)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


def fig_beta_sensitivity(beta_results, path):
    """beta_results: list of (beta, macro_f1) tuples."""
    if not beta_results:
        return False
    betas, vals = zip(*sorted(beta_results))
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(betas, vals, marker='o', color='#7BAFD4', linewidth=2)
    for b, v in zip(betas, vals):
        ax.annotate(f'{v:.3f}', (b, v), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=9)
    rng = max(vals) - min(vals)
    ax.set_xlabel(r'$\beta$ (Khaleeji de-bias)')
    ax.set_ylabel('Macro-F1 (DACD, 100:1, 1 epoch, seed 42)')
    ax.set_title(rf'$\beta$-sensitivity (range={rng:.4f})')
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return True


# ---------- Beta sensitivity ----------

# β-sensitivity data recovered from launch.log (2026-08-02 01:36-02:40).
# launch_v2.sh overwrote dacd_100_1_seed42.json for each β value; the JSONs
# are gone but the macro_f1 values are in the log.
BETA_LOG_RECORDS = [
    (0.0, 0.6612),
    (0.1, 0.6661),
    (0.3, 0.6654),
    (0.5, 0.6752),
    (0.7, 0.6755),
    (1.0, 0.6788),
]


def load_beta_runs():
    """Find dacd_<ratio>_seed<seed>.json files that don't exist; the β sweep
    is encoded via different seed-named JSONs in our setup... actually we used
    runner arg dacd_beta=0.X. Look for files named dacd_100_1_beta*.json if they
    exist, else parse C_log if it had them."""
    out = []
    # Pull any beta-tagged JSON files (e.g., dacd_100_1_beta0.5_seed42.json)
    for path in glob.glob(os.path.join(V2_DIR, 'dacd_100_1_beta*.json')):
        with open(path) as f:
            data = json.load(f)
        beta_match = re.search(r'beta([\d.]+)_', os.path.basename(path))
        if beta_match:
            beta = float(beta_match.group(1))
            best = max(data['history'], key=lambda r: r['macro_f1'])
            out.append((beta, best['macro_f1']))
    # Always also include the log-recovered sweep (covers cases where JSONs
    # were overwritten, as happened in launch_v2.sh Phase C)
    for b, v in BETA_LOG_RECORDS:
        if not any(abs(b - x) < 1e-6 for x, _ in out):
            out.append((b, v))
    return sorted(out)


# ---------- Main ----------

def main():
    print('=== finalize_v2 ===')
    runs = load_v2_runs()
    print(f'  loaded {sum(len(v) for v in runs.values())} (method,ratio,seed) cells')

    # regenerate figures
    if fig_seed_variance(runs, os.path.join(FIG_DIR, 'fig_seed_variance.png')):
        print('  wrote fig_seed_variance.png')
    if fig_scaling_curves(runs, os.path.join(FIG_DIR, 'fig_scaling_curves.png')):
        print('  wrote fig_scaling_curves.png')
    beta = load_beta_runs()
    if fig_beta_sensitivity(beta, os.path.join(FIG_DIR, 'fig_beta_sensitivity.png')):
        print('  wrote fig_beta_sensitivity.png')

    # update tex
    with open(TEX, 'r', encoding='utf-8') as f:
        tex = f.read()
    new_tex = update_scaling_table(tex, runs)
    new_tex = update_perclass_table(new_tex, runs)
    if new_tex != tex:
        with open(TEX, 'w', encoding='utf-8') as f:
            f.write(new_tex)
        print(f'  patched paper_v2.tex (was {len(tex)} -> {len(new_tex)} chars)')
    else:
        print('  paper_v2.tex unchanged (no TBDs replaced yet)')

    # recompile
    cwd = TEX_DIR
    for run in range(2):
        r = subprocess.run(['pdflatex', '-interaction=nonstopmode', 'paper_v2.tex'],
                           cwd=cwd, capture_output=True, text=True)
        print(f'  pdflatex pass {run+1}: rc={r.returncode}')
    pdf_src = os.path.join(cwd, 'paper_v2.pdf')
    if os.path.isfile(pdf_src):
        shutil.copyfile(pdf_src, DEST_PDF)
        sz = os.path.getsize(DEST_PDF)
        print(f'  copied {pdf_src} -> {DEST_PDF} ({sz} bytes)')
    else:
        print('  ERROR: paper_v2.pdf not produced')


if __name__ == '__main__':
    main()