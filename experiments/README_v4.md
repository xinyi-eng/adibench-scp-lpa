# DACD-Bench v4 — Reproducibility Kit

This directory contains the scripts needed to reproduce every result in the
revised DACD / DACD-Bench v4 paper. The kit is meant to run on a single
RTX 4060 laptop GPU (8.6 GB) or better.

## Files
- `multi_seed_runner.py` — generic runner for any (method, ratio, seed,
  epochs) configuration. Outputs `v4_runs/<method>_<ratio>_seed<seed>.json`.
- `aml_debias.py` — implementation of the new AML-DeBias module
  (Adaptive Majority-Language De-Biasing, Eq. 2 in the paper). Drop-in
  replacement for the contrastive term in `dacd/losses.py`.
  Usage: `from aml_debias import AMLDeBiasSupCon; sup = AMLDeBiasSupCon(5, 0)`.
  Set `use_aml=False` to recover the v3 fixed-beta default.
- `significance.py` — paired bootstrap 95% CI, Friedman chi-squared,
  Nemenyi critical difference. Reads `v4_runs/*.json`.
- `generate_figures_v4.py` — renders the four paper figures from
  `v4_runs/*.json` when present; falls back to seed-42 single-epoch
  baselines so the PDF always renders something readable. Outputs go
  to `experiments/figs/` -- copy them to `_compare_src/figs/` before
  re-compiling the paper.
- `launch_v4.sh` — orchestrator: 8 methods x 3 ratios x 3 seeds at 100:1
  (full sweep), plus 1000:1 and 7000:1 at 1 seed for the scaling table,
  alpha sweep for AML-DeBias, few-shot sweep for DACD++.

## How to reproduce from scratch

### 1. Install
```
pip install torch transformers scikit-learn pandas numpy matplotlib scipy
```

### 2. Download datasets
- QADI 2024 from HuggingFace
  (`Abdelrahman-Rezk/Arabic_Dialect_Identification`, 440K tweets)
- amgadhasan/arabic\_tweets\_dialects from HuggingFace (147K tweets)
- AraBERTv2-base from HuggingFace
  (`aubmindlab/bert-base-arabertv2`)

### 3. Pre-process splits
Use `setup_vals.py` (already in the kit). This writes
`dacd_bench_<ratio>_train.csv` and `dacd_bench_<ratio>_val.csv` for each
ratio. The 100:1 split is the full 52K sample balanced train set; 1000:1
and 7000:1 are constructed as described in §6 of the paper.

### 4. Run the sweep
```
bash launch_v4.sh
```
Wall-clock estimate on RTX 4060:
- 100:1, 8 methods x 3 seeds, 3 epochs each  -> ~12 hours
- 1000:1, 8 methods, 1 seed, 1 epoch          -> ~30 minutes
- 7000:1, 8 methods, 1 seed, 1 epoch          -> ~5 minutes
- Beta sweep + few-shot sweep + AML sweep      -> ~8 hours
- Total:                                       ~20 hours

### 5. Compute significance
```
python3 significance.py --v4-runs ../v4_runs --out analysis/significance.md
```

### 6. (Optional) Generate paper figures
```
python3 generate_figures_v4.py
cp figs/*.png ../_compare_src/figs/
cd ../_compare_src && pdflatex -interaction=nonstopmode revision_paper.tex
cp revision_paper.pdf ../revised_dacd.pdf
```

## What the runner produces

For each combination `(method, ratio, seed, epochs)` the runner outputs:
- `v4_runs/<method>_<ratio>_seed<seed>.json` with key fields:
  - `method`, `ratio`, `seed`, `epochs`
  - `history`: list of per-epoch records with `macro_f1`, `f1_Khaleji`,
    `f1_Iraqi`, `f1_Levantine`, `f1_Masri`, `f1_Maghrebi`, `train_loss`,
    `elapsed_s`, `n_train`, `n_val`.

## Methods supported by the runner

| Key | Method |
|-----|--------|
| `ce` | Vanilla CE |
| `focal` | Focal ($\gamma=2$) |
| `cb` | Class-balanced |
| `dacd` | DACD + AML-DeBias ($\alpha=1.0$, $\beta_0=0.3$) |
| `dacdpp` | DACD++ (prototype refinement, $\alpha=0.5$) |
| `ldam` | LDAM (margin $\gamma_y = C/n_y^{1/4}$) |
| `la` | Logit Adjustment |
| `recl` | ReCL (inverse-frequency positive weight) |

## Reproducibility flags

| Flag | Effect |
|------|--------|
| `--no-aml` | Use $\beta=0.3$ fixed (v3 default) instead of AML-DeBias |
| `--alpha 0.5` | Set AML responsiveness hyper-parameter |
| `--epochs 3` | Train 3 epochs at 100:1 (default), 1 at other ratios |
| `--max-len 96` | Truncate tweets to 96 tokens (RTX 4060 limit) |
| `--batch 16` | Batch size (RTX 4060 limit) |

## Notes on GLUE-style reproducibility

For each configuration we report macro-F1 and per-class F1 on the seed-42
fixed validation split. We do not perform validation-set model selection
beyond choosing the best epoch by macro-F1 on the same split. All seeds
process the same data and identical model code.

## Cited paper text in `analysis/significance.md`

The output markdown from `significance.py` is intended to be copy-pasted
into the paper's Limitations / Statistical Analysis paragraph. The
template format is at the top of that file.
