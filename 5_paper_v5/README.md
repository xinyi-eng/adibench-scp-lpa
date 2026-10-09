# L2C v5: Linguistic-Distance-Aware Long-Tail ADI

**Working directory on D drive**: `D:\dacd2026\5_paper_v5\`

## What's new in v5 (4 contributions)

1. **L2C** (Linguistic-Distance-Weighted Contrastive) - `dacd/losses.py::L2CLoss`
2. **CEDA** (Cross-Encoder Distillation Augmentation) - `dacd/losses.py::CEDALoss` + `dacd/models.py::CEDAClassifier`
3. **AMI** (Adaptive Margin for Imbalance) - `dacd/losses.py::AMILoss`
4. **ARC** (Adaptive Ratio Curriculum) - `dacd/losses.py::ARCLoss`

Combined as **L2C++** = L2C + CEDA + AMI + ARC, implemented in `dacd/losses.py::DACDv5Loss`.

## Layout

```
D:\dacd2026\5_paper_v5\
├── dacd\                          # Python package (losses, models, data)
│   ├── losses.py                  # 4 new losses + 8 v4 baselines
│   ├── models.py                  # SingleEncoder + MEC + CEDA (dual-encoder)
│   ├── data.py                    # TextDataset, build_dacd_bench
│   ├── train.py                   # (legacy; not used by v5 runner)
│   ├── utils.py
│   └── __init__.py                # exports
├── experiments\
│   ├── multi_seed_runner_v5.py    # main runner; 11 methods x 3 ratios x seeds
│   ├── multi_seed_runner.py       # (v4 runner, kept for reference)
│   ├── multi_seed_runner_aml.py   # (v4 AML runner, kept for reference)
│   ├── multi_seed_runner_fewshot.py
│   ├── feats_extract.py           # extract features for t-SNE
│   ├── render_tsne_real.py
│   ├── render_gradnorm_real.py
│   ├── significance.py            # bootstrap CI + Friedman
│   ├── generate_figures_v4.py     # 4 figures from v2_runs JSONs
│   ├── run_v5_week1.sh            # Week 1 plan (Table 7 + Table 1 + t-SNE)
│   └── run_v5_week2.sh            # Week 2 plan (4 innovations + MARBERTv2)
├── paper\
│   ├── paper_v5.tex               # v5 paper draft
│   └── figs\                      # paper figures (4)
├── results\                       # JSON outputs of all runs
└── outputs\                       # log files, intermediate
```

## Run experiments

```bash
# Activate venv if needed (D:\dacd2026\site-packages is already on PYTHONPATH)
cd /d/dacd2026/5_paper_v5

# Week 1: ~10-12 GPU hours
bash experiments/run_v5_week1.sh

# Week 2: ~12-15 GPU hours
bash experiments/run_v5_week2.sh
```

## Method invocation

```bash
# Baseline methods (v4 compatible)
python -u experiments/multi_seed_runner_v5.py ce        100_1 42 3
python -u experiments/multi_seed_runner_v5.py focal     100_1 42 3
python -u experiments/multi_seed_runner_v5.py cb        100_1 42 3
python -u experiments/multi_seed_runner_v5.py dacd      100_1 42 3
python -u experiments/multi_seed_runner_v5.py dacdpp    100_1 42 3
python -u experiments/multi_seed_runner_v5.py ldam      100_1 42 3
python -u experiments/multi_seed_runner_v5.py la        100_1 42 3
python -u experiments/multi_seed_runner_v5.py recl      100_1 42 3
python -u experiments/multi_seed_runner_v5.py marbert   100_1 42 3

# v5 innovations
python -u experiments/multi_seed_runner_v5.py l2c       100_1 42 3
python -u experiments/multi_seed_runner_v5.py ami       100_1 42 3
python -u experiments/multi_seed_runner_v5.py arc       100_1 42 3
python -u experiments/multi_seed_runner_v5.py ceda      100_1 42 1
python -u experiments/multi_seed_runner_v5.py dacdv5    100_1 42 3
```

## Hardware requirements

- Single GPU with **>= 8 GB VRAM** (RTX 4060 / 3060 / A100 etc.)
- CUDA 12.x, Python 3.12
- `torch 2.6.0+cu124`, `transformers 4.47.0`, `scikit-learn`, `pandas`, `numpy`

## Key v5 files (changed/added)

| File | What's in it |
|------|--------------|
| `dacd/losses.py` | + L2CLoss, AMILoss, ARCLoss, CEDALoss, DACDv5Loss, LINGUISTIC_DISTANCE |
| `dacd/models.py` | + CEDAClassifier (frozen MARBERTv2 teacher) |
| `dacd/__init__.py` | exports for v5 |
| `experiments/multi_seed_runner_v5.py` | unified 11-method runner |
| `experiments/run_v5_week1.sh` | Week 1 plan |
| `experiments/run_v5_week2.sh` | Week 2 plan |
| `paper/paper_v5.tex` | v5 paper draft |

## Submission targets (高标准, 强创新)

1. **TACL** (ACM journal, CCF A) - best fit; 4 innovations + benchmark + analysis
2. **ACL 2027** (CCF A) - via ARR October 2026 cycle (deadline 2026-10-12)
3. **EMNLP 2027** (CCF B) - backup via same ARR cycle
4. **TALLIP** (CCF B journal) - if TACL too long

## Status as of 2026-09-04

- Code: v5 complete (4 innovations implemented and exported)
- Paper: v5 draft written (16K chars, 4 sections, 5 tables)
- Experiments: ready to run (no data yet)
- Submission target: ARR Oct 2026 cycle → commit to ACL 2027 or TACL

## Next steps

1. Run `bash experiments/run_v5_week1.sh` (1-2 days)
2. Run `bash experiments/run_v5_week2.sh` (1-2 days)
3. Fill in real numbers in paper_v5.tex tables
4. Recompile to PDF
5. Polish writing, add refs, internal review
6. Submit to ARR October 2026 cycle (deadline 2026-10-12)
