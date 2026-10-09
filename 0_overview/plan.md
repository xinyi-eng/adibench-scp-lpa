# DACD++ Project Plan -- Path 3 for ArabicNLP 2026

**Target venue**: ArabicNLP 2026 (workshop at EMNLP 2026)
**Hardware**: NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)
**Compute budget**: ~2 weeks of on-and-off training
**Status**: 2026-07-17 setup phase -- cleanup done, CUDA torch working, both models verified on GPU

---

## 1. Why Path 3 (rejected-EMNLP to ArabicNLP)

The original EMNLP draft (rejected for: refs, equations, limitations) had
structural issues that a Path 3 rewrite must fix:

| Issue | Root cause | Path-3 fix |
|-------|-----------|-----------|
| DataFlare private/unreleased | "data-use agreement" placeholder | DACD-Bench from public NADI/MADAR/QADI |
| Stage 1 = 26.41 reverse-engineered | No actual Stage-1-only run | Run Stage-1-only training explicitly |
| Single seed | No multi-seed protocol | Run 3+ seeds for all main results |
| Fig 13b 2D heatmap was mock data | No 2D grid sweep | 1D sweep + 1D figure, no fake 2D |
| MADAR+QADI as fair comparison | Uses external pre-training | Compare to *public* MADAR models only |
| Abstract said "even surpassing" | Mild overclaim | Tone down to "competitive with" |

## 2. Paper title (working)

"Curriculum Imbalance Schedule and Differentiable beta for Extreme Long-Tail Arabic Dialect Identification"

Shorter: "DACD++: Curriculum, Differentiable Debiasing, and Multi-Encoder Consistency for 7000:1 Arabic Dialect Identification"

Final title TBD after experiments.

## 3. Three innovations

### 3.1 CIS -- Curriculum Imbalance Schedule
**Hypothesis**: at extreme imbalance (7000:1), minority gradients are noisy. Start at 1000:1, gradually increase to 7000:1 -- let model first learn mid-frequency distinctions before being asked to separate extreme ones.

**Implementation**: sampler with ratio $r(e) = 1000 + 6000 \cdot (e/E)^2$.

**Novelty**: no prior paper does curriculum on imbalance ratio. Standard curriculum operates on sample difficulty, not class-prior difficulty.

### 3.2 DLB -- Differentiable Lower Bound on beta
**Hypothesis**: Khaleei-debiasing weight beta=0.3 was found by grid search (a magic number). Make it data-dependent and differentiable:
   beta = sigma(w^T h_class + b)
where h_class is the current class-frequency embedding and w, b are learned.

**Novelty**: existing rebalanced-SupCon (Cao 2021) uses fixed beta. We make it data-dependent and differentiable.

### 3.3 MEC -- Multi-Encoder Consistency
**Hypothesis**: AraBERTv2 (general-domain) and MARBERTv2 (Twitter-domain) have complementary inductive biases. Train both on a shared classifier with a consistency loss on shared representations.

**Implementation**:
- shared linear head W on [h_AraBERT; h_MARBERT]
- supervised: L_sup = CE(W*h_AraBERT, y) + CE(W*h_MARBERT, y)
- consistency: L_cons = || sigmoid(W*h_AraBERT) - sigmoid(W*h_MARBERT) ||^2
- total: L = L_sup + alpha * L_cons

**Novelty**: dual-encoder consistency is used in semi-sup learning (Xie 2020) but not in extreme long-tail text classification.

## 4. Data plan -- DACD-Bench (open-release target)

Combine 3 publicly available Arabic dialect corpora:

| Source | URL | Size | Classes |
|--------|-----|------|---------|
| NADI 2024 shared task | HF: ARBERT/NADI2024 | ~30K | 18 dialects -> 5-way |
| MADAR Twitter | HF: MADAR-Twitter | ~15K | 25 cities -> 5-way |
| QADI | HF: qadi | ~5K | 5 dialects (already aligned!) |

5-way mapping: Khaleji / Levantine / Masri / Iraqi / Maghrebi

Imbalance control: subsample to 100:1, 1000:1, 10000:1, 70000:1 to test DACD++ across full long-tail spectrum.

Release: HuggingFace Datasets dataset card under our own org.

## 5. Experiments

### Main (Table 1)
5 methods (CE, Focal, CB, DACD, DACD++) x 3 datasets x 5 dialects x 3 seeds = 225 cells

### Ablation (Table 2)
DACD++ minus each innovation (-CIS, -DLB, -MEC) x 3 seeds = 18 cells

### Imbalance-ratio scaling (Fig 3)
4 methods x 6 ratios x 3 seeds = 72 cells, log-scale x

### Cross-dialect transfer (Table 3)
5 methods x 5 leave-one-out x 3 seeds = 75 cells

### Noise robustness (Fig 4)
5 noise levels (0-20%) x 3 methods x 3 seeds = 45 cells

### Calibration (Table 4)
ECE before/after temperature scaling, 5 methods x 3 seeds = 15 cells

## 6. Code structure

D:\dacd2026\2_models\dacd++\
  dacd/
    __init__.py
    models.py     # AraBERTv2, MARBERTv2, DACD, DACD++ encoders
    losses.py     # CIS sampler, DLB beta, MEC consistency, combined
    data.py       # load_nadi / load_madar / load_qadi / build_bench
    train.py      # 1 training script, 5 methods x 3 seeds x 3 datasets
    eval.py       # F1, ECE, per-class
    utils.py
  scripts/
    run_main.sh
    run_ablation.sh
    run_scaling.sh
    run_transfer.sh
  configs/        # per-experiment YAML

## 7. Timeline

| Day | Task | Time |
|-----|------|------|
| 1 | Build code skeleton | 3-4h |
| 2 | Download datasets, build DACD-Bench | 2h |
| 3 | Run baselines (CE, Focal, CB) on QADI | 2h |
| 4 | Run DACD on QADI (3 seeds) | 4h |
| 5 | Run DACD++ variants on QADI | 4h |
| 6 | Run on NADI + MADAR | 6h |
| 7 | Ablation, scaling, transfer | 4h |
| 8-10 | Paper writing | 12h |
| 11 | Polish figures, citations | 4h |
| 12 | Final pass + submit | 2h |

## 8. Anti-patterns (lessons from prior round)

- DO NOT use "DataFlare" (unreproducible)
- DO NOT cite imaginary papers (liu2019open, zhong2021understanding)
- DO NOT show figures with mock data labeled as measured
- DO NOT claim "even surpassing" without fairness note
- DO NOT hide where Stage 1 = 26.44 came from -- we run it now
- DO NOT use single seed -- 3+ seeds for sigma
