# ADIBench + SCP-LPA

**Benchmark and method code for the ARR October 2026 submission.**

This work introduces **ADIBench**, a unified benchmark for long-tail Arabic dialect identification across three public datasets and eight baselines, and **SCP-LPA** (Supervised Contrastive + Linguistic-Prior Aggregation), a method that combines supervised contrastive learning with a hand-crafted linguistic-distance prior to address extreme class imbalance.

## Contents

```
adibench/        # ADIBench benchmark package (protocol, baselines, data loaders)
alps/            # SCP-LPA method package (ProtoNet, contrastive loss, LPA)
dacd++/          # DACD++ contrastive pre-training (prior work)
experiments/     # Experiment scripts (multi-seed, transfer, sensitivity, figures)
paper/           # BibTeX
adibench_paper.tex             # LaTeX source
adibench_paper_FINAL.pdf       # Compiled PDF (10 pp, ACL style)
adibench_submission.zip        # Full submission package
```

## Datasets

The benchmark reuses three public Arabic dialect datasets. They are **not** redistributed here — download them from the original sources and place them under `data/`:

| Dataset | Source | Notes |
|---|---|---|
| NADI 2024 (18 / 5 dialects) | [NADI shared task](https://nadi.dlnlp.org/) | Country-level dialects |
| amgadhasan / qadi | [HuggingFace](https://huggingface.co/datasets) | Smaller, higher-resource |

A small processed sample lives under `processed/` for smoke testing only.

## Models

The encoder is MARBERTv2 / AraBERTv02. Pretrained weights are **not** included — pull them from HuggingFace:

- [`UBC-NLP/MARBERTv2`](https://huggingface.co/UBC-NLP/MARBERTv2)
- [`aubmindlab/bert-base-arabertv02`](https://huggingface.co/aubmindlab/bert-base-arabertv02)

Place the downloaded `pytorch_model.bin` / `model.safetensors` under `marbertv2/` and `arabertv02/` respectively, or set `HF_HOME` so transformers finds them.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate      # or .venv\Scripts\activate on Windows
pip install -r experiments/requirements.txt
```

For LaTeX compilation of the paper:

```bash
# TeX Live 2025 with the acl package; install via:
sudo tlmgr install acl
```

## Reproducing the main results

```bash
# from repo root
python experiments/adibench_multiseed.py --config configs/main.yaml
python experiments/adibench_scp_lpa_multiseed_summary.py
```

See `experiments/` for individual scripts (per-cell results, sensitivity sweeps, cross-corpus transfer, significance tests).

## Method summary

**SCP-LPA** adds two components on top of a Prototypical Network encoder:

1. **Supervised Contrastive (SCP)** — pulls same-class examples together, pushes different-class examples apart in the embedding space.
2. **Linguistic-Prior Aggregation (LPA)** — reweights prototypes using a hand-coded linguistic dissimilarity matrix. Justified empirically: Spearman ρ(linguistic distance, embedding distance) = 0.435 (p < 1e-15) on NADI 18.

The method wins on all six (dataset, shot) cells with 4-seed mean Macro F1, average Δ +0.028 over vanilla ProtoNet.

## Citation

Anonymous submission to ARR October 2026. The full paper is in `adibench_paper_FINAL.pdf`. Bibliographic entry will be added after the review cycle.

## License

TBD (will be set on acceptance per venue requirements).

## Status

- Submitted: 2026-09-30
- Forum: [openreview.net/forum?id=OK3D3ViXUq](https://openreview.net/forum?id=OK3D3ViXUq)
- Reviewer registration deadline: 2026-10-20
- Reviews due: 2026-11-16