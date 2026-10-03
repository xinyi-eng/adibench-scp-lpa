"""ADIBench: HuggingFace evaluate wrapper.

Provides an evaluate-compatible metric for the ADIBench benchmark.
Users can call `evaluate.load("adibench/accuracy")` to get accuracy + ECE.

Usage:
    import evaluate
    metric = evaluate.load("adibench/accuracy")
    results = metric.compute(predictions=preds, references=refs)
"""
import numpy as np
from typing import Dict, List

import evaluate


_CITATION = """\
@inproceedings{adibench2026,
  title={ADIBench: A Multi-Granularity Benchmark for Few-Shot Arabic Dialect Identification},
  author={Anonymous},
  year={2026},
}
"""

_DESCRIPTION = """\
ADIBench few-shot evaluation metric. Computes accuracy, Expected
Calibration Error (ECE), and Brier score.
"""


def accuracy_with_ece(predictions, references, n_bins=10):
    """Compute accuracy, ECE, and Brier.

    Args:
        predictions: array of shape (N, n_classes) of class probabilities.
        references: array of shape (N,) of int labels.

    Returns:
        Dict with keys: accuracy, ece, brier, n_samples.
    """
    preds = np.asarray(predictions)
    refs = np.asarray(references)
    n_samples = len(refs)
    pred_labels = preds.argmax(axis=1)
    acc = float((pred_labels == refs).mean())
    # ECE
    confidences = preds.max(axis=1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            bin_acc = (pred_labels[in_bin] == refs[in_bin]).mean()
            bin_conf = confidences[in_bin].mean()
            ece += (in_bin.sum() / n_samples) * abs(bin_acc - bin_conf)
    # Brier
    onehot = np.zeros_like(preds)
    onehot[np.arange(n_samples), refs] = 1
    brier = float(((preds - onehot) ** 2).sum(axis=1).mean())
    return {
        "accuracy": acc,
        "ece": ece,
        "brier": brier,
        "n_samples": n_samples,
    }


@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION)
class AdibenchAccuracy(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description="Predictions as (N, n_classes) probabilities; references as (N,) int labels.",
        )

    def _compute(self, predictions, references, n_bins=10):
        return accuracy_with_ece(predictions, references, n_bins=n_bins)