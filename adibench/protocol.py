"""ADIBench evaluation protocol and results aggregation.

Standardized protocol:
  - For each (dataset, n_way, k_shot) combination, run 1000 episodes
  - Report mean accuracy, std, 95% CI
  - Multi-seed (3 seeds) for statistical significance

Results aggregator: load all per-cell JSON results and produce
paper-ready tables.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


class EvaluationProtocol:
    """Standardized few-shot evaluation protocol."""

    def __init__(self, n_episodes: int = 1000, n_seeds: int = 3,
                 seeds: List[int] = [0, 7, 42], q_query: int = 15):
        self.n_episodes = n_episodes
        self.n_seeds = n_seeds
        self.seeds = seeds
        self.q_query = q_query

    def summary(self) -> str:
        return (f"EvalProtocol(n_episodes={self.n_episodes}, "
                f"n_seeds={self.n_seeds}, seeds={self.seeds}, "
                f"q_query={self.q_query})")


class ResultsAggregator:
    """Aggregate per-cell JSON results into a paper-ready table.

    Expected file naming: {method}_{dataset}_{n}way_{k}shot_seed{s}.json
    """

    def __init__(self, results_dir: str = "D:/dacd2026/adibench_v1/results"):
        self.results_dir = results_dir

    def load_all(self):
        """Load all results into a nested dict: results[method][dataset][n_way][k_shot][seed]."""
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
        for fname in os.listdir(self.results_dir):
            if not fname.endswith(".json"):
                continue
            # Try to parse: {method}_{dataset}_{n}way_{k}shot_seed{s}.json
            parts = fname[:-5].split("_")
            if len(parts) < 5:
                continue
            try:
                method = parts[0]
                dataset = parts[1]
                n_way = int(parts[2].replace("w", ""))
                k_shot = int(parts[3].replace("shot", ""))
                seed = int(parts[4].replace("seed", ""))
            except (ValueError, IndexError):
                continue
            with open(os.path.join(self.results_dir, fname)) as f:
                data = json.load(f)
            if "best_acc" in data:
                results[method][dataset][n_way][k_shot][seed] = data["best_acc"]
        return results

    def to_markdown(self):
        """Generate a paper-ready markdown table."""
        results = self.load_all()
        # Collect unique (dataset, n_way, k_shot) combinations
        cells = set()
        for m, dsets in results.items():
            for d, nways in dsets.items():
                for n, kshots in nways.items():
                    for k, seeds in kshots.items():
                        cells.add((d, n, k))
        cells = sorted(cells)
        methods = sorted(results.keys())

        # Header
        out = "| Method | " + " | ".join([f"{d} {n}w{k}s" for d, n, k in cells]) + " |\n"
        out += "|--------|" + "|".join(["-" * 8] * len(cells)) + "\n"
        for m in methods:
            row = [m]
            for d, n, k in cells:
                seeds = results[m].get(d, {}).get(n, {}).get(k, {})
                if not seeds:
                    row.append("-")
                else:
                    vals = list(seeds.values())
                    mean = np.mean(vals)
                    std = np.std(vals) if len(vals) > 1 else 0.0
                    row.append(f"{mean:.3f}±{std:.3f}")
            out += "| " + " | ".join(row) + " |\n"
        return out

    def to_csv(self, output_path: str = None):
        """Save aggregated results to CSV."""
        results = self.load_all()
        cells = set()
        for m, dsets in results.items():
            for d, nways in dsets.items():
                for n, kshots in nways.items():
                    for k, seeds in kshots.items():
                        cells.add((d, n, k))
        cells = sorted(cells)
        methods = sorted(results.keys())
        rows = []
        for m in methods:
            for d, n, k in cells:
                seeds = results[m].get(d, {}).get(n, {}).get(k, {})
                if not seeds:
                    rows.append({"method": m, "dataset": d,
                                 "n_way": n, "k_shot": k,
                                 "mean": None, "std": None, "n_seeds": 0})
                else:
                    vals = list(seeds.values())
                    rows.append({"method": m, "dataset": d,
                                 "n_way": n, "k_shot": k,
                                 "mean": float(np.mean(vals)),
                                 "std": float(np.std(vals)) if len(vals) > 1 else 0.0,
                                 "n_seeds": len(vals)})
        import csv
        if output_path is None:
            output_path = os.path.join(self.results_dir, "summary.csv")
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["method", "dataset", "n_way", "k_shot", "mean", "std", "n_seeds"])
            writer.writeheader()
            writer.writerows(rows)
        return output_path
