"""Geo-Curriculum: Geographic-difficulty-based training schedule.

Two curriculum strategies based on the Arabic dialect continuum:

  1. **Distance-from-center curriculum**: At step t, sample training
     episodes that contain at least one pair of dialects whose
     linguistic distance is above threshold tau(t). tau(t) starts
     high (force "distant" pairs early) and decays to 0 (allow all
     pairs late).

  2. **Easy-to-hard pair curriculum**: Sample within-class pairs early
     (same class = trivially close) and gradually shift to cross-class
     pairs that span the maximum distance first.

Both strategies are designed to make the encoder learn coarse
geographic structure first, then refine fine-grained distinctions.
"""
from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch
import pandas as pd

from .distance import get_linguistic_distance
from .data import build_fewshot_episode


class GeoCurriculum:
    """Geo-Curriculum: episode difficulty scheduling by class pair distance.

    At step t, the curriculum has a difficulty threshold tau(t):
      - Easy episodes (tau=0.0): all class pairs allowed
      - Hard episodes (tau=1.0): only pairs with dist > 0.5 allowed
      - tau(t) decays from 1.0 (hardest) to 0.0 (easiest) over training
    """

    def __init__(self, num_classes, total_steps, start_tau=1.0, end_tau=0.0):
        self.num_classes = num_classes
        self.total_steps = total_steps
        self.start_tau = start_tau
        self.end_tau = end_tau
        D = get_linguistic_distance(num_classes)
        if isinstance(D, torch.Tensor):
            D = D.cpu().numpy()
        self.D = D
        # Pre-compute hard pair list for each "tau level" (0..10)
        # tau=0: all pairs, tau=1: only pairs with dist > 0.5
        self.pair_cache = {}
        for level in range(11):
            threshold = level / 10.0
            mask = (D > threshold) & (~np.eye(num_classes, dtype=bool))
            pairs = [(i, j) for i in range(num_classes) for j in range(i+1, num_classes) if mask[i, j]]
            if not pairs:
                pairs = [(i, j) for i in range(num_classes) for j in range(i+1, num_classes)]
            self.pair_cache[level] = pairs

    def get_tau(self, step: int) -> float:
        """Decay schedule: linear from start_tau to end_tau."""
        if self.total_steps <= 1:
            return self.end_tau
        progress = min(1.0, step / (self.total_steps - 1))
        return self.start_tau + (self.end_tau - self.start_tau) * progress

    def sample_episode(self, df, n_way, k_shot, q_query, step, rng=None):
        """Sample an episode, optionally constrained to high-distance pairs."""
        if rng is None:
            rng = random.Random()
        tau = self.get_tau(step)
        level = int(round(tau * 10))
        # With probability 1-tau, sample a fully-random episode.
        # With probability tau, force the episode to contain a
        # high-distance pair.
        if rng.random() < tau:
            # Choose a hard pair
            hard_pairs = self.pair_cache[level]
            pair = hard_pairs[rng.randint(0, len(hard_pairs))]
            # Sample other (n_way - 2) classes randomly
            other_classes = [c for c in range(self.num_classes) if c not in pair]
            rng.shuffle(other_classes)
            chosen = list(pair) + other_classes[:n_way - 2]
        else:
            chosen = None
        # Build episode
        if chosen is None:
            return build_fewshot_episode(df, n_way, k_shot, q_query, rng=rng)
        # Use chosen classes
        by_class = {c: df.index[df["label"] == c].tolist() for c in chosen}
        support_idx, query_idx = [], []
        support_labels, query_labels = [], []
        for ep_label, orig in enumerate(chosen):
            pool = by_class[orig]
            need = k_shot + q_query
            if len(pool) < need:
                pool = rng.choices(pool, k=need)
            else:
                pool = rng.sample(pool, need)
            support_idx.extend(pool[:k_shot])
            query_idx.extend(pool[k_shot:k_shot + q_query])
            support_labels.extend([ep_label] * k_shot)
            query_labels.extend([ep_label] * q_query)
        return support_idx, query_idx, support_labels, query_labels
