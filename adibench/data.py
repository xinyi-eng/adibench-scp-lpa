"""ADIBench data loaders: 3 standardized Arabic dialect datasets.

Datasets:
  - NADI 2024 18-way: 18 country dialects, 440K tweets (QADI parquet)
  - NADI 2024 5-way: 5 coarse dialect groups (mapped from NADI 18)
  - amgadhasan 5-city: 5 city dialects (EG, LY, LB, SD, MA), 147K tweets
"""
from __future__ import annotations

import os
import random
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd


# 18-way NADI 2024 country labels (canonical order)
NADI_18_COUNTRIES = [
    "OM", "SD", "SA", "KW", "QA", "LB", "JO", "SY",
    "IQ", "MA", "EG", "PL", "YE", "BH", "DZ", "AE", "TN", "LY",
]
QADI_5_NAMES = ["Khaleeji", "Iraqi", "Levantine", "Masri", "Maghrebi"]
AMGHADHASAN_5_CITIES = ["EG", "LY", "LB", "SD", "MA"]


# 5-way coarse mapping from 18-way
QADI_18_TO_5 = {
    0: 0, 1: 3, 2: 0, 3: 0, 4: 0, 5: 2, 6: 2, 7: 2, 8: 1,
    9: 4, 10: 3, 11: 2, 12: 0, 13: 0, 14: 4, 15: 0, 16: 4, 17: 4,
}

# amgadhasan 5-city - keep original city labels
AMGHADHASAN_CITY_TO_IDX = {"EG": 0, "LY": 1, "LB": 2, "SD": 3, "MA": 4}
AMGHADHASAN_5_NAMES = ["Egypt-Cairo", "Libya-Tripoli", "Lebanon-Beirut",
                         "Sudan-Khartoum", "Morocco-Rabat"]


def load_nadi_18way(path: str = "D:/dacd2026/1_data/raw/qadi.parquet") -> pd.DataFrame:
    """Load NADI 2024 18-way (QADI parquet)."""
    df = pd.read_parquet(path)
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]].reset_index(drop=True)


def load_nadi_5way(path: str = "D:/dacd2026/1_data/raw/qadi.parquet") -> pd.DataFrame:
    """Load NADI 2024 mapped to 5-way coarse."""
    df = load_nadi_18way(path)
    df["label"] = df["label"].map(QADI_18_TO_5)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def load_amgadhasan_5city(path: str = "D:/dacd2026/1_data/raw/amgadhasan.parquet") -> pd.DataFrame:
    """Load amgadhasan with 5-way city labels (EG, LY, LB, SD, MA)."""
    df = pd.read_parquet(path)
    df["label"] = df["dialect"].map(AMGHADHASAN_CITY_TO_IDX)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]].reset_index(drop=True)


def build_fewshot_episode(
    df: pd.DataFrame,
    n_way: int,
    k_shot: int,
    q_query: int,
    rng: Optional[random.Random] = None,
):
    """Sample a single N-way K-shot episode."""
    if rng is None:
        rng = random.Random()
    by_class = {}
    for idx, label in zip(df.index, df["label"]):
        by_class.setdefault(int(label), []).append(idx)
    available_classes = list(by_class.keys())
    if len(available_classes) < n_way:
        chosen_classes = rng.choices(available_classes, k=n_way)
    else:
        chosen_classes = rng.sample(available_classes, n_way)
    support_idx, query_idx = [], []
    support_labels, query_labels = [], []
    for episode_label, orig_label in enumerate(chosen_classes):
        candidates = by_class[orig_label]
        if len(candidates) < k_shot + q_query:
            sampled = rng.choices(candidates, k=k_shot + q_query)
        else:
            sampled = rng.sample(candidates, k_shot + q_query)
        support_idx.extend(sampled[:k_shot])
        query_idx.extend(sampled[k_shot:k_shot + q_query])
        support_labels.extend([episode_label] * k_shot)
        query_labels.extend([episode_label] * q_query)
    return support_idx, query_idx, support_labels, query_labels


# Dataset registry
DATASETS = {
    "nadi_18": {
        "loader": load_nadi_18way,
        "num_classes": 18,
        "size": 440052,
        "name": "NADI 2024 18-way",
        "granularity": "country",
        "domain": "twitter",
    },
    "nadi_5": {
        "loader": load_nadi_5way,
        "num_classes": 5,
        "size": 440052,
        "name": "NADI 2024 5-way (coarse)",
        "granularity": "coarse_dialect",
        "domain": "twitter",
    },
    "amgadhasan_5": {
        "loader": load_amgadhasan_5city,
        "num_classes": 5,
        "size": 147725,
        "name": "amgadhasan 5-city",
        "granularity": "city",
        "domain": "twitter",
    },
}


def get_dataset(name: str) -> Tuple[pd.DataFrame, dict]:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    meta = DATASETS[name]
    df = meta["loader"]()
    return df, meta


def list_datasets() -> List[str]:
    return list(DATASETS.keys())
