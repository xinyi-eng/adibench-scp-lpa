"""Data loaders for ALPS: Arabic few-shot dialect identification.

Datasets:
  - NADI 2024 (QADI parquet): 18 country labels, 440K tweets
  - amgadhasan: 5 city labels, 147K tweets
  - 5-way coarse mapping: from 18-way / 5-way to {Kha, IRQ, Lev, Mas, Mag}
"""
from __future__ import annotations

import os
import random
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# 18-way NADI 2024 country labels (canonical order)
NADI_18_COUNTRIES = [
    "OM", "SD", "SA", "KW", "QA", "LB", "JO", "SY",
    "IQ", "MA", "EG", "PL", "YE", "BH", "DZ", "AE", "TN", "LY",
]

# 5-way amgadhasan city labels
AMGHADHASAN_5_CITIES = ["EG", "LY", "LB", "SD", "MA"]


# 5-way coarse mapping
# Khaleeji: SA, KW, AE, BH, QA, OM, YE
# Iraqi: IQ
# Levantine: LB, SY, JO, PL
# Masri: EG, SD
# Maghrebi: MA, DZ, TN, LY
QADI_18_TO_5 = {
    0: 0,   # OM -> Kha
    1: 3,   # SD -> Mas
    2: 0,   # SA -> Kha
    3: 0,   # KW -> Kha
    4: 0,   # QA -> Kha
    5: 2,   # LB -> Lev
    6: 2,   # JO -> Lev
    7: 2,   # SY -> Lev
    8: 1,   # IQ -> IRQ
    9: 4,   # MA -> Mag
    10: 3,  # EG -> Mas
    11: 2,  # PL -> Lev
    12: 0,  # YE -> Kha
    13: 0,  # BH -> Kha
    14: 4,  # DZ -> Mag
    15: 0,  # AE -> Kha
    16: 4,  # TN -> Mag
    17: 4,  # LY -> Mag
}

AMGHADHASAN_TO_5 = {
    "EG": 3,  # Masri
    "LY": 4,  # Maghrebi
    "LB": 2,  # Levantine
    "SD": 3,  # Masri (Sudan is part of Masri in 5-way)
    "MA": 4,  # Maghrebi
}

QADI_5_NAMES = ["Khaleeji", "Iraqi", "Levantine", "Masri", "Maghrebi"]


def load_nadi_18way(path: str = "D:/dacd2026/1_data/raw/qadi.parquet") -> pd.DataFrame:
    """Load NADI 2024 (QADI parquet) with 18-way labels."""
    df = pd.read_parquet(path)
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]].reset_index(drop=True)


def load_nadi_5way(path: str = "D:/dacd2026/1_data/raw/qadi.parquet") -> pd.DataFrame:
    """Load NADI 2024 mapped to 5-way coarse labels."""
    df = load_nadi_18way(path)
    df["label"] = df["label"].map(QADI_18_TO_5)
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    return df.reset_index(drop=True)


def load_amgadhasan_5way(path: str = "D:/dacd2026/1_data/raw/amgadhasan.parquet") -> pd.DataFrame:
    """Load amgadhasan with 5-way mapped labels."""
    df = pd.read_parquet(path)
    df["label"] = df["dialect"].map(AMGHADHASAN_TO_5)
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
    """Sample a single N-way K-shot episode from a labeled dataframe.

    Returns:
      support_idx: list of K indices per class, total N*K
      query_idx: list of Q indices per class, total N*Q
      support_labels: list of N class labels (0..N-1 within episode)
      query_labels: list of N class labels (0..N-1) for each query
    """
    if rng is None:
        rng = random.Random()
    # Group indices by class
    by_class = {}
    for idx, label in zip(df.index, df["label"]):
        by_class.setdefault(int(label), []).append(idx)
    available_classes = list(by_class.keys())
    if len(available_classes) < n_way:
        # Sample with replacement
        chosen_classes = rng.choices(available_classes, k=n_way)
    else:
        chosen_classes = rng.sample(available_classes, n_way)
    # Build support and query
    support_idx, query_idx = [], []
    support_labels, query_labels = [], []
    for episode_label, orig_label in enumerate(chosen_classes):
        candidates = by_class[orig_label]
        if len(candidates) < k_shot + q_query:
            # With replacement
            sampled = rng.choices(candidates, k=k_shot + q_query)
        else:
            sampled = rng.sample(candidates, k_shot + q_query)
        support_idx.extend(sampled[:k_shot])
        query_idx.extend(sampled[k_shot:k_shot + q_query])
        support_labels.extend([episode_label] * k_shot)
        query_labels.extend([episode_label] * q_query)
    return support_idx, query_idx, support_labels, query_labels


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 96):
        self.texts = df["text"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }
