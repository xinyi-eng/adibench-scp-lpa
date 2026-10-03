"""Data loaders for DACD-Bench.

Loaders for:
    - NADI 2024 shared task
    - MADAR Twitter corpus
    - QADI (5-way)

Plus a builder that combines all three into DACD-Bench.
"""
from __future__ import annotations

import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# 5-way dialect mapping (consistent with original DACD paper)
DIALECT_5WAY = {
    "kha": 0, "khaleeji": 0, "gulf": 0, "sa": 0, "ae": 0, "kw": 0, "om": 0, "qa": 0, "bh": 0, "ye": 0,
    "irq": 1, "iraqi": 1, "iq": 1,
    "lev": 2, "levantine": 2, "syr": 2, "lb": 2, "jo": 2, "ps": 2,
    "msa": 3, "egyptian": 3, "egy": 3, "sudan": 3, "sd": 3, "masri": 3,
    "maghrebi": 4, "mor": 4, "alg": 4, "tun": 4, "ly": 4, "ma": 4, "dza": 4, "tn": 4,
}
DIALECT_NAMES = ["Khaleji", "Iraqi", "Levantine", "Masri", "Maghrebi"]

# QADI 18-way -> 5-way mapping (label order from Abdelrahman-Rezk/Arabic_Dialect_Identification)
# https://huggingface.co/datasets/Abdelrahman-Rezk/Arabic_Dialect_Identification
# 18 labels: OM, SD, SA, KW, QA, LB, JO, SY, IQ, MA, EG, PL, YE, BH, DZ, AE, TN, LY
QADI_18_TO_5 = {
    # Khaleji (Gulf)
    0: 0,   # OM
    2: 0,   # SA
    3: 0,   # KW
    4: 0,   # QA
    12: 0,  # YE
    13: 0,  # BH
    15: 0,  # AE
    # Iraqi
    8: 1,   # IQ
    # Levantine
    5: 2,   # LB
    6: 2,   # JO
    7: 2,   # SY
    11: 2,  # PL
    # Masri (Egyptian + Sudanese)
    1: 3,   # SD
    10: 3,  # EG
    # Maghrebi
    9: 4,   # MA
    14: 4,  # DZ
    16: 4,  # TN
    17: 4,  # LY
}


def load_nadi(path: str = "D:/dacd2026/1_data/raw/nadi.csv") -> pd.DataFrame:
    """NADI 2024 shared task: 18 dialect labels -> 5-way via QADI_18_TO_5 mapping.

    Reads the parquet file from Abdelrahman-Rezk/Arabic_Dialect_Identification which
    IS the QADI 2024 data, since the parquet is in our D drive from earlier download.
    """
    if path.endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # QADI uses integer labels 0-17; map to our 5-way scheme
    if "label" in df.columns and df["label"].dtype.kind in ("i", "u"):
        df["label_5way"] = df["label"].map(QADI_18_TO_5)
    elif "dialect" in df.columns:
        df["label_5way"] = df["dialect"].apply(_normalize_country_to_5)
    else:
        raise ValueError("NADI/QADI file must have 'label' (int) or 'dialect' (str) column")
    df["text"] = df["text"].astype(str)
    # drop original label/dialect, keep only mapped 5-way
    drop_cols = [c for c in ("label", "dialect", "id") if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.rename(columns={"label_5way": "label"})
    df = df[["text", "label"]].dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df = df[["text", "label"]].reset_index(drop=True)
    return df


def _normalize_country_to_5(country_code: str) -> int:
    """Fallback: ISO country code -> 5-way class (for amgadhasan-style data)."""
    c = str(country_code).upper().strip()
    if c in ("EG", "SD"):
        return 3
    if c in ("LB", "SY", "JO", "PS", "PL"):
        return 2
    if c in ("SA", "KW", "AE", "OM", "QA", "BH", "YE"):
        return 0
    if c in ("IQ",):
        return 1
    if c in ("MA", "TN", "DZ", "LY"):
        return 4
    return -1


def load_amgadhasan(path: str = "D:/dacd2026/1_data/raw/amgadhasan.parquet") -> pd.DataFrame:
    """amgadhasan/arabic_tweets_dialects: 5 city codes (EG, LY, LB, SD, MA) -> 5-way.

    Note: this only has 3 of our 5-way classes (Levantine, Masri, Maghrebi).
    Missing: Khaleji, Iraqi.
    """
    df = pd.read_parquet(path)
    if "dialect" in df.columns:
        df["label"] = df["dialect"].apply(_normalize_country_to_5)
    elif "label" in df.columns:
        df["label"] = df["label"].apply(_normalize_country_to_5)
    df = df[["text", "label"]].dropna()
    df = df[df["label"] >= 0].copy()
    df["text"] = df["text"].astype(str)
    return df.reset_index(drop=True)



def normalize_label(raw: str) -> int:
    """Map a raw label string to one of the 5 dialect indices, or -1 if unknown."""
    if raw is None:
        return -1
    s = str(raw).strip().lower()
    s = re.sub(r"[^a-z]+", "", s)
    if s in DIALECT_5WAY:
        return DIALECT_5WAY[s]
    # city code mapping fallback
    for k, v in DIALECT_5WAY.items():
        if k in s:
            return v
    return -1


def _load_csv_with_label(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna()
    df["text"] = df[text_col].astype(str)
    df["label"] = df[label_col].apply(normalize_label)
    df = df[df["label"] >= 0]
    return df.reset_index(drop=True)


def subsample_imbalance(df: pd.DataFrame, target_ratio: float, majority_class: int = 0,
                         random_state: int = 42) -> pd.DataFrame:
    """Subsample minority classes so majority:minority ratio = target_ratio.

    All minority classes are scaled to have the same n.
    """
    rng = np.random.default_rng(random_state)
    counts = df["label"].value_counts().to_dict()
    n_majority = counts.get(majority_class, max(counts.values()))
    n_minority = max(1, int(n_majority / target_ratio))
    pieces = []
    for label, n in counts.items():
        n_keep = n_majority if label == majority_class else n_minority
        sub = df[df["label"] == label]
        if len(sub) >= n_keep:
            sub = sub.sample(n=n_keep, random_state=random_state)
        else:
            # upsample
            sub = sub.sample(n=n_keep, replace=True, random_state=random_state)
        pieces.append(sub)
    out = pd.concat(pieces, ignore_index=True)
    out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return out


def build_dacd_bench(
    qadi_path: str = "D:/dacd2026/1_data/raw/qadi.parquet",
    madar_path: str = "D:/dacd2026/1_data/raw/madar.csv",
    nadi_path: str = "D:/dacd2026/1_data/raw/nadi.csv",
    amgadhasan_path: str = "D:/dacd2026/1_data/raw/amgadhasan.parquet",
    imbalance: float = 7000.0,
    out_train_path: str = "D:/dacd2026/1_data/processed/train.csv",
    out_val_path: str = "D:/dacd2026/1_data/processed/val.csv",
    val_frac: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the unified DACD-Bench train/val splits.

    Combines QADI + MADAR + NADI, subsamples to target imbalance,
    stratified 85/15 train/val split.
    """
    qadi = load_nadi(qadi_path)  # QADI 18-way via parquet
    amgad = load_amgadhasan(amgadhasan_path)  # 5-way already
    print(f"QADI:    {len(qadi):>5d} rows")
    print(f"amgad:   {len(amgad):>5d} rows")
    # explicit column alignment to avoid duplicate-column concat issues
    df = pd.concat([qadi[["text","label"]].reset_index(drop=True),
                    amgad[["text","label"]].reset_index(drop=True)], ignore_index=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Combined (deduped): {len(df):>5d} rows")
    # Find majority class
    majority = int(df["label"].value_counts().idxmax())
    df = subsample_imbalance(df, target_ratio=imbalance,
                            majority_class=majority, random_state=random_state)
    print(f"After subsample to 1:{imbalance}: {len(df):>5d} rows")
    # stratified 85/15 split
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(df, test_size=val_frac, stratify=df["label"],
                              random_state=random_state)
    tr = tr.reset_index(drop=True)
    va = va.reset_index(drop=True)
    os.makedirs(os.path.dirname(out_train_path), exist_ok=True)
    tr.to_csv(out_train_path, index=False)
    va.to_csv(out_val_path, index=False)
    print(f"Saved train ({len(tr)}) and val ({len(va)}) to {out_train_path}, {out_val_path}")
    return tr, va


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 64):
        self.texts = df["text"].tolist()
        raw_labels = df["label"].astype(int).tolist()
        # remap labels to [0, K-1] contiguous range
        unique = sorted(set(raw_labels))
        self.label_map = {v: i for i, v in enumerate(unique)}
        self.num_classes_from_data = len(unique)
        self.labels = [self.label_map[v] for v in raw_labels]
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
