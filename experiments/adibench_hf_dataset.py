"""ADIBench: HuggingFace datasets wrapper.

A unified loader for ADIBench's three Arabic dialect identification datasets.
Designed to be uploaded to HuggingFace as a community dataset and called
via `load_dataset("adibench/nadi_18", split="train")` etc.

Usage:
    from datasets import load_dataset
    ds = load_dataset("adibench/nadi_18", split="train")
    print(ds[0])  # {'text': '...', 'label': 5}
"""
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import datasets
from datasets import (
    DatasetDict,
    Dataset,
    Features,
    Value,
    ClassLabel,
    Split,
    BuilderConfig,
    GeneratorBasedBuilder,
)


_CITATION = """\
@inproceedings{adibench2026,
  title={ADIBench: A Multi-Granularity Benchmark for Few-Shot Arabic Dialect Identification},
  author={Anonymous},
  booktitle={Anonymous submission},
  year={2026},
}
"""

_DESCRIPTION = """\
ADIBench is a benchmark for evaluating few-shot learning methods on
long-tail Arabic dialect identification. It unifies three standard
public datasets:

- **NADI 2024 18-way** (QADI parquet): 18 country dialects, 440K tweets.
- **NADI 2024 5-way (coarse)**: 5 dialect groups (Khaleeji/Iraqi/Levantine/Masri/Maghrebi).
- **amgadhasan 5-city**: 5 city dialects (EG/LY/LB/SD/MA), 147K tweets.

Each dataset is designed for 5-way or N-way K-shot evaluation.
"""

_HOMEPAGE = "https://github.com/anonymous/adibench"


class AdibenchDataset(GeneratorBasedBuilder):
    """ADIBench: A Multi-Granularity Benchmark for Few-Shot Arabic Dialect ID."""

    BUILDER_CONFIGS = [
        BuilderConfig(name="nadi_18", version=datasets.Version("1.0.0"),
                      description="NADI 2024 18 country dialects"),
        BuilderConfig(name="nadi_5", version=datasets.Version("1.0.0"),
                      description="NADI 2024 5 coarse dialect groups"),
        BuilderConfig(name="amgadhasan_5", version=datasets.Version("1.0.0"),
                      description="amgadhasan 5 city dialects"),
    ]
    DEFAULT_CONFIG_NAME = "nadi_18"

    def _info(self):
        n_classes = {"nadi_18": 18, "nadi_5": 5, "amgadhasan_5": 5}[self.config.name]
        label_names = self._get_label_names()
        features = Features({
            "text": Value("string"),
            "label": ClassLabel(names=label_names),
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION + f"\n\nLoaded config: {self.config.name}",
            features=features,
            citation=_CITATION,
            homepage=_HOMEPAGE,
        )

    def _get_label_names(self):
        if self.config.name == "nadi_18":
            return ["OM", "SD", "SA", "KW", "QA", "LB", "JO", "SY",
                    "IQ", "MA", "EG", "PL", "YE", "BH", "DZ", "AE", "TN", "LY"]
        if self.config.name == "nadi_5":
            return ["Khaleeji", "Iraqi", "Levantine", "Masri", "Maghrebi"]
        if self.config.name == "amgadhasan_5":
            return ["EG", "LY", "LB", "SD", "MA"]

    def _split_generators(self, dl_manager):
        # Path to local parquet files (hosted on HF datasets hub as well)
        data_dir = Path(os.environ.get("ADIBENCH_DATA_DIR",
                                       "D:/dacd2026/1_data/raw"))
        paths = {
            "nadi_18": data_dir / "qadi.parquet",
            "nadi_5": data_dir / "qadi.parquet",
            "amgadhasan_5": data_dir / "amgadhasan.parquet",
        }
        parquet_path = paths[self.config.name]
        return [
            datasets.SplitGenerator(
                name=Split.TRAIN,
                gen_kwargs={"parquet_path": str(parquet_path),
                            "config": self.config.name},
            ),
        ]

    def _generate_examples(self, parquet_path, config):
        df = pd.read_parquet(parquet_path)
        if "label" not in df.columns and "dialect" in df.columns:
            df["label"] = df["dialect"]
        df["label"] = df["label"].astype(int)
        if config == "nadi_5":
            # Map 18-way to 5-way via QADI_18_TO_5
            mapping = {0: 0, 1: 3, 2: 0, 3: 0, 4: 0, 5: 2, 6: 2, 7: 2, 8: 1,
                       9: 4, 10: 3, 11: 2, 12: 0, 13: 0, 14: 4, 15: 0, 16: 4, 17: 4}
            df["label"] = df["label"].map(mapping)
            df = df.dropna(subset=["label"]).copy()
            df["label"] = df["label"].astype(int)
        for i, row in df[["text", "label"]].iterrows():
            yield i, {"text": str(row["text"]), "label": int(row["label"])}


if __name__ == "__main__":
    # Smoke test
    for cfg_name in ["nadi_18", "nadi_5", "amgadhasan_5"]:
        ds = AdibenchDataset(config=BuilderConfig(name=cfg_name))
        info = ds._info()
        print(f"\n=== {cfg_name} ===")
        for k, v in info.features.items():
            print(f"  {k}: {v}")
        parquet_path = ("D:/dacd2026/1_data/raw/qadi.parquet" if "nadi" in cfg_name
                       else "D:/dacd2026/1_data/raw/amgadhasan.parquet")
        n_samples = sum(1 for _ in ds._generate_examples(parquet_path, cfg_name))
        print(f"  total samples: {n_samples}")


def load_adibench(config: str = "nadi_18", data_dir: str = None):
    """Local convenience loader (no HF Hub needed).

    Usage:
        from experiments.adibench_hf_dataset import load_adibench
        ds = load_adibench('nadi_18')
        print(ds[0])  # {'text': '...', 'label': 5}
    """
    import datasets as hfds
    if data_dir is None:
        data_dir = os.environ.get("ADIBENCH_DATA_DIR", "D:/dacd2026/1_data/raw")
    parquet = ("qadi.parquet" if "nadi" in config else "amgadhasan.parquet")
    parquet_path = os.path.join(data_dir, parquet)
    df = pd.read_parquet(parquet_path)
    if "label" not in df.columns and "dialect" in df.columns:
        df["label"] = df["dialect"]
    df["label"] = df["label"].astype(int)
    if config == "nadi_5":
        mapping = {0: 0, 1: 3, 2: 0, 3: 0, 4: 0, 5: 2, 6: 2, 7: 2, 8: 1,
                   9: 4, 10: 3, 11: 2, 12: 0, 13: 0, 14: 4, 15: 0, 16: 4, 17: 4}
        df["label"] = df["label"].map(mapping)
        df = df.dropna(subset=["label"]).copy()
        df["label"] = df["label"].astype(int)
    return hfds.Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))