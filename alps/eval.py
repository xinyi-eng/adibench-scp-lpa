"""Multi-granularity evaluation for ALPS few-shot Arabic dialect ID.

Evaluates trained models on three benchmarks:
  - NADI 2024 18-way (country-level)
  - QADI 5-way (coarse dialect)
  - amgadhasan 5-way (city-level)

Each benchmark uses N-way K-shot evaluation with 1000 episodes.
Reports mean accuracy and 95% confidence interval.
"""
from __future__ import annotations

import json
import os
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .data import (
    load_nadi_18way, load_nadi_5way, load_amgadhasan_5way,
    build_fewshot_episode, NADI_18_COUNTRIES,
)
from .train import make_model, set_seed, get_device, freeze_early_layers
from .protonet import ProtoNet, L2CProtoNet
from .distance import (
    NADI_18_DISTANCE, QADI_5_DISTANCE, AMGHADHASAN_5_DISTANCE,
    get_linguistic_distance,
)


ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"


def evaluate_fewshot(
    model,
    df: pd.DataFrame,
    n_way: int,
    k_shot: int,
    q_query: int,
    n_episodes: int,
    device,
    tok,
    method: str = "protonet",
    seed: int = 42,
):
    """Run n_episodes random episodes and return mean accuracy."""
    set_seed(seed)
    model.eval()
    correct = 0
    total = 0
    per_episode_acc = []
    with torch.no_grad():
        for _ in range(n_episodes):
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(
                df, n_way, k_shot, q_query,
            )
            s_texts = [df.iloc[i]["text"] for i in s_idx]
            q_texts = [df.iloc[i]["text"] for i in q_idx]
            s_enc = tok(s_texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
            q_enc = tok(q_texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            s_orig = torch.tensor([int(df.iloc[i]["label"]) for i in s_idx], dtype=torch.long, device=device)
            q_orig = torch.tensor([int(df.iloc[i]["label"]) for i in q_idx], dtype=torch.long, device=device)
            if method == "protonet":
                logits, _, _ = model(s_enc["input_ids"], s_enc["attention_mask"], s_y,
                                      q_enc["input_ids"], q_enc["attention_mask"])
            else:
                logits, _, _ = model(s_enc["input_ids"], s_enc["attention_mask"], s_y,
                                      q_enc["input_ids"], q_enc["attention_mask"],
                                      query_orig_labels=q_orig)
            preds = logits.argmax(-1)
            ep_correct = (preds == q_y).sum().item()
            ep_total = q_y.size(0)
            per_episode_acc.append(ep_correct / ep_total)
            correct += ep_correct
            total += ep_total
    acc = correct / max(1, total)
    return {
        "accuracy": acc,
        "n_episodes": n_episodes,
        "std": float(np.std(per_episode_acc)),
        "ci95": 1.96 * float(np.std(per_episode_acc)) / np.sqrt(n_episodes),
        "per_episode_acc": per_episode_acc,
    }


def multi_granularity_eval(
    model,
    method: str,
    n_way: int = 5,
    k_shots: List[int] = [1, 3, 5],
    n_episodes: int = 1000,
    seed: int = 42,
    encoder_path: str = ARABERT_PATH,
):
    """Evaluate on all 3 benchmarks at multiple K-shots."""
    device = get_device()
    tok = AutoTokenizer.from_pretrained(encoder_path, use_fast=True)
    out = {"method": method, "n_way": n_way, "results": {}}
    # NADI 18-way: subsample to 5-way at runtime via df sampling
    print("Loading NADI 18-way...", flush=True)
    df_nadi = load_nadi_18way()
    for k in k_shots:
        res = evaluate_fewshot(model, df_nadi, n_way=n_way, k_shot=k, q_query=15,
                               n_episodes=n_episodes, device=device, tok=tok,
                               method=method, seed=seed)
        out["results"][f"nadi18_{k}shot"] = res
        print(f"  NADI 18-way {k}-shot: acc={res['accuracy']:.4f} ± {res['ci95']:.4f}", flush=True)
    # QADI 5-way (coarse)
    print("Loading QADI 5-way...", flush=True)
    df_qadi = load_nadi_5way()
    for k in k_shots:
        res = evaluate_fewshot(model, df_qadi, n_way=min(n_way, 5), k_shot=k, q_query=15,
                               n_episodes=n_episodes, device=device, tok=tok,
                               method=method, seed=seed)
        out["results"][f"qadi5_{k}shot"] = res
        print(f"  QADI 5-way {k}-shot: acc={res['accuracy']:.4f} ± {res['ci95']:.4f}", flush=True)
    # amgadhasan 5-way (city)
    print("Loading amgadhasan 5-way...", flush=True)
    df_amg = load_amgadhasan_5way()
    for k in k_shots:
        res = evaluate_fewshot(model, df_amg, n_way=min(n_way, 5), k_shot=k, q_query=15,
                               n_episodes=n_episodes, device=device, tok=tok,
                               method=method, seed=seed)
        out["results"][f"amgadhasan5_{k}shot"] = res
        print(f"  amgadhasan 5-way {k}-shot: acc={res['accuracy']:.4f} ± {res['ci95']:.4f}", flush=True)
    return out
