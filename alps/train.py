"""Training and evaluation routines for ALPS few-shot.

Episodic training: sample N-way K-shot episodes from a labeled
dataframe, train the encoder+prototype model end-to-end.

We do NOT fine-tune the encoder fully (would be slow on 8GB GPU).
We use:
  - Encoder: AraBERTv2-base, last 4 layers unfrozen, rest frozen
  - Projection: small linear head, fully trainable
  - Episodic: 1 episode = 1 batch, AdamW lr=1e-4 on trainable params
"""
from __future__ import annotations

import os
import time
import json
import random
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .protonet import ProtoNet, L2CProtoNet
from .data import build_fewshot_episode, TextDataset
from .curriculum import GeoCurriculum
from .distance import get_linguistic_distance


ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def freeze_early_layers(encoder, n_unfreeze_last=4):
    """Freeze all encoder parameters except last n_unfreeze_last layers."""
    # AraBERTv2 has encoder.layer.0..11
    for name, p in encoder.named_parameters():
        if "embeddings" in name:
            p.requires_grad = False
        elif "encoder.layer" in name:
            # Check layer number
            try:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                if layer_num < 12 - n_unfreeze_last:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            except (IndexError, ValueError):
                p.requires_grad = False
        elif "pooler" in name:
            p.requires_grad = False
        else:
            p.requires_grad = False


def make_model(method, num_classes, encoder_path=ARABERT_PATH):
    """Create the model based on method name."""
    if method == "protonet":
        return ProtoNet(encoder_path)
    if method == "l2c":
        return L2CProtoNet(encoder_path, num_classes=num_classes, alpha=1.0, beta=0.5, gamma=2.0)
    if method == "l2c_strong":
        return L2CProtoNet(encoder_path, num_classes=num_classes, alpha=1.0, beta=1.0, gamma=3.0)
    if method == "l2c_weak":
        return L2CProtoNet(encoder_path, num_classes=num_classes, alpha=1.0, beta=0.2, gamma=1.0)
    raise ValueError(method)


def train_fewshot(
    method: str,
    df_train: pd.DataFrame,
    num_classes: int,
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 15,
    n_episodes: int = 2000,
    eval_every: int = 200,
    n_eval_episodes: int = 200,
    df_eval: Optional[pd.DataFrame] = None,
    use_curriculum: bool = False,
    seed: int = 42,
    lr: float = 1e-4,
    encoder_path: str = ARABERT_PATH,
    n_unfreeze_last: int = 4,
    log_file: Optional[str] = None,
):
    """Train a few-shot model with episodic training.

    Args:
      method: "protonet" | "l2c" | "l2c_strong" | "l2c_weak"
      df_train: training dataframe (text, label)
      num_classes: total number of classes in df_train
      n_way, k_shot, q_query: episode config
      n_episodes: total training episodes
      eval_every: evaluate on validation set every N episodes
      df_eval: optional validation dataframe; if None, use df_train
      use_curriculum: whether to use GeoCurriculum
    """
    set_seed(seed)
    device = get_device()
    # Build model
    model = make_model(method, num_classes, encoder_path=encoder_path).to(device)
    # Freeze early layers
    freeze_early_layers(model.encoder.encoder, n_unfreeze_last=n_unfreeze_last)
    # Count trainable params
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{method}] trainable params: {n_train/1e6:.2f}M / total {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    # Build curriculum
    if use_curriculum:
        curriculum = GeoCurriculum(num_classes, total_steps=n_episodes)
    # Build tokenizer (just for verification, not used in encode step)
    tok = AutoTokenizer.from_pretrained(encoder_path, use_fast=True)
    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    # Training loop
    history = []
    t0 = time.time()
    best_acc = 0.0
    for ep in range(1, n_episodes + 1):
        model.train()
        if use_curriculum:
            s_idx, q_idx, s_lab, q_lab = curriculum.sample_episode(
                df_train, n_way, k_shot, q_query, ep
            )
        else:
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(
                df_train, n_way, k_shot, q_query,
            )
        s_texts = [df_train.iloc[i]["text"] for i in s_idx]
        q_texts = [df_train.iloc[i]["text"] for i in q_idx]
        # Map original labels to episode labels (already done in build_fewshot_episode)
        # Tokenize support
        s_enc = tok(s_texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
        q_enc = tok(q_texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        # Original labels for L2C
        s_orig = torch.tensor([int(df_train.iloc[i]["label"]) for i in s_idx], dtype=torch.long, device=device)
        q_orig = torch.tensor([int(df_train.iloc[i]["label"]) for i in q_idx], dtype=torch.long, device=device)
        # Forward
        if method == "protonet":
            logits, _, _ = model(s_enc["input_ids"], s_enc["attention_mask"], s_y,
                                  q_enc["input_ids"], q_enc["attention_mask"])
        else:  # l2c variants
            logits, _, _ = model(s_enc["input_ids"], s_enc["attention_mask"], s_y,
                                  q_enc["input_ids"], q_enc["attention_mask"],
                                  query_orig_labels=q_orig)
        # Loss = cross-entropy on query
        loss = F.cross_entropy(logits, q_y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if ep % eval_every == 0 or ep == n_episodes:
            eval_acc = evaluate_fewshot(
                model, df_eval if df_eval is not None else df_train,
                n_way, k_shot, q_query, n_eval_episodes, device, tok, method=method,
            )
            elapsed = time.time() - t0
            log_msg = f"[{method}] ep={ep}/{n_episodes}  loss={loss.item():.4f}  eval_acc={eval_acc:.4f}  elapsed={elapsed:.0f}s"
            print(log_msg, flush=True)
            if log_file is not None:
                with open(log_file, "a") as f:
                    f.write(log_msg + "\n")
            history.append({"episode": ep, "loss": loss.item(), "eval_acc": eval_acc})
            if eval_acc > best_acc:
                best_acc = eval_acc
        if ep % 50 == 0:
            torch.cuda.empty_cache()
    return {"method": method, "best_acc": best_acc, "history": history,
            "n_episodes": n_episodes, "n_way": n_way, "k_shot": k_shot,
            "q_query": q_query, "use_curriculum": use_curriculum}


def evaluate_fewshot(
    model,
    df,
    n_way,
    k_shot,
    q_query,
    n_episodes,
    device,
    tok,
    method: str = "protonet",
):
    """Evaluate model on n_episodes random episodes."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for _ in range(n_episodes):
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)
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
            correct += (preds == q_y).sum().item()
            total += q_y.size(0)
    return correct / max(1, total)
