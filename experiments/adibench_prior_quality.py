"""LPA prior-quality ablation: prove the gain comes from linguistic
knowledge, not from any arbitrary smoothing.

Compares 4 prototype-aggregation priors on the 5-way datasets
(nadi_5, amgadhasan_5) and NADI 18 5-way (with correct episode-class
mapping), 5-shot, seed 42:

  1. none       — vanilla ProtoNet (no aggregation)
  2. random     — random symmetric matrix (no linguistic signal)
  3. uniform    — uniform off-diagonal (pure smoothing)
  4. linguistic — the Versteegh/Habash distance matrix (real LPA)

If linguistic > {random, uniform} on all cells, the gain is due to
linguistic knowledge. Uses the SAME encoder config as the main table
(last 4 layers unfrozen).
"""
import sys, os, json
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from adibench.data import build_fewshot_episode_ex, get_dataset
from adibench.dialects import similarity_matrix
from adibench.baselines import Encoder, freeze_early_layers

ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)


class PriorProtoNet(nn.Module):
    """ProtoNet with a pluggable prototype-aggregation prior matrix."""
    def __init__(self, prior_matrix, dataset, alpha=0.7):
        super().__init__()
        self.encoder = Encoder(embed_dim=256)
        freeze_early_layers(self.encoder.encoder, n_unfreeze=4)
        self.alpha = alpha
        self.dataset = dataset
        # prior_matrix is (num_classes, num_classes), row-normalised similarity
        self.prior = None if prior_matrix is None else torch.from_numpy(prior_matrix).float()
        self._episode_classes = None

    def set_episode_classes(self, chosen):
        self._episode_classes = list(chosen)

    def forward(self, sx, sm, sy, qx, qm):
        sf = self.encoder(sx, sm)
        qf = self.encoder(qx, qm)
        nw = int(sy.max().item()) + 1
        protos = torch.stack([sf[sy == c].mean(0) for c in range(nw)])
        if self.prior is not None and self._episode_classes is not None and nw > 1:
            W = self.prior.to(sf.device)
            idx = torch.tensor(self._episode_classes, dtype=torch.long, device=sf.device)
            W_ep = W[idx][:, idx]  # (nw, nw) — subset rows/cols for this episode
            mixed = self.alpha * protos + (1 - self.alpha) * (W_ep @ protos)
        else:
            mixed = protos
        return -torch.cdist(qf, mixed)


def make_prior(kind, dataset, seed=42):
    n_classes = {"nadi_18": 18, "nadi_5": 5, "amgadhasan_5": 5}[dataset]
    if kind == "none":
        return None
    rng = np.random.RandomState(seed)
    if kind == "random":
        M = rng.rand(n_classes, n_classes)
        np.fill_diagonal(M, 0)
        M = M / (M.sum(axis=1, keepdims=True) + 1e-9)
        return M
    if kind == "uniform":
        M = np.ones((n_classes, n_classes))
        np.fill_diagonal(M, 0)
        M = M / (M.sum(axis=1, keepdims=True) + 1e-9)
        return M
    if kind == "linguistic":
        # use each dataset's OWN linguistic similarity matrix
        return similarity_matrix(dataset)
    raise ValueError(kind)


def train(model, df, n_way, k_shot, q_query, n_episodes=500):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    for _ in range(n_episodes):
        s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)
        model.set_episode_classes(chosen)
        s_ids, s_mask = encode_batch([df.iloc[i]["text"] for i in s_idx])
        q_ids, q_mask = encode_batch([df.iloc[i]["text"] for i in q_idx])
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        logits = model(s_ids, s_mask, s_y, q_ids, q_mask)
        loss = F.cross_entropy(logits, q_y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()


def encode_batch(texts):
    e = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return e["input_ids"], e["attention_mask"]


def eval_model(model, df, n_way, k_shot, q_query, n_eval=100, seed=42):
    import random
    rng = random.Random(seed)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_eval):
            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query, rng=rng)
            model.set_episode_classes(chosen)
            s_ids, s_mask = encode_batch([df.iloc[i]["text"] for i in s_idx])
            q_ids, q_mask = encode_batch([df.iloc[i]["text"] for i in q_idx])
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits = model(s_ids, s_mask, s_y, q_ids, q_mask)
            correct += (logits.argmax(-1) == q_y).sum().item()
            total += q_y.size(0)
    return correct / max(1, total)


def run_cell(dataset, kind, n_way=5, k_shot=5, q_query=15, n_episodes=500, n_eval=100):
    df, meta = get_dataset(dataset)
    prior = make_prior(kind, dataset)
    model = PriorProtoNet(prior, dataset).to(device)
    train(model, df, n_way, k_shot, q_query, n_episodes)
    acc = eval_model(model, df, n_way, k_shot, q_query, n_eval)
    del model
    torch.cuda.empty_cache()
    return acc


def main():
    datasets = ["nadi_18", "nadi_5", "amgadhasan_5"]
    priors = ["none", "random", "uniform", "linguistic"]
    results = {}
    for k_shot in [1, 5]:
        print("=" * 76)
        print(f"LPA prior-quality ablation (5-way {k_shot}-shot, seed 42, frozen enc)")
        print("=" * 76)
        print(f"{'Dataset':<16}{'none':<10}{'random':<10}{'uniform':<10}{'linguistic':<12}")
        print("-" * 76)
        for ds in datasets:
            row = {}
            for p in priors:
                acc = run_cell(ds, p, k_shot=k_shot)
                row[p] = acc
                results[f"{ds}/{k_shot}shot/{p}"] = acc
            ling = row["linguistic"] - row["none"]
            rand = row["random"] - row["none"]
            unif = row["uniform"] - row["none"]
            print(f"{ds:<16}{row['none']:<10.4f}{row['random']:<10.4f}{row['uniform']:<10.4f}{row['linguistic']:<12.4f}")
            print(f"{'':<16}gain vs none: linguistic={ling:+.4f}  random={rand:+.4f}  uniform={unif:+.4f}")
            print(flush=True)

    with open(r"D:/dacd2026/adibench_v1/results/prior_quality.json", "w") as f:
        json.dump(results, f, indent=2)

    # Verdict per shot setting
    for k in [1, 5]:
        ling_best = all(
            results[f"{ds}/{k}shot/linguistic"] >= max(results[f"{ds}/{k}shot/random"],
                                                        results[f"{ds}/{k}shot/uniform"])
            for ds in datasets)
        ling_pos = all(results[f"{ds}/{k}shot/linguistic"] > results[f"{ds}/{k}shot/none"]
                       for ds in datasets)
        print(f"\n[{k}-shot] linguistic is best-or-tied among {random,uniform} on ALL datasets: {ling_best}")
        print(f"[{k}-shot] linguistic beats 'none' (no aggregation) on ALL datasets: {ling_pos}")
    print("\nSaved prior_quality.json")


if __name__ == "__main__":
    main()