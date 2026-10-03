"""P3: corpus-statistics prior for the prior-quality ablation (1-shot).

kind = 'corpus': cosine distance between per-class word-frequency
centroids built from the dataset's own (unlabelled-for-this-purpose)
tweets -> row-softmax(-D/tau_ling) with the SAME tau as the linguistic
prior, for a fair comparison. Runs only the 3 'corpus' cells at 1-shot;
merges with existing results/prior_quality.json for the table.
"""
import sys, os, json, time, random
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from transformers import AutoTokenizer
from adibench.data import get_dataset, build_fewshot_episode_ex, build_fewshot_episode
from adibench.baselines import Encoder, freeze_early_layers

ARABERT = r"D:\dacd2026\2_models\arabertv02"
OUT = r"D:/dacd2026/adibench_v1/results/prior_quality_corpus.json"
OLD = r"D:/dacd2026/adibench_v1/results/prior_quality.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ARABERT, use_fast=True)
VOCAB_CAP, PER_CLASS_CAP, TAU = 3000, 1500, 0.5


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def enc(texts):
    e = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return e["input_ids"], e["attention_mask"]


def corpus_prior(df, num_classes, seed=42):
    """Row-normalised softmax(-D/tau) from word-frequency centroids."""
    rng = np.random.RandomState(seed)
    # token counts per class (subword ids), capped per class
    counters = [Counter() for _ in range(num_classes)]
    for c in range(num_classes):
        idx = df.index[df["label"] == c].tolist()
        if len(idx) > PER_CLASS_CAP:
            idx = list(rng.choice(idx, PER_CLASS_CAP, replace=False))
        batch = [df.loc[i, "text"] for i in idx]
        for j in range(0, len(batch), 64):
            e = tok(batch[j:j + 64], truncation=True, padding=True,
                    max_length=96, return_tensors="pt")
            for row in e["input_ids"]:
                for t in row.tolist():
                    if t > 2:  # skip specials
                        counters[c][t] += 1
    # global vocab cap by frequency
    total = Counter()
    for cnt in counters:
        total.update(cnt)
    vocab = [w for w, _ in total.most_common(VOCAB_CAP)]
    widx = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    # tf-idf-ish: tf counts, idf from class document frequency
    mat = np.zeros((num_classes, V), dtype=np.float64)
    dfreq = np.zeros(V)
    for c in range(num_classes):
        for w, v in counters[c].items():
            if w in widx:
                mat[c, widx[w]] = float(v)
                dfreq[widx[w]] += 1
    idf = np.log((num_classes + 1) / (dfreq + 1)) + 1.0
    mat *= idf[None, :]
    # L2 normalise rows -> cosine distance matrix
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
    mat = mat / norms
    D = 1.0 - mat @ mat.T
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    # row-softmax(-D/tau), same transform as the linguistic prior
    S = np.exp(-D / TAU)
    S /= (S.sum(axis=1, keepdims=True) + 1e-9)
    print(f"corpus prior: V={V}, diag={np.diag(S).mean():.3f}, "
          f"offdiag={ (S.sum()-np.trace(S))/(num_classes*(num_classes-1)):.4f}", flush=True)
    return S.astype(np.float32)


class PriorProtoNet(nn.Module):
    """CE ProtoNet with pluggable inference-time aggregation (as prior_quality.py)."""
    def __init__(self, prior):
        super().__init__()
        self.encoder = Encoder(embed_dim=256)
        freeze_early_layers(self.encoder.encoder, n_unfreeze=4)
        self.prior = torch.from_numpy(prior)
        self._chosen = None

    def set_episode_classes(self, chosen):
        self._chosen = list(chosen)

    def forward(self, sx, sm, sy, qx, qm):
        sf = self.encoder(sx, sm)
        qf = self.encoder(qx, qm)
        nw = int(sy.max().item()) + 1
        protos = torch.stack([sf[sy == c].mean(0) for c in range(nw)])
        W = self.prior.to(sf.device)
        idx = torch.tensor(self._chosen, dtype=torch.long, device=sf.device)
        W_ep = W[idx][:, idx]
        mixed = 0.7 * protos + 0.3 * (W_ep @ protos)   # same alpha=0.7 as Table 4
        return -torch.cdist(qf, mixed)


def run_cell(dataset, k_shot=1):
    set_seed(42)
    df, meta = get_dataset(dataset)
    prior = corpus_prior(df, meta["num_classes"])
    model = PriorProtoNet(prior).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    for _ in range(500):
        s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, 5, k_shot, 15)
        model._chosen = chosen
        s_ids, s_m = enc([df.iloc[i]["text"] for i in s_idx])
        q_ids, q_m = enc([df.iloc[i]["text"] for i in q_idx])
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        logits = model(s_ids, s_m, s_y, q_ids, q_m)
        loss = F.cross_entropy(logits, q_y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
    model.eval()
    accs = []
    rng = random.Random(42)
    with torch.no_grad():
        for _ in range(100):
            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, 5, k_shot, 15, rng=rng)
            model._chosen = chosen
            s_ids, s_m = enc([df.iloc[i]["text"] for i in s_idx])
            q_ids, q_m = enc([df.iloc[i]["text"] for i in q_idx])
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits = model(s_ids, s_m, s_y, q_ids, q_m)
            accs.append(float((logits.argmax(-1) == q_y).float().mean().item()))
    acc = float(np.mean(accs))
    print(f"[corpus/{dataset}/1shot] acc={acc:.4f}", flush=True)
    del model; torch.cuda.empty_cache()
    return acc


def main():
    out = {}
    for ds in ["nadi_18", "nadi_5", "amgadhasan_5"]:
        out[f"{ds}/1shot/corpus"] = run_cell(ds, 1)
    # merge print with old results
    old = {}
    if os.path.exists(OLD):
        with open(OLD) as f:
            old = json.load(f)
    print("\n== Prior-quality table (1-shot), merged ==")
    print(f"{'dataset':<16}{'none':>8}{'random':>8}{'uniform':>9}{'corpus':>9}{'linguistic':>12}")
    for ds in ["nadi_18", "nadi_5", "amgadhasan_5"]:
        key = lambda k: old.get(f"{ds}/1shot/{k}", float("nan"))
        print(f"{ds:<16}{key('none'):>8.3f}{key('random'):>8.3f}{key('uniform'):>9.3f}"
              f"{out[f'{ds}/1shot/corpus']:>9.3f}{key('linguistic'):>12.3f}")
    with open(OUT, "w") as f:
        json.dump({"corpus": out, "old": old}, f, indent=2)
    print("Saved", OUT)


if __name__ == "__main__":
    main()