"""ADIBench unified baseline runner.

Runs all 8 baselines x 3 datasets x (5w1s, 5w5s) x 3 seeds.
Each run writes a JSON to results/ with naming:
  {method}_{dataset}_{n}way_{k}shot_seed{s}.json
"""
import sys, os, json, time, argparse
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
sys.path.insert(0, r"D:/dacd2026/site-packages")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import Counter
from transformers import AutoTokenizer

from adibench.data import (
    load_nadi_18way, load_nadi_5way, load_amgadhasan_5city,
    build_fewshot_episode, list_datasets, get_dataset, DATASETS,
)
from adibench.baselines import (
    ProtoNet, MatchingNet, RelationNet, FineTune, Focal, CB, MAML, Random,
    CFProtoNet, cf_protonet_supcon_step, LinguisticProtoNet, lpa_protonet_step, SoftAnchorProtoNet, softanchor_step,
)


ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
OUT_DIR = r"D:/dacd2026/adibench_v1/results"
LOG_DIR = r"D:/dacd2026/adibench_v1/experiments/logs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(method, num_classes):
    if method == "protonet":
        return ProtoNet()
    if method == "matchingnet":
        return MatchingNet()
    if method == "relationnet":
        return RelationNet()
    if method == "finetune":
        return FineTune(num_classes=num_classes)
    if method == "focal":
        return Focal(num_classes=num_classes)
    if method == "cb":
        return CB(num_classes=num_classes)
    if method == "maml":
        return MAML(num_classes=num_classes)
    if method == "random":
        return Random(num_classes=min(num_classes, 5))
    if method == "cf_protonet":
        return CFProtoNet()
    if method == "lpa_protonet":
        return LinguisticProtoNet()
    if method == "scp_lpa":
        return LinguisticProtoNet()
    if method in ("softanchor", "sa_proto"):
        return SoftAnchorProtoNet()
    raise ValueError(method)


def compute_acc(logits, labels):
    return (logits.argmax(-1) == labels).float().mean().item()


def evaluate(model, df, n_way, k_shot, q_query, n_episodes, device, tok, method):
    model.eval()
    correct, total = 0, 0
    per_ep_acc = []
    with torch.no_grad():
        for _ in range(n_episodes):
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)
            s_texts = [df.iloc[i]["text"] for i in s_idx]
            q_texts = [df.iloc[i]["text"] for i in q_idx]
            s_enc = tok(s_texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
            q_enc = tok(q_texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            if method == "random":
                logits, _, _ = model(s_enc["input_ids"], s_enc["attention_mask"], s_y,
                                      q_enc["input_ids"], q_enc["attention_mask"])
            else:
                logits, _, _ = model(s_enc["input_ids"], s_enc["attention_mask"], s_y,
                                      q_enc["input_ids"], q_enc["attention_mask"])
            ep_correct = (logits.argmax(-1) == q_y).sum().item()
            ep_total = q_y.size(0)
            per_ep_acc.append(ep_correct / ep_total)
            correct += ep_correct
            total += ep_total
    return correct / max(1, total), float(np.std(per_ep_acc))


def train_method(model, df, n_way, k_shot, q_query, n_episodes, device, tok, method):
    """Train a model with episodic training (only for methods that need training)."""
    if method == "random":
        # No training needed; evaluate only
        return None
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return None
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for ep in range(1, n_episodes + 1):
        s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)
        s_texts = [df.iloc[i]["text"] for i in s_idx]
        q_texts = [df.iloc[i]["text"] for i in q_idx]
        s_enc = tok(s_texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
        q_enc = tok(q_texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        if method in ("finetune", "focal", "cb"):
            loss = model.loss(s_enc["input_ids"], s_enc["attention_mask"], s_y,
                              q_enc["input_ids"], q_enc["attention_mask"], q_y)
        elif method == "maml":
            # MAML uses predict + adapt
            q_logits, _, _ = model.adapt_and_predict(s_enc["input_ids"], s_enc["attention_mask"], s_y,
                                                    q_enc["input_ids"], q_enc["attention_mask"])
            loss = F.cross_entropy(q_logits, q_y)
        elif method == "cf_protonet":
            loss, _, _ = cf_protonet_supcon_step(
                model, s_enc["input_ids"], s_enc["attention_mask"], s_y,
                q_enc["input_ids"], q_enc["attention_mask"], q_y,
                temperature=model.supcon_temperature, ce_weight=0.5)
        else:
            # Few-shot methods: use query loss
            logits, _, _ = model(s_enc["input_ids"], s_enc["attention_mask"], s_y,
                                  q_enc["input_ids"], q_enc["attention_mask"])
            loss = F.cross_entropy(logits, q_y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        if ep % 100 == 0:
            torch.cuda.empty_cache()
    return time.time() - t0


def run_one(method, dataset, n_way, k_shot, seed, n_episodes=500, n_eval=200):
    """Run one (method, dataset, n_way, k_shot, seed) cell."""
    set_seed(seed)
    device = get_device()
    tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)
    df, meta = get_dataset(dataset)
    num_classes = meta["num_classes"]
    n_way = min(n_way, num_classes)
    model = make_model(method, num_classes).to(device)
    # Set class frequency for CF-ProtoNet
    if method == "cf_protonet":
        freq = torch.tensor(
            [df[df["label"] == c].shape[0] for c in range(num_classes)],
            dtype=torch.float)
        model.set_class_freq(freq.to(device))
    if method in ("lpa_protonet", "scp_lpa"):
        model.set_dataset(dataset)
    if method in ("softanchor", "sa_proto"):
        model.set_dataset(dataset, num_classes)
    # Train
    t_train = train_method(model, df, n_way, k_shot, q_query=15,
                            n_episodes=n_episodes, device=device, tok=tok, method=method)
    # Evaluate
    t0 = time.time()
    acc, std = evaluate(model, df, n_way, k_shot, q_query=15,
                        n_episodes=n_eval, device=device, tok=tok, method=method)
    t_eval = time.time() - t0
    result = {
        "method": method, "dataset": dataset, "n_way": n_way, "k_shot": k_shot,
        "seed": seed, "n_episodes": n_episodes, "n_eval": n_eval,
        "accuracy": acc, "std": std, "ci95": 1.96 * std / np.sqrt(n_eval),
        "train_time_s": t_train, "eval_time_s": t_eval,
    }
    # Save
    out_name = f"{method}_{dataset}_{n_way}way_{k_shot}shot_seed{seed}.json"
    with open(os.path.join(OUT_DIR, out_name), "w") as f:
        json.dump(result, f, indent=2)
    t_train_str = f"{t_train:.0f}s" if t_train is not None else "n/a"
    print(f"[{method}/{dataset}/{n_way}w{k_shot}s/s{seed}] acc={acc:.4f} ± {result['ci95']:.4f}  train={t_train_str}  eval={t_eval:.0f}s", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=["protonet", "matchingnet", "finetune"])
    parser.add_argument("--datasets", nargs="+", default=["nadi_18", "nadi_5", "amgadhasan_5"])
    parser.add_argument("--n_ways", nargs="+", type=int, default=[5])
    parser.add_argument("--k_shots", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--n_eval", type=int, default=200)
    args = parser.parse_args()
    print(f"ADIBench runner: methods={args.methods}, datasets={args.datasets}, "
          f"n_ways={args.n_ways}, k_shots={args.k_shots}, seeds={args.seeds}", flush=True)
    for method in args.methods:
        for dataset in args.datasets:
            for n_way in args.n_ways:
                for k_shot in args.k_shots:
                    for seed in args.seeds:
                        try:
                            run_one(method, dataset, n_way, k_shot, seed,
                                    args.n_episodes, args.n_eval)
                        except Exception as e:
                            print(f"FAILED: {method} {dataset} {n_way}w{k_shot}s s{seed}: {e}", flush=True)
    print("DONE", flush=True)
