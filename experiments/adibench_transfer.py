"""Cross-dataset transfer for ADIBench.

Trains a method on source_dataset, then evaluates few-shot on target_dataset
(frozen encoder). This tests whether the learned encoder + LPA prior transfer
to a different ADI corpus.
"""
import sys, os, json, time, argparse, io
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from transformers import AutoTokenizer

from adibench.baselines import (
    ProtoNet, MatchingNet, RelationNet, FineTune, Focal, CB, MAML,
    CFProtoNet, cf_protonet_supcon_step, LinguisticProtoNet, lpa_protonet_step,
)
from adibench.data import build_fewshot_episode, build_fewshot_episode_ex, get_dataset

ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
OUT_DIR = r"D:/dacd2026/adibench_v1/results"
LOG_DIR = r"D:/dacd2026/adibench_v1/experiments/logs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)


def encode_batch(texts):
    enc = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return enc["input_ids"], enc["attention_mask"]


def make_model(method, num_classes):
    if method == "protonet":
        return ProtoNet()
    if method == "scp":
        return CFProtoNet(ldam_margin=0.0, prior_alpha=0.0)
    if method == "lpa":
        return LinguisticProtoNet()
    if method == "scp_lpa":
        return LinguisticProtoNet()
    raise ValueError(method)


def train_model(model, df, n_way, k_shot, q_query, n_episodes, method):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    t0 = time.time()
    for ep in range(n_episodes):
        if method in ("cf_protonet", "scp", "scp_lpa"):
            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)
            model.set_episode_classes(chosen)
        else:
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)
        s_texts = [df.iloc[i]["text"] for i in s_idx]
        q_texts = [df.iloc[i]["text"] for i in q_idx]
        s_ids, s_mask = encode_batch(s_texts)
        q_ids, q_mask = encode_batch(q_texts)
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        if method in ("cf_protonet", "scp"):
            loss, _, _ = cf_protonet_supcon_step(
                model, s_ids, s_mask, s_y, q_ids, q_mask, q_y,
                temperature=model.supcon_temperature,
                ce_weight=getattr(model, "ce_weight", 0.5),
                supcon_weight=getattr(model, "supcon_weight", 1.0))
        elif method in ("lpa", "scp_lpa"):
            loss, _, _ = lpa_protonet_step(
                model, s_ids, s_mask, s_y, q_ids, q_mask, q_y,
                temperature=model.supcon_temperature,
                ce_weight=getattr(model, "ce_weight", 0.5),
                supcon_weight=getattr(model, "supcon_weight", 1.0))
        else:
            logits, _, _ = model(s_ids, s_mask, s_y, q_ids, q_mask)
            loss = F.cross_entropy(logits, q_y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
    print(f"  trained {n_episodes} episodes in {time.time()-t0:.0f}s", flush=True)


def eval_model(model, df, n_way, k_shot, q_query, n_eval, method):
    model.eval()
    correct, total = 0, 0
    per_ep = []
    with torch.no_grad():
        for _ in range(n_eval):
            if method in ("cf_protonet", "scp", "scp_lpa", "lpa"):
                s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)
                model.set_episode_classes(chosen)
            else:
                s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)
            s_texts = [df.iloc[i]["text"] for i in s_idx]
            q_texts = [df.iloc[i]["text"] for i in q_idx]
            s_ids, s_mask = encode_batch(s_texts)
            q_ids, q_mask = encode_batch(q_texts)
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits, _, _ = model(s_ids, s_mask, s_y, q_ids, q_mask)
            ep_correct = (logits.argmax(-1) == q_y).sum().item()
            ep_total = q_y.size(0)
            per_ep.append(ep_correct / ep_total)
            correct += ep_correct
            total += ep_total
    return correct / max(1, total), float(np.std(per_ep))


def run_transfer(method, source_ds, target_ds, seed, n_episodes=500, n_eval=200):
    np.random.seed(seed); torch.manual_seed(seed)
    import random; random.seed(seed)
    print(f"\n=== Transfer: {method} trained on {source_ds} -> eval on {target_ds} ===", flush=True)
    src_df, src_meta = get_dataset(source_ds)
    tgt_df, tgt_meta = get_dataset(target_ds)

    model = make_model(method, src_meta["num_classes"]).to(device)
    # Set up LPA dataset
    if method in ("lpa", "scp_lpa"):
        # Source dataset similarity
        model.set_dataset(source_ds)
    if method in ("cf_protonet", "scp"):
        freq = torch.tensor([src_df[src_df["label"] == c].shape[0] for c in range(src_meta["num_classes"])],
                            dtype=torch.float)
        model.set_class_freq(freq.to(device))

    # Train on source (use 5-way 5-shot for encoder training, common to all eval)
    train_model(model, src_df, 5, 5, 15, n_episodes, method)

    # Swap LPA dataset to target (so the dialectal matrix used at inference matches the eval corpus)
    if method in ("lpa", "scp_lpa"):
        model.set_dataset(target_ds)

    # Eval on target
    acc, std = eval_model(model, tgt_df, 5, 5, 15, n_eval, method)
    ci = 1.96 * std / np.sqrt(n_eval)
    result = {
        "method": method, "source": source_ds, "target": target_ds,
        "seed": seed, "accuracy": acc, "std": std, "ci95": ci,
    }
    out_name = f"transfer_{method}_{source_ds}_to_{target_ds}_seed{seed}.json"
    with open(os.path.join(OUT_DIR, out_name), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [{method}/{source_ds}->{target_ds}/s{seed}] acc={acc:.4f} +/- {ci:.4f}", flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["protonet", "scp", "lpa", "scp_lpa"])
    ap.add_argument("--direction", choices=["nadi_to_amgad", "amgad_to_nadi"], default="nadi_to_amgad")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_episodes", type=int, default=500)
    ap.add_argument("--n_eval", type=int, default=200)
    args = ap.parse_args()

    if args.direction == "nadi_to_amgad":
        source, target = "nadi_18", "amgadhasan_5"
    else:
        source, target = "amgadhasan_5", "nadi_18"

    results = []
    for m in args.methods:
        r = run_transfer(m, source, target, args.seed, args.n_episodes, args.n_eval)
        results.append(r)
        # Free GPU
        del r

    # Save combined
    out_name = f"transfer_{args.direction}_seed{args.seed}.json"
    with open(os.path.join(OUT_DIR, out_name), "w") as f:
        json.dump(results, f, indent=2)
    summary = [(r["method"], r["accuracy"]) for r in results]
    print("\nDONE {}: {}".format(args.direction, summary), flush=True)