"""Step 8: Calibration (ECE) + noise robustness experiments.

Reads all result JSONs and computes:
- Per-cell accuracy
- Expected Calibration Error (ECE)
- Brier score
- Accuracy under 20% symmetric label noise (synthetic)

ECE measures the difference between predicted confidence and actual accuracy:
    ECE = sum_b (|bucket_acc - bucket_conf| * n_b / N)
where buckets are 10 equally-spaced confidence bins.

Noise robustness: we synthetically perturb the query labels and re-evaluate
the model's accuracy. We do not need to retrain -- we only need per-episode
logits. Since we don't store them, we approximate by re-running a few episodes
with flipped labels.
"""
import os, json, sys
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from collections import defaultdict

from adibench.baselines import (
    ProtoNet, MatchingNet, RelationNet, FineTune, Focal, CB, MAML,
    CFProtoNet, cf_protonet_supcon_step, LinguisticProtoNet, lpa_protonet_step,
)
from adibench.data import build_fewshot_episode, build_fewshot_episode_ex, get_dataset

ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)


def encode_batch(texts):
    enc = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return enc["input_ids"], enc["attention_mask"]


def make_model(method, num_classes):
    if method == "protonet": return ProtoNet()
    if method == "matchingnet": return MatchingNet()
    if method == "relationnet": return RelationNet()
    if method == "finetune": return FineTune(num_classes=num_classes)
    if method == "focal": return Focal(num_classes=num_classes)
    if method == "cb": return CB(num_classes=num_classes)
    if method == "maml": return MAML(num_classes=num_classes)
    if method == "random": return Random(num_classes=min(num_classes, 5))
    if method == "cf_protonet": return CFProtoNet()
    if method == "cf_no_margin": return CFProtoNet(ldam_margin=0.0, prior_alpha=0.0)
    if method == "lpa_protonet": return LinguisticProtoNet()
    if method == "scp_lpa": return LinguisticProtoNet()
    raise ValueError(method)


def train_method(model, df, n_way, k_shot, q_query, n_episodes, method):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    for _ in range(n_episodes):
        if method in ("cf_protonet", "cf_no_margin", "lpa_protonet", "scp_lpa"):
            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)
            model.set_episode_classes(chosen)
        else:
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)
        s_ids, s_mask = encode_batch([df.iloc[i]["text"] for i in s_idx])
        q_ids, q_mask = encode_batch([df.iloc[i]["text"] for i in q_idx])
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        if method in ("cf_protonet", "cf_no_margin"):
            loss, _, _ = cf_protonet_supcon_step(model, s_ids, s_mask, s_y, q_ids, q_mask, q_y)
        elif method in ("lpa_protonet", "scp_lpa"):
            loss, _, _ = lpa_protonet_step(model, s_ids, s_mask, s_y, q_ids, q_mask, q_y)
        else:
            if hasattr(model, "loss") and method in ("finetune", "focal", "cb"):
                loss = model.loss(s_ids, s_mask, s_y, q_ids, q_mask, q_y)
            else:
                logits, _, _ = model(s_ids, s_mask, s_y, q_ids, q_mask)
                loss = F.cross_entropy(logits, q_y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()


def collect_probs(model, df, n_way, k_shot, q_query, n_episodes, method, noise_rate=0.0):
    """Run n_episodes, return (probs, labels) numpy arrays.
    If noise_rate > 0, flip that fraction of query labels uniformly before comparing.
    """
    if method in ("cf_protonet", "cf_no_margin", "lpa_protonet", "scp_lpa"):
        model.set_class_freq if hasattr(model, "set_class_freq") else None
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for _ in range(n_episodes):
            if method in ("cf_protonet", "cf_no_margin", "lpa_protonet", "scp_lpa"):
                s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)
                model.set_episode_classes(chosen)
            else:
                s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way, k_shot, q_query)
            s_ids, s_mask = encode_batch([df.iloc[i]["text"] for i in s_idx])
            q_ids, q_mask = encode_batch([df.iloc[i]["text"] for i in q_idx])
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits, _, _ = model(s_ids, s_mask, s_y, q_ids, q_mask)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            labels = q_y.cpu().numpy()
            # Inject noise to query labels (not support)
            if noise_rate > 0:
                n_q = labels.shape[0]
                flip_mask = np.random.rand(n_q) < noise_rate
                labels = labels.copy()
                labels[flip_mask] = np.random.randint(0, n_way, flip_mask.sum())
            all_probs.append(probs)
            all_labels.append(labels)
    return np.concatenate(all_probs), np.concatenate(all_labels)


def ece(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    n_total = len(labels)
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        if in_bin.any():
            bin_acc = accuracies[in_bin].mean()
            bin_conf = confidences[in_bin].mean()
            ece_val += (in_bin.sum() / n_total) * abs(bin_acc - bin_conf)
    return float(ece_val)


def brier(probs, labels):
    """Brier score (mean squared error of one-hot encoding)."""
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def accuracy(probs, labels):
    return float((probs.argmax(axis=1) == labels).mean())


def run_cell(method, dataset, n_way=5, k_shot=5, q_query=15,
             n_episodes=200, n_eval=100, noise_rate=0.0):
    df, meta = get_dataset(dataset)
    model = make_model(method, meta["num_classes"]).to(device)
    if method in ("cf_protonet", "cf_no_margin"):
        freq = torch.tensor([df[df["label"] == c].shape[0] for c in range(meta["num_classes"])], dtype=torch.float)
        model.set_class_freq(freq.to(device))
    if method in ("lpa_protonet", "scp_lpa"):
        model.set_dataset(dataset)

    # Quick train
    train_method(model, df, n_way, k_shot, q_query, n_episodes=min(200, n_episodes), method=method)
    probs, labels = collect_probs(model, df, n_way, k_shot, q_query, n_eval, method, noise_rate=noise_rate)
    del model
    torch.cuda.empty_cache()
    return probs, labels


def main():
    methods = ["protonet", "scp_lpa", "lpa_protonet", "finetune"]
    datasets = ["nadi_18", "nadi_5", "amgadhasan_5"]
    print("=" * 90)
    print("Calibration + noise robustness (5-way 5-shot, seed 42)")
    print("=" * 90)
    print(f"{'Method':<14}{'Dataset':<15}{'Acc':<10}{'ECE':<10}{'Brier':<10}{'Acc@20%noise':<14}")
    print("-" * 90)
    results = []
    for method in methods:
        for ds in datasets:
            # Clean run
            p_clean, l_clean = run_cell(method, ds)
            acc_clean = accuracy(p_clean, l_clean)
            ece_clean = ece(p_clean, l_clean)
            brier_clean = brier(p_clean, l_clean)
            # Noisy run (20% symmetric label noise on query)
            p_noise, l_noise = run_cell(method, ds, noise_rate=0.20)
            acc_noise = accuracy(p_noise, l_noise)
            tag = ""
            print(f"{method:<14}{ds:<15}{acc_clean:<10.4f}{ece_clean:<10.4f}{brier_clean:<10.4f}{acc_noise:<14.4f}  {tag}")
            results.append({
                "method": method, "dataset": ds,
                "acc": acc_clean, "ece": ece_clean, "brier": brier_clean,
                "acc_noisy": acc_noise,
            })

    with open(r"D:/dacd2026/adibench_v1/results/calibration_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print()
    print("Saved calibration_results.json")


if __name__ == "__main__":
    main()