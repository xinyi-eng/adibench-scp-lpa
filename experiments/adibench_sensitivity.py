"""LPA hyperparameter sensitivity analysis.

Sweeps alpha and tau_ling on the hardest cell (NADI 18 1-shot, seed 42)
and plots a heatmap.
"""
import sys, os, json, time
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from adibench.baselines import LinguisticProtoNet, lpa_protonet_step
from adibench.data import build_fewshot_episode, build_fewshot_episode_ex, get_dataset

ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ARABERT_PATH, use_fast=True)


def encode_batch(texts):
    enc = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return enc["input_ids"], enc["attention_mask"]


def train_and_eval(alpha, tau_ling, dataset, n_way=5, k_shot=1, q_query=15,
                  n_episodes=500, n_eval=200, seed=42):
    import random
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    df, meta = get_dataset(dataset)
    n_classes = meta["num_classes"]

    model = LinguisticProtoNet(alpha=alpha, tau_ling=tau_ling).to(device)
    model.set_dataset(dataset)
    # Temporarily override the similarity matrix to use this tau_ling
    from adibench.dialects import similarity_matrix
    S = similarity_matrix(dataset, tau=tau_ling)
    model._similarity = S
    import torch as _torch
    model._similarity_t = _torch.from_numpy(S).float()

    # Train (SupCon + LPA-style combined loss)
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
        loss, _, _ = lpa_protonet_step(model, s_ids, s_mask, s_y, q_ids, q_mask, q_y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()

    # Eval
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(n_eval):
            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)
            model.set_episode_classes(chosen)
            s_ids, s_mask = encode_batch([df.iloc[i]["text"] for i in s_idx])
            q_ids, q_mask = encode_batch([df.iloc[i]["text"] for i in q_idx])
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits, _, _ = model(s_ids, s_mask, s_y, q_ids, q_mask)
            correct += (logits.argmax(-1) == q_y).sum().item()
            total += q_y.size(0)
    del model
    torch.cuda.empty_cache()
    return correct / max(1, total)


def main():
    alphas = [0.3, 0.5, 0.7, 0.9]
    taus = [0.3, 0.5, 1.0]
    dataset = "nadi_18"

    results = np.zeros((len(alphas), len(taus)))
    print(f"LPA sensitivity on {dataset} 5w1s, seed 42")
    print(f"  alpha x tau_ling grid: {len(alphas)} x {len(taus)} = {len(alphas)*len(taus)} cells")
    for i, a in enumerate(alphas):
        for j, t in enumerate(taus):
            t0 = time.time()
            acc = train_and_eval(a, t, dataset, n_episodes=500, n_eval=200)
            results[i, j] = acc
            print(f"  alpha={a:.1f} tau={t:.1f}: acc={acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # Save grid
    out = {
        "dataset": dataset, "seed": 42,
        "alphas": alphas, "taus": taus,
        "grid": results.tolist(),
        "best": {"alpha": alphas[int(np.argmax(results) // len(taus))],
                 "tau_ling": taus[int(np.argmax(results) % len(taus))],
                 "acc": float(results.max())},
        "vanilla_protonet_nadi18_1s": 0.388,
    }
    with open(r"D:/dacd2026/adibench_v1/results/lpa_sensitivity.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved sensitivity grid. Best: {out['best']}")

    # Plot heatmap
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(results, cmap="viridis", vmin=0.30, vmax=0.55)
    ax.set_xticks(range(len(taus)))
    ax.set_xticklabels([f"{t:.1f}" for t in taus])
    ax.set_yticks(range(len(alphas)))
    ax.set_yticklabels([f"{a:.1f}" for a in alphas])
    ax.set_xlabel(r"$\tau_{\mathrm{ling}}$")
    ax.set_ylabel(r"$\alpha$ (trust raw proto)")
    ax.set_title(f"LPA sensitivity on {dataset} 5-way 1-shot (seed 42)")
    for i in range(len(alphas)):
        for j in range(len(taus)):
            ax.text(j, i, f"{results[i,j]:.3f}", ha="center", va="center",
                    color="white" if results[i,j] < 0.42 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="5-way 1-shot accuracy")
    plt.tight_layout()
    plt.savefig(r"D:/dacd2026/adibench_v1/paper/figs/fig4_lpa_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.savefig(r"D:/dacd2026/adibench_v1/paper/figs/fig4_lpa_sensitivity.pdf", bbox_inches="tight")
    print("Saved fig4_lpa_sensitivity.png/.pdf")


if __name__ == "__main__":
    main()