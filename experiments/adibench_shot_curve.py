"""P2: shot-sweep curve for FineTune vs ProtoNet on NADI 18.
k in {1,3,5,10,20}; seed 42; 500 train / 200 eval; accuracy only.
Writes results/shot_curve.json and figs/figA_shotcurve.{png,pdf}.
"""
import sys, os, json, time, random
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from adibench.data import get_dataset, build_fewshot_episode
from adibench.baselines import ProtoNet, FineTune

ARABERT = r"D:\dacd2026\2_models\arabertv02"
OUT = r"D:/dacd2026/adibench_v1/results/shot_curve.json"
FIG_DIR = r"D:/dacd2026/adibench_v1/paper/figs"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ARABERT, use_fast=True)
SHOTS = [1, 3, 5, 10, 20]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def enc(texts):
    e = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return e["input_ids"], e["attention_mask"]


def train(model, df, k, n_episodes=500):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    for _ in range(n_episodes):
        s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, 5, k, 15)
        s_ids, s_m = enc([df.iloc[i]["text"] for i in s_idx])
        q_ids, q_m = enc([df.iloc[i]["text"] for i in q_idx])
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        if isinstance(model, FineTune):
            loss = model.loss(s_ids, s_m, s_y, q_ids, q_m, q_y)
        else:
            logits, _, _ = model(s_ids, s_m, s_y, q_ids, q_m)
            loss = F.cross_entropy(logits, q_y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()


def evaluate(model, df, k, n_eval=200):
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_eval):
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, 5, k, 15)
            s_ids, s_m = enc([df.iloc[i]["text"] for i in s_idx])
            q_ids, q_m = enc([df.iloc[i]["text"] for i in q_idx])
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits, _, _ = model(s_ids, s_m, s_y, q_ids, q_m)
            accs.append(float((logits.argmax(-1) == q_y).float().mean().item()))
    return float(np.mean(accs)), float(np.std(accs))


def run(method, k):
    set_seed(42)
    df, meta = get_dataset("nadi_18")
    model = (FineTune(num_classes=meta["num_classes"]) if method == "finetune" else ProtoNet()).to(device)
    t0 = time.time(); train(model, df, k); t_tr = time.time() - t0
    acc, sd = evaluate(model, df, k)
    print(f"[{method}/nadi_18/{k}shot] acc={acc:.4f}+-{sd:.4f} train={t_tr:.0f}s", flush=True)
    del model; torch.cuda.empty_cache()
    return {"k": k, "accuracy": acc, "std": sd}


def main():
    res = {}
    for method in ["finetune", "protonet"]:
        res[method] = [run(method, k) for k in SHOTS]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print("Saved", OUT)

    # ---- figure (appendix) ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    xs = SHOTS
    ft = res["finetune"]; pn = res["protonet"]
    ax.errorbar(xs, [r["accuracy"] for r in ft], yerr=[r["std"] for r in ft],
                marker="s", color="#B3403B", capsize=2.5, lw=1.5, ms=5, label="FineTune")
    ax.errorbar(xs, [r["accuracy"] for r in pn], yerr=[r["std"] for r in pn],
                marker="o", color="#1565C0", capsize=2.5, lw=1.5, ms=5, label="ProtoNet")
    ax.axhline(0.20, color="#888", ls=":", lw=0.9)
    ax.text(xs[-1], 0.205, "chance (0.20)", fontsize=7.5, ha="right", color="#666")
    ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([str(k) for k in xs])
    ax.set_xlabel("shot $k$ per class (5-way episodes)", fontsize=9)
    ax.set_ylabel("5-way accuracy", fontsize=9)
    ax.set_title("Support size vs.\\ accuracy on NADI 18 (seed 42)",
                 fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # crossing annotation: first k where finetune > chance+margin
    for r in ft:
        pass
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, "figA_shotcurve.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "figA_shotcurve.pdf"), bbox_inches="tight")
    print("Saved figure figA_shotcurve")

    # summary for write-up
    ft_vals = [r["accuracy"] for r in ft]; pn_vals = [r["accuracy"] for r in pn]
    print("\nSUMMARY:")
    print("  finetune by shot:", [f"{v:.3f}" for v in ft_vals])
    print("  protonet by shot:", [f"{v:.3f}" for v in pn_vals])
    for i, k in enumerate(SHOTS):
        if ft_vals[i] >= 0.30:
            print(f"  FineTune first recovers (>=0.30) at k={k}")
            break
    else:
        print("  FineTune never reaches 0.30 within tested shots")


if __name__ == "__main__":
    main()