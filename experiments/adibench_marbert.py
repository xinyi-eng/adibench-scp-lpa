"""P4a: MARBERT encoder generalisation check.

Core methods {finetune, protonet, scp_lpa} on amgadhasan_5-city,
k in {1,5}, seed 42, same protocol (500 train / 200 eval).
Swaps in the local marbertv2 encoder (hidden 768, same as AraBERT).
Writes results/marbert_core.json.
"""
import sys, os, json, time, random
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from adibench.data import get_dataset, build_fewshot_episode, build_fewshot_episode_ex
from adibench.baselines import (
    ProtoNet, FineTune, LinguisticProtoNet, lpa_protonet_step,
    Encoder, freeze_early_layers,
)

MODEL = r"D:\dacd2026\2_models\marbertv2"
OUT = r"D:/dacd2026/adibench_v1/results/marbert_core.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
except Exception:
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def enc(texts):
    e = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return e["input_ids"], e["attention_mask"]


def swap_encoder(model):
    model.encoder = Encoder(encoder_name=MODEL, embed_dim=256).to(device)
    freeze_early_layers(model.encoder.encoder, n_unfreeze=4)
    return model


def make(method, num_classes=5):
    if method == "protonet":
        m = ProtoNet()
    elif method == "finetune":
        m = FineTune(num_classes=num_classes)
    else:
        m = LinguisticProtoNet()
        m.set_dataset("amgadhasan_5")
    swap_encoder(m)
    return m.to(device)


def train(model, df, method, k, n_episodes=500):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    for _ in range(n_episodes):
        if method == "scp_lpa":
            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, 5, k, 15)
            model.set_episode_classes(chosen)
        else:
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, 5, k, 15)
        s_ids, s_m = enc([df.iloc[i]["text"] for i in s_idx])
        q_ids, q_m = enc([df.iloc[i]["text"] for i in q_idx])
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        if method == "finetune":
            loss = model.loss(s_ids, s_m, s_y, q_ids, q_m, q_y)
        elif method == "protonet":
            logits, _, _ = model(s_ids, s_m, s_y, q_ids, q_m)
            loss = F.cross_entropy(logits, q_y)
        else:
            loss, _, _ = lpa_protonet_step(
                model, s_ids, s_m, s_y, q_ids, q_m, q_y,
                temperature=model.supcon_temperature, ce_weight=0.5, supcon_weight=1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()


def evaluate(model, df, method, k, n_eval=200):
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_eval):
            if method == "scp_lpa":
                s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, 5, k, 15)
                model.set_episode_classes(chosen)
            else:
                s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, 5, k, 15)
            s_ids, s_m = enc([df.iloc[i]["text"] for i in s_idx])
            q_ids, q_m = enc([df.iloc[i]["text"] for i in q_idx])
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits, _, _ = model(s_ids, s_m, s_y, q_ids, q_m)
            accs.append(float((logits.argmax(-1) == q_y).float().mean().item()))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    set_seed(42)
    df, meta = get_dataset("amgadhasan_5")
    res = {}
    # sanity: check marbert loads + dims
    probe = AutoModel.from_pretrained(MODEL)
    print(f"marbert hidden={probe.config.hidden_size}", flush=True)
    del probe
    for method in ["finetune", "protonet", "scp_lpa"]:
        for k in [1, 5]:
            set_seed(42)
            model = make(method, meta["num_classes"])
            t0 = time.time(); train(model, df, method, k); tr = time.time() - t0
            acc, sd = evaluate(model, df, method, k)
            res[f"{method}_{k}shot"] = {"accuracy": acc, "std": sd}
            print(f"[marbert/{method}/amgadhasan/{k}shot] acc={acc:.4f}+-{sd:.4f} train={tr:.0f}s", flush=True)
            del model; torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print("Saved", OUT)
    print("\n== MARBERT summary (AraBERT Table-2 refs in parens) ==")
    refs = {("finetune", 1): 0.202, ("finetune", 5): 0.192,
            ("protonet", 1): 0.593, ("protonet", 5): 0.768,
            ("scp_lpa", 1): 0.641, ("scp_lpa", 5): 0.785}
    for (m, k), ref in refs.items():
        a = res[f"{m}_{k}shot"]["accuracy"]
        print(f"  {m:9s} {k}shot: MARBERT {a:.3f}  (AraBERT {ref:.3f})")


if __name__ == "__main__":
    main()