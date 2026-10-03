"""P4b: REVERSE transfer  amgadhasan_5 (source, 5 city classes)
                      -> nadi_5 (target, 5 coarse groups).

LPA variants carry the SOURCE (city) similarity matrix onto target
episodes: same 5x5 shape, mismatched semantics -- the deliberate test
of 'LPA prior is corpus-specific, SCP is corpus-agnostic'.
Methods: protonet, scp (SupCon, no prior), scp_lpa (SupCon + source prior).
5-way, k=1 and k=5 on target; 500 source-train episodes; 200 target-eval.
Writes results/transfer_reverse.json.
"""
import sys, os, json, time, random
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from adibench.data import get_dataset, build_fewshot_episode, build_fewshot_episode_ex
from adibench.baselines import ProtoNet, LinguisticProtoNet, lpa_protonet_step

ARABERT = r"D:\dacd2026\2_models\arabertv02"
OUT = r"D:/dacd2026/adibench_v1/results/transfer_reverse.json"
SRC, TGT = "amgadhasan_5", "nadi_5"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ARABERT, use_fast=True)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def enc(texts):
    e = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return e["input_ids"], e["attention_mask"]


def make(method):
    if method == "protonet":
        return ProtoNet()
    m = LinguisticProtoNet()
    if method == "scp_lpa":
        m.set_dataset(SRC)          # SOURCE prior carried to target
    return m


def train_src(model, method, k, df_src, n_episodes=500):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    for _ in range(n_episodes):
        if method in ("scp", "scp_lpa"):
            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df_src, 5, k, 15)
            model.set_episode_classes(chosen)
        else:
            s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df_src, 5, k, 15)
        s_ids, s_m = enc([df_src.iloc[i]["text"] for i in s_idx])
        q_ids, q_m = enc([df_src.iloc[i]["text"] for i in q_idx])
        s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
        q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
        if method == "protonet":
            logits, _, _ = model(s_ids, s_m, s_y, q_ids, q_m)
            loss = F.cross_entropy(logits, q_y)
        else:
            # scp: no dataset set -> raw protos; scp_lpa: source prior
            loss, _, _ = lpa_protonet_step(
                model, s_ids, s_m, s_y, q_ids, q_m, q_y,
                temperature=model.supcon_temperature, ce_weight=0.5, supcon_weight=1.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()


def eval_tgt(model, method, k, df_tgt, n_eval=200):
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(n_eval):
            if method in ("scp", "scp_lpa"):
                s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df_tgt, 5, k, 15)
                model.set_episode_classes(chosen)
            else:
                s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df_tgt, 5, k, 15)
            s_ids, s_m = enc([df_tgt.iloc[i]["text"] for i in s_idx])
            q_ids, q_m = enc([df_tgt.iloc[i]["text"] for i in q_idx])
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits, _, _ = model(s_ids, s_m, s_y, q_ids, q_m)
            accs.append(float((logits.argmax(-1) == q_y).float().mean().item()))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    df_src, _ = get_dataset(SRC)
    df_tgt, _ = get_dataset(TGT)
    res = {}
    for method in ["protonet", "scp", "scp_lpa"]:
        for k in [1, 5]:
            set_seed(42)
            model = make(method).to(device)
            t0 = time.time(); train_src(model, method, k, df_src); tr = time.time() - t0
            acc, sd = eval_tgt(model, method, k, df_tgt)
            res[f"{method}_{k}shot"] = {"accuracy": acc, "std": sd, "train_s": tr}
            print(f"[reverse {SRC}->{TGT} / {method} / {k}shot] acc={acc:.4f}+-{sd:.4f}", flush=True)
            del model; torch.cuda.empty_cache()
    # forward-direction reference values (paper Finding 4, NADI18->amgad, 5-shot)
    fwd = {"protonet_5shot": 0.670, "scp_5shot": 0.705, "scp_lpa_5shot": 0.670}
    res["_forward_ref_nadi18_to_amgad_5shot"] = fwd
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print("Saved", OUT)
    print("\n== Reverse transfer summary (forward refs from Finding 4) ==")
    for m in ["protonet", "scp", "scp_lpa"]:
        for k in [1, 5]:
            tag = f"{m}_{k}shot"
            extra = f"  (forward 5s ref: {fwd.get(tag, '-')})" if k == 5 else ""
            print(f"  {m:9s} {k}shot: {res[tag]['accuracy']:.3f}{extra}")


if __name__ == "__main__":
    main()