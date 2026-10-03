"""P1: Macro-F1 (+ accuracy from the SAME eval) for core methods.

Methods: finetune, protonet, scp (SupCon, raw protos), lpa (CE over LPA
protos, supcon off = test-time story), scp_lpa (SupCon + LPA).
Datasets: nadi_18, nadi_5, amgadhasan_5  x  k in {1,5}
seed 42, 500 train episodes, 200 eval episodes (same protocol as Table 2).

Validation gate: seed-42 accuracy must be within +-0.03 of the paper's
single-seed / multi-seed-mean values, else the recipe is rejected.
"""
import sys, os, json, time, random
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from adibench.data import get_dataset, build_fewshot_episode, build_fewshot_episode_ex
from adibench.baselines import (
    ProtoNet, FineTune, LinguisticProtoNet, lpa_protonet_step,
)

ARABERT = r"D:\dacd2026\2_models\arabertv02"
OUT = r"D:/dacd2026/adibench_v1/results/macro_f1.json"
device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(ARABERT, use_fast=True)

# Paper reference (Table 2 / multi-seed means; +-0.03 gate)
REF = {
    ("protonet", "nadi_18", 1): 0.388, ("protonet", "nadi_18", 5): 0.545,
    ("protonet", "nadi_5", 1): 0.499, ("protonet", "nadi_5", 5): 0.677,
    ("protonet", "amgadhasan_5", 1): 0.593, ("protonet", "amgadhasan_5", 5): 0.768,
    ("scp", "nadi_18", 1): 0.425, ("scp", "nadi_18", 5): 0.549,
    ("scp", "nadi_5", 1): 0.520, ("scp", "nadi_5", 5): 0.689,
    ("scp", "amgadhasan_5", 1): 0.660, ("scp", "amgadhasan_5", 5): 0.788,
    ("lpa", "nadi_18", 1): 0.420, ("lpa", "nadi_18", 5): 0.549,
    ("lpa", "nadi_5", 1): 0.537, ("lpa", "nadi_5", 5): 0.689,
    ("lpa", "amgadhasan_5", 1): 0.656, ("lpa", "amgadhasan_5", 5): 0.785,
    ("scp_lpa", "nadi_18", 1): 0.416, ("scp_lpa", "nadi_18", 5): 0.547,
    ("scp_lpa", "nadi_5", 1): 0.532, ("scp_lpa", "nadi_5", 5): 0.686,
    ("scp_lpa", "amgadhasan_5", 1): 0.641, ("scp_lpa", "amgadhasan_5", 5): 0.785,
    ("finetune", "nadi_18", 1): 0.198, ("finetune", "nadi_18", 5): 0.201,
    ("finetune", "nadi_5", 1): 0.199, ("finetune", "nadi_5", 5): 0.198,
    ("finetune", "amgadhasan_5", 1): 0.202, ("finetune", "amgadhasan_5", 5): 0.192,
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enc(texts):
    e = tok(texts, truncation=True, padding="max_length", max_length=96, return_tensors="pt").to(device)
    return e["input_ids"], e["attention_mask"]


def make(method):
    if method == "protonet":
        return ProtoNet()
    if method == "finetune":
        return FineTune(num_classes=5)
    # scp / lpa / scp_lpa share LinguisticProtoNet
    return LinguisticProtoNet()


def macro_f1_np(y_true, y_pred, n_way):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    f1s = []
    for c in range(n_way):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s))


def train(model, df, method, k, n_episodes=500):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=0.01)
    model.train()
    for _ in range(n_episodes):
        if method in ("scp", "lpa", "scp_lpa"):
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
        elif method in ("scp", "scp_lpa"):
            loss, _, _ = lpa_protonet_step(
                model, s_ids, s_m, s_y, q_ids, q_m, q_y,
                temperature=model.supcon_temperature, ce_weight=0.5, supcon_weight=1.0)
        else:  # lpa: CE over LPA protos only (test-time story)
            loss, _, _ = lpa_protonet_step(
                model, s_ids, s_m, s_y, q_ids, q_m, q_y,
                temperature=model.supcon_temperature, ce_weight=1.0, supcon_weight=0.0)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        if trainable and (trainable[0].grad is None):
            pass


def evaluate(model, df, method, k, n_eval=200):
    model.eval()
    accs, f1s = [], []
    with torch.no_grad():
        for _ in range(n_eval):
            if method in ("scp", "lpa", "scp_lpa"):
                s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, 5, k, 15)
                model.set_episode_classes(chosen)
            else:
                s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, 5, k, 15)
            s_ids, s_m = enc([df.iloc[i]["text"] for i in s_idx])
            q_ids, q_m = enc([df.iloc[i]["text"] for i in q_idx])
            s_y = torch.tensor(s_lab, dtype=torch.long, device=device)
            q_y = torch.tensor(q_lab, dtype=torch.long, device=device)
            logits, _, _ = model(s_ids, s_m, s_y, q_ids, q_m)
            pred = logits.argmax(-1).cpu().numpy()
            true = q_y.cpu().numpy()
            accs.append(float((pred == true).mean()))
            f1s.append(macro_f1_np(true, pred, 5))
    return float(np.mean(accs)), float(np.mean(f1s)), float(np.std(f1s)), float(np.std(accs))


def run(method, dataset, k):
    set_seed(42)
    df, meta = get_dataset(dataset)
    model = make(method).to(device)
    if method in ("lpa", "scp_lpa"):
        model.set_dataset(dataset)
    # scp: NO set_dataset -> similarity None -> raw prototypes (pure SupCon+CE)
    t0 = time.time()
    train(model, df, method, k)
    t_tr = time.time() - t0
    t0 = time.time()
    acc, mf1, f1std, accstd = evaluate(model, df, method, k)
    t_ev = time.time() - t0
    ref = REF[(method, dataset, k)]
    ok = abs(acc - ref) <= 0.03
    print(f"[{method}/{dataset}/{k}shot] acc={acc:.4f} (ref {ref:.3f}, gate {'PASS' if ok else 'FAIL'}) "
          f"macroF1={mf1:.4f}+-{f1std:.4f}  train={t_tr:.0f}s eval={t_ev:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
    return {"method": method, "dataset": dataset, "k_shot": k, "seed": 42,
            "accuracy": acc, "accuracy_ref": ref, "gate_pass": bool(ok),
            "macro_f1": mf1, "macro_f1_std": f1std, "acc_std": accstd,
            "train_s": t_tr, "eval_s": t_ev}


def main():
    results, all_pass = [], True
    for method in ["finetune", "protonet", "scp", "lpa", "scp_lpa"]:
        for dataset in ["nadi_18", "nadi_5", "amgadhasan_5"]:
            for k in [1, 5]:
                r = run(method, dataset, k)
                results.append(r)
                all_pass &= r["gate_pass"]
                os.makedirs(os.path.dirname(OUT), exist_ok=True)
                with open(OUT, "w") as f:
                    json.dump({"results": results, "all_gates_pass": all_pass}, f, indent=2)
    print("\nRECIPE GATE:", "ALL PASS" if all_pass else "*** SOME CELLS OFF BY >0.03 - DO NOT USE WITHOUT REVIEW ***")
    # compact table
    print("\n== Macro-F1 table (acc in parens) ==")
    for method in ["finetune", "protonet", "scp", "lpa", "scp_lpa"]:
        row = []
        for dataset in ["nadi_18", "nadi_5", "amgadhasan_5"]:
            for k in [1, 5]:
                r = next(x for x in results if x["method"] == method and x["dataset"] == dataset and x["k_shot"] == k)
                row.append(f"{r['macro_f1']:.3f}({r['accuracy']:.3f})")
        print(f"{method:9s} " + " ".join(f"{c:>14s}" for c in row))
    print("\nSaved", OUT)


if __name__ == "__main__":
    main()