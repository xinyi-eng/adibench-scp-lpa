"""Smoke test for ADIBench."""
import sys
sys.path.insert(0, r"D:/dacd2026/adibench_v1")
sys.path.insert(0, r"D:/dacd2026/site-packages")
import torch
from adibench.data import (
    NADI_18_COUNTRIES, QADI_5_NAMES, AMGHADHASAN_5_NAMES,
    load_nadi_18way, load_nadi_5way, load_amgadhasan_5city,
    build_fewshot_episode, list_datasets, get_dataset, DATASETS,
)
from adibench.baselines import ProtoNet, MatchingNet, Random, ALL_BASELINES

print("=== ADIBench smoke test ===", flush=True)
for name in list_datasets():
    df, meta = get_dataset(name)
    print(f"{name}: {len(df)} examples, {df['label'].nunique()} classes, type={meta['granularity']}", flush=True)

df_nadi, _ = get_dataset("nadi_18")
s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df_nadi, n_way=5, k_shot=5, q_query=15)
print(f"NADI 18 5-way 5-shot: support {len(s_idx)}, query {len(q_idx)}", flush=True)

df_amg, _ = get_dataset("amgadhasan_5")
s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df_amg, n_way=5, k_shot=5, q_query=15)
print(f"amgadhasan 5-city 5-way 5-shot: support {len(s_idx)}, query {len(q_idx)}", flush=True)

device = torch.device("cuda")
print(f"cuda: {torch.cuda.is_available()}", flush=True)
for name, cls in [("protonet", ProtoNet), ("matchingnet", MatchingNet), ("random", Random)]:
    m = cls(num_classes=5) if name == "random" else cls()
    m = m.to(device)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{name}: {n_params/1e6:.2f}M trainable params", flush=True)
print("=== SMOKE TEST PASSED ===", flush=True)
