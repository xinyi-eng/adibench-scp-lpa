"""Multi-seed summary for ProtoNet (top baseline)."""
import os, json
import numpy as np

RESULTS_DIR = r"D:/dacd2026/adibench_v1/results"

# Load all ProtoNet results
proto_results = {}
for fname in os.listdir(RESULTS_DIR):
    if not fname.startswith("protonet_") or not fname.endswith(".json"):
        continue
    parts = fname[:-5].split("_")
    # protonet_nadi_18_5way_1shot_seed42.json
    dataset = "_".join(parts[1:3])
    n_way = int(parts[3].replace("way", ""))
    k_shot = int(parts[4].replace("shot", ""))
    seed = int(parts[5].replace("seed", ""))
    if n_way != 5: continue  # only 5-way
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        data = json.load(f)
    if "accuracy" not in data: continue
    proto_results[(dataset, k_shot, seed)] = data["accuracy"]

# Aggregate by (dataset, shot)
datasets = ["nadi_18", "nadi_5", "amgadhasan_5"]
k_shots = [1, 5]
seeds = [0, 7, 42, 123]

print("=" * 90)
print("ProtoNet Multi-Seed Accuracy Summary (5-way)")
print("=" * 90)
print(f"{'Dataset':<15}{'Shot':<8}{'s=0':<10}{'s=7':<10}{'s=42':<10}{'s=123':<10}{'mean±std':<15}")
print("-" * 90)
rows = {}
for ds in datasets:
    for k in k_shots:
        vals = []
        per_seed = []
        for s in seeds:
            v = proto_results.get((ds, k, s))
            per_seed.append(v if v else float('nan'))
            if v: vals.append(v)
        if not vals: continue
        m, sd = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0
        print(f"{ds:<15}{k:<8}{per_seed[0]:<10.4f}{per_seed[1]:<10.4f}{per_seed[2]:<10.4f}{per_seed[3]:<10.4f}{m:.4f}±{sd:.4f}")
        rows[f"{ds}_{k}shot"] = {"seeds": per_seed, "mean": m, "std": sd, "n": len(vals)}

# Save
with open(os.path.join(RESULTS_DIR, "multiseed_summary.json"), "w") as f:
    json.dump(rows, f, indent=2)
print()
print(f"Saved multiseed_summary.json ({len(rows)} cells)")