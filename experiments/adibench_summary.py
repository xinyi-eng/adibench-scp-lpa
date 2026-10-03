"""Generate summary table from all ADIBench results."""
import json
import os
import csv
from collections import defaultdict

RESULTS_DIR = r"D:/dacd2026/adibench_v1/results"

# Method, dataset, n_way, k_shot, seed
results = defaultdict(list)
for fname in os.listdir(RESULTS_DIR):
    if not fname.endswith(".json"):
        continue
    parts = fname[:-5].split("_")
    if len(parts) < 6:
        continue
    try:
        method = parts[0]
        # parts[1] is dataset (e.g., nadi, 18, 5way, 1shot, seed42)
        # Actually let me re-check the naming convention
        # {method}_{dataset}_{n}way_{k}shot_seed{s}.json
        # e.g., protonet_nadi_18_5way_1shot_seed42.json
        # so parts = [protonet, nadi, 18, 5way, 1shot, seed42]
        dataset = "_".join(parts[1:3])  # nadi_18, nadi_5, amgadhasan_5
        n_way = int(parts[3].replace("way", ""))
        k_shot = int(parts[4].replace("shot", ""))
        seed = int(parts[5].replace("seed", ""))
    except (ValueError, IndexError):
        continue
    with open(os.path.join(RESULTS_DIR, fname)) as f:
        data = json.load(f)
    if "accuracy" in data:
        results[(method, dataset, n_way, k_shot)].append({
            "seed": seed, "acc": data["accuracy"],
            "std": data.get("std", 0), "ci95": data.get("ci95", 0),
        })

# Print markdown table
print("=" * 80)
print("ADIBench Final Summary Table (5-way, seed 42)")
print("=" * 80)
print()
methods = ["random", "protonet", "matchingnet", "relationnet", "finetune", "focal", "cb", "maml"]
datasets = [("nadi_18", "NADI 18"), ("nadi_5", "NADI 5"), ("amgadhasan_5", "amgad")]
ks = [1, 5]
print("| Method | " + " | ".join([f"{d[1]} {k}shot" for d in datasets for k in ks]) + " |")
print("|--------|" + "|".join(["-" * 8] * (len(datasets) * len(ks))) + "|")
for m in methods:
    row = [m]
    for d, _ in datasets:
        for k in ks:
            vals = results.get((m, d, 5, k), [])
            if not vals:
                row.append("-")
            else:
                acc = vals[0]["acc"]
                ci = vals[0]["ci95"]
                row.append(f"{acc:.3f}±{ci:.3f}")
    print("| " + " | ".join(row) + " |")

# Write CSV
with open(os.path.join(RESULTS_DIR, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["method", "dataset", "n_way", "k_shot", "seed", "accuracy", "ci95"])
    for fname in os.listdir(RESULTS_DIR):
        if not fname.endswith(".json"):
            continue
        parts = fname[:-5].split("_")
        if len(parts) < 6:
            continue
        try:
            method = parts[0]
            dataset = "_".join(parts[1:3])
            n_way = int(parts[3].replace("way", ""))
            k_shot = int(parts[4].replace("shot", ""))
            seed = int(parts[5].replace("seed", ""))
        except (ValueError, IndexError):
            continue
        with open(os.path.join(RESULTS_DIR, fname)) as f2:
            data = json.load(f2)
        if "accuracy" in data:
            w.writerow([method, dataset, n_way, k_shot, seed,
                        f"{data['accuracy']:.4f}", f"{data.get('ci95', 0):.4f}"])
print()
print(f"Saved {os.path.join(RESULTS_DIR, 'summary.csv')}")
