"""Multi-seed summary for FineTune (verify chance-level collapse)."""
import os, json
import numpy as np

RESULTS_DIR = r"D:/dacd2026/adibench_v1/results"

datasets = ["nadi_18", "nadi_5", "amgadhasan_5"]
k_shots = [1, 5]
seeds = [0, 7, 42, 123]

print("=" * 90)
print("FineTune Multi-Seed Accuracy Summary (5-way, verifying chance-level)")
print("=" * 90)
print(f"{'Dataset':<15}{'Shot':<8}{'s=0':<10}{'s=7':<10}{'s=42':<10}{'s=123':<10}{'mean±std':<15}")
print("-" * 90)
all_passed = True
for ds in datasets:
    for k in k_shots:
        vals = []
        per_seed = []
        for s in seeds:
            fname = f"finetune_{ds}_5way_{k}shot_seed{s}.json"
            fpath = os.path.join(RESULTS_DIR, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    data = json.load(f)
                v = data["accuracy"]
                per_seed.append(v)
                vals.append(v)
            else:
                per_seed.append(float('nan'))
        if not vals: continue
        m, sd = np.mean(vals), np.std(vals, ddof=1) if len(vals) > 1 else 0
        max_v = max(vals)
        # All cells should be ~0.20
        status = "PASS (chance)" if max_v < 0.22 else "FAIL (above chance)"
        if max_v >= 0.22:
            all_passed = False
        print(f"{ds:<15}{k:<8}{per_seed[0]:<10.4f}{per_seed[1]:<10.4f}{per_seed[2]:<10.4f}{per_seed[3]:<10.4f}{m:.4f}±{sd:.4f}  {status}")

print()
print(f"Verdict: {'ALL cells at chance level (collapse confirmed across seeds)' if all_passed else 'Some cells above chance - investigate'}")