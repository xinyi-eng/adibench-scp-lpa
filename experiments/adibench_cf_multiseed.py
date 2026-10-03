"""CF-ProtoNet multi-seed summary and comparison vs ProtoNet."""
import os, json
import numpy as np

RESULTS_DIR = r"D:/dacd2026/adibench_v1/results"
datasets = ["nadi_18", "nadi_5", "amgadhasan_5"]
seeds = [0, 7, 42, 123]

print("=" * 95)
print("CF-ProtoNet Multi-Seed Accuracy (5-way) + comparison vs ProtoNet")
print("=" * 95)
hdr = f"{'Dataset':<15}{'Shot':<6}{'s0':<8}{'s7':<8}{'s42':<8}{'s123':<8}{'mean+-std':<16}{'ProtoNet':<10}{'Delta':<10}"
print(hdr)
print("-" * 95)

summary = {}
for ds in datasets:
    for k in [1, 5]:
        cf_vals, pt_vals = [], []
        per_seed = []
        for s in seeds:
            fcf = os.path.join(RESULTS_DIR, f"cf_protonet_{ds}_5way_{k}shot_seed{s}.json")
            fpt = os.path.join(RESULTS_DIR, f"protonet_{ds}_5way_{k}shot_seed{s}.json")
            if os.path.exists(fcf):
                with open(fcf) as f:
                    v = json.load(f)["accuracy"]
                per_seed.append(v); cf_vals.append(v)
            else:
                per_seed.append(float("nan"))
            if os.path.exists(fpt):
                with open(fpt) as f:
                    pt_vals.append(json.load(f)["accuracy"])
        if not cf_vals:
            continue
        m = np.mean(cf_vals)
        sd = np.std(cf_vals, ddof=1) if len(cf_vals) > 1 else 0.0
        pm = np.mean(pt_vals) if pt_vals else float("nan")
        delta = m - pm
        wins = "WIN" if delta > 0 else "loss"
        row = f"{ds:<15}{k:<6}"
        for v in per_seed:
            row += f"{v:<8.4f}" if not np.isnan(v) else f"{'--':<8}"
        row += f"{m:.4f}+-{sd:.4f}  {pm:.4f}  {delta:+.4f} {wins}"
        print(row)
        summary[f"{ds}_{k}shot"] = {
            "cf_mean": float(m), "cf_std": float(sd),
            "proto_mean": float(pm), "delta": float(delta),
            "cf_seeds": [float(v) if not np.isnan(v) else None for v in per_seed],
        }

with open(os.path.join(RESULTS_DIR, "cf_protonet_multiseed_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print()
n_win = sum(1 for v in summary.values() if v["delta"] > 0)
print(f"CF-ProtoNet beats ProtoNet (4-seed means) on {n_win}/{len(summary)} cells")
