"""Aggregate SCP + LPA multi-seed results into a single comparison table."""
import os, json
import numpy as np

RESULTS_DIR = r"D:/dacd2026/adibench_v1/results"
datasets = ["nadi_18", "nadi_5", "amgadhasan_5"]
seeds = [0, 7, 42, 123]
shots = [1, 5]

# method key -> file prefix
methods = {
    "ProtoNet": "protonet",
    "SCP": "cf_no_margin",
    "LPA": "lpa_protonet",
}

def load(method_prefix, ds, k, s):
    fname = f"{method_prefix}_{ds}_5way_{k}shot_seed{s}.json"
    fpath = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath) as f:
            return json.load(f)["accuracy"]
    return None

print("=" * 100)
print("Multi-seed comparison (4 seeds: 0, 7, 42, 123) — mean ± std")
print("=" * 100)
print(f"{'Cell':<20}{'ProtoNet':<20}{'SCP':<20}{'LPA':<20}{'SCP win':<10}{'LPA win':<10}")
print("-" * 100)

summary = {}
for ds in datasets:
    for k in shots:
        key = f"{ds}_{k}shot"
        row = {}
        for name, prefix in methods.items():
            vals = [load(prefix, ds, k, s) for s in seeds]
            vals = [v for v in vals if v is not None]
            if vals:
                row[name] = (np.mean(vals), np.std(vals, ddof=1))
            else:
                row[name] = (float('nan'), float('nan'))
        p_m, p_s = row["ProtoNet"]
        s_m, s_s = row["SCP"]
        l_m, l_s = row["LPA"]
        scp_win = "WIN" if s_m > p_m else "loss"
        lpa_win = "WIN" if l_m > p_m else "loss"
        print(f"{key:<20}{p_m:.4f}+-{p_s:.3f}   {s_m:.4f}+-{s_s:.3f}   {l_m:.4f}+-{l_s:.3f}   {scp_win:<10}{lpa_win:<10}")
        summary[key] = {
            "protonet": {"mean": p_m, "std": p_s},
            "scp": {"mean": s_m, "std": s_s},
            "lpa": {"mean": l_m, "std": l_s},
            "scp_delta": s_m - p_m,
            "lpa_delta": l_m - p_m,
        }

with open(os.path.join(RESULTS_DIR, "scp_lpa_multiseed_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print()
scp_wins = sum(1 for v in summary.values() if v["scp_delta"] > 0)
lpa_wins = sum(1 for v in summary.values() if v["lpa_delta"] > 0)
print(f"SCP beats ProtoNet on {scp_wins}/6 cells (mean delta {np.mean([v['scp_delta'] for v in summary.values()]):+.4f})")
print(f"LPA beats ProtoNet on {lpa_wins}/6 cells (mean delta {np.mean([v['lpa_delta'] for v in summary.values()]):+.4f})")