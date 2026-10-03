"""Run multi-seed 1-shot + Geo-Curriculum + QADI 5-way for full evaluation."""
import sys, os, json
sys.path.insert(0, r"D:/dacd2026/alps_v1")
sys.path.insert(0, r"D:/dacd2026/site-packages")
from alps.data import load_nadi_18way, load_nadi_5way, load_amgadhasan_5way
from alps.train import train_fewshot

OUT_DIR = r"D:/dacd2026/alps_v1/results"
LOG_DIR = r"D:/dacd2026/alps_v1/experiments/logs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("Loading datasets...", flush=True)
df_nadi = load_nadi_18way()
df_qadi = load_nadi_5way()
df_amg = load_amgadhasan_5way()

def run(name, method, df, num_classes, n_way, k_shot, n_ep=500, use_curr=False, seed=42):
    print(f"\n===== {name} (seed={seed}) =====", flush=True)
    log_file = os.path.join(LOG_DIR, f"{name}.log")
    result = train_fewshot(
        method=method, df_train=df, num_classes=num_classes,
        n_way=n_way, k_shot=k_shot, q_query=15,
        n_episodes=n_ep, eval_every=100, n_eval_episodes=200,
        use_curriculum=use_curr, seed=seed, lr=1e-4,
        log_file=log_file,
    )
    out = os.path.join(OUT_DIR, f"{name}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{name} best: {result['best_acc']:.4f}", flush=True)
    return result

# Multi-seed for 1-shot (most important)
for seed in [0, 7, 42]:
    run(f"protonet_nadi18_5w1s_s{seed}", "protonet", df_nadi, 18, 5, 1, seed=seed)
    run(f"l2c_nadi18_5w1s_s{seed}", "l2c", df_nadi, 18, 5, 1, seed=seed)

# Geo-Curriculum test
print("\n--- Geo-Curriculum ---", flush=True)
run("protonet_nadi18_curr_5w1s", "protonet", df_nadi, 18, 5, 1, use_curr=True)
run("l2c_nadi18_curr_5w1s", "l2c", df_nadi, 18, 5, 1, use_curr=True)

# QADI 5-way 1-shot (multi-granularity)
print("\n--- QADI 5-way ---", flush=True)
run("protonet_qadi5_5w1s", "protonet", df_qadi, 5, 5, 1)
run("l2c_qadi5_5w1s", "l2c", df_qadi, 5, 5, 1)

# amgadhasan 5-way 1-shot
print("\n--- amgadhasan 5-way ---", flush=True)
run("protonet_amg5_5w1s", "protonet", df_amg, 5, 5, 1)
run("l2c_amg5_5w1s", "l2c", df_amg, 5, 5, 1)
