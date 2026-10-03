"""Run more experiments: 3-shot, 18-way 1-shot, Geo-Curriculum."""
import sys, os, json
sys.path.insert(0, r"D:/dacd2026/alps_v1")
sys.path.insert(0, r"D:/dacd2026/site-packages")
from alps.data import load_nadi_18way
from alps.train import train_fewshot

OUT_DIR = r"D:/dacd2026/alps_v1/results"
LOG_DIR = r"D:/dacd2026/alps_v1/experiments/logs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("Loading NADI 2024...", flush=True)
df = load_nadi_18way()

def run(name, method, n_way, k_shot, n_ep=500, use_curr=False):
    print(f"\n===== {name} ({n_way}-way {k_shot}-shot) =====", flush=True)
    log_file = os.path.join(LOG_DIR, f"{name}.log")
    result = train_fewshot(
        method=method, df_train=df, num_classes=18,
        n_way=n_way, k_shot=k_shot, q_query=15,
        n_episodes=n_ep, eval_every=100, n_eval_episodes=200,
        use_curriculum=use_curr, seed=42, lr=1e-4,
        log_file=log_file,
    )
    out = os.path.join(OUT_DIR, f"{name}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{name} best: {result['best_acc']:.4f}", flush=True)
    return result

# 3-shot
run("protonet_3shot", "protonet", 5, 3)
run("l2c_3shot", "l2c", 5, 3)

# 18-way 1-shot (harder, more classes)
run("protonet_18w1s", "protonet", 18, 1)
run("l2c_18w1s", "l2c", 18, 1)

# Geo-Curriculum
run("protonet_curr_1shot", "protonet", 5, 1, use_curr=True)
run("l2c_curr_1shot", "l2c", 5, 1, use_curr=True)
