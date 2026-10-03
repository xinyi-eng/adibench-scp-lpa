"""Run L2C-ProtoNet on NADI 18-way 5-shot. Compares against ProtoNet baseline."""
import sys, os, json
sys.path.insert(0, r"D:/dacd2026/alps_v1")
sys.path.insert(0, r"D:/dacd2026/site-packages")
from alps.data import load_nadi_18way
from alps.train import train_fewshot

OUT_DIR = r"D:/dacd2026/alps_v1/results"
LOG_DIR = r"D:/dacd2026/alps_v1/experiments/logs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

print("Loading NADI 2024 (18-way)...", flush=True)
df = load_nadi_18way()
print(f"Loaded {len(df)} examples", flush=True)

print("\n===== L2C-ProtoNet (beta=0.5) on NADI 18-way 5-shot =====", flush=True)
log_file = os.path.join(LOG_DIR, "l2c_5w5s.log")
result = train_fewshot(
    method="l2c",
    df_train=df,
    num_classes=18,
    n_way=5, k_shot=5, q_query=15,
    n_episodes=500,
    eval_every=100,
    n_eval_episodes=200,
    use_curriculum=False,
    seed=42,
    lr=1e-4,
    log_file=log_file,
)
out = os.path.join(OUT_DIR, "l2c_nadi18_5w5s.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nL2C best: {result['best_acc']:.4f}", flush=True)

print("\n===== L2C-ProtoNet (strong, beta=1.0) on NADI 18-way 5-shot =====", flush=True)
log_file = os.path.join(LOG_DIR, "l2c_strong_5w5s.log")
result = train_fewshot(
    method="l2c_strong",
    df_train=df,
    num_classes=18,
    n_way=5, k_shot=5, q_query=15,
    n_episodes=500,
    eval_every=100,
    n_eval_episodes=200,
    use_curriculum=False,
    seed=42,
    lr=1e-4,
    log_file=log_file,
)
out = os.path.join(OUT_DIR, "l2c_strong_nadi18_5w5s.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nL2C-strong best: {result['best_acc']:.4f}", flush=True)
