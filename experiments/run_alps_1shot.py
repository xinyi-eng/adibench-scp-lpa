"""Run ProtoNet and L2C at 1-shot on NADI 18."""
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

# ProtoNet 1-shot
print("\n===== ProtoNet 5-way 1-shot =====", flush=True)
result = train_fewshot(
    method="protonet", df_train=df, num_classes=18,
    n_way=5, k_shot=1, q_query=15,
    n_episodes=500, eval_every=100, n_eval_episodes=200,
    use_curriculum=False, seed=42, lr=1e-4,
    log_file=os.path.join(LOG_DIR, "protonet_5w1s.log"),
)
out = os.path.join(OUT_DIR, "protonet_nadi18_5w1s.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"ProtoNet 1-shot best: {result['best_acc']:.4f}", flush=True)

# L2C 1-shot
print("\n===== L2C-ProtoNet 5-way 1-shot =====", flush=True)
result = train_fewshot(
    method="l2c", df_train=df, num_classes=18,
    n_way=5, k_shot=1, q_query=15,
    n_episodes=500, eval_every=100, n_eval_episodes=200,
    use_curriculum=False, seed=42, lr=1e-4,
    log_file=os.path.join(LOG_DIR, "l2c_5w1s.log"),
)
out = os.path.join(OUT_DIR, "l2c_nadi18_5w1s.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"L2C 1-shot best: {result['best_acc']:.4f}", flush=True)

# L2C-strong 1-shot
print("\n===== L2C-strong 5-way 1-shot =====", flush=True)
result = train_fewshot(
    method="l2c_strong", df_train=df, num_classes=18,
    n_way=5, k_shot=1, q_query=15,
    n_episodes=500, eval_every=100, n_eval_episodes=200,
    use_curriculum=False, seed=42, lr=1e-4,
    log_file=os.path.join(LOG_DIR, "l2c_strong_5w1s.log"),
)
out = os.path.join(OUT_DIR, "l2c_strong_nadi18_5w1s.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"L2C-strong 1-shot best: {result['best_acc']:.4f}", flush=True)
