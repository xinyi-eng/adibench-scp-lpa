"""run_alps_protonet.py: Quick test of ProtoNet on NADI 18-way 5-shot.

This is the FIRST experiment to verify the pipeline works end-to-end
on the new ALPS framework. Expected runtime: ~15 min for 1000 episodes
on RTX 4060.

Usage:
  python run_alps_protonet.py
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, r"D:/dacd2026/alps_v1")
sys.path.insert(0, r"D:/dacd2026/site-packages")
from alps.data import load_nadi_18way
from alps.train import train_fewshot

OUT_DIR = r"D:/dacd2026/alps_v1/results"
LOG_DIR = r"D:/dacd2026/alps_v1/experiments/logs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Load NADI 18-way
print("Loading NADI 2024 (18-way)...")
df = load_nadi_18way()
print(f"Loaded {len(df)} examples, {df['label'].nunique()} classes")
print(f"Class distribution: {df['label'].value_counts().sort_index().tolist()}")

# Train ProtoNet baseline: 5-way 5-shot on NADI 18
print("\n" + "=" * 60)
print("Protonet baseline: 5-way 5-shot on NADI 18-way")
print("=" * 60)
log_file = os.path.join(LOG_DIR, "protonet_5w5s.log")
result = train_fewshot(
    method="protonet",
    df_train=df,
    num_classes=18,
    n_way=5,
    k_shot=5,
    q_query=15,
    n_episodes=500,
    eval_every=100,
    n_eval_episodes=200,
    use_curriculum=False,
    seed=42,
    lr=1e-4,
    log_file=log_file,
)
# Save result
out = os.path.join(OUT_DIR, "protonet_nadi18_5w5s.json")
with open(out, "w") as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {out}")
print(f"Best eval acc: {result['best_acc']:.4f}")
