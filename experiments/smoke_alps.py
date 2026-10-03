"""Smoke test for ALPS framework."""
import sys
sys.path.insert(0, r"D:/dacd2026/alps_v1")
sys.path.insert(0, r"D:/dacd2026/site-packages")
import torch
from alps.distance import NADI_18_DISTANCE, QADI_5_DISTANCE, get_linguistic_distance
from alps.data import load_nadi_18way, build_fewshot_episode, NADI_18_COUNTRIES
from alps.protonet import ProtoNet, L2CProtoNet
from alps.curriculum import GeoCurriculum

print("=== ALPS smoke test ===", flush=True)
print(f"NADI_18_DISTANCE shape: {NADI_18_DISTANCE.shape}", flush=True)
print(f"QADI_5_DISTANCE shape: {QADI_5_DISTANCE.shape}", flush=True)
print(f"NADI 18 countries: {NADI_18_COUNTRIES}", flush=True)
# Load data
df = load_nadi_18way()
print(f"Loaded {len(df)} examples, {df['label'].nunique()} classes", flush=True)
# Test episode builder
s_idx, q_idx, s_lab, q_lab = build_fewshot_episode(df, n_way=5, k_shot=5, q_query=15)
print(f"Episode: support {len(s_idx)} (labels: {set(s_lab)}), query {len(q_idx)} (labels: {set(q_lab)})", flush=True)
# Test models
device = torch.device("cuda")
proto = ProtoNet(r"D:\dacd2026\2_models\arabertv02").to(device)
print(f"ProtoNet loaded, params: {sum(p.numel() for p in proto.parameters())/1e6:.2f}M", flush=True)
l2c = L2CProtoNet(r"D:\dacd2026\2_models\arabertv02", num_classes=18).to(device)
print(f"L2CProtoNet loaded, params: {sum(p.numel() for p in l2c.parameters())/1e6:.2f}M", flush=True)
# Test forward pass
B, L = 5, 32
sx = torch.randint(0, 1000, (B, L)).to(device)
sm = torch.ones(B, L).to(device)
sy = torch.tensor([0, 0, 1, 1, 2]).to(device)
qx = torch.randint(0, 1000, (15, L)).to(device)
qm = torch.ones(15, L).to(device)
q_orig = torch.randint(0, 18, (15,)).to(device)
logits, _, _ = proto(sx, sm, sy, qx, qm)
print(f"ProtoNet logits: {logits.shape}", flush=True)
logits2, _, _ = l2c(sx, sm, sy, qx, qm, query_orig_labels=q_orig)
print(f"L2CProtoNet logits: {logits2.shape}", flush=True)
# Test curriculum
curr = GeoCurriculum(num_classes=18, total_steps=1000)
print(f"Curriculum tau(0)={curr.get_tau(0):.3f}, tau(500)={curr.get_tau(500):.3f}, tau(999)={curr.get_tau(999):.3f}", flush=True)
print("=== SMOKE TEST PASSED ===", flush=True)
