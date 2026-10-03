"""Patch C drive runner to support lpa_protonet/scp_lpa/softanchor."""
import io

P = r"C:/Users/zsndz/Desktop/dacd_bench_v4_20260826_153119/experiments/run_adibench_baseline.py"

with io.open(P, "r", encoding="utf-8") as f:
    s = f.read()

# Step 1: imports
old = "CFProtoNet, cf_protonet_supcon_step,\n)"
new = "CFProtoNet, cf_protonet_supcon_step, LinguisticProtoNet, lpa_protonet_step, SoftAnchorProtoNet, softanchor_step,\n)"
assert old in s, "imports pattern not found"
s = s.replace(old, new, 1)

# Step 2: make_model entries
old = '    if method == "cf_protonet":\n        return CFProtoNet()\n    raise ValueError(method)'
new = ('    if method == "cf_protonet":\n'
       '        return CFProtoNet()\n'
       '    if method == "lpa_protonet":\n'
       '        return LinguisticProtoNet()\n'
       '    if method == "scp_lpa":\n'
       '        return LinguisticProtoNet()\n'
       '    if method in ("softanchor", "sa_proto"):\n'
       '        return SoftAnchorProtoNet()\n'
       '    raise ValueError(method)')
assert old in s, "make_model pattern not found"
s = s.replace(old, new, 1)

# Step 3: set_dataset for LPA/SA
old = ('    if method == "cf_protonet":\n'
       '        freq = torch.tensor(\n'
       '            [df[df["label"] == c].shape[0] for c in range(num_classes)],\n'
       '            dtype=torch.float)\n'
       '        model.set_class_freq(freq.to(device))')
new = old + ('\n'
             '    if method in ("lpa_protonet", "scp_lpa"):\n'
             '        model.set_dataset(dataset)\n'
             '    if method in ("softanchor", "sa_proto"):\n'
             '        model.set_dataset(dataset, num_classes)')
assert old in s, "set_freq pattern not found"
s = s.replace(old, new, 1)

# Step 4: episode gate (train + eval)
for old, indent in [
    ('            if method in ("cf_protonet", "cf_no_margin", "cf_no_supcon"):', 12),
    ('        if method in ("cf_protonet", "cf_no_margin", "cf_no_supcon"):', 8),
]:
    if old in s:
        new = old.rstrip(":") + ', "lpa_protonet", "scp_lpa", "softanchor", "sa_proto"):'
        s = s.replace(old, new, 1)
        print(f"Patched {indent}-space episode gate")

# Step 5: loss branches
old = ('        elif method in ("cf_protonet", "cf_no_margin", "cf_no_supcon"):\n'
       '            loss, _, _ = cf_protonet_supcon_step(\n'
       '                model, s_enc["input_ids"], s_enc["attention_mask"], s_y,\n'
       '                q_enc["input_ids"], q_enc["attention_mask"], q_y,\n'
       '                temperature=model.supcon_temperature,\n'
       '                ce_weight=getattr(model, "ce_weight", 0.5),\n'
       '                supcon_weight=getattr(model, "supcon_weight", 1.0))\n')
new = old + ('\n'
             '        elif method in ("lpa_protonet", "scp_lpa"):\n'
             '            loss, _, _ = lpa_protonet_step(\n'
             '                model, s_enc["input_ids"], s_enc["attention_mask"], s_y,\n'
             '                q_enc["input_ids"], q_enc["attention_mask"], q_y,\n'
             '                temperature=model.supcon_temperature,\n'
             '                ce_weight=getattr(model, "ce_weight", 0.5),\n'
             '                supcon_weight=getattr(model, "supcon_weight", 1.0))\n'
             '        elif method in ("softanchor", "sa_proto"):\n'
             '            loss, _, _ = softanchor_step(\n'
             '                model, s_enc["input_ids"], s_enc["attention_mask"], s_y,\n'
             '                q_enc["input_ids"], q_enc["attention_mask"], q_y,\n'
             '                temperature=model.supcon_temperature,\n'
             '                ce_weight=getattr(model, "ce_weight", 0.5),\n'
             '                supcon_weight=getattr(model, "supcon_weight", 1.0),\n'
             '                anchor_gamma=getattr(model, "anchor_gamma", 0.1))\n')
assert old in s, "loss branch pattern not found"
s = s.replace(old, new, 1)

with io.open(P, "w", encoding="utf-8", newline="\n") as f:
    f.write(s)
print("All patches applied successfully")