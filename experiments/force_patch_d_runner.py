"""Force-add softanchor support to D drive runner."""
import io

P = r"D:/dacd2026/adibench_v1/experiments/run_adibench_baseline.py"

with io.open(P, "r", encoding="utf-8") as f:
    s = f.read()

# Print current imports
import re
m = re.search(r"from adibench\.baselines import \(\n.*?\)", s, re.DOTALL)
if m:
    print("Current imports:")
    print(m.group(0))
m = re.search(r"def make_model\(.*?\n(?=\ndef )", s, re.DOTALL)
if m:
    print("\nCurrent make_model:")
    print(m.group(0))

# Add import
old = "CFProtoNet, cf_protonet_supcon_step,\n)"
new = "CFProtoNet, cf_protonet_supcon_step, LinguisticProtoNet, lpa_protonet_step, SoftAnchorProtoNet, softanchor_step,\n)"
if old in s:
    s = s.replace(old, new, 1)
    print("\nAdded import")
else:
    print("Import not found, manual fix needed")
    # Find the existing import
    m = re.search(r"from adibench\.baselines import \(\n(.*?)\)", s, re.DOTALL)
    if m:
        print("Existing:", m.group(0))

# Add make_model entries
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
if old in s:
    s = s.replace(old, new, 1)
    print("Added make_model")

# set_dataset patch
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
if old in s:
    s = s.replace(old, new, 1)
    print("Added set_dataset")

# Episode gates
for old in ['            if method in ("cf_protonet", "cf_no_margin", "cf_no_supcon"):',
             '        if method in ("cf_protonet", "cf_no_margin", "cf_no_supcon"):']:
    if old in s:
        s = s.replace(old, old.rstrip(":") + ', "lpa_protonet", "scp_lpa", "softanchor", "sa_proto"):', 1)
        print("Patched episode gate")

# Loss branches
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
if old in s:
    s = s.replace(old, new, 1)
    print("Added loss branches")

with io.open(P, "w", encoding="utf-8", newline="\n") as f:
    f.write(s)
print("D drive runner patched")