"""Add ablation variants for CF-ProtoNet:
- cf_no_margin: SupCon + CE only (no LDAM, no prior scaling)
- cf_no_supcon: CE + LDAM + prior (no contrastive pre-training)
Mechanism flags are model attributes read by the runner.
"""
import io, sys

BASE = r"D:/dacd2026/adibench_v1/adibench/baselines.py"
RUNNER = r"D:/dacd2026/adibench_v1/experiments/run_adibench_baseline.py"


def read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


def replace_once(src, old, new, label):
    n = src.count(old)
    if n != 1:
        print(f"FAIL [{label}]: {n} occurrences")
        sys.exit(1)
    print(f"ok  [{label}]")
    return src.replace(old, new)


# baselines: supcon_weight actually used in step fn
src = read(BASE)
src = replace_once(
    src,
    "def cf_protonet_supcon_step(model, support_x, support_mask, support_y,\n"
    "                            query_x, query_mask, query_y,\n"
    "                            temperature=0.07, ce_weight=0.5):",
    "def cf_protonet_supcon_step(model, support_x, support_mask, support_y,\n"
    "                            query_x, query_mask, query_y,\n"
    "                            temperature=0.07, ce_weight=0.5, supcon_weight=1.0):",
    "step fn signature")
src = replace_once(
    src,
    "    supcon = supervised_contrastive_loss(all_feat, all_y, temperature)\n",
    "    if supcon_weight > 0:\n"
    "        supcon = supervised_contrastive_loss(all_feat, all_y, temperature) * supcon_weight\n"
    "    else:\n"
    "        supcon = torch.zeros((), device=all_feat.device)\n",
    "step fn supcon weight")
write(BASE, src)

# runner: new variants in make_model + attribute-driven weights
src = read(RUNNER)
src = replace_once(
    src,
    "    if method == \"cf_protonet\":\n        return CFProtoNet()\n",
    "    if method == \"cf_protonet\":\n        return CFProtoNet()\n"
    "    if method == \"cf_no_margin\":\n"
    "        m = CFProtoNet(ldam_margin=0.0, prior_alpha=0.0)\n"
    "        return m\n"
    "    if method == \"cf_no_supcon\":\n"
    "        m = CFProtoNet()\n"
    "        m.ce_weight = 1.0\n"
    "        m.supcon_weight = 0.0\n"
    "        return m\n",
    "runner variants")
src = replace_once(
    src,
    "    # Set class frequency for CF-ProtoNet\n"
    "    if method == \"cf_protonet\":\n",
    "    # Set class frequency for CF-ProtoNet variants\n"
    "    if method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n",
    "runner class-freq gate")
src = replace_once(
    src,
    "        elif method == \"cf_protonet\":\n"
    "            loss, _, _ = cf_protonet_supcon_step(\n"
    "                model, s_enc[\"input_ids\"], s_enc[\"attention_mask\"], s_y,\n"
    "                q_enc[\"input_ids\"], q_enc[\"attention_mask\"], q_y,\n"
    "                temperature=model.supcon_temperature, ce_weight=0.5)",
    "        elif method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n"
    "            loss, _, _ = cf_protonet_supcon_step(\n"
    "                model, s_enc[\"input_ids\"], s_enc[\"attention_mask\"], s_y,\n"
    "                q_enc[\"input_ids\"], q_enc[\"attention_mask\"], q_y,\n"
    "                temperature=model.supcon_temperature,\n"
    "                ce_weight=getattr(model, 'ce_weight', 0.5),\n"
    "                supcon_weight=getattr(model, 'supcon_weight', 1.0))",
    "runner loss branch")
# evaluate(): chosen-classes threading must cover variants too
src = replace_once(
    src,
    "            if method == \"cf_protonet\":\n"
    "                s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)\n"
    "                model.set_episode_classes(chosen)",
    "            if method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n"
    "                s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)\n"
    "                model.set_episode_classes(chosen)",
    "runner eval variants gate")
src = replace_once(
    src,
    "        if method == \"cf_protonet\":\n"
    "            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)\n"
    "            model.set_episode_classes(chosen)",
    "        if method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n"
    "            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)\n"
    "            model.set_episode_classes(chosen)",
    "runner train variants gate")
write(RUNNER, src)
print("ablation patch done")
