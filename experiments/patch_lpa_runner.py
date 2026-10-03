"""Apply LPA patches to runner.py (baselines.py already has LPA)."""
import io, sys

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


src = read(RUNNER)

# Add LPA import
src = replace_once(
    src,
    "    CFProtoNet, cf_protonet_supcon_step,\n"
    ")",
    "    CFProtoNet, cf_protonet_supcon_step, LinguisticProtoNet, lpa_protonet_step,\n"
    ")",
    "import LPA",
)

# Add LPA make_model entry
src = replace_once(
    src,
    "    if method == \"cf_no_supcon\":\n"
    "        m = CFProtoNet()\n"
    "        m.ce_weight = 1.0\n"
    "        m.supcon_weight = 0.0\n"
    "        return m\n"
    "    raise ValueError(method)",
    "    if method == \"cf_no_supcon\":\n"
    "        m = CFProtoNet()\n"
    "        m.ce_weight = 1.0\n"
    "        m.supcon_weight = 0.0\n"
    "        return m\n"
    "    if method in (\"lpa_protonet\", \"scp_lpa\"):\n"
    "        return LinguisticProtoNet()\n"
    "    raise ValueError(method)",
    "make_model LPA",
)

# set_dataset + episode gate
src = replace_once(
    src,
    "    # Set class frequency for CF-ProtoNet variants\n"
    "    if method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n",
    "    # Set class frequency for CF-ProtoNet variants and LPA similarity matrix\n"
    "    if method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n",
    "class-freq comment",
)
# Insert set_dataset() right after set_class_freq, with unique anchor
src = replace_once(
    src,
    "        model.set_class_freq(freq.to(device))\n"
    "    # Train\n"
    "    t_train = train_method(model, df, n_way, k_shot, q_query=15,",
    "        model.set_class_freq(freq.to(device))\n"
    "    if method in (\"lpa_protonet\", \"scp_lpa\"):\n"
    "        model.set_dataset(dataset)\n"
    "    # Train\n"
    "    t_train = train_method(model, df, n_way, k_shot, q_query=15,",
    "set_dataset insert",
)

# eval/train episode gate (extend method tuple)
src = replace_once(
    src,
    "            if method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n"
    "                s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)\n"
    "                model.set_episode_classes(chosen)",
    "            if method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\", \"lpa_protonet\", \"scp_lpa\"):\n"
    "                s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)\n"
    "                model.set_episode_classes(chosen)",
    "eval episode gate",
)
src = replace_once(
    src,
    "        if method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n"
    "            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)\n"
    "            model.set_episode_classes(chosen)",
    "        if method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\", \"lpa_protonet\", \"scp_lpa\"):\n"
    "            s_idx, q_idx, s_lab, q_lab, chosen = build_fewshot_episode_ex(df, n_way, k_shot, q_query)\n"
    "            model.set_episode_classes(chosen)",
    "train episode gate",
)

# Loss branch
src = replace_once(
    src,
    "        elif method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n"
    "            loss, _, _ = cf_protonet_supcon_step(\n"
    "                model, s_enc[\"input_ids\"], s_enc[\"attention_mask\"], s_y,\n"
    "                q_enc[\"input_ids\"], q_enc[\"attention_mask\"], q_y,\n"
    "                temperature=model.supcon_temperature,\n"
    "                ce_weight=getattr(model, 'ce_weight', 0.5),\n"
    "                supcon_weight=getattr(model, 'supcon_weight', 1.0))\n",
    "        elif method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n"
    "            loss, _, _ = cf_protonet_supcon_step(\n"
    "                model, s_enc[\"input_ids\"], s_enc[\"attention_mask\"], s_y,\n"
    "                q_enc[\"input_ids\"], q_enc[\"attention_mask\"], q_y,\n"
    "                temperature=model.supcon_temperature,\n"
    "                ce_weight=getattr(model, 'ce_weight', 0.5),\n"
    "                supcon_weight=getattr(model, 'supcon_weight', 1.0))\n"
    "        elif method in (\"lpa_protonet\", \"scp_lpa\"):\n"
    "            loss, _, _ = lpa_protonet_step(\n"
    "                model, s_enc[\"input_ids\"], s_enc[\"attention_mask\"], s_y,\n"
    "                q_enc[\"input_ids\"], q_enc[\"attention_mask\"], q_y,\n"
    "                temperature=model.supcon_temperature,\n"
    "                ce_weight=getattr(model, 'ce_weight', 0.5),\n"
    "                supcon_weight=getattr(model, 'supcon_weight', 1.0))\n",
    "loss branch LPA",
)

write(RUNNER, src)
print("runner patched successfully")
