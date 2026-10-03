"""Add LinguisticProtoNet (LPA) to baselines.py + register variants in runner."""
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


# ---------------------------------------------------------------- baselines.py
src = read(BASE)

# Insert new class BEFORE ALL_BASELINES. We'll add right after cf_protonet_supcon_step.
new_classes = '''
SKIPPED (already applied)
'''


class LinguisticProtoNet(nn.Module):
    """Linguistic-Prior Prototype Aggregation (LPA-ProtoNet).

    Adds an inductive bias from Arabic dialectology to ProtoNet: at
    inference, the prototype of class c is a weighted mixture of the
    raw prototype and its linguistically related siblings:

        p_c^LPA = alpha * p_c + (1 - alpha) * sum_{c'} w(c, c') * p_{c'}

    where w(c, c') = exp(-d_ling(c, c') / tau_ling), normalised over c'
    so each row sums to 1.

    Aggregation is in-episode: only classes that appear in the current
    episode contribute, weighted by their dialectal similarity to c.
    Self-similarity is 1.0 (after normalisation), so the original
    prototype always keeps the largest weight.

    Hyperparameters:
      alpha (default 0.7): trust the raw prototype vs. linguistic
        siblings. Higher = closer to vanilla ProtoNet.
      tau_ling (default 0.5): softmax temperature over dialectal
        distances. Smaller = sharper (only close siblings contribute).
    """
    def __init__(self, embed_dim=256, alpha=0.7, tau_ling=0.5,
                 supcon_temperature=0.07, supcon_weight=1.0,
                 ce_weight=0.5):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        freeze_early_layers(self.encoder.encoder, n_unfreeze=4)
        self.alpha = alpha
        self.tau_ling = tau_ling
        self.supcon_temperature = supcon_temperature
        self.supcon_weight = supcon_weight
        self.ce_weight = ce_weight
        # dataset name and similarity matrix (set externally)
        self._dataset_name = None
        self._similarity = None  # np.ndarray (num_classes x num_classes), rows normalised
        self._episode_classes = None  # list[int], original ids of episode

    def set_dataset(self, name: str):
        """Load the dialectal similarity matrix for the given dataset."""
        from adibench.dialects import similarity_matrix
        self._dataset_name = name
        self._similarity = similarity_matrix(name, tau=self.tau_ling)
        # Force float32 tensor
        import torch
        self._similarity_t = torch.from_numpy(self._similarity).float()

    def set_episode_classes(self, chosen):
        self._episode_classes = list(chosen)

    def forward(self, support_x, support_mask, support_y, query_x, query_mask):
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        n_way = int(support_y.max().item()) + 1
        protos = torch.stack([s_feat[support_y == c].mean(0) for c in range(n_way)])
        # Apply LPA if dataset similarity is loaded and episode classes known
        if self._similarity is not None and self._episode_classes is not None and n_way > 1:
            sim = self._similarity_t.to(s_feat.device)
            # sim_mat[i, j] = similarity between episode class i and original class j
            idx = torch.tensor(self._episode_classes, dtype=torch.long, device=s_feat.device)
            sim_episode = sim[idx][:, idx]  # (n_way, n_way)
            # mix: p_c^LPA = alpha * p_c + (1-alpha) * sum_{c'} sim(c,c') * p_{c'}
            # use episode-class-prototypes as both ends; aggregation matrix sim_episode
            # self-weight (diagonal) is high since sim is row-normalised
            mixed = self.alpha * protos + (1.0 - self.alpha) * sim_episode @ protos
            protos_lpa = mixed
        else:
            protos_lpa = protos
        d = torch.cdist(q_feat, protos_lpa)
        return -d, q_feat, protos_lpa


def lpa_protonet_step(model, support_x, support_mask, support_y,
                       query_x, query_mask, query_y,
                       temperature=0.07, ce_weight=0.5, supcon_weight=1.0):
    """Combined SupCon + CE step for LinguisticProtoNet."""
    s_feat = model.encoder(support_x, support_mask)
    q_feat = model.encoder(query_x, query_mask)
    if supcon_weight > 0:
        all_feat = torch.cat([s_feat, q_feat], dim=0)
        all_y = torch.cat([support_y, query_y], dim=0)
        supcon = supervised_contrastive_loss(all_feat, all_y, temperature) * supcon_weight
    else:
        supcon = torch.zeros((), device=s_feat.device)
    n_way = int(support_y.max().item()) + 1
    protos = torch.stack([s_feat[support_y == c].mean(0) for c in range(n_way)])
    # compute LPA prototypes
    if model._similarity is not None and model._episode_classes is not None and n_way > 1:
        sim = model._similarity_t.to(s_feat.device)
        idx = torch.tensor(model._episode_classes, dtype=torch.long, device=s_feat.device)
        sim_episode = sim[idx][:, idx]
        protos_lpa = model.alpha * protos + (1.0 - model.alpha) * sim_episode @ protos
    else:
        protos_lpa = protos
    d = torch.cdist(q_feat, protos_lpa)
    ce = F.cross_entropy(-d, query_y)
    return supcon + ce_weight * ce, supcon.item(), ce.item()
'''

# Insert before ALL_BASELINES (already applied previously)
# write(BASE, src)
print("baselines.py already patched (skipped)")

# ---------------------------------------------------------------- runner
src = read(RUNNER)
# Add import
src = replace_once(
    src,
    "from adibench.baselines import (\n"
    "    ProtoNet, MatchingNet, RelationNet, FineTune, Focal, CB, MAML, Random,\n"
    "    CFProtoNet, cf_protonet_supcon_step,\n"
    ")",
    "from adibench.baselines import (\n"
    "    ProtoNet, MatchingNet, RelationNet, FineTune, Focal, CB, MAML, Random,\n"
    "    CFProtoNet, cf_protonet_supcon_step, LinguisticProtoNet, lpa_protonet_step,\n"
    ")",
    "runner import LPA",
)
# make_model entries: insert LPA block after cf_no_supcon
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
    "runner make_model LPA",
)
# Set dataset + episode classes
src = replace_once(
    src,
    "    # Set class frequency for CF-ProtoNet variants\n"
    "    if method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n",
    "    # Set class frequency for CF-ProtoNet variants and LPA's similarity matrix\n"
    "    if method in (\"cf_protonet\", \"cf_no_margin\", \"cf_no_supcon\"):\n",
    "class freq gate comment",
)
src = src.replace(
    "        freq = torch.tensor(\n"
    "            [df[df[\"label\"] == c].shape[0] for c in range(num_classes)],\n"
    "            dtype=torch.float)\n"
    "        model.set_class_freq(freq.to(device))",
    "        freq = torch.tensor(\n"
    "            [df[df[\"label\"] == c].shape[0] for c in range(num_classes)],\n"
    "            dtype=torch.float)\n"
    "        model.set_class_freq(freq.to(device))\n"
    "    if method in (\"lpa_protonet\", \"scp_lpa\"):\n"
    "        model.set_dataset(dataset)",
)
# Episode classes threading (eval and train)
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
print("runner patched")
