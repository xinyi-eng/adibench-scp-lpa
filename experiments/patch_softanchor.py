"""Add SoftAnchorProtoNet to baselines.py and register method in runner."""
import io, sys

BASE = r"D:/dacd2026/adibench_v1/adibench/baselines.py"
RUNNER = r"D:/dacd2026/adibench_v1/experiments/run_adibench_baseline.py"

NEW_CLASS = '''


class SoftAnchorProtoNet(nn.Module):
    """Soft-Anchor Prototype Aggregation (SA-ProtoNet).

    Replaces the hand-coded linguistic distance matrix with a
    learnable matrix that is softly anchored to the expert
    Versteegh matrix. This is the key novelty of the paper:
    experts provide a weak prior, and the model can correct
    mistakes in that prior if the data warrants.

    Forward uses the learned matrix; a soft-anchor loss pulls
    the learned matrix toward the expert matrix at training time.
    """
    def __init__(self, embed_dim=256, alpha=0.7, tau_ling=0.5,
                 manifold_dim=16, anchor_gamma=0.1,
                 supcon_temperature=0.07, supcon_weight=1.0,
                 ce_weight=0.5):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        freeze_early_layers(self.encoder.encoder, n_unfreeze=4)
        self.alpha = alpha
        self.tau_ling = tau_ling
        self.manifold_dim = manifold_dim
        self.anchor_gamma = anchor_gamma
        self.supcon_temperature = supcon_temperature
        self.supcon_weight = supcon_weight
        self.ce_weight = ce_weight
        self.class_embeddings = None
        self._dataset_name = None
        self._episode_classes = None

    def set_dataset(self, name, num_classes):
        from adibench.dialects import similarity_matrix
        self._dataset_name = name
        sim = similarity_matrix(name, tau=self.tau_ling)
        self.register_buffer("_expert_similarity_t", torch.from_numpy(sim).float())
        if self.class_embeddings is None or self.class_embeddings.shape[0] != num_classes:
            self.class_embeddings = nn.Parameter(
                torch.randn(num_classes, self.manifold_dim) * 0.1
            )

    def set_episode_classes(self, chosen):
        self._episode_classes = list(chosen)

    def learned_distance(self):
        """Compute W_learned[c, c'] = ||e_c - e_c'||^2 over all classes.
        Returns (num_classes, num_classes) tensor."""
        if self.class_embeddings is None:
            return None
        return torch.cdist(self.class_embeddings, self.class_embeddings, p=2) ** 2

    def learned_similarity_for_episode(self, num_classes):
        """Compute W_learned over current episode's classes; row-normalised."""
        if self._episode_classes is None:
            return None
        idx = torch.tensor(self._episode_classes, dtype=torch.long,
                            device=self.class_embeddings.device)
        emb = self.class_embeddings[idx]  # (n_way, d)
        d = torch.cdist(emb, emb, p=2) ** 2
        sim = torch.exp(-d / self.tau_ling)
        sim = sim / sim.sum(dim=1, keepdim=True)
        return sim

    def anchor_loss(self):
        """Frobenius distance between learned and expert similarity matrices."""
        if self.class_embeddings is None or not hasattr(self, "_expert_similarity_t"):
            return torch.zeros((), device=self.class_embeddings.device)
        d = self.learned_distance()
        sim = torch.exp(-d / self.tau_ling)
        sim = sim / sim.sum(dim=1, keepdim=True)
        return ((sim - self._expert_similarity_t) ** 2).sum()

    def forward(self, support_x, support_mask, support_y, query_x, query_mask):
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        n_way = int(support_y.max().item()) + 1
        protos = torch.stack([s_feat[support_y == c].mean(0) for c in range(n_way)])
        if self._episode_classes is not None and n_way > 1:
            sim = self.learned_similarity_for_episode(n_way)
            protos_lpa = self.alpha * protos + (1.0 - self.alpha) * sim @ protos
        else:
            protos_lpa = protos
        d = torch.cdist(q_feat, protos_lpa)
        return -d, q_feat, protos_lpa


def softanchor_step(model, support_x, support_mask, support_y,
                    query_x, query_mask, query_y,
                    temperature=0.07, ce_weight=0.5, supcon_weight=1.0,
                    anchor_gamma=0.1):
    """Combined SupCon + CE + anchor loss for SoftAnchorProtoNet."""
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
    if model._episode_classes is not None and n_way > 1:
        sim = model.learned_similarity_for_episode(n_way)
        protos_lpa = model.alpha * protos + (1.0 - model.alpha) * sim @ protos
    else:
        protos_lpa = protos
    d = torch.cdist(q_feat, protos_lpa)
    ce = F.cross_entropy(-d, query_y)
    anc = model.anchor_loss() if anchor_gamma > 0 else torch.zeros((), device=s_feat.device)
    return supcon + ce_weight * ce + anchor_gamma * anc, supcon.item(), ce.item(), anc.item()
'''


def read(p):
    with io.open(p, "r", encoding="utf-8") as f:
        return f.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)


# Patch baselines.py
src = read(BASE)
if "SoftAnchorProtoNet" not in src:
    src += NEW_CLASS
    # Register in ALL_BASELINES
    src = src.replace(
        '    "lpa_protonet": LinguisticProtoNet,\n'
        '    "scp_lpa": LinguisticProtoNet,\n',
        '    "lpa_protonet": LinguisticProtoNet,\n'
        '    "scp_lpa": LinguisticProtoNet,\n'
        '    "softanchor": SoftAnchorProtoNet,\n'
        '    "sa_proto": SoftAnchorProtoNet,\n'
    )
    print("baselines.py patched")
    write(BASE, src)
else:
    print("baselines.py already has SoftAnchorProtoNet")

# Patch runner
src = read(RUNNER)
if "SoftAnchorProtoNet" not in src:
    src = src.replace(
        "from adibench.baselines import (\n"
        "    ProtoNet, MatchingNet, RelationNet, FineTune, Focal, CB, MAML, Random,\n"
        "    CFProtoNet, cf_protonet_supcon_step, LinguisticProtoNet, lpa_protonet_step,\n"
        ")",
        "from adibench.baselines import (\n"
        "    ProtoNet, MatchingNet, RelationNet, FineTune, Focal, CB, MAML, Random,\n"
        "    CFProtoNet, cf_protonet_supcon_step, LinguisticProtoNet, lpa_protonet_step,\n"
        "    SoftAnchorProtoNet, softanchor_step,\n"
        ")",
    )
    # Add to make_model
    src = src.replace(
        '    if method in ("lpa_protonet", "scp_lpa"):\n'
        "        return LinguisticProtoNet()\n",
        '    if method in ("lpa_protonet", "scp_lpa"):\n'
        "        return LinguisticProtoNet()\n"
        '    if method in ("softanchor", "sa_proto"):\n'
        "        return SoftAnchorProtoNet()\n",
    )
    # Add set_dataset call
    src = src.replace(
        "    if method in (\"lpa_protonet\", \"scp_lpa\"):\n"
        "        model.set_dataset(dataset)\n",
        "    if method in (\"lpa_protonet\", \"scp_lpa\"):\n"
        "        model.set_dataset(dataset)\n"
        "    if method in (\"softanchor\", \"sa_proto\"):\n"
        "        model.set_dataset(dataset, num_classes)\n",
    )
    # Add episode gate
    src = src.replace(
        '"lpa_protonet", "scp_lpa"',
        '"lpa_protonet", "scp_lpa", "softanchor", "sa_proto"',
    )
    # Add loss branch
    src = src.replace(
        "        elif method in (\"lpa_protonet\", \"scp_lpa\"):\n"
        "            loss, _, _ = lpa_protonet_step(\n"
        "                model, s_enc[\"input_ids\"], s_enc[\"attention_mask\"], s_y,\n"
        "                q_enc[\"input_ids\"], q_enc[\"attention_mask\"], q_y,\n"
        "                temperature=model.supcon_temperature,\n"
        "                ce_weight=getattr(model, \"ce_weight\", 0.5),\n"
        "                supcon_weight=getattr(model, \"supcon_weight\", 1.0))\n",
        "        elif method in (\"lpa_protonet\", \"scp_lpa\"):\n"
        "            loss, _, _ = lpa_protonet_step(\n"
        "                model, s_enc[\"input_ids\"], s_enc[\"attention_mask\"], s_y,\n"
        "                q_enc[\"input_ids\"], q_enc[\"attention_mask\"], q_y,\n"
        "                temperature=model.supcon_temperature,\n"
        "                ce_weight=getattr(model, \"ce_weight\", 0.5),\n"
        "                supcon_weight=getattr(model, \"supcon_weight\", 1.0))\n"
        "        elif method in (\"softanchor\", \"sa_proto\"):\n"
        "            loss, _, _ = softanchor_step(\n"
        "                model, s_enc[\"input_ids\"], s_enc[\"attention_mask\"], s_y,\n"
        "                q_enc[\"input_ids\"], q_enc[\"attention_mask\"], q_y,\n"
        "                temperature=model.supcon_temperature,\n"
        "                ce_weight=getattr(model, \"ce_weight\", 0.5),\n"
        "                supcon_weight=getattr(model, \"supcon_weight\", 1.0),\n"
        "                anchor_gamma=getattr(model, \"anchor_gamma\", 0.1))\n",
    )
    print("runner patched")
    write(RUNNER, src)
else:
    print("runner already has SoftAnchorProtoNet")

print("done")