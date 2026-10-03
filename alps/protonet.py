"""ProtoNet and L2C-ProtoNet for few-shot Arabic dialect identification.

ProtoNet: Encode each example, compute per-class prototype as mean
of support features, classify queries by nearest prototype.

L2C-ProtoNet (Ours): Weight prototype distances by the linguistic
distance matrix. Classes that are linguistically close are expected
to have closer prototypes; this prior is combined with the
observed distance in a Bayesian-style log-posterior.

Math:
  ProtoNet:    log p(y|x) = -||f(x) - mu_y||^2
  L2C-ProtoNet: log p(y|x) = -alpha * ||f(x) - mu_y||^2
                            + beta  * log D_prior(y)
where D_prior(y) is a soft prior derived from the linguistic distance
matrix. The simplest form: D_prior(y) = 1 - dist[query_class, y]
for all candidate y, normalized.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .distance import get_linguistic_distance


class ProtoNetEncoder(nn.Module):
    """Encoder: AraBERTv2 + dropout + linear projection to embedding dim."""
    def __init__(self, encoder_name, embed_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(h, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        return self.proj(cls)


class ProtoNet(nn.Module):
    """Standard Prototypical Network (Snell et al. 2017)."""
    def __init__(self, encoder_name, embed_dim=256):
        super().__init__()
        self.encoder = ProtoNetEncoder(encoder_name, embed_dim)

    def forward(self, support_x, support_mask, support_y, query_x, query_mask):
        """Compute per-class prototype and query logits.

        support_y: (N*K,) int tensor with episode-wise labels 0..N-1
        query_x: (N*Q, L)
        Returns: (N*Q, N) logits = -||f(query) - mu_y||^2
        """
        s_feat = self.encoder(support_x, support_mask)  # (N*K, D)
        q_feat = self.encoder(query_x, query_mask)  # (N*Q, D)
        n_way = int(support_y.max().item()) + 1
        # Compute prototypes
        protos = torch.stack([
            s_feat[support_y == c].mean(0) for c in range(n_way)
        ])  # (N, D)
        # Distances: (N*Q, N)
        dists = -torch.cdist(q_feat, protos)  # negative distance = log p
        return dists, q_feat, protos


class L2CProtoNet(nn.Module):
    """L2C-ProtoNet: Linguistic-distance-weighted prototype network.

    log p(y|x) = -alpha * ||f(x) - mu_y||^2 + beta * log p_ling(y)
    where p_ling(y) is computed from a soft prior over the linguistic
    distance matrix. We use:
       p_ling(y) ∝ exp(-gamma * dist[query, y])
    This makes close classes (low dist) more likely a priori, but
    can be overridden by evidence (f(x) features).
    """
    def __init__(self, encoder_name, num_classes, embed_dim=256,
                 alpha=1.0, beta=0.5, gamma=2.0):
        super().__init__()
        self.encoder = ProtoNetEncoder(encoder_name, embed_dim)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # Cache the distance matrix (will be moved to device in forward)
        self.register_buffer("distance_matrix", get_linguistic_distance(num_classes))
        self.num_classes = num_classes

    def forward(self, support_x, support_mask, support_y, query_x, query_mask,
                query_orig_labels=None):
        """Compute L2C-ProtoNet logits.

        query_orig_labels: (N*Q,) original class indices in the dataset
                          (NOT episode-wise 0..N-1). Used to compute
                          linguistic prior for each query.
        """
        s_feat = self.encoder(support_x, support_mask)  # (N*K, D)
        q_feat = self.encoder(query_x, query_mask)  # (N*Q, D)
        n_way = int(support_y.max().item()) + 1
        # Episode-wise prototypes
        protos = torch.stack([
            s_feat[support_y == c].mean(0) for c in range(n_way)
        ])  # (N, D)
        # Build a mapping from episode label -> original class label
        # The support_y tensor is in episode labels; the prototype
        # for episode class c is built from samples with support_y == c.
        # To get the original class, we look at the first sample in
        # each support set.
        device = s_feat.device
        # Episode label to original label mapping
        ep_to_orig = []
        for c in range(n_way):
            mask_c = (support_y == c)
            if mask_c.any():
                ep_to_orig.append(int(support_y[mask_c][0].item()))
            else:
                ep_to_orig.append(c)
        # The actual original labels per query (provided as input)
        if query_orig_labels is None:
            # Fallback: use support_y mapping
            query_orig_labels = torch.tensor([
                ep_to_orig[min(int(q.item()) if isinstance(q, torch.Tensor) else int(q), n_way-1)]
                for q in range(q_feat.size(0))
            ], device=device)
        # Evidence: negative squared distance
        evidence = -self.alpha * torch.cdist(q_feat, protos)  # (N*Q, N)
        # Linguistic prior: exp(-gamma * D[query_orig, episode_orig])
        # Build the (N*Q, N) prior matrix
        D = self.distance_matrix.to(device)  # (num_classes, num_classes)
        prior = torch.zeros(q_feat.size(0), n_way, device=device)
        for q_i, q_orig in enumerate(query_orig_labels):
            for e_i, e_orig in enumerate(ep_to_orig):
                prior[q_i, e_i] = -self.gamma * D[int(q_orig), int(e_orig)]
        prior = F.log_softmax(prior, dim=-1)  # normalize over episode classes
        # Combined log-posterior
        logits = evidence + self.beta * prior
        return logits, q_feat, protos
