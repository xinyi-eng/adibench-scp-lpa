"""ADIBench baseline implementations.

8 baseline methods for few-shot Arabic dialect identification:
  - ProtoNet: Prototypical Networks (Snell 2017)
  - MatchingNet: Matching Networks (Vinyals 2016)
  - RelationNet: Relation Networks (Sung 2018)
  - FineTune: Fine-tune encoder with cross-entropy
  - Focal: Focal loss
  - CB: Class-balanced loss
  - MAML: Model-agnostic meta-learning (Finn 2017)
  - Random: Random baseline (sanity check)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


ARABERT_PATH = r"D:\dacd2026\2_models\arabertv02"


class Encoder(nn.Module):
    """AraBERTv2 encoder + projection to embed_dim."""
    def __init__(self, encoder_name=ARABERT_PATH, embed_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(h, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(self.proj(out.last_hidden_state[:, 0]))


def freeze_early_layers(encoder, n_unfreeze=4):
    """Freeze all encoder parameters except last n_unfreeze layers."""
    for name, p in encoder.named_parameters():
        if "embeddings" in name or "pooler" in name:
            p.requires_grad = False
        elif "encoder.layer" in name:
            try:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                p.requires_grad = (layer_num >= 12 - n_unfreeze)
            except (IndexError, ValueError):
                p.requires_grad = False
        else:
            p.requires_grad = False


class ProtoNet(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        freeze_early_layers(self.encoder.encoder, n_unfreeze=4)

    def forward(self, support_x, support_mask, support_y, query_x, query_mask):
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        n_way = int(support_y.max().item()) + 1
        protos = torch.stack([s_feat[support_y == c].mean(0) for c in range(n_way)])
        return -torch.cdist(q_feat, protos), q_feat, protos


class MatchingNet(nn.Module):
    """Matching Networks with full context embedding (simplified)."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        freeze_early_layers(self.encoder.encoder, n_unfreeze=4)

    def forward(self, support_x, support_mask, support_y, query_x, query_mask):
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        # Cosine similarity
        s_norm = F.normalize(s_feat, dim=-1)
        q_norm = F.normalize(q_feat, dim=-1)
        n_way = int(support_y.max().item()) + 1
        # Softmax attention over support, weighted by class
        sim = q_norm @ s_norm.t()  # (Q, N*K)
        attn = F.softmax(sim, dim=-1)
        # Class predictions
        out = torch.zeros(q_feat.size(0), n_way, device=q_feat.device)
        for c in range(n_way):
            mask = (support_y == c).float()
            out[:, c] = (attn * mask).sum(-1)
        return out.log(), q_feat, None


class RelationNet(nn.Module):
    """Relation Networks (Sung 2018) - simplified."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        self.relation = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        freeze_early_layers(self.encoder.encoder, n_unfreeze=4)

    def forward(self, support_x, support_mask, support_y, query_x, query_mask):
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        n_way = int(support_y.max().item()) + 1
        n_per = s_feat.size(0) // n_way
        # Reshape support to (N, K, D), compute mean per class
        s_feat = s_feat.view(n_way, n_per, -1).mean(1)  # (N, D)
        # Compute relation: for each query, each class proto
        q_exp = q_feat.unsqueeze(1).expand(-1, n_way, -1)  # (Q, N, D)
        s_exp = s_feat.unsqueeze(0).expand(q_feat.size(0), -1, -1)  # (Q, N, D)
        pair = torch.cat([q_exp, s_exp], dim=-1)  # (Q, N, 2D)
        rel = self.relation(pair).squeeze(-1)  # (Q, N)
        return rel, q_feat, None


class FineTune(nn.Module):
    """Fine-tune encoder with cross-entropy on support+query."""
    def __init__(self, embed_dim=256, num_classes=5):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        freeze_early_layers(self.encoder.encoder, n_unfreeze=4)

    def forward(self, support_x, support_mask, support_y, query_x, query_mask):
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        s_logits = self.classifier(s_feat)
        q_logits = self.classifier(q_feat)
        return q_logits, q_feat, None

    def loss(self, support_x, support_mask, support_y, query_x, query_mask, query_y):
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        s_logits = self.classifier(s_feat)
        q_logits = self.classifier(q_feat)
        return F.cross_entropy(s_logits, support_y) + F.cross_entropy(q_logits, query_y)


class Focal(nn.Module):
    """Focal loss for imbalanced data (Lin 2017)."""
    def __init__(self, embed_dim=256, num_classes=5, gamma=2.0):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.gamma = gamma
        freeze_early_layers(self.encoder.encoder, n_unfreeze=4)

    def forward(self, support_x, support_mask, support_y, query_x, query_mask):
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        s_logits = self.classifier(s_feat)
        q_logits = self.classifier(q_feat)
        return q_logits, q_feat, None

    def loss(self, support_x, support_mask, support_y, query_x, query_mask, query_y):
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        s_logits = self.classifier(s_feat)
        q_logits = self.classifier(q_feat)
        ce = F.cross_entropy(s_logits, support_y, reduction='none')
        pt = torch.exp(-ce)
        s_loss = ((1 - pt) ** self.gamma * ce).mean()
        ce_q = F.cross_entropy(q_logits, query_y, reduction='none')
        pt_q = torch.exp(-ce_q)
        q_loss = ((1 - pt_q) ** self.gamma * ce_q).mean()
        return s_loss + q_loss


class CB(nn.Module):
    """Class-balanced loss (Cui 2019)."""
    def __init__(self, embed_dim=256, num_classes=5, beta=0.999):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.beta = beta
        freeze_early_layers(self.encoder.encoder, n_unfreeze=4)

    def get_weights(self, labels):
        counts = torch.bincount(labels, minlength=self.num_classes).float()
        eff = (1 - self.beta ** counts) / (1 - self.beta)
        eff = torch.where(eff > 0, eff, torch.ones_like(eff))
        per_class = 1.0 / eff
        per_class = per_class / per_class.sum() * self.num_classes
        return per_class.to(labels.device)

    def forward(self, support_x, support_mask, support_y, query_x, query_mask):
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        s_logits = self.classifier(s_feat)
        q_logits = self.classifier(q_feat)
        return q_logits, q_feat, None

    def loss(self, support_x, support_mask, support_y, query_x, query_mask, query_y):
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        s_logits = self.classifier(s_feat)
        q_logits = self.classifier(q_feat)
        w = self.get_weights(support_y)
        return F.cross_entropy(s_logits, support_y, weight=w) + F.cross_entropy(q_logits, query_y)


class MAML(nn.Module):
    """MAML-style meta-learning (simplified - first-order approximation)."""
    def __init__(self, embed_dim=256, num_classes=5):
        super().__init__()
        self.encoder = Encoder(embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        freeze_early_layers(self.encoder.encoder, n_unfreeze=4)

    def forward(self, support_x, support_mask, support_y, query_x, query_mask):
        q_feat = self.encoder(query_x, query_mask)
        q_logits = self.classifier(q_feat)
        return q_logits, q_feat, None

    def adapt_and_predict(self, support_x, support_mask, support_y, query_x, query_mask, inner_lr=0.01):
        """Simple MAML: fine-tune classifier on support, predict query."""
        s_feat = self.encoder(support_x, support_mask)
        q_feat = self.encoder(query_x, query_mask)
        # Inner loop: SGD on classifier weights
        s_logits = self.classifier(s_feat)
        inner_loss = F.cross_entropy(s_logits, support_y)
        grads = torch.autograd.grad(inner_loss, self.classifier.parameters(), create_graph=True)
        # Fast weights
        fast_weights = [p - inner_lr * g for p, g in zip(self.classifier.parameters(), grads)]
        # Predict query with fast weights
        q_logits = F.linear(q_feat, fast_weights[0], fast_weights[1])
        return q_logits, q_feat, None


class Random(nn.Module):
    """Random baseline - uniform prediction over classes."""
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, support_x, support_mask, support_y, query_x, query_mask):
        # Random logits
        n_query = query_x.size(0)
        return torch.zeros(n_query, self.num_classes), None, None


ALL_BASELINES = {
    "protonet": ProtoNet,
    "matchingnet": MatchingNet,
    "relationnet": RelationNet,
    "finetune": FineTune,
    "focal": Focal,
    "cb": CB,
    "maml": MAML,
    "random": Random,
}
