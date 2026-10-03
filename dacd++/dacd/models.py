"""Encoder architectures for DACD++.

Encoders:
    - AraBERTv2Encoder: aubmindlab/bert-base-arabertv02
    - MARBERTv2Encoder: UBC-NLP/MARBERTv2
    - SingleClassifier: standard head
    - MECClassifier: shared head across two encoders with consistency loss
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from .utils import count_params, get_device


class SingleEncoderClassifier(nn.Module):
    """One encoder + linear head. Used as baseline (CE, Focal, CB, DACD)."""

    def __init__(self, encoder_name: str, num_labels: int = 5, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # use CLS token
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        return self.classifier(cls)

    def features(self, input_ids, attention_mask, **kwargs):
        """Return pre-classifier features (for MEC)."""
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]


class MECClassifier(nn.Module):
    """Two-encoder MEC: shared head on concatenated [h_Ara; h_MAR] with consistency loss.

    The consistency loss is computed in eval-friendly way:
        L_cons = || sigmoid(W h_A) - sigmoid(W h_M) ||^2
    where W is the shared classifier.
    """

    def __init__(
        self,
        encoder1_name: str,
        encoder2_name: str,
        num_labels: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder1 = AutoModel.from_pretrained(encoder1_name)
        self.encoder2 = AutoModel.from_pretrained(encoder2_name)
        h1 = self.encoder1.config.hidden_size
        h2 = self.encoder2.config.hidden_size
        assert h1 == h2, f"Encoders must have same hidden dim, got {h1} vs {h2}"
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h1, num_labels)
        self.num_labels = num_labels

    def _features(self, encoder, input_ids, attention_mask):
        out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.dropout(out.last_hidden_state[:, 0])

    def forward(self, input_ids, attention_mask, **kwargs):
        h1 = self._features(self.encoder1, input_ids, attention_mask)
        h2 = self._features(self.encoder2, input_ids, attention_mask)
        # use AraBERT as primary
        return self.classifier(h1)

    def forward_both(self, input_ids, attention_mask, alpha: float = 0.5):
        """Returns (logits1, logits2, consistency_loss) for MEC training."""
        h1 = self._features(self.encoder1, input_ids, attention_mask)
        h2 = self._features(self.encoder2, input_ids, attention_mask)
        logits1 = self.classifier(h1)
        logits2 = self.classifier(h2)
        p1 = torch.sigmoid(logits1)
        p2 = torch.sigmoid(logits2)
        cons = ((p1 - p2) ** 2).sum(dim=-1).mean()
        return logits1, logits2, cons * alpha
