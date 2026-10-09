"""Encoder architectures for DACD++ v5.

This file contains the v5 multi-encoder model used for CEDA (Cross-Encoder
Distillation Augmentation): a dual-encoder AraBERTv2 + MARBERTv2 model that
frozen MARBERTv2 acts as a teacher for AraBERTv2 training.

Encoders:
    - AraBERTv2Encoder: aubmindlab/bert-base-arabertv02 (general Arabic)
    - MARBERTv2Encoder: UBC-NLP/MARBERTv2 (Twitter-domain Arabic)
    - SingleEncoderClassifier: one encoder + linear head
    - CEDAClassifier: AraBERTv2 (student) + frozen MARBERTv2 (teacher)
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

from .utils import count_params, get_device


class SingleEncoderClassifier(nn.Module):
    """One encoder + linear head. Used as baseline."""

    def __init__(self, encoder_name, num_labels=5, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h, num_labels)
        self.num_labels = num_labels
        self.encoder_name = encoder_name

    def forward(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]
        cls = self.dropout(cls)
        return self.classifier(cls)

    def features(self, input_ids, attention_mask, **kwargs):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]


class MECClassifier(nn.Module):
    """Two-encoder MEC: shared head on concatenated features with consistency loss."""

    def __init__(self, encoder1_name, encoder2_name, num_labels=5, dropout=0.1):
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
        return self.classifier(h1)

    def forward_both(self, input_ids, attention_mask, alpha=0.5):
        h1 = self._features(self.encoder1, input_ids, attention_mask)
        h2 = self._features(self.encoder2, input_ids, attention_mask)
        logits1 = self.classifier(h1)
        logits2 = self.classifier(h2)
        p1 = torch.sigmoid(logits1)
        p2 = torch.sigmoid(logits2)
        cons = ((p1 - p2) ** 2).sum(dim=-1).mean()
        return logits1, logits2, cons * alpha


class CEDAClassifier(nn.Module):
    """CEDA: Cross-Encoder Distillation Augmentation.

    Architecture:
      - Student: AraBERTv2 (trainable)
      - Teacher: MARBERTv2 (frozen, only forward pass used for KD)

    The student takes both input and produces features+logits via the
    standard head. The teacher also takes the same input, producing its
    own features and logits which are used by the CEDA loss to distill
    minority-class knowledge from MARBERTv2 (Twitter-domain) into
    AraBERTv2 (general-domain).

    Gradient checkpointing is enabled on both encoders to fit in 8GB VRAM.
    """

    def __init__(self, student_name, teacher_name, num_labels=5, dropout=0.1,
                 use_teacher=True):
        super().__init__()
        self.student = AutoModel.from_pretrained(student_name)
        # Gradient checkpointing for memory
        try:
            self.student.gradient_checkpointing_enable()
        except Exception:
            pass
        self.teacher = AutoModel.from_pretrained(teacher_name)
        try:
            self.teacher.gradient_checkpointing_enable()
        except Exception:
            pass
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        h_s = self.student.config.hidden_size
        h_t = self.teacher.config.hidden_size
        # Project teacher features to student dim if different
        if h_s != h_t:
            self.teacher_proj = nn.Linear(h_t, h_s)
        else:
            self.teacher_proj = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h_s, num_labels)
        self.num_labels = num_labels
        self.use_teacher = use_teacher

    def _features(self, encoder, input_ids, attention_mask):
        out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0]

    def forward(self, input_ids, attention_mask, **kwargs):
        h_s = self.dropout(self._features(self.student, input_ids, attention_mask))
        return self.classifier(h_s)

    def features(self, input_ids, attention_mask, **kwargs):
        return self._features(self.student, input_ids, attention_mask)

    def forward_with_teacher(self, input_ids, attention_mask):
        """Returns (student_logits, student_features, teacher_logits, teacher_features)."""
        h_s = self.dropout(self._features(self.student, input_ids, attention_mask))
        s_logits = self.classifier(h_s)
        if self.use_teacher:
            with torch.no_grad():
                h_t = self._features(self.teacher, input_ids, attention_mask)
            t_logits = self.classifier(self.teacher_proj(self.dropout(h_t)))
        else:
            t_logits = s_logits.detach()
            h_t = h_s
        return s_logits, h_s, t_logits, h_t
