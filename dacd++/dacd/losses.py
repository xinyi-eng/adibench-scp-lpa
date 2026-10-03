"""Loss functions and samplers for DACD++."""
from __future__ import annotations

import math
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


class CISSampler(Sampler):
    def __init__(self, labels, epoch, total_epochs, data_imbalance=150.0,
                 max_ratio_start=1.0, max_ratio_end=7000.0, seed=42):
        self.labels = np.asarray(labels)
        self.epoch = epoch
        self.total_epochs = total_epochs
        peak = max(data_imbalance, max_ratio_end)
        f = (epoch / max(1, total_epochs - 1)) ** 2
        self.target_ratio = max_ratio_start + (peak - max_ratio_start) * f
        self.seed = seed
        counts = np.bincount(self.labels)
        self.majority_class = int(np.argmax(counts))
        n_majority = int(counts[self.majority_class])
        n_minority = max(1, int(n_majority / self.target_ratio))
        self.class_target_n = {}
        for c in range(len(counts)):
            if c == self.majority_class:
                self.class_target_n[c] = int(counts[c])
            else:
                self.class_target_n[c] = n_minority
        self.length = sum(self.class_target_n.values())

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        per_class_indices = {}
        for c in range(int(self.labels.max()) + 1):
            per_class_indices[c] = np.where(self.labels == c)[0].tolist()
        batch_indices = []
        for c, idx_list in per_class_indices.items():
            target = self.class_target_n.get(c, len(idx_list))
            if len(idx_list) > target:
                chosen = rng.sample(idx_list, target)
            else:
                chosen = [rng.choice(idx_list) for _ in range(target)]
            batch_indices.extend(chosen)
        rng.shuffle(batch_indices)
        return iter(batch_indices)

    def __len__(self):
        return self.length


def class_balanced_weights(labels, num_classes, beta=0.999):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts_safe = np.where(counts > 0, counts, 1.0)
    eff = (1.0 - np.power(beta, counts_safe)) / (1.0 - beta)
    per_class = 1.0 / eff
    per_class = np.where(counts > 0, per_class, 1.0)
    s_total = per_class.sum()
    if s_total > 0:
        per_class = per_class / s_total * num_classes
    return torch.tensor(per_class, dtype=torch.float32)


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, targets):
        return self.ce(logits, targets)


class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2.0, class_weights=None):
        super().__init__()
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        nll = -logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        w = (1 - p.gather(1, targets.unsqueeze(1)).squeeze(1)) ** self.gamma
        loss = w * nll
        if self.class_weights is not None:
            cw = self.class_weights.to(loss.device)[targets]
            loss = loss * cw
        return loss.mean()


class DACDLoss(nn.Module):
    def __init__(self, num_classes, lam=0.5, beta=0.3, temperature=0.07, class_weights=None):
        super().__init__()
        self.lam = lam
        self.beta = beta
        self.temperature = temperature
        self.class_weights = class_weights
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction="mean")

    def forward(self, logits, targets, features=None):
        loss_ce = self.ce(logits, targets)
        if features is None:
            return loss_ce
        feats = F.normalize(features, dim=-1)
        B = feats.size(0)
        sim = feats @ feats.t() / self.temperature
        y = targets.unsqueeze(0) == targets.unsqueeze(1)
        eye = torch.eye(B, dtype=torch.bool, device=feats.device)
        pos_mask = y & ~eye
        neg_mask = ~y
        weighted_neg = neg_mask.float() * self.beta + pos_mask.float() * 1.0
        exp_sim = torch.exp(sim) * (~eye).float()
        denom = (exp_sim * weighted_neg).sum(dim=-1) + 1e-12
        log_prob = sim - torch.log(denom)
        pos_count = pos_mask.float().sum(dim=-1)
        loss_supcon = -(log_prob * pos_mask.float()).sum(dim=-1) / (pos_count + 1e-12)
        if self.class_weights is not None:
            cw = self.class_weights.to(loss_supcon.device)[targets]
            loss_supcon = loss_supcon * cw
        loss_supcon = loss_supcon.mean()
        return (1 - self.lam) * loss_ce + self.lam * loss_supcon


class MECLoss(nn.Module):
    def __init__(self, ce_loss, alpha=0.5):
        super().__init__()
        self.ce = ce_loss
        self.alpha = alpha

    def forward(self, logits1, logits2, targets, cons):
        return self.ce(logits1, targets) + self.ce(logits2, targets) + self.alpha * cons


class PrototypeLoss(nn.Module):
    """Contrastive loss vs class prototypes (computed dynamically from batch).

    For each batch:
    1. compute per-class mean embedding (prototype)
    2. each sample pulls toward its own class prototype, pushes away from others
    3. weight by inverse class frequency (rare classes get stronger gradient)
    """
    def __init__(self, num_classes, temperature=0.07, alpha=0.5, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.class_weights = class_weights

    def forward(self, logits, targets, features=None):
        if features is None:
            return torch.tensor(0., device=logits.device, requires_grad=True)
        feats = F.normalize(features, dim=-1)
        # compute per-class prototypes
        protos = []
        proto_exists = []
        for c in range(self.num_classes):
            mask = (targets == c)
            if mask.sum() > 0:
                protos.append(feats[mask].mean(0))
                proto_exists.append(True)
            else:
                protos.append(torch.zeros_like(feats[0]))
                proto_exists.append(False)
        protos = torch.stack(protos)  # (C, D)
        proto_exists = torch.tensor(proto_exists, device=feats.device)
        # similarity: (B, C)
        sim = feats @ protos.t() / self.temperature
        # contrastive: positive is own class, negative is other classes
        pos_mask = F.one_hot(targets, self.num_classes).float()  # (B, C)
        neg_mask = 1.0 - pos_mask
        # log prob over present classes only
        valid = proto_exists.float().unsqueeze(0)  # (1, C)
        exp_sim = torch.exp(sim) * valid
        denom = exp_sim.sum(dim=-1, keepdim=True) + 1e-12
        log_prob = sim - torch.log(denom)
        # only positive contribution
        loss = -(log_prob * pos_mask).sum(dim=-1)
        # weight by inverse class frequency
        if self.class_weights is not None:
            cw = self.class_weights.to(loss.device)[targets]
            loss = loss * cw
        return self.alpha * loss.mean()



class LDAMLoss(nn.Module):
    """LDAM Loss (Cao et al. 2019) - Label-Distribution-Aware Margin.

    Adds a class-specific margin to logits:
        margin_k = C / n_k^(1/4)
    where C is a hyperparameter and n_k is the count of class k.

    For training, uses a deferred re-weighting (only applied after warmup).
    """
    def __init__(self, num_classes, max_m=0.5, s=30.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.max_m = max_m
        self.s = s
        self.class_weights = class_weights

    def forward(self, logits, targets):
        # logits: (B, C); targets: (B,)
        # Compute per-class margin scaled by 1/n^0.25 (proxy for inverse frequency)
        # Without per-class counts here, we approximate via uniform max_m
        # but apply stronger margin to minority classes via class_weights
        idx = torch.arange(self.num_classes, device=logits.device).unsqueeze(0)
        target_idx = targets.unsqueeze(1)
        # Standard LDAM: add margin based on class
        # Without n_k here, use class_weights as a proxy (rare classes get larger margin)
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            # Invert and normalize: rare (high cw) -> large margin
            margin = (cw / cw.max()) * self.max_m  # (C,)
        else:
            margin = torch.full((self.num_classes,), self.max_m, device=logits.device)
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        target_logits = logits[batch_idx, targets]
        # Subtract margin from target logits
        margin_applied = torch.zeros_like(logits)
        margin_applied[batch_idx, targets] = margin[targets]
        adjusted_logits = logits - margin_applied
        # Scale and compute cross-entropy
        scaled_logits = adjusted_logits * self.s
        loss = F.cross_entropy(scaled_logits, targets, weight=self.class_weights)
        return loss


class LogitAdjustmentLoss(nn.Module):
    """Logit Adjustment (Menon et al. 2021) - adjusts logits by class prior.

    Adds tau * log(pi_y) to logits at training time, where pi_y is the
    empirical class prior. This makes the model optimize a balanced objective
    without explicitly reweighting the loss.
    """
    def __init__(self, num_classes, class_priors=None, tau=1.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau
        self.class_weights = class_weights
        if class_priors is not None:
            priors = torch.tensor(class_priors, dtype=torch.float32)
            self.adjustment = tau * torch.log(priors + 1e-12)
        else:
            self.adjustment = None

    def forward(self, logits, targets):
        if self.adjustment is not None:
            adj = self.adjustment.to(logits.device)
            logits = logits + adj.unsqueeze(0)
        return F.cross_entropy(logits, targets, weight=self.class_weights)


class ReCLLoss(nn.Module):
    """ReCL (Cao et al. NeurIPS 2021) simplified.
    L = (1 - alpha) * L_CE + alpha * L_SupCon class-reweighted
    Positives weighted by inverse-class-frequency (rare = more anchor weight).
    Negatives weighted uniformly (no per-class assignment needed).
    """
    def __init__(self, num_classes, alpha=0.5, temperature=0.07, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.class_weights = class_weights
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    def forward(self, logits, targets, features=None):
        loss_ce = self.ce(logits, targets)
        if features is None:
            return loss_ce
        feats = F.normalize(features, dim=-1)
        sim = feats @ feats.t() / self.temperature
        B = feats.size(0)
        y_eq = targets.unsqueeze(0) == targets.unsqueeze(1)
        eye = torch.eye(B, dtype=torch.bool, device=feats.device)
        pos_mask = (y_eq & ~eye).float()
        neg_mask = (~y_eq).float()
        # class-frequency-based positive weight
        if self.class_weights is not None:
            cw = self.class_weights.to(feats.device)[targets]  # (B,)
            pos_weight = pos_mask * cw.unsqueeze(0)  # each anchor weighted by its class weight
        else:
            pos_weight = pos_mask
        exp_sim = torch.exp(sim) * (~eye).float()
        denom = (exp_sim * (pos_mask + neg_mask)).sum(dim=-1) + 1e-12
        log_prob = sim - torch.log(denom)
        pos_count = pos_mask.sum(dim=-1)
        per_anchor = -(log_prob * pos_weight).sum(dim=-1) / (pos_count + 1e-12)
        loss_supcon = per_anchor.mean()
        return (1 - self.alpha) * loss_ce + self.alpha * loss_supcon
