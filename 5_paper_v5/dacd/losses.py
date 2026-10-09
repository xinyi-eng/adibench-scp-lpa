"""Loss functions and samplers for DACD++ v5.

This file contains FOUR new methodological contributions for the v5 paper:
  1. L2C - Linguistic-Distance-Weighted Contrastive
  2. AMI - Adaptive Margin for Imbalance
  3. ARC - Adaptive Ratio Curriculum
  4. CEDA - Cross-Encoder Distillation Augmentation
All v4 losses (CE, Focal, CB, DACD, MEC, Prototype, LDAM, LA, ReCL) are
preserved for fair baseline comparison.
"""
from __future__ import annotations
import math
import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Sampler


LINGUISTIC_DISTANCE = torch.tensor([
    [0.00, 0.45, 0.55, 0.65, 0.85],
    [0.45, 0.00, 0.40, 0.50, 0.75],
    [0.55, 0.40, 0.00, 0.20, 0.70],
    [0.65, 0.50, 0.20, 0.00, 0.65],
    [0.85, 0.75, 0.70, 0.65, 0.00],
], dtype=torch.float32)


def get_linguistic_distance(device=None):
    d = LINGUISTIC_DISTANCE.clone()
    if device is not None:
        d = d.to(device)
    return d


class CISSampler(Sampler):
    def __init__(self, labels, epoch, total_epochs,
                 start_ratio=100.0, peak_ratio=7000.0, seed=42):
        self.labels = np.asarray(labels)
        self.epoch = epoch
        self.total_epochs = total_epochs
        if total_epochs <= 1:
            f = 1.0
        else:
            f = (epoch / (total_epochs - 1)) ** 2
        self.target_ratio = start_ratio + (peak_ratio - start_ratio) * f
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
        protos = torch.stack(protos)
        proto_exists = torch.tensor(proto_exists, device=feats.device)
        sim = feats @ protos.t() / self.temperature
        pos_mask = F.one_hot(targets, self.num_classes).float()
        valid = proto_exists.float().unsqueeze(0)
        exp_sim = torch.exp(sim) * valid
        denom = exp_sim.sum(dim=-1, keepdim=True) + 1e-12
        log_prob = sim - torch.log(denom)
        loss = -(log_prob * pos_mask).sum(dim=-1)
        if self.class_weights is not None:
            cw = self.class_weights.to(loss.device)[targets]
            loss = loss * cw
        return self.alpha * loss.mean()


class LDAMLoss(nn.Module):
    def __init__(self, num_classes, max_m=0.5, s=30.0, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.max_m = max_m
        self.s = s
        self.class_weights = class_weights

    def forward(self, logits, targets):
        if self.class_weights is not None:
            cw = self.class_weights.to(logits.device)
            margin = (cw / cw.max()) * self.max_m
        else:
            margin = torch.full((self.num_classes,), self.max_m, device=logits.device)
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        margin_applied = torch.zeros_like(logits)
        margin_applied[batch_idx, targets] = margin[targets]
        adjusted_logits = logits - margin_applied
        scaled_logits = adjusted_logits * self.s
        return F.cross_entropy(scaled_logits, targets, weight=self.class_weights)


class LogitAdjustmentLoss(nn.Module):
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
        if self.class_weights is not None:
            cw = self.class_weights.to(feats.device)[targets]
            pos_weight = pos_mask * cw.unsqueeze(0)
        else:
            pos_weight = pos_mask
        exp_sim = torch.exp(sim) * (~eye).float()
        denom = (exp_sim * (pos_mask + neg_mask)).sum(dim=-1) + 1e-12
        log_prob = sim - torch.log(denom)
        pos_count = pos_mask.sum(dim=-1)
        per_anchor = -(log_prob * pos_weight).sum(dim=-1) / (pos_count + 1e-12)
        loss_supcon = per_anchor.mean()
        return (1 - self.alpha) * loss_ce + self.alpha * loss_supcon


# v5 INNOVATION #1: L2C
class L2CLoss(nn.Module):
    def __init__(self, num_classes, lam=0.5, alpha_ling=0.7,
                 base_beta=0.3, temperature=0.07, class_weights=None,
                 distance_matrix=None):
        super().__init__()
        self.lam = lam
        self.alpha_ling = alpha_ling
        self.base_beta = base_beta
        self.temperature = temperature
        self.class_weights = class_weights
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        if distance_matrix is None:
            distance_matrix = LINGUISTIC_DISTANCE
        d = distance_matrix.clone()
        mean_d = d[~torch.eye(num_classes, dtype=torch.bool)].mean()
        self.register_buffer('w_matrix', base_beta + alpha_ling * (d - mean_d))

    def forward(self, logits, targets, features=None):
        loss_ce = self.ce(logits, targets)
        if features is None:
            return loss_ce
        feats = F.normalize(features, dim=-1)
        B = feats.size(0)
        sim = feats @ feats.t() / self.temperature
        y_eq = targets.unsqueeze(0) == targets.unsqueeze(1)
        eye = torch.eye(B, dtype=torch.bool, device=feats.device)
        pos_mask = (y_eq & ~eye).float()
        neg_mask = (~y_eq).float()
        w_full = self.w_matrix[targets.unsqueeze(0), targets.unsqueeze(1)]
        w_full = torch.clamp(w_full, min=0.05, max=2.0)
        exp_sim = torch.exp(sim) * (~eye).float()
        denom = (exp_sim * (pos_mask + neg_mask * w_full)).sum(dim=-1) + 1e-12
        log_prob = sim - torch.log(denom)
        pos_count = pos_mask.sum(dim=-1)
        per_anchor = -(log_prob * pos_mask).sum(dim=-1) / (pos_count + 1e-12)
        if self.class_weights is not None:
            cw = self.class_weights.to(per_anchor.device)[targets]
            per_anchor = per_anchor * cw
        loss_supcon = per_anchor.mean()
        return (1 - self.lam) * loss_ce + self.lam * loss_supcon


# v5 INNOVATION #2: AMI
class AMILoss(nn.Module):
    def __init__(self, num_classes, max_m=0.5, s=30.0, alpha=1.0,
                 class_counts=None, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.max_m = max_m
        self.s = s
        self.alpha = alpha
        self.class_weights = class_weights
        if class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float32)
            counts_safe = torch.where(counts > 0, counts, torch.ones_like(counts))
            norm_factor = counts_safe.pow(-0.25)
            norm_factor = norm_factor / norm_factor.median()
        else:
            norm_factor = torch.ones(num_classes)
        self.register_buffer('inv_freq_factor', norm_factor)
        self.register_buffer('grad_norms', torch.zeros(num_classes, 2))
        self.grad_norms[:, 0] = 1.0
        self.grad_norms[:, 1] = 0.0

    def set_grad_norms(self, grad_norms_per_class):
        if grad_norms_per_class is None:
            return
        device = self.grad_norms.device
        gn = grad_norms_per_class.detach().to(device)
        self.grad_norms[:, 0] = 0.9 * self.grad_norms[:, 0] + 0.1 * gn
        self.grad_norms[:, 1] = 0.9 * self.grad_norms[:, 1] + 0.1 * (gn - self.grad_norms[:, 0]).pow(2)

    def _compute_margin(self):
        g_mean = self.grad_norms[:, 0]
        g_max = g_mean.max().clamp(min=1e-6)
        adaptive = 1.0 + self.alpha * (g_mean / g_max)
        return self.max_m * self.inv_freq_factor * adaptive

    def forward(self, logits, targets):
        margin = self._compute_margin().to(logits.device)
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        margin_applied = torch.zeros_like(logits)
        margin_applied[batch_idx, targets] = margin[targets]
        adjusted_logits = logits - margin_applied
        scaled_logits = adjusted_logits * self.s
        return F.cross_entropy(scaled_logits, targets, weight=self.class_weights)


# v5 INNOVATION #3: ARC
class ARCLoss(nn.Module):
    def __init__(self, base_contrastive_loss, num_epochs,
                 tau_start=0.3, tau_target=0.07,
                 beta_start=0.7, beta_target=0.3,
                 lam_start=0.7, lam_target=0.5):
        super().__init__()
        self.base = base_contrastive_loss
        self.num_epochs = num_epochs
        self.tau_start = tau_start
        self.tau_target = tau_target
        self.beta_start = beta_start
        self.beta_target = beta_target
        self.lam_start = lam_start
        self.lam_target = lam_target
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch
        if self.num_epochs <= 1:
            f = 1.0
        else:
            f = (epoch / (self.num_epochs - 1)) ** 2
        decay = 1.0 - f
        if hasattr(self.base, 'temperature'):
            self.base.temperature = self.tau_target + (self.tau_start - self.tau_target) * decay
        if hasattr(self.base, 'beta'):
            self.base.beta = self.beta_target + (self.beta_start - self.beta_target) * decay
        if hasattr(self.base, 'lam'):
            self.base.lam = self.lam_target + (self.lam_start - self.lam_target) * decay

    def forward(self, logits, targets, features=None):
        return self.base(logits, targets, features=features)


# v5 INNOVATION #4: CEDA
class CEDALoss(nn.Module):
    def __init__(self, num_classes, base_loss, T=2.0, alpha_kd=0.4,
                 alpha_align=0.2, class_weights=None):
        super().__init__()
        self.base = base_loss
        self.T = T
        self.alpha_kd = alpha_kd
        self.alpha_align = alpha_align
        self.num_classes = num_classes
        self.class_weights = class_weights

    def forward(self, student_logits, student_features, targets,
                teacher_logits, teacher_features):
        loss_base = self.base(student_logits, targets, features=student_features)
        s_log = F.log_softmax(student_logits / self.T, dim=-1)
        t_prob = F.softmax(teacher_logits / self.T, dim=-1).detach()
        kl = F.kl_div(s_log, t_prob, reduction='none').sum(dim=-1)
        if self.class_weights is not None:
            cw = self.class_weights.to(kl.device)[targets]
            kl = kl * cw
        loss_kd = (self.T ** 2) * kl.mean()
        if self.class_weights is not None and student_features is not None and teacher_features is not None:
            minority_mask = self.class_weights.to(targets.device)[targets] > 1.0
        else:
            minority_mask = torch.zeros(targets.size(0), dtype=torch.bool, device=targets.device)
        if minority_mask.any() and student_features is not None and teacher_features is not None:
            sf = F.normalize(student_features[minority_mask], dim=-1)
            tf = F.normalize(teacher_features[minority_mask], dim=-1)
            align = ((sf - tf) ** 2).sum(dim=-1).mean()
        else:
            align = torch.tensor(0.0, device=student_logits.device)
        return loss_base + self.alpha_kd * loss_kd + self.alpha_align * align


# Combined v5 method
class DACDv5Loss(nn.Module):
    def __init__(self, num_classes, lam=0.5, alpha_ling=0.7, base_beta=0.3,
                 temperature=0.07, max_m=0.5, s=30.0, ami_alpha=1.0,
                 class_counts=None, class_weights=None):
        super().__init__()
        self.l2c = L2CLoss(num_classes=num_classes, lam=lam,
                           alpha_ling=alpha_ling, base_beta=base_beta,
                           temperature=temperature, class_weights=class_weights)
        self.ami = AMILoss(num_classes=num_classes, max_m=max_m, s=s,
                           alpha=ami_alpha, class_counts=class_counts,
                           class_weights=class_weights)
        self.class_weights = class_weights
        self.num_classes = num_classes

    def forward(self, logits, targets, features=None):
        loss_contrastive = self.l2c(logits, targets, features=features)
        loss_ami = self.ami(logits, targets)
        return 0.5 * loss_contrastive + 0.5 * loss_ami

    def set_epoch(self, epoch):
        if hasattr(self.l2c, 'temperature'):
            self.l2c.temperature = max(0.05, 0.3 - (0.3 - 0.07) * (epoch / max(1, 4)))
        if hasattr(self.l2c, 'beta'):
            self.l2c.beta = max(0.05, 0.7 - (0.7 - 0.3) * (epoch / max(1, 4)))

    def set_grad_norms(self, grad_norms_per_class):
        self.ami.set_grad_norms(grad_norms_per_class)
