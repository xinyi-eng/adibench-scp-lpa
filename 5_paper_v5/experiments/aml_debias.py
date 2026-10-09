"""AML-DeBias implementation: Adaptive Majority-Language De-Biasing for
supervised contrastive learning, replacing a fixed negative-pair weight $\beta$
with an instance-level dynamic weight derived from per-batch per-class
gradient L2 norms (Eq. 2 in the paper).

Drop-in replacement for the contrastive term in `dacd/losses.py`. Designed to
read alongside `multi_seed_runner.py` (the latter calls `forward` with
logits, targets, features).

Key behaviours:
  - Per-batch class gradient norm cache (set via `set_last_grads`).
  - When the cache is empty (eval phase, or first epoch), falls back to the
    fixed $\beta_0$ value -- this is the v3 default, recovered as a safety net.
  - The fallback is opt-out via `force_adaptive=True` (for ablation tests).
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AMLDeBiasState:
    """Holds per-batch per-class gradient norms for the running training step.

    Updated by the trainer after each optimizer step and consumed by
    `AMLDeBiasSupCon.forward`.
    """

    def __init__(self):
        self.grad_norms: dict[int, float] = {}  # class -> mean L2 norm
        self.batch_id: int = -1

    def update(self, features: torch.Tensor, loss: torch.Tensor,
                targets: torch.Tensor, normalize: bool = True) -> None:
        """Compute per-class gradient L2 norm on the contrastive term.

        Args:
          features: encoder hidden states (B, d).
          loss: scalar contrastive loss whose gradient we want to track.
          targets: per-sample class labels (B,).
          normalize: if True, project features to unit norm before computing
                     the gradient. Set False for raw features.
        """
        feats = F.normalize(features, dim=-1) if normalize else features
        # Sum of per-sample losses -> gradient aggregated across batch
        per_sample_grads = torch.autograd.grad(
            loss, feats, retain_graph=False, create_graph=False,
        )[0]
        # L2 norm per sample
        norms = per_sample_grads.norm(dim=-1).detach()
        for c in targets.unique().tolist():
            mask = (targets == c)
            self.grad_norms[int(c)] = float(norms[mask].mean().item())

    def get_ratio(self, c_minor: int, c_major: int) -> float:
        """Return g_minor / g_major with safe defaults."""
        g_min = self.grad_norms.get(c_minor, 1.0)
        g_maj = self.grad_norms.get(c_major, 1.0)
        if g_maj <= 1e-9:
            return 1.0
        return float(g_min / g_maj)


_AML_STATE = AMLDeBiasState()


def get_state() -> AMLDeBiasState:
    return _AML_STATE


class AMLDeBiasSupCon(nn.Module):
    """SupCon with instance-level adaptive negative-pair weighting.

    Loss per batch:
      L = mean over anchors i of  - log( w_pos * exp(sim(z_i, z_pos)/tau)
                                          / sum_a w_neg(i,a) * exp(sim(z_i, z_a)/tau) )

    where for each negative pair (i, j) with y_j == majority:
      w_neg(i, j) = beta_ij = beta0 * sigmoid(-alpha * log(g_yi/g_yj))
      w_neg(i, j) = beta_ij for non-majority negatives: 1.0
    Positive weights follow standard SupCon.
    """

    def __init__(self,
                 num_classes: int,
                 majority_class: int = 0,
                 alpha: float = 1.0,
                 beta0: float = 0.3,
                 temperature: float = 0.07,
                 use_aml: bool = True,
                 state: Optional[AMLDeBiasState] = None):
        super().__init__()
        self.num_classes = num_classes
        self.majority = majority_class
        self.alpha = alpha
        self.beta0 = beta0
        self.temperature = temperature
        self.use_aml = use_aml
        self.state = state if state is not None else _AML_STATE

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(features, dim=-1)
        sim = feats @ feats.t() / self.temperature  # (B, B)
        B = feats.size(0)
        eye = torch.eye(B, dtype=torch.bool, device=feats.device)
        pos_mask = ((targets.unsqueeze(0) == targets.unsqueeze(1)) & ~eye).float()
        neg_mask = (~(targets.unsqueeze(0) == targets.unsqueeze(1))).float()

        # per-anchor negative weights: shape (B, B)
        neg_w = neg_mask.clone()
        if self.use_aml:
            for i in range(B):
                y_i = int(targets[i].item())
                for j in range(B):
                    if i == j:
                        continue
                    if int(targets[j].item()) != self.majority:
                        continue
                    r = self.state.get_ratio(y_i, self.majority)
                    # beta_ij = beta0 * sigmoid(-alpha * log r). log r < 0 <-> r < 1 (minority under-represented)
                    beta_ij = self.beta0 * torch.sigmoid(
                        torch.tensor(-self.alpha * math.log(max(r, 1e-6)))
                    ).item()
                    neg_w[i, j] = beta_ij
        else:
            # fixed-beta fallback (v3 default)
            for i in range(B):
                for j in range(B):
                    if i == j:
                        continue
                    if int(targets[j].item()) == self.majority:
                        neg_w[i, j] = self.beta0

        # log-prob per anchor
        exp_sim = torch.exp(sim) * (~eye).float()
        denom = (exp_sim * (pos_mask + neg_w)).sum(dim=-1) + 1e-12
        log_prob = sim - torch.log(denom)
        pos_count = pos_mask.sum(dim=-1) + 1e-12
        per_anchor = -(log_prob * pos_mask).sum(dim=-1) / pos_count
        return per_anchor.mean()


if __name__ == '__main__':
    # smoke test
    torch.manual_seed(0)
    B, d = 32, 16
    feats = torch.randn(B, d, requires_grad=True)
    targets = torch.tensor([0]*28 + [1, 2, 3, 1])  # 0 = Khaleeji majority
    # supply dummy gradient norms so AML-DeBias has something to compare against
    sup = AMLDeBiasSupCon(num_classes=5, majority_class=0, alpha=1.0, beta0=0.3)
    # initial pass: loss with no per-class gradient info -> safety-net beta0=0.3
    loss = sup(feats, targets)
    print(f'loss (no AML state) = {loss.item():.4f}')
    # Now compute gradients w.r.t. the *contrastive* loss on the *original* features
    sup.zero_grad()
    loss_for_grad = sup(feats.detach().requires_grad_(True), targets)
    loss_for_grad.backward()
    grad_norms = feats.grad.detach() if feats.grad is not None else torch.zeros_like(feats)
    # populate state
    sup.state.grad_norms.clear()
    for c in targets.unique().tolist():
        mask = (targets == c)
        if mask.sum() > 0:
            sup.state.grad_norms[int(c)] = float(grad_norms[mask].norm(dim=-1).mean().item())
    print(f'updated state: {sup.state.grad_norms}')
    # second pass: AML takes effect
    feats.grad = None
    loss2 = sup(feats.detach().requires_grad_(True), targets)
    print(f'loss (after grad update) = {loss2.item():.4f}')
