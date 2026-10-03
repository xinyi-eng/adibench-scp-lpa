"""Append ReCLLoss class to dacd/losses.py.

ReCL (Cao et al. NeurIPS 2021): simple supervised contrastive with
class-reweighted positive / negative pairs, combined with CE.

    L = (1 - alpha) * L_CE + alpha * L_SupCon_reweighted

Key simplification vs DACDLoss (which we already have):
- No Khaleeji-specific de-biasing (every negative pair shares one weight)
- Class-aware positive weighting (rare classes contribute more to anchor terms)
- Symmetric / single-direction SupCon (not SupCon with self-contrast exclusions)
"""
import os
path = r'D:/dacd2026/2_models/dacd++/dacd/losses.py'

ReCL_BLOCK = '''

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
'''

# append if not already present
with open(path, 'r') as f:
    cur = f.read()
if 'class ReCLLoss' in cur:
    print('ReCLLoss already present, skipping append')
else:
    with open(path, 'a') as f:
        f.write(ReCL_BLOCK)
    print(f'appended ReCLLoss to {path}')
