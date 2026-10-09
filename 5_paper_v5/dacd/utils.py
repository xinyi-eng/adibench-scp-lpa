"""Shared utilities for DACD++."""
import os, random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set all relevant seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
