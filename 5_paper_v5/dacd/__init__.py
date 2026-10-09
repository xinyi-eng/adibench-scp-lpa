"""DACD++ v5: package init."""
from .losses import (
    CISSampler, class_balanced_weights,
    CrossEntropyLoss, FocalLoss, DACDLoss, MECLoss, PrototypeLoss,
    LDAMLoss, LogitAdjustmentLoss, ReCLLoss,
    L2CLoss, AMILoss, ARCLoss, CEDALoss, DACDv5Loss,
    LINGUISTIC_DISTANCE, get_linguistic_distance,
)
from .models import SingleEncoderClassifier, MECClassifier, CEDAClassifier
from .data import (
    DIALECT_NAMES, DIALECT_5WAY, QADI_18_TO_5,
    load_nadi, load_amgadhasan, normalize_label, subsample_imbalance,
    build_dacd_bench, TextDataset,
)
