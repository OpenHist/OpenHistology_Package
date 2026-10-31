"""
Loss utilities for ResGAT training.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight


def build_class_weighted_nll(train_labels, n_classes, device):
    """Build an NLLLoss with class-balanced weights from training labels.

    Args:
        train_labels: Array-like of integer training labels.
        n_classes: Total number of classes.
        device: Target torch device.

    Returns:
        nn.NLLLoss with per-class weights.
    """
    train_labels = np.asarray(train_labels)
    classes = np.arange(n_classes)
    present = np.intersect1d(classes, np.unique(train_labels))
    cw = compute_class_weight("balanced", classes=present, y=train_labels)
    weights = torch.ones(n_classes, dtype=torch.float32)
    for c, w in zip(present, cw):
        weights[c] = w
    return nn.NLLLoss(weight=weights.to(device))
