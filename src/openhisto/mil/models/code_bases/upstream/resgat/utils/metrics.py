"""
Evaluation helpers: prediction collection and metric computation.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


def collect_predictions(outputs, labels):
    """Convert log-softmax outputs and labels to numpy arrays.

    Returns:
        preds:       (N,) int   -- argmax class indices
        true_labels: (N,) int
        confidences: (N,) float -- max predicted probability
        probs:       (N, C) float -- class probabilities
    """
    preds = outputs.argmax(dim=1).cpu().numpy()
    true_labels = labels.cpu().numpy()
    probs = outputs.exp().cpu().numpy()
    confidences = probs.max(axis=1)
    return preds, true_labels, confidences, probs


def evaluate_metrics(preds, labels, probs):
    """Compute accuracy (%), ROC-AUC, and weighted F1.

    For binary tasks, AUC is computed using the class-1 probability.
    """
    acc = float((preds == labels).mean() * 100)

    if probs.ndim == 1:
        y_score = probs
    else:
        y_score = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    try:
        auc_roc = float(roc_auc_score(labels, y_score))
    except Exception:
        auc_roc = float("nan")

    f1 = float(f1_score(labels, preds, average="weighted", zero_division=0))
    return acc, auc_roc, f1
