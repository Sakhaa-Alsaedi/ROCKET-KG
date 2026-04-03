"""
Evaluation metrics for all four clinical prediction tasks.

Task mapping:
  - mortality      : binary classification  → compute_metrics
  - readmission    : binary classification  → compute_metrics
  - drugrec        : multi-label classification → compute_multilabel_metrics
  - lenofstay      : multi-class (10 bins)  → compute_multiclass_metrics

All functions accept numpy arrays and return a flat dict of floats.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Binary classification (mortality, readmission)
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        y_true  : [N] integer labels (0 or 1).
        y_score : [N] predicted probabilities for class 1.
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        dict with keys: roc_auc, pr_auc, f1, accuracy, precision, recall.
    """
    y_pred = (y_score >= threshold).astype(int)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_score)
    except ValueError:
        pr_auc = float("nan")

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Multi-label classification (drug recommendation)
# ---------------------------------------------------------------------------

def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute multi-label classification metrics.

    Args:
        y_true  : [N, L] binary ground truth labels.
        y_score : [N, L] predicted scores (sigmoid outputs).
        threshold: Binarisation threshold.

    Returns:
        dict with keys: roc_auc, pr_auc, f1, jaccard, precision, recall.
    """
    y_pred = (y_score >= threshold).astype(int)

    try:
        roc_auc = roc_auc_score(y_true, y_score, average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    try:
        pr_auc = average_precision_score(y_true, y_score, average="macro")
    except ValueError:
        pr_auc = float("nan")

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "jaccard": float(jaccard_score(y_true, y_pred, average="samples", zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average="samples", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="samples", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Multi-class classification (length of stay, 10 bins)
# ---------------------------------------------------------------------------

def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """Compute multi-class classification metrics.

    Args:
        y_true  : [N] integer class labels.
        y_score : [N, C] predicted class probabilities (softmax outputs).

    Returns:
        dict with keys: roc_auc, pr_auc, f1, accuracy, kappa.
    """
    y_pred = np.argmax(y_score, axis=1)
    n_classes = y_score.shape[1]

    try:
        roc_auc = roc_auc_score(
            y_true, y_score, average="macro", multi_class="ovr",
            labels=np.arange(n_classes),
        )
    except ValueError:
        roc_auc = float("nan")

    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=np.arange(n_classes))
        pr_auc = average_precision_score(y_bin, y_score, average="macro")
    except ValueError:
        pr_auc = float("nan")

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


# ---------------------------------------------------------------------------
# Task-aware dispatcher
# ---------------------------------------------------------------------------

def compute_task_metrics(
    task: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Dispatch to the correct metric function based on task name.

    Args:
        task    : One of "mortality", "readmission", "drugrec", "lenofstay".
        y_true  : Ground truth labels.
        y_score : Predicted scores / probabilities.
        threshold: Decision threshold (binary / multi-label only).

    Returns:
        Metric dict.
    """
    if task in ("mortality", "readmission"):
        return compute_metrics(y_true, y_score, threshold)
    elif task == "drugrec":
        return compute_multilabel_metrics(y_true, y_score, threshold)
    elif task == "lenofstay":
        return compute_multiclass_metrics(y_true, y_score)
    else:
        raise ValueError(f"Unknown task: {task!r}. Choose from mortality/readmission/drugrec/lenofstay.")
