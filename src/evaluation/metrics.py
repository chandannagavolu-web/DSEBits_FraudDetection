"""Shared evaluation metrics for the dissertation.

Every experiment (baseline, GNN, temporal, autoencoder) should score its
predictions through :func:`compute_all_metrics` so that results are
comparable in a single results table (see ``src/utils/io.py::log_results``).

Includes:
- Standard metrics: accuracy, precision, recall, F1, ROC-AUC.
- Fraud-detection metrics: PR-AUC (average precision), MCC.
- Credit-scoring metrics: KS statistic, Gini coefficient.
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between the cumulative
    true-positive-rate and false-positive-rate curves.

    Industry rule of thumb: KS > 0.40 is considered a strong credit model.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2 * AUC - 1, a 0-1 rescaling of ROC-AUC familiar to risk teams."""
    auc = roc_auc_score(y_true, y_score)
    return float(2 * auc - 1)


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute the full shared metric set for a binary classification task.

    Args:
        y_true: ground-truth labels (0/1), shape (n,).
        y_score: predicted probability / score for the positive class,
            shape (n,).
        threshold: decision threshold used to derive ``y_pred`` from
            ``y_score`` if ``y_pred`` is not supplied.
        y_pred: optional precomputed hard predictions (0/1). If omitted,
            derived from ``y_score >= threshold``.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, roc_auc, pr_auc,
        mcc, ks_statistic, gini.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_pred is None:
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "ks_statistic": ks_statistic(y_true, y_score),
        "gini": gini_coefficient(y_true, y_score),
    }
