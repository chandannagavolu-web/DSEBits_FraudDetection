import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.metrics import compute_all_metrics, gini_coefficient


def test_compute_all_metrics_perfect_separation():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    metrics = compute_all_metrics(y_true, y_score)

    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["mcc"] == 1.0
    assert metrics["ks_statistic"] == 1.0
    assert metrics["gini"] == 1.0


def test_compute_all_metrics_bounds_on_random_data():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=1000)
    y_score = rng.random(1000)

    metrics = compute_all_metrics(y_true, y_score)

    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert -1.0 <= metrics["mcc"] <= 1.0
    assert 0.0 <= metrics["ks_statistic"] <= 1.0
    assert -1.0 <= metrics["gini"] <= 1.0


def test_gini_matches_auc_relation():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.2, 0.8, 0.4, 0.6])

    auc = roc_auc_score(y_true, y_score)

    assert np.isclose(gini_coefficient(y_true, y_score), 2 * auc - 1)
