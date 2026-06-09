import numpy as np
import pandas as pd

from predictive_maintenance_ml.evaluation import ClassificationEvaluator


def test_evaluator_returns_expected_metric_keys() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.9])

    evaluator = ClassificationEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_proba)

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "confusion_matrix",
        "classification_report",
    }

    assert set(metrics.keys()) == expected_keys


def test_evaluator_confusion_matrix_shape() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.9])

    evaluator = ClassificationEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_proba)

    assert metrics["confusion_matrix"].shape == (2, 2)


def test_evaluator_metrics_are_valid_numbers() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.9])

    evaluator = ClassificationEvaluator()
    metrics = evaluator.evaluate(y_true, y_pred, y_proba)

    for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        assert 0.0 <= metrics[metric_name] <= 1.0