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

def test_evaluate_thresholds_returns_dataframe() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.7, 0.9])
    thresholds = np.array([0.3, 0.5, 0.8])

    evaluator = ClassificationEvaluator()
    results = evaluator.evaluate_thresholds(
        y_true=y_true,
        y_proba=y_proba,
        thresholds=thresholds
    )

    assert isinstance(results, pd.DataFrame)

def test_default_threshold_dataframe_shape() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.7, 0.9])

    evaluator = ClassificationEvaluator()
    results = evaluator.evaluate_thresholds(
        y_true=y_true,
        y_proba=y_proba
    )

    assert results.shape[0] == 9

def test_threshold_dataframe_columns() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.7, 0.9])

    evaluator = ClassificationEvaluator()
    results = evaluator.evaluate_thresholds(
        y_true=y_true,
        y_proba=y_proba
    )
    expected_columns = {"threshold", "accuracy", "precision", "recall", "f1", "true_negatives", "false_positives", "false_negatives", "true_positives"}

    assert expected_columns.issubset(results.columns)

def test_threshold_metrics_are_valid_numbers() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.7, 0.9])

    evaluator = ClassificationEvaluator()
    results = evaluator.evaluate_thresholds(
        y_true=y_true,
        y_proba=y_proba
    )
    
    for metric_name in ["accuracy", "precision", "recall", "f1"]:
        assert results[metric_name].between(0.0, 1.0).all()

def test_tp_fn_tp_tn_are_valid_numbers() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.7, 0.9])

    evaluator = ClassificationEvaluator()
    results = evaluator.evaluate_thresholds(
        y_true=y_true,
        y_proba=y_proba
    )
    
    for metric_name in ["true_negatives", "true_positives", "false_negatives", "false_positives"]:
        assert (0.0 <= results[metric_name] ).all()

def test_custom_thresholds() -> None:
    y_true = pd.Series([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.7, 0.9])
    thresholds = np.array([0.3, 0.5, 0.8])

    evaluator = ClassificationEvaluator()
    results = evaluator.evaluate_thresholds(
        y_true=y_true,
        y_proba=y_proba,
        thresholds=thresholds
    )

    assert list(results["threshold"]) == [0.3, 0.5, 0.8]