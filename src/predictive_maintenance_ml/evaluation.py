from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ClassificationEvaluator:
    """Evaluate binary classification predictions."""

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
    ) -> dict[str, float | np.ndarray | str]:
        """Compute classification metrics."""

        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "confusion_matrix": confusion_matrix(y_true, y_pred),
            "classification_report": classification_report(
                y_true,
                y_pred,
                zero_division=0,
            ),
        }
    
    def evaluate_thresholds(
        self,
        y_true: pd.Series | np.ndarray,
        y_proba: np.ndarray,
        thresholds: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Evaluate classification metrics across decision thresholds."""

        if thresholds is None:
            thresholds = np.arange(0.1, 1.0, 0.1)

        rows = []

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()

            row = {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp,
            }

            rows.append(row)

        return pd.DataFrame(rows)