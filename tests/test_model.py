import numpy as np
import pandas as pd

from predictive_maintenance_ml.model import BaselineClassifier


def test_baseline_classifier_can_fit_and_predict() -> None:
    X = pd.DataFrame(
        {
            "feature_1": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
            "feature_2": [1.0, 1.1, 1.2, 0.0, 0.1, 0.2],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])

    model = BaselineClassifier()
    model.fit(X, y)

    predictions = model.predict(X)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (6,)


def test_baseline_classifier_predict_proba_returns_two_columns() -> None:
    X = pd.DataFrame(
        {
            "feature_1": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
            "feature_2": [1.0, 1.1, 1.2, 0.0, 0.1, 0.2],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])

    model = BaselineClassifier()
    model.fit(X, y)

    probabilities = model.predict_proba(X)

    assert isinstance(probabilities, np.ndarray)
    assert probabilities.shape == (6, 2)


def test_baseline_classifier_fit_returns_self() -> None:
    X = pd.DataFrame(
        {
            "feature_1": [0.0, 0.1, 0.2, 1.0, 1.1, 1.2],
            "feature_2": [1.0, 1.1, 1.2, 0.0, 0.1, 0.2],
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])

    model = BaselineClassifier()
    fitted_model = model.fit(X, y)

    assert fitted_model is model