import pandas as pd
import numpy as np
import pytest
from sklearn.utils.validation import check_is_fitted

from predictive_maintenance_ml.experiment import BaselineExperiment

@pytest.fixture(scope="module")
def experiment_and_results() -> tuple[
    BaselineExperiment,
    dict[str, pd.Series | np.ndarray],
]:
    experiment = BaselineExperiment()
    results = experiment.run()

    return experiment, results

@pytest.fixture(scope="module")
def experiment(
    experiment_and_results: tuple[
        BaselineExperiment,
        dict[str, pd.Series | np.ndarray],
    ],
) -> BaselineExperiment:
    experiment, _ = experiment_and_results

    return experiment


@pytest.fixture(scope="module")
def experiment_results(
    experiment_and_results: tuple[
        BaselineExperiment,
        dict[str, pd.Series | np.ndarray],
    ],
) -> dict[str, pd.Series | np.ndarray]:
    _, results = experiment_and_results

    return results

def test_run_returns_dict(experiment_results: dict[str, pd.Series | np.ndarray]) -> None:
    assert isinstance(experiment_results, dict)

def test_dict_content(experiment_results: dict[str, pd.Series | np.ndarray]) -> None:
    expected_items = {"y_test", "y_pred", "y_proba"}
    assert set(experiment_results.keys()) == expected_items 

def test_returned_arrays_have_same_length(experiment_results: dict[str, pd.Series | np.ndarray]) -> None:
    lengths = [item.shape[0] for item in experiment_results.values()]
    assert len(set(lengths)) == 1

def test_y_pred_values(experiment_results: dict[str, pd.Series | np.ndarray]) -> None:
    assert set(experiment_results["y_pred"]).issubset({0, 1})

def test_y_proba_values(experiment_results: dict[str, pd.Series | np.ndarray]) -> None:
    y_proba = experiment_results["y_proba"]

    assert np.all(y_proba >= 0)
    assert np.all(y_proba <= 1)

def test_classifier_is_fitted_after_run(experiment: BaselineExperiment) -> None:
    assert experiment.classifier is not None

    check_is_fitted(experiment.classifier.model)


def test_preprocessor_is_fitted_after_run(experiment: BaselineExperiment) -> None:
    assert experiment.preprocessor is not None
    assert experiment.preprocessor.encoder is not None

    check_is_fitted(experiment.preprocessor.encoder)

    if experiment.preprocessor.scale:
        assert experiment.preprocessor.scaler is not None
        check_is_fitted(experiment.preprocessor.scaler)