import pandas as pd
import pytest

from predictive_maintenance_ml.data_loader import AI4IDataLoader
from predictive_maintenance_ml.feature_engineering import FeaturePreprocessor


@pytest.fixture(scope="module")
def raw_features() -> pd.DataFrame:
    loader = AI4IDataLoader()
    X, _ = loader.load_data()
    return X


@pytest.fixture(scope="module")
def processed_features(raw_features: pd.DataFrame) -> pd.DataFrame:
    preprocessor = FeaturePreprocessor()
    return preprocessor.fit_transform(raw_features)


def test_fit_transform_returns_dataframe(processed_features: pd.DataFrame) -> None:
    assert isinstance(processed_features, pd.DataFrame)


def test_processed_features_have_expected_number_of_rows(
    processed_features: pd.DataFrame,
) -> None:
    assert processed_features.shape[0] == 10000


def test_type_column_is_removed(processed_features: pd.DataFrame) -> None:
    assert 'Type' not in processed_features.columns


def test_one_hot_type_columns_are_created(processed_features: pd.DataFrame) -> None:
    expected_columns = {"Type_H", "Type_L", "Type_M"}
    assert expected_columns.issubset(processed_features.columns)


def test_processed_features_have_no_object_columns(
    processed_features: pd.DataFrame,
) -> None:
    assert processed_features.select_dtypes(include=['object']).columns.size == 0


def test_transform_before_fit_raises_error(raw_features: pd.DataFrame) -> None:
    preprocessor = FeaturePreprocessor()
    with pytest.raises(ValueError) as excinfo:
        preprocessor.transform(raw_features)
    assert "The preprocessor has not been fitted yet." in str(excinfo.value)