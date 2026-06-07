import pandas as pd
import pytest

from predictive_maintenance_ml.data_loader import AI4IDataLoader

# The fixture loads the dataset once for this test file, then reuses it in all tests
@pytest.fixture(scope="module")
def loaded_data() -> tuple[pd.DataFrame, pd.Series]:
    loader = AI4IDataLoader()
    return loader.load_data()

# 1. X has 10000 rows
def test_X_has_expected_number_of_rows(
    loaded_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = loaded_data

    assert X.shape[0] == 10000

# 2. y has 10000 rows
def test_y_has_expected_number_of_rows(
    loaded_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    _, y = loaded_data

    assert y.shape[0] == 10000

# 3. y contains only 0 and 1
def test_y_contains_only_binary_values(
    loaded_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    _, y = loaded_data

    assert set(y.unique()) == {0, 1}

# 4. X contains the "Type" column
def test_X_contains_type_column(
    loaded_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, _ = loaded_data

    assert "Type" in X.columns

# 5. y is named "Machine failure"
def test_y_has_expected_name(
    loaded_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    _, y = loaded_data

    assert y.name == "Machine failure"