import pandas as pd
from ucimlrepo import fetch_ucirepo

class AI4IDataLoader:
    """Load the AI4I 2020 Predictive Maintenance dataset."""

    DEFAULT_TARGET_COLUMN = 'Machine failure'

    def __init__(
        self, 
        dataset_id: int = 601,
        target_column: str = DEFAULT_TARGET_COLUMN
    ) -> None:
        self.dataset_id = dataset_id # ai4i_2020_predictive_maintenance_dataset
        self.target_column = target_column

    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Fetch the dataset and return the features and selected target."""

        dataset = fetch_ucirepo(id=self.dataset_id)

        X: pd.DataFrame = dataset.data.features.copy() 
        targets: pd.DataFrame = dataset.data.targets

        if self.target_column not in targets.columns:
            available_columns = ", ".join(targets.columns)
            raise ValueError(
                f"Target column '{self.target_column}' not found."
                f"Available target columns: {available_columns}"
            )
        
        y:pd.Series = targets[self.target_column].copy()

        return X,y