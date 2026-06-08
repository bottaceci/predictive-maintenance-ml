import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class FeaturePreprocessor:
    """Preprocess AI4I features for machine learning models."""

    DEFAULT_CATEGORICAL_COLUMNS = ["Type"]

    def __init__(
            self, 
            categorical_columns: list[str] | None = None, 
            scale: bool = False,
            scaling_columns: list[str] | None = None
    ) -> None:
        self.categorical_columns = (
            categorical_columns
            if categorical_columns is not None
            else self.DEFAULT_CATEGORICAL_COLUMNS
        )
        self.scale = scale
        self.scaling_columns = scaling_columns

        self.encoder: OneHotEncoder | None = None
        self.scaler: StandardScaler | None = None

    def _fit(self, X: pd.DataFrame) -> None:
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoder.fit(X[self.categorical_columns])

        if self.scale:
            if self.scaling_columns is None:
                self.scaling_columns = [
                    column
                    for column in X.columns
                    if column not in self.categorical_columns
                ]
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.scaling_columns])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform features using the fitted preprocessing steps."""

        if self.encoder is None:
            raise ValueError("The preprocessor has not been fitted yet.")

        if self.scale and self.scaler is None:
            raise ValueError("Scaling is enabled, but the scaler has not been fitted yet.")
        
        encoded_data = self.encoder.transform(X[self.categorical_columns])
        encoded_df = pd.DataFrame(
            encoded_data,
            columns = self.encoder.get_feature_names_out(self.categorical_columns),
            index=X.index,
        )
        X_processed = pd.concat(
            [X.drop(columns=self.categorical_columns), encoded_df],
            axis=1
        )

        if self.scale:
            scaled_data = self.scaler.transform(X[self.scaling_columns])
            scaled_df = pd.DataFrame(
                scaled_data,
                columns = self.scaling_columns,
                index=X.index,
            )
            X_processed = pd.concat(
                [X_processed.drop(columns=self.scaling_columns), scaled_df],
                axis=1
            )
        
        return X_processed
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit the preprocessing steps and transform the input features."""
        
        self._fit(X)
        return self.transform(X)

    

    