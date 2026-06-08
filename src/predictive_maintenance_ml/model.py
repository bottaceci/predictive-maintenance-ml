from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# The dataset is highly imbalanced, so balanced class weights help
# prevent the model from ignoring the minority failure class, assigning
# each class a weight inversely proportional to the number of samples

class BaselineClassifier:
    """Baseline binary classifier for predictive maintenance."""

    def __init__(
            self, 
            max_iter: int = 1000, 
            class_weight: str | dict[int, float] | None = "balanced",
            random_state: int = 42
    ) -> None:
        self.model = LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight, 
            random_state=random_state,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaselineClassifier":
        """Fit the classifier on preprocessed features."""
        
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""

        return self.model.predict_proba(X)