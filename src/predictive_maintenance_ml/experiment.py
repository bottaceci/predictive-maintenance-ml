from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from predictive_maintenance_ml.data_loader import AI4IDataLoader
from predictive_maintenance_ml.feature_engineering import FeaturePreprocessor
from predictive_maintenance_ml.model import BaselineClassifier
from predictive_maintenance_ml.evaluation import ClassificationEvaluator


class BaselineExperiment:
    """Run the baseline predictive maintenance classification experiment."""

    def __init__(
        self,
        dataset_id: int = 601,
        target_column: str = "Machine failure",
        test_size: float = 0.2,
        random_state: int = 42,
        scale_features: bool = True,
        model_max_iter: int = 1000,
        model_class_weight: str | dict[int, float] | None = "balanced",
    ) -> None:
        self.dataset_id = dataset_id
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        self.scale_features = scale_features
        self.model_max_iter = model_max_iter
        self.model_class_weight = model_class_weight

        self.loader: AI4IDataLoader | None = None
        self.preprocessor: FeaturePreprocessor | None = None
        self.classifier: BaselineClassifier | None = None


    def run(self) -> dict[str, pd.Series | np.ndarray]:
        """Run the full baseline experiment and return test predictions."""
        
        # Load the dataset
        self.loader = AI4IDataLoader(
            dataset_id=self.dataset_id, 
            target_column=self.target_column,
        )
        X, y = self.loader.load_data()

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Preprocess the data
        self.preprocessor = FeaturePreprocessor(
            scale=self.scale_features,
        )
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Fit baseline classifier
        self.classifier = BaselineClassifier(
            max_iter=self.model_max_iter,
            random_state=self.random_state,
            class_weight=self.model_class_weight,
        )
        self.classifier.fit(X_train_processed, y_train)

        # Predict on X_test
        y_pred = self.classifier.predict(X_test_processed)
        y_proba = self.classifier.predict_proba(X_test_processed)[:, 1]

        # Return the results
        return {
            "y_test": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    
    
def main() -> None:
    experiment = BaselineExperiment()
    results = experiment.run()

    evaluator = ClassificationEvaluator()
    metrics = evaluator.evaluate(
        y_true=results["y_test"],
        y_pred=results["y_pred"],
        y_proba=results["y_proba"],
    )

    print("Accuracy:", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall:", metrics["recall"])
    print("F1:", metrics["f1"])
    print("ROC-AUC:", metrics["roc_auc"])
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print("Classification report:")
    print(metrics["classification_report"])


if __name__ == "__main__":
    main()