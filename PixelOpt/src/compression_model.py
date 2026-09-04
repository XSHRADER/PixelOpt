from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import joblib
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


class AdaptiveCompressionModel:
    """Multi-output regression model that predicts resize factor and JPEG quality."""

    def __init__(self, model_path: str = "models/adaptive_compression_model.joblib"):
        self.model_path = Path(model_path)
        self.model = None
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def train(self, features: Iterable[np.ndarray], targets: Iterable[np.ndarray]) -> None:
        X = np.asarray(list(features), dtype=np.float32)
        y = np.asarray(list(targets), dtype=np.float32)

        if X.shape[0] == 0 or y.shape[0] == 0:
            raise ValueError("Training data is empty.")

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        base_model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=8,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            task_type="GPU",
        )
        model = MultiOutputRegressor(base_model)
        model.fit(X_train, y_train)

        predictions = model.predict(X_valid)
        mae = mean_absolute_error(y_valid, predictions)
        r2 = r2_score(y_valid, predictions, multioutput="uniform_average")

        self.model = model
        joblib.dump(model, self.model_path)

        return {"mae": float(mae), "r2": float(r2)}

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        self.model = joblib.load(self.model_path)
        return True

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained or loaded.")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        predictions = self.model.predict(features)
        return np.asarray(predictions, dtype=np.float32)

    def save(self):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        joblib.dump(self.model, self.model_path)

    def predict_single(self, feature_vector: np.ndarray) -> Tuple[float, float]:
        prediction = self.predict(feature_vector)[0]
        resize_factor = float(np.clip(prediction[0], 0.2, 1.0))
        quality = float(np.clip(prediction[1], 12, 95))
        return resize_factor, quality
