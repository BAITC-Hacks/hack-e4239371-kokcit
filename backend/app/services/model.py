import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.services.features import FEATURE_COLUMNS

HISTOGRAM_PROFILES = {
    "balanced": {"learning_rate": 0.07, "max_iter": 240, "l2_regularization": 0.1},
    "accurate": {"learning_rate": 0.05, "max_iter": 400, "l2_regularization": 0.2},
}
MODEL_PROFILES = (*HISTOGRAM_PROFILES, "extra_trees")


class ForecastEnsemble:
    """Weighted independent regressors, each with its own training window/features."""

    def __init__(self, members, weights):
        self.members = members
        self.weights = weights

    def predict(self, features):
        predictions = [model.predict(features[columns]) for model, columns in self.members]
        return np.average(predictions, axis=0, weights=self.weights)


def create_baseline_model(profile: str = "balanced") -> Any:
    if profile == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=120,
            min_samples_leaf=4,
            max_features=1.0,
            n_jobs=-1,
            random_state=42,
        )
    return HistGradientBoostingRegressor(
        max_leaf_nodes=31,
        random_state=42,
        **HISTOGRAM_PROFILES[profile],
    )


def evaluate(model: Any, frame: pd.DataFrame) -> dict[str, float]:
    prediction = np.clip(model.predict(frame[list(FEATURE_COLUMNS)]), 0, 1)
    actual = frame["power"].to_numpy()
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
        "samples": len(frame),
    }


def save_model(model: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination, compress=3)


@lru_cache(maxsize=16)
def _load_cached(path: str, modified: int) -> Any:
    model = joblib.load(path)
    # Keep inference portable in environments without multiprocessing permissions.
    estimator = model.get("estimator") if isinstance(model, dict) else model
    if hasattr(estimator, "n_jobs"):
        estimator.n_jobs = 1
    return model


def load_model(path: str | Path) -> Any:
    path = Path(path)
    return _load_cached(str(path.resolve()), path.stat().st_mtime_ns)


@lru_cache(maxsize=16)
def _model_digest(path: str, modified: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_digest(path: Path) -> str:
    return _model_digest(str(path.resolve()), path.stat().st_mtime_ns)
