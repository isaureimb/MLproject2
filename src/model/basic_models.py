import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.features import build_xy


def make_models(target: str = "y_mean") -> dict:
    """Build three fresh models with fixed, deliberately modest settings.

    Settings are chosen in advance and never tuned on validation results,
    to limit overfitting.

    Each model is a pipeline:
    - SimpleImputer: fills a missing value with the training median of that column.
    - StandardScaler (Ridge only): puts all features on the same scale.
    """
    alpha_ridge = 10.0 if target == "y_mean" else 5.0
    return {
        "Ridge": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=alpha_ridge),
        ),
        "RandomForest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "GradientBoosting": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=3,
                random_state=42,
            ),
        ),
    }


def get_xy(raw: pd.DataFrame, k: int, add_extra: bool):
    """Build X (features from the k input days) and y (targets from days 8-12).

    If add_extra is True, the three features from features_extra.py
    (night mean, day mean, longest gap) are added to those from windowings.py.
    """
    X, y = build_xy(raw, k_days=k)
    if not add_extra:
        X = X.drop(columns=["cgm_night_mean", "cgm_day_mean"])
    return X, y
