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


def evaluate_models(horizons=(1, 3, 5, 7)) -> pd.DataFrame:
    """Fit each model on the train set and score it on the validation set.

    Loops over input length, feature set, target and model.
    Saves MAE, RMSE, bias and train MAE to results/models_val_results.csv.
    """
    project_root = Path(__file__).resolve().parents[3]
    processed = project_root / "data" / "processed"

    train_raw = pd.read_csv(processed / "train_12d.csv")
    val_raw = pd.read_csv(processed / "val_12d.csv")

    feature_sets = {"base": False, "base+extra": True}
    records = []

    for k in horizons:
        for feature_set_name, add_extra in feature_sets.items():
            X_tr, y_tr = get_xy(train_raw, k, add_extra)
            X_va, y_va = get_xy(val_raw, k, add_extra)

            for target in ("y_mean", "y_cv"):
                for nom_modele, modele in make_models(target).items():
                    modele.fit(X_tr, y_tr[target])

                    preds = modele.predict(X_va)
                    preds_train = modele.predict(X_tr)

                    records.append({
                        "k_days": k,
                        "features": feature_set_name,
                        "target": target,
                        "model": nom_modele,
                        "MAE": round(float(mean_absolute_error(y_va[target], preds)), 4),
                        "RMSE": round(float(root_mean_squared_error(y_va[target], preds)), 4),
                        "bias": round(float((preds - y_va[target]).mean()), 4),
                        "MAE_train": round(float(mean_absolute_error(y_tr[target], preds_train)), 4),
                    })

    results = pd.DataFrame(records)
    print(results.to_string(index=False))

    out_dir = project_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "models_val_results.csv", index=False)
    print(f"\nResults saved to {out_dir / 'models_val_results.csv'}")
    return results


if __name__ == "__main__":
    evaluate_models()
