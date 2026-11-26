import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from MLproject2.src.trainer.evaluate import get_xy


def tune_ridge_alphas(
    horizons=(1, 3, 5, 7),
    alphas=(0.01, 0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0),
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Grid-search Ridge regularization parameter alpha using 5-Fold CV on (Train + Val)
    while keeping test_12d.csv completely untouched.
    """
    project_root = Path(__file__).resolve().parents[3]
    processed = project_root / "data" / "processed"

    train_raw = pd.read_csv(processed / "train_12d.csv")
    val_raw = pd.read_csv(processed / "val_12d.csv")
    dev_raw = pd.concat([train_raw, val_raw], ignore_index=True)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }

    feature_sets = {"base": False, "base+extra": True}
    records = []

    for k in horizons:
        for feat_name, add_extra in feature_sets.items():
            X_dev, y_dev = get_xy(dev_raw, k, add_extra)

            for target in ("y_mean", "y_cv"):
                y_vec = y_dev[target]

                for alpha in alphas:
                    pipe = make_pipeline(
                        SimpleImputer(strategy="median"),
                        StandardScaler(),
                        Ridge(alpha=alpha),
                    )
                    cv_out = cross_validate(
                        pipe,
                        X_dev,
                        y_vec,
                        cv=kf,
                        scoring=scoring,
                        n_jobs=1,  
                    )

                    rmse_scores = -cv_out["test_rmse"]
                    mae_scores = -cv_out["test_mae"]

                    records.append({
                        "k_days": k,
                        "features": feat_name,
                        "target": target,
                        "alpha": alpha,
                        "CV_RMSE_mean": round(float(np.mean(rmse_scores)), 4),
                        "CV_RMSE_std": round(float(np.std(rmse_scores)), 4),
                        "CV_MAE_mean": round(float(np.mean(mae_scores)), 4),
                    })

    all_results = pd.DataFrame(records)

    # Automatically extract the optimal alpha (lowest CV_RMSE_mean) per configuration
    best_idx = all_results.groupby(["target", "k_days", "features"])["CV_RMSE_mean"].idxmin()
    best_results = all_results.loc[best_idx].sort_values(["target", "k_days", "features"])

    print("\n=== BEST ALPHA PER CONFIGURATION (5-FOLD CV) ===")
    print(best_results.to_string(index=False))

    out_dir = project_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(out_dir / "ridge_alpha_tuning.csv", index=False)
    
    print("\n=== BEST OVERALL ALPHA (5-FOLD CV) ===")
    for target in ("y_mean", "y_cv"):
        sub = all_results[all_results["target"] == target]
        mean_by_alpha = sub.groupby("alpha")["CV_RMSE_mean"].mean()
        best_alpha = mean_by_alpha.idxmin()
        best_rmse = mean_by_alpha.min()
        print(f"The best alpha for {target} is alpha = {best_alpha} (mean CV RMSE: {best_rmse:.4f}).")
        
    return all_results


if __name__ == "__main__":
    tune_ridge_alphas()