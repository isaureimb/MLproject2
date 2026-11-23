import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate

from src.trainer.eval_models import get_xy, make_models


def run_cross_validation(
    horizons=(1, 3, 5, 7),
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run 5-Fold Cross-Validation on (Train + Val) participants using
    src.datasets.features.build_xy while keeping test_12d.csv untouched.
    """
    project_root = Path(__file__).resolve().parents[3]
    processed = project_root / "data" / "processed"

    # 1. groupe Train + Val pour le CV
    train_raw = pd.read_csv(processed / "train_12d.csv")
    val_raw = pd.read_csv(processed / "val_12d.csv")
    dev_raw = pd.concat([train_raw, val_raw], ignore_index=True)

    # 2. split en k-folds
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }

    game = {"base": False, "base+extra": True}
    records = []

    for k in horizons:
        for nom_jeu, add_extra in game.items():
            X_dev, y_dev = get_xy(dev_raw, k, add_extra)

            for target in ("y_mean", "y_cv"):
                y_vec = y_dev[target]

                for nom_modele, modele in make_models().items():
                    cv_out = cross_validate(
                        modele,
                        X_dev,
                        y_vec,
                        cv=kf,
                        scoring=scoring,
                        n_jobs=-1,
                    )

                    mae_scores = -cv_out["test_mae"]
                    rmse_scores = -cv_out["test_rmse"]

                    records.append({
                        "k_days": k,
                        "features": nom_jeu,
                        "target": target,
                        "model": nom_modele,
                        "CV_MAE_mean": round(float(np.mean(mae_scores)), 4),
                        "CV_MAE_std": round(float(np.std(mae_scores)), 4),
                        "CV_RMSE_mean": round(float(np.mean(rmse_scores)), 4),
                        "CV_RMSE_std": round(float(np.std(rmse_scores)), 4),
                    })

    results = pd.DataFrame(records)
    print(f"\n=== {n_splits}-FOLD CROSS-VALIDATION ===")
    print(results.to_string(index=False))

    # save in results
    out_dir = project_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "models_cv_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\n Results CV save in{out_path}")
    return results


if __name__ == "__main__":
    run_cross_validation()