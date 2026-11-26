import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def evaluate_basics_models(horizons=(1, 3, 5, 7)) -> pd.DataFrame:
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
    evaluate_basics_models()
