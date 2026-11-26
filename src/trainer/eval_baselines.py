import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.features import build_xy
from src.metrics.calculate_error import evaluation_metrics


def evaluate_baselines(horizons: list[int] = [1, 3, 5, 7]) -> pd.DataFrame:
    """
    Evaluate Constant and Persistence on the
    Validation set across observation windows k in {1, 3, 5, 7} days.
    """
    project_root = Path(__file__).resolve().parents[3]
    processed_dir = project_root / "data" / "processed"

    train_raw = pd.read_csv(processed_dir / "train_12d.csv")
    val_raw = pd.read_csv(processed_dir / "val_12d.csv")

    records = []

    for k in horizons:
        X_train, y_train = build_xy(train_raw, k_days=k)
        X_val, y_val = build_xy(val_raw, k_days=k)

        for target, persist_col in [("y_mean", "cgm_mean"), ("y_cv", "cgm_cv")]:
            y_tr = y_train[target].to_numpy()
            y_va = y_val[target].to_numpy()

            # 1. baseline constante (mean of the training target)
            pred_const = np.full(len(y_va), np.mean(y_tr))

            # 2. baseline of persistence (value observed on the first k days)
            pred_persist = X_val[persist_col].to_numpy()

            for model_name, preds in [
                ("Constant", pred_const),
                ("Persistence", pred_persist),
            ]:
                records.append({"k_days": k, "target": target, "model": model_name,
                                **evaluation_metrics(y_va, preds)})

    results_df = pd.DataFrame(records)
    print("\n=== VALIDATION BASELINE RESULTS (CGM ONLY) ===")
    print(results_df.to_string(index=False))

    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / "baseline_val_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved results -> {out_path}")
    return results_df


if __name__ == "__main__":
    evaluate_baselines()
    