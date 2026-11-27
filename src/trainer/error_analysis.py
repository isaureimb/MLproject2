"""Participant-level out-of-fold error analysis and diagnostic plotting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from src.metrics.calculate_error import evaluation_metrics
from src.model.basic_models import get_xy, make_models


TARGET_CONFIG = {
    "y_mean": ("cgm_mean", "mean glucose (mmol/L)", 10.0),
    "y_cv": ("cgm_cv", "CV (% points)", 5.0),
}


def top_error_share(squared_error: np.ndarray, fraction: float) -> float:
    """Percentage of total squared error contributed by the largest errors."""
    sorted_error = np.sort(squared_error)[::-1]
    n_top = max(1, int(np.ceil(fraction * len(sorted_error))))
    total_error = sorted_error.sum()
    return 0.0 if total_error == 0 else 100.0 * sorted_error[:n_top].sum() / total_error


def get_oof_and_participant_errors(
    dev_raw: pd.DataFrame,
    k: int = 3,
    add_extra: bool = False,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute out-of-fold Ridge predictions and participant-level errors."""
    X_dev, y_dev = get_xy(dev_raw, k=k, add_extra=add_extra)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    df_err = pd.DataFrame({
        "participant_id": X_dev.index,
        "k_days": k,
    })

    for target, (persist_col, _, _) in TARGET_CONFIG.items():
        y_true = y_dev[target].to_numpy()
        y_persist = X_dev[persist_col].fillna(X_dev[persist_col].median()).to_numpy()

        ridge = make_models(target)["Ridge"]
        y_pred = cross_val_predict(ridge, X_dev, y_true, cv=kf, n_jobs=1)

        df_err[f"{target}_true"] = y_true
        df_err[f"{target}_pred"] = y_pred
        df_err[f"{target}_persist"] = y_persist
        df_err[f"{target}_regime_shift"] = y_true - y_persist
        df_err[f"{target}_residual"] = y_pred - y_true
        df_err[f"{target}_abs_err"] = np.abs(y_pred - y_true)
        df_err[f"{target}_sq_err"] = (y_pred - y_true) ** 2
        df_err[f"{target}_persist_abs_err"] = np.abs(y_persist - y_true)

        ridge_metrics = evaluation_metrics(y_true, y_pred)
        persist_metrics = evaluation_metrics(y_true, y_persist)
        sq_err = df_err[f"{target}_sq_err"].to_numpy()

        print(
            f"\n--- {target}, k={k}d, Ridge={ridge_metrics} "
            f"| Persistence={persist_metrics} ---"
        )
        print(f"Top 5% of participants:  {top_error_share(sq_err, 0.05):.1f}% of squared error")
        print(f"Top 10% of participants: {top_error_share(sq_err, 0.10):.1f}% of squared error")

    return df_err


def plot_observed_vs_predicted(
    df_oof: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """Plot Observed vs. Predicted highlighting the Top 10% error contributors."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    configs = [
        ("y_mean", 3, "Future Mean Glucose (days 8–12)", "mmol/L", "alpha = 10.0"),
        ("y_cv", 7, "Future Glucose Variability CV (days 8–12)", "%", "alpha = 5.0"),
    ]

    for ax, (target, best_k, label, unit, alpha_str) in zip(axes, configs):
        sub = df_oof[df_oof["k_days"] == best_k].copy()
        thresh = sub[f"{target}_sq_err"].quantile(0.90)
        is_top10 = sub[f"{target}_sq_err"] >= thresh

        rest = sub[~is_top10]
        top10 = sub[is_top10].sort_values(f"{target}_sq_err", ascending=False)

        ax.scatter(
            sub[f"{target}_true"], sub[f"{target}_persist"],
            alpha=0.25, s=18, color="gray", label="Persistence (input period)"
        )
        ax.scatter(
            rest[f"{target}_true"], rest[f"{target}_pred"],
            alpha=0.60, s=24, color="#1f77b4", label=f"Tuned Ridge — 90% cohort ({alpha_str})"
        )
        ax.scatter(
            top10[f"{target}_true"], top10[f"{target}_pred"],
            alpha=0.90, s=42, color="#d62728", edgecolor="black", lw=0.5,
            label="Top 10% error contributors"
        )

        for _, row in top10.head(5).iterrows():
            ax.annotate(
                str(row["participant_id"]),
                (row[f"{target}_true"], row[f"{target}_pred"]),
                textcoords="offset points", xytext=(5, 4), fontsize=8, color="#8b0000"
            )

        lims = [
            min(sub[f"{target}_true"].min(), sub[f"{target}_pred"].min(), sub[f"{target}_persist"].min()),
            max(sub[f"{target}_true"].max(), sub[f"{target}_pred"].max(), sub[f"{target}_persist"].max()),
        ]
        ax.plot(lims, lims, "k--", lw=1.2, label="Ideal (y = x)")
        ax.set_xlabel(f"Observed {label} [{unit}]")
        ax.set_ylabel(f"Predicted {label} [{unit}]")
        ax.set_title(f"Observed vs. Predicted — {target} (k = {best_k} days)")
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_outlier_diagnostics(
    df_oof: pd.DataFrame,
    save_path: Path | None = None,
) -> None:
    """Plot residuals vs. observed values and actual behavioral shifts (2x2 grid)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    configs = [
        ("y_mean", 3, "Mean Glucose (mmol/L)"),
        ("y_cv", 7, "Glucose CV (%)"),
    ]

    for row_idx, (target, best_k, unit_label) in enumerate(configs):
        sub = df_oof[df_oof["k_days"] == best_k].copy()
        thresh = sub[f"{target}_sq_err"].quantile(0.90)
        is_top10 = sub[f"{target}_sq_err"] >= thresh

        # 1. Residuals vs. Observed
        ax = axes[row_idx, 0]
        ax.scatter(sub.loc[~is_top10, f"{target}_true"], sub.loc[~is_top10, f"{target}_residual"],
                   alpha=0.5, s=20, color="#1f77b4", label="90% cohort")
        ax.scatter(sub.loc[is_top10, f"{target}_true"], sub.loc[is_top10, f"{target}_residual"],
                   alpha=0.9, s=36, color="#d62728", edgecolor="black", lw=0.5, label="Top 10% errors")
        ax.axhline(0, color="k", linestyle="--", lw=1)
        ax.set_xlabel(f"Observed {unit_label}")
        ax.set_ylabel("Residual (Pred - True)")
        ax.set_title(f"{target} (k={best_k}d): Residual vs. Observed")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

        # 2. Residuals vs. Actual Shift (Days 8-12 minus Input)
        ax = axes[row_idx, 1]
        ax.scatter(sub.loc[~is_top10, f"{target}_regime_shift"], sub.loc[~is_top10, f"{target}_residual"],
                   alpha=0.5, s=20, color="#1f77b4", label="90% cohort")
        ax.scatter(sub.loc[is_top10, f"{target}_regime_shift"], sub.loc[is_top10, f"{target}_residual"],
                   alpha=0.9, s=36, color="#d62728", edgecolor="black", lw=0.5, label="Top 10% errors")
        ax.axhline(0, color="k", linestyle="--", lw=1)
        ax.axvline(0, color="gray", linestyle=":", lw=1)
        ax.set_xlabel(f"Actual Shift (Days 8–12 minus Input) [{unit_label}]")
        ax.set_ylabel("Residual (Pred - True)")
        ax.set_title(f"{target} (k={best_k}d): Error vs. Behavioral Shift")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def run_error_analysis() -> None:
    project_root = Path(__file__).resolve().parents[3]
    processed = project_root / "data" / "processed"
    out_dir = project_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_raw = pd.read_csv(processed / "train_12d.csv")
    val_raw = pd.read_csv(processed / "val_12d.csv")
    dev_raw = pd.concat([train_raw, val_raw], ignore_index=True)

    oof_frames = [
        get_oof_and_participant_errors(dev_raw, k=k, add_extra=False)
        for k in (1, 3, 5, 7)
    ]
    df_oof = pd.concat(oof_frames, ignore_index=True)
    df_oof.to_csv(out_dir / "oof_participant_errors.csv", index=False)


def run_top10_ablation(
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compare 5-fold CV Ridge trained on all participants vs. trained without Top 10% error outliers."""
    project_root = Path(__file__).resolve().parents[3]
    processed_dir = project_root / "data" / "processed"
    results_dir = project_root / "results"

    dev_raw = pd.concat(
        [
            pd.read_csv(processed_dir / "train_12d.csv"),
            pd.read_csv(processed_dir / "val_12d.csv"),
        ],
        ignore_index=True,
    )
    df_oof = pd.read_csv(results_dir / "oof_participant_errors.csv")

    rows = []
    for target, k in [("y_mean", 3), ("y_cv", 7)]:
        X_dev, y_dev = get_xy(dev_raw, k=k, add_extra=False)
        y_all = y_dev[target].to_numpy()

        sub_oof = df_oof[df_oof["k_days"] == k].reset_index(drop=True)
        thresh = sub_oof[f"{target}_sq_err"].quantile(0.90)
        inlier_mask = (sub_oof[f"{target}_sq_err"] < thresh).to_numpy()

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        preds_full = np.zeros(len(y_all))
        preds_no_top10 = np.zeros(len(y_all))

        for tr_idx, va_idx in kf.split(X_dev):
            X_tr, y_tr = X_dev.iloc[tr_idx], y_all[tr_idx]
            X_va = X_dev.iloc[va_idx]

            # 1. Trained on 100% of training fold
            m_full = make_models(target)["Ridge"]
            m_full.fit(X_tr, y_tr)
            preds_full[va_idx] = m_full.predict(X_va)

            # 2. Trained only on 90% inliers (strictly inside training fold, no leakage)
            tr_inliers = inlier_mask[tr_idx]
            m_clean = make_models(target)["Ridge"]
            m_clean.fit(X_tr.iloc[tr_inliers], y_tr[tr_inliers])
            preds_no_top10[va_idx] = m_clean.predict(X_va)

        scenarios = [
            ("1. Train 100% -> Eval 100%", y_all, preds_full),
            ("2. Train 90% (no Top10) -> Eval 100%", y_all, preds_no_top10),
            ("3. Train 100% -> Eval 90% stable", y_all[inlier_mask], preds_full[inlier_mask]),
            ("4. Train 90% (no Top10) -> Eval 90% stable", y_all[inlier_mask], preds_no_top10[inlier_mask]),
        ]

        for scenario, yt, yp in scenarios:
            m = evaluation_metrics(yt, yp)
            rows.append({
                "target": target,
                "k_days": k,
                "n_eval": len(yt),
                "scenario": scenario,
                "MAE": round(m["MAE"], 4),
                "RMSE": round(m["RMSE"], 4),
                "Bias": round(m["Bias"], 4),
            })

    
    return pd.DataFrame(rows)


def plot_top10_ablation(save_path: Path | None = None) -> None:
    """Visualize MAE and RMSE across the 4 Top-10% outlier ablation scenarios."""
    df_ablation = run_top10_ablation()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    configs = [
        ("y_mean", 3, "Mean Glucose Error [mmol/L]"),
        ("y_cv", 7, "Glucose CV Error [% points]"),
    ]
    short_labels = [
        "Train 100%\n→ Eval 100%",
        "Train 90%\n→ Eval 100%",
        "Train 100%\n→ Eval 90%",
        "Train 90%\n→ Eval 90%",
    ]

    x = np.arange(len(short_labels))
    width = 0.35

    for ax, (target, k, ylabel) in zip(axes, configs):
        sub = df_ablation[(df_ablation["target"] == target) & (df_ablation["k_days"] == k)].reset_index(drop=True)
        mae_vals = sub["MAE"].to_numpy()
        rmse_vals = sub["RMSE"].to_numpy()

        bars_mae = ax.bar(x - width / 2, mae_vals, width, label="MAE", color="#1f77b4", alpha=0.9)
        bars_rmse = ax.bar(x + width / 2, rmse_vals, width, label="RMSE", color="#ff7f0e", alpha=0.9)

        ax.axvline(1.5, color="gray", linestyle="--", lw=1, alpha=0.7)

        for bar in list(bars_mae) + list(bars_rmse):
            h = bar.get_height()
            ax.annotate(
                f"{h:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9.5)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{target} (k = {k} days) — Impact of Top 10% Outliers")
        ax.set_ylim(0, max(rmse_vals) * 1.18)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()



if __name__ == "__main__":
    run_error_analysis()