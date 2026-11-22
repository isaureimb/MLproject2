import numpy as np
import pandas as pd


def build_xy_cgm_only(
    split_df: pd.DataFrame,
    k_days: int = 3,
    target_start_day: int = 7,  # Jour 8 en indexation 0 (jours 7, 8, 9, 10, 11)
    timezone: str = "Europe/Zurich",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build feature matrix X (from days 1..k) and target matrix y (from days 8 to 12)
    using only CGM glucose readings, with 1 row per participant.
    """
    if not (1 <= k_days <= target_start_day):
        raise ValueError(f"k_days must be between 1 and {target_start_day}, got {k_days}")

    df = split_df.copy()
    ts_local = pd.to_datetime(df["read_at"], format="ISO8601", utc=True).dt.tz_convert(timezone)
    df["local_date"] = ts_local.dt.date

    # Numérotation relative des jours 0..11 dans le bloc de 12 jours de chaque participant
    df = df.sort_values(["user_id", "local_date", "read_at"])
    df["day_idx"] = df.groupby("user_id")["local_date"].transform(
        lambda s: pd.factorize(s)[0]
    )

    X_rows = []
    y_rows = []

    for uid, g in df.groupby("user_id", sort=True):
        past = g.loc[g["day_idx"] < k_days, "val"].to_numpy()
        future = g.loc[g["day_idx"] >= target_start_day, "val"].to_numpy()

        past_mean = float(np.mean(past))
        past_std = float(np.std(past, ddof=1))
        past_cv = 100.0 * past_std / past_mean

        X_rows.append({
            "user_id": uid,
            "cgm_mean": past_mean,
            "cgm_std": past_std,
            "cgm_cv": past_cv,
            "cgm_min": float(np.min(past)),
            "cgm_q25": float(np.percentile(past, 25)),
            "cgm_median": float(np.median(past)),
            "cgm_q75": float(np.percentile(past, 75)),
            "cgm_max": float(np.max(past)),
            "cgm_iqr": float(np.percentile(past, 75) - np.percentile(past, 25)),
            "cgm_tir_39_100": float(np.mean((past >= 3.9) & (past <= 10.0)) * 100.0),
            "cgm_coverage": float(len(past) / (k_days * 96.0)),
        })

        fut_mean = float(np.mean(future))
        fut_std = float(np.std(future, ddof=1))
        fut_cv = 100.0 * fut_std / fut_mean

        y_rows.append({
            "user_id": uid,
            "y_mean": fut_mean,
            "y_cv": fut_cv,
        })

    X_df = pd.DataFrame(X_rows).set_index("user_id")
    y_df = pd.DataFrame(y_rows).set_index("user_id")
    return X_df, y_df