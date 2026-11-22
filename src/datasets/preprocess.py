from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

def filter_physiological_outliers(
    df: pd.DataFrame,
    min_mmol: float = 2.0,
    max_mmol: float = 27.8,
) -> pd.DataFrame:
    """
    Remove missing values and readings outside standard CGM physiological bounds
    (2.0 to 27.8 mmol/L, i.e., ~36 to 500 mg/dL) before bin aggregation.
    """
    initial_rows = len(df)
    valid_mask = df["val"].notna() & (df["val"] >= min_mmol) & (df["val"] <= max_mmol)
    cleaned = df.loc[valid_mask].copy()
    removed = initial_rows - len(cleaned)
    print(f"[1/3] Outlier filter [{min_mmol}, {max_mmol} mmol/L]: removed {removed} rows ({len(cleaned)} remaining)")
    return cleaned


def preprocess_glucose(
    input_filename: str = "glucose_cohort_b.csv",
    output_filename: str = "glucose_cleaned.csv",
    timezone: str = "Europe/Zurich",
    bin_freq: str = "15min",
    min_gap_minutes: float = 10.0,
) -> pd.DataFrame:
    """
    Convert UTC timestamps to Europe/Zurich local time, snap readings to 15-min bins
    (shifting a reading to the next bin if the target bin is occupied and >= 10 min
    have elapsed since the previous reading), and average true duplicates (< 10 min).
    """
    project_root = Path(__file__).resolve().parents[3]
    in_path = project_root / "data" / "processed" / input_filename
    out_path = project_root / "data" / "processed" / output_filename

    df = pd.read_csv(in_path)
    original_cols = df.columns.tolist()

    # 1. Parse UTC timestamps and sort chronologically per user
    df["ts_utc"] = pd.to_datetime(df["read_at"], format="ISO8601", utc=True)
    df = df.sort_values(["user_id", "ts_utc"]).reset_index(drop=True)

    # 2. Floor to 15-min grid in UTC (avoids DST AmbiguousTimeError), then shift if gap >= 10 min
    df["bin_utc"] = df["ts_utc"].dt.floor(bin_freq)
    same_user = df["user_id"] == df["user_id"].shift(1)
    same_bin = df["bin_utc"] == df["bin_utc"].shift(1)
    gap_ok = (df["ts_utc"] - df["ts_utc"].shift(1)).dt.total_seconds() >= (min_gap_minutes * 60)

    df.loc[same_user & same_bin & gap_ok, "bin_utc"] += pd.Timedelta(bin_freq)

    # Convert the clean 15-min bins to Europe/Zurich local time
    df["read_at"] = df["bin_utc"].dt.tz_convert(timezone)

    # 3. Average any remaining true duplicates (< 10 min apart) inside the same bin
    cleaned_df = (
        df.groupby(["user_id", "read_at"], as_index=False)["val"]
        .mean()
        .sort_values(["user_id", "read_at"])
        .reset_index(drop=True)
    )[original_cols]

    cleaned_df.to_csv(out_path, index=False)
    print(f"Preprocessed {len(cleaned_df)} bins across {cleaned_df['user_id'].nunique()} participants -> {out_path}")
    return cleaned_df


def extract_eligible_blocks(
    cleaned_df: pd.DataFrame,
    output_filename: str = "glucose_eligible_12d.csv",
    timezone: str = "Europe/Zurich",
    cov_thresh: float = 0.75,
    block_days: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute daily coverage, identify participants with >= `block_days` consecutive valid days
    (coverage >= `cov_thresh`), save their first contiguous `block_days` block, and return
    (eligible_df, daily_df, stats_df) so audit plots reuse the exact same logic.
    """
    project_root = Path(__file__).resolve().parents[3]
    out_path = project_root / "data" / "processed" / output_filename

    df = cleaned_df.copy()
    df["local_date"] = pd.to_datetime(df["read_at"], format="ISO8601", utc=True).dt.tz_convert(timezone).dt.date

    daily = df.groupby(["user_id", "local_date"]).size().reset_index(name="bins")
    daily["cov"] = (daily["bins"] / 96.0).clip(upper=1.0)
    daily["valid"] = daily["cov"] >= cov_thresh
    daily["date_dt"] = pd.to_datetime(daily["local_date"])

    stats = []
    eligible_windows = []
    for uid, g in daily.groupby("user_id"):
        span = (g["date_dt"].max() - g["date_dt"].min()).days + 1
        v_dates = g.loc[g["valid"], "date_dt"].sort_values().reset_index(drop=True)
        consec = v_dates.groupby(v_dates.diff().dt.days.ne(1).cumsum()).size().max() if len(v_dates) else 0
        stats.append({"user_id": uid, "span": span, "consec": consec})

        if consec >= block_days:
            streak_ids = v_dates.diff().dt.days.ne(1).cumsum()
            for _, streak in v_dates.groupby(streak_ids):
                if len(streak) >= block_days:
                    eligible_windows.append((uid, set(streak.iloc[:block_days].dt.date)))
                    break

    stats_df = pd.DataFrame(stats)
    eligible_df = pd.concat(
        [df[(df["user_id"] == uid) & (df["local_date"].isin(dates))] for uid, dates in eligible_windows],
        ignore_index=True,
    ).drop(columns=["local_date"])

    eligible_df.to_csv(out_path, index=False)
    print(f"Eligible cohort ({block_days}d consecutive >= {int(cov_thresh * 100)}%): {eligible_df['user_id'].nunique()} participants -> {out_path}")
    return eligible_df, daily, stats_df

from sklearn.model_selection import train_test_split


def split_and_save_cohort(
    eligible_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split eligible participants into disjoint Train (70%), Val (15%), and Test (15%)
    sets by `user_id` and save them to `data/processed/`.
    """
    project_root = Path(__file__).resolve().parents[3]
    processed_dir = project_root / "data" / "processed"

    users = sorted(eligible_df["user_id"].unique())
    train_users, temp_users = train_test_split(
        users, train_size=train_ratio, random_state=random_state
    )
    val_users, test_users = train_test_split(
        temp_users, train_size=val_ratio / (1.0 - train_ratio), random_state=random_state
    )

    splits = {
        "train_12d.csv": eligible_df[eligible_df["user_id"].isin(train_users)].copy(),
        "val_12d.csv": eligible_df[eligible_df["user_id"].isin(val_users)].copy(),
        "test_12d.csv": eligible_df[eligible_df["user_id"].isin(test_users)].copy(),
    }

    for filename, split_df in splits.items():
        split_df.to_csv(processed_dir / filename, index=False)
        print(f"Saved {filename:<15} : {split_df['user_id'].nunique()} participants ({len(split_df)} bins)")

    return splits["train_12d.csv"], splits["val_12d.csv"], splits["test_12d.csv"]


if __name__ == "__main__":
    cleaned = preprocess_glucose()
    eligible, _, _ = extract_eligible_blocks(cleaned)
    split_and_save_cohort(eligible)