from pathlib import Path
import pandas as pd


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


if __name__ == "__main__":
    preprocess_glucose()