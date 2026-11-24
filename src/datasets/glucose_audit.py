"""Quality checks on the raw Cohort B glucose data (before cleaning).

Checks, in order:
1. Timezone: are the timestamps in UTC or already in Swiss time?
2. Outliers: count suspicious values (nothing is removed here).
3. Units: are all values in mmol/L?
4. Gaps: how long are the stretches without any reading?

Run from the MLproject2 folder:
    python src/datasets/glucose_audit.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --- Paths and settings -----------------------------------------
DATA = Path(__file__).resolve().parents[3] / "data"
FIG = Path(__file__).resolve().parents[2] / "figures"
COHORT_FILE = DATA / "processed" / "cohort_b_glucose_data.csv"
TIMEZONE = "Europe/Zurich"

LOW, HIGH = 2.0, 27.8
HIGH_WARN = 25.0
FLAT_RUN = 12
GAP_MIN = 30


def load_cohort_b() -> pd.DataFrame:
    """Load the raw Cohort B readings and parse timestamps as UTC."""
    df = pd.read_csv(COHORT_FILE)
    df["ts"] = pd.to_datetime(df["read_at"], format="ISO8601", utc=True)
    return df.sort_values(["user_id", "ts"]).reset_index(drop=True)


def night_trough(profile: pd.Series) -> int:
    """Return the hour (0-9 h) at which the mean glucose is lowest."""
    return int(profile.loc[0:9].idxmin())


def check_timezone(df: pd.DataFrame) -> None:
    """Check whether timestamps are UTC or Swiss time.

    Glucose is lowest at night (around 3-5 h). We compute mean glucose per hour of
    the day in two ways and see which one puts the trough at a plausible hour.
    Winter vs summer is the most reliable test: if timestamps are UTC, the summer
    curve is shifted about 1 h earlier (Swiss time moves with daylight saving, UTC
    does not). If they are already Swiss time, both curves line up.
    """
    hour_raw = df["ts"].dt.hour
    hour_swiss = df["ts"].dt.tz_convert(TIMEZONE).dt.hour

    month = df["ts"].dt.month
    winter = df[month.isin([12, 1, 2])]
    summer = df[month.isin([6, 7, 8])]

    profiles = {
        "raw": df.groupby(hour_raw)["val"].mean(),
        "swiss": df.groupby(hour_swiss)["val"].mean(),
        "winter": winter.groupby(winter["ts"].dt.hour)["val"].mean(),
        "summer": summer.groupby(summer["ts"].dt.hour)["val"].mean(),
    }

    print("\n== 1. Timezone: hour of the night trough ==")
    print(f"  raw hours in the file       : {night_trough(profiles['raw'])} h")
    print(f"  converted to Swiss time     : {night_trough(profiles['swiss'])} h")
    print(f"  winter (raw hours)          : {night_trough(profiles['winter'])} h")
    print(f"  summer (raw hours)          : {night_trough(profiles['summer'])} h")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    ax[0].plot(profiles["raw"], marker="o", label="raw hours")
    ax[0].plot(profiles["swiss"], marker="o", label="converted to Swiss time")
    ax[0].set_title("Raw vs converted")
    ax[1].plot(profiles["winter"], marker="o", label="winter")
    ax[1].plot(profiles["summer"], marker="o", label="summer")
    ax[1].set_title("Winter vs summer (raw hours)")
    for a in ax:
        a.set_xlabel("hour of the day")
        a.set_xticks(range(0, 24, 2))
        a.legend()
    ax[0].set_ylabel("mean glucose (mmol/L)")
    plt.tight_layout()
    fig.savefig(FIG / "timezone_check.png", dpi=150)
    plt.close(fig)


def check_outliers(df: pd.DataFrame) -> None:
    """Count suspicious values. Nothing is removed: the rule is decided as a team."""
    v = df["val"]
    print("\n== 2. Outliers ==")
    print(f"  min / max value                 : {v.min():.2f} / {v.max():.2f}")
    print(f"  missing or <= 0                 : {(v.isna() | (v <= 0)).sum()}")
    print(f"  below {LOW}                       : {(v < LOW).sum()}")
    print(f"  above {HIGH}                     : {(v > HIGH).sum()}")
    print(f"  above {HIGH_WARN}                     : {(v > HIGH_WARN).sum()}")

    new_run = (df["val"] != df["val"].shift()) | (df["user_id"] != df["user_id"].shift())
    run_length = df.groupby(new_run.cumsum())["val"].size()
    print(f"  flat runs of {FLAT_RUN}+ identical readings : {(run_length >= FLAT_RUN).sum()}")


def check_units(df: pd.DataFrame) -> None:
    """Check units with each participant's mean (about 4-10 in mmol/L, 90-150 in mg/dL)."""
    means = df.groupby("user_id")["val"].mean()
    print("\n== 3. Units ==")
    print(f"  participant means, min / max          : {means.min():.2f} / {means.max():.2f}")
    print(f"  participants with a mean above 30     : {(means > 30).sum()} (likely mg/dL)")


def check_gaps(df: pd.DataFrame) -> None:
    """Describe pauses between consecutive readings of the same participant."""
    gap_min = df.groupby("user_id")["ts"].diff().dt.total_seconds() / 60
    gaps = gap_min[gap_min > GAP_MIN]

    print(f"\n== 4. Gaps (> {GAP_MIN} min) ==")
    print(f"  number of gaps: {len(gaps)}")
    bins = pd.cut(
        gaps,
        [GAP_MIN, 60, 120, 360, 1440, float("inf")],
        labels=["30 min-1 h", "1-2 h", "2-6 h", "6-24 h", "over 24 h"],
    )
    print(bins.value_counts().sort_index().to_string())

    longest_h = gap_min.groupby(df["user_id"]).max() / 60
    print(f"  participants with a gap over 24 h : {(longest_h > 24).sum()}")
    print(f"  longest gap, median / max         : {longest_h.median():.1f} h / {longest_h.max():.1f} h")


def main() -> None:
    FIG.mkdir(exist_ok=True)
    df = load_cohort_b()
    print(f"Participants: {df['user_id'].nunique()} | Readings: {len(df)}")

    check_timezone(df)
    check_outliers(df)
    check_units(df)
    check_gaps(df)


if __name__ == "__main__":
    main()