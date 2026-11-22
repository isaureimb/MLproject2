import numpy as np
import pandas as pd

BLOCK_DAYS = 12
TARGET_START_DAY = 7

def split_input_target(df, k_days=3, target_start_day=TARGET_START_DAY,
                       block_days=BLOCK_DAYS, id_col="user_id", ts_col="read_at",
                       timezone="Europe/Zurich"):
    """
    Split each participant's 12-day block into an input window and a target window.

    Input window  = the k days just before the target: day_idx in [target_start_day - k, target_start_day)
                    (1-indexed: days 8-k .. 7).
    Target window = day_idx >= target_start_day (1-indexed: days 8 .. 12).

    Args:
        df (pd.DataFrame): cleaned glucose dataset
        k_days (int): number of input days (1, 3, 5 or 7).
        target_start_day (int): 0-indexed first target day (7 = day 8).
        block_days (int): expected number of days per participant.
        id_col (str): participant identifier column.
        ts_col (str): full timestamp column (UTC or with offset).
        timezone (str): local timezone used to define days and hours.

    Returns:
        input_df (pd.DataFrame): readings of the input window.
        target_df (pd.DataFrame): readings of the target window.
    """
    if not 1 <= k_days <= target_start_day:
        raise ValueError(f"k_days must be between 1 and {target_start_day}, got {k_days}")

    out = df.copy()
    out["ts_local"] = pd.to_datetime(out[ts_col], format="ISO8601", utc=True).dt.tz_convert(timezone)
    out = out.sort_values([id_col, "ts_local"])
    local_date = out["ts_local"].dt.date
    out["day_idx"] = local_date.groupby(out[id_col]).transform(lambda s: pd.factorize(s)[0])

    n_days = out.groupby(id_col)["day_idx"].max() + 1
    bad = n_days[n_days != block_days]
    if len(bad):
        raise ValueError(f"{len(bad)} participant(s) do not have exactly {block_days} days "
                         f"(e.g. {bad.index[0]} has {bad.iloc[0]})")

    first_input_day = target_start_day - k_days
    input_df = out[(out["day_idx"] >= first_input_day) & (out["day_idx"] < target_start_day)]
    target_df = out[out["day_idx"] >= target_start_day]
    return input_df, target_df


def compute_cgm_features(input_df, id_col="user_id", val_col="val",
                         night_hours=(0, 6), day_hours=(8, 20)):
    """
    CGM features per participant, computed on the input window.

    Features: mean, std, CV (%), min, P10, median, P90, max,
    night mean and day mean.

    Args:
        input_df (pd.DataFrame): readings of the input window.
        id_col (str): participant identifier column.
        val_col (str): glucose value column.
        night_hours (tuple[int, int]): night range in local hours, start < end (e.g. (0, 6)).
        day_hours (tuple[int, int]): day range in local hours, start < end (e.g. (8, 20)).

    Returns:
        pd.DataFrame: one row per participant.
    """
    g = input_df.groupby(id_col)[val_col]
    features = pd.DataFrame({
        "cgm_mean": g.mean(),
        "cgm_std": g.std(),  
        "cgm_min": g.min(),
        "cgm_p10": g.quantile(0.10),
        "cgm_q25": g.quantile(0.25),
        "cgm_median": g.median(),
        "cgm_q75": g.quantile(0.75),
        "cgm_p90": g.quantile(0.90),
        "cgm_max": g.max(),
    })
    features.insert(2, "cgm_cv", 100 * features["cgm_std"] / features["cgm_mean"])

    # Night and day means (ranges must not cross midnight)
    hour = input_df["ts_local"].dt.hour
    is_night = (hour >= night_hours[0]) & (hour < night_hours[1])
    is_day = (hour >= day_hours[0]) & (hour < day_hours[1])
    features["cgm_night_mean"] = input_df[is_night].groupby(id_col)[val_col].mean()
    features["cgm_day_mean"] = input_df[is_day].groupby(id_col)[val_col].mean()
    return features

def compute_targets_features(target_df, id_col="user_id", val_col="val"):
    """
    Targets per participant on the target window: mean glucose and CV (%)

    Returns:
        pd.DataFrame: columns y_mean and y_cv (index = id_col).
    """
    g = target_df.groupby(id_col)[val_col]
    targets = pd.DataFrame({"y_mean": g.mean()})
    targets["y_cv"] = 100 * g.std() / targets["y_mean"]
    return targets
