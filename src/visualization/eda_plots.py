import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def inter_individual_spread(df, id_col="user_id", val_col="val", stat="median"):
    """
    compute the median glucose value within each participants,and compute the minimum, 
    the first quartile, the third quartile and the maxmimum of CGM values over all participants.

    Args:
        df (pd.DataFrame): one row per reading.
        id_col (str): participant identifier column.
        val_col (str): glucose value column.
        stat (str | callable): per-participant statistic
            ("median", "mean", "std", or any function).

    Returns:
        pd.Series: min, Q1, median, Q3, max, IQR and number of participants.
    """
    per_id = df.groupby(id_col)[val_col].agg(stat)

    stats = per_id.quantile([0, 0.25, 0.5, 0.75, 1])
    stats.index = ["min", "Q1", "median", "Q3", "max"]

    stats["IQR"] = stats["Q3"] - stats["Q1"]
    stats["n_participants"] = per_id.count()
    return stats


def plot_participant_boxplots(df, id_col="user_id", val_col="val", n_participants=30,
                              seed=42, show_outliers=False):
    """
    Side-by-side box plots for a random sample of participants, sorted by median.

    Args:
        df (pd.DataFrame): one row per reading.
        id_col (str): participant identifier column.
        val_col (str): glucose value column.
        n_participants (int): number of participants to draw at random.
        seed (int): random seed, so the same sample is drawn at every run.
        show_outliers (bool): show points beyond 1.5 

    """
    # Draw a random sample of participants 
    all_ids = df[id_col].dropna().unique()
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(all_ids, size=min(n_participants, len(all_ids)), replace=False)
    sample = df[df[id_col].isin(sample_ids)]

    order = sample.groupby(id_col)[val_col].median().sort_values().index
    values = [sample.loc[sample[id_col] == pid, val_col].dropna().to_numpy() for pid in order]
    labels = [f"P{i:02d}" for i in range(1, len(order) + 1)]

    #plot
    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(order)), 5))
    ax.boxplot(values, showfliers=show_outliers, widths=0.6)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.axhline(df[val_col].median(), color="red", linestyle="--", linewidth=1,
               label="population median")
    ax.set_xlabel("participants (sorted by median)")
    ax.set_ylabel("glucose (mmol/L)")
    ax.set_title(f"Glucose distribution of {len(order)} random participants")
    ax.legend()
    fig.tight_layout()



def plot_participant_mean_cv(df, id_col="user_id", val_col="val", min_readings=96,
                             sort_by="mean"):
    """
    Plot mean glucose and CV (%) per participant

    Args:
        df (pd.DataFrame): one row per reading.
        id_col (str): participant identifier column.
        val_col (str): glucose value column.
        min_readings (int): participants with fewer readings are dropped
            (96 readings ≈ one full day at 15-minute sampling).
        sort_by (str): "mean" or "cv", order of participants on the x axis.

    Returns:
        stats (pd.DataFrame): one row per participant (n_readings, mean, std, cv).
    """
   
    stats = df.groupby(id_col)[val_col].agg(n_readings="count", mean="mean", std="std")
    stats = stats[stats["n_readings"] >= min_readings]

    stats["cv"] = 100 * stats["std"] / stats["mean"]

    stats = stats.sort_values(sort_by).reset_index()
    stats.index = stats.index + 1
    stats.index.name = "participant_rank"

    fig, ax_mean = plt.subplots(figsize=(12, 5))
    ax_cv = ax_mean.twinx()

    ax_mean.scatter(stats.index, stats["mean"], s=8, color="tab:blue",
                    label=f"mean glucose (between-participant SD = {stats['mean'].std():.2f} mmol/L)")
    ax_cv.scatter(stats.index, stats["cv"], s=8, color="tab:orange", alpha=0.7,
                  label=f"CV (between-participant SD = {stats['cv'].std():.1f} %)")

    ax_mean.axhline(stats["mean"].mean(), color="tab:blue", linestyle="--", linewidth=1)
    ax_cv.axhline(stats["cv"].mean(), color="tab:orange", linestyle="--", linewidth=1)

    ax_mean.set_xlabel(f"participants (sorted by {sort_by})")
    ax_mean.set_ylabel("mean glucose (mmol/L)", color="tab:blue")
    ax_cv.set_ylabel("CV (%)", color="tab:orange")
    ax_mean.tick_params(axis="y", colors="tab:blue")
    ax_cv.tick_params(axis="y", colors="tab:orange")
    ax_mean.set_title(f"Mean glucose and CV of {len(stats)} participants "
                      f"(≥ {min_readings} readings); dashed lines = averages")

    handles = ax_mean.get_legend_handles_labels()[0] + ax_cv.get_legend_handles_labels()[0]
    ax_mean.legend(handles=handles, loc="upper left")

    fig.tight_layout()
    return stats

def plot_glucose_evol_day(df, id_col="user_id", date_col="date", time_col="read_at", val_col="val",
                   n_participants=5, start_hour=9, seed=42, max_gap_minutes=60):
    """
    Glucose curve over the first 24 h window of randomly drawn participants.

    Args:
        df (pd.DataFrame): one row per reading
        id_col (str): participant identifier column.
        date_col (str): date column.
        time_col (str): time-of-day column.
        val_col (str): glucose value column.
        n_participants (int): number of participants drawn at random.
        start_hour (int): hour at which the 24 h window starts.
        seed (int): random seed, so the same participants are drawn at every run.
        max_gap_minutes (float): gaps longer than this break the line.

    """
    #Draw a random sample of participants
    all_ids = df[id_col].dropna().unique()
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(all_ids, size=min(n_participants, len(all_ids)), replace=False)

    sample = df.loc[df[id_col].isin(sample_ids), [id_col, date_col, time_col, val_col]].copy()
    sample["datetime"] = pd.to_datetime(sample[date_col] + " " + sample[time_col])

    #Window start = start_hour on each participant's first recorded date
    first_date = pd.to_datetime(sample[date_col]).groupby(sample[id_col]).transform("min")
    window_start = first_date + pd.Timedelta(hours=start_hour)
    sample["hours"] = (sample["datetime"] - window_start).dt.total_seconds() / 3600
    sample = sample[(sample["hours"] >= 0) & (sample["hours"] <= 24)]

    fig, ax = plt.subplots(figsize=(12, 4.5))
    colors = plt.cm.tab10.colors

    for i, pid in enumerate(sample_ids):
        p = sample[sample[id_col] == pid].sort_values("hours")
        label = f"P{i + 1:02d}"
        if p.empty:
            ax.plot([], [], color=colors[i], label=f"{label} (no data in window)")
            continue

        hours = p["hours"].to_numpy()
        values = p[val_col].to_numpy(dtype=float)
        gap = np.diff(hours) * 60 > max_gap_minutes
        hours = np.insert(hours, np.where(gap)[0] + 1, np.nan)
        values = np.insert(values, np.where(gap)[0] + 1, np.nan)

        ax.plot(hours, values, color=colors[i], linewidth=1, label=label)

    ticks = range(0, 25, 3)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{(start_hour + t) % 24}h" for t in ticks])
    ax.axvline(24 - start_hour, color="grey", linestyle=":", linewidth=1)  # midnight
    ax.set_xlim(0, 24)
    ax.set_xlabel(f"time of day")
    ax.set_ylabel("glucose (mmol/L)")
    ax.set_title(f"Glucose over the first day of {len(sample_ids)} random participants")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()


def plot_daily_mean(df, id_col="user_id", date_col="date", val_col="val",
                                    n_participants=30, min_readings_per_day=72,
                                    median_over_all=True, min_participants_for_median=20,
                                    seed=42):
    """
    Daily mean glucose over the whole recording of randomly drawn participants,
    with the median of the daily means for each day.

    Args:
        df (pd.DataFrame): one row per reading, with a date column ("YYYY-MM-DD").
        id_col (str): participant identifier column.
        date_col (str): date column.
        val_col (str): glucose value column.
        n_participants (int): number of participants drawn at random (thin lines).
        min_readings_per_day (int): days with fewer readings are left out.
        median_over_all (bool): True = median computed over all participants,
            False = median computed over the drawn participants only.
        min_participants_for_median (int): the median is only drawn on days with at
            least this many participants (avoids a median based on 1 or 2 people).
        seed (int): random seed, so the same participants are drawn at every run.

    """
    #Mean glucose and number of readings for each participant and each date
    daily = (df.groupby([id_col, date_col])[val_col]
               .agg(n_readings="count", daily_mean="mean")
               .reset_index())

    #Recorded-day number
    dates = pd.to_datetime(daily[date_col])
    daily["day"] = dates.groupby(daily[id_col]).rank(method="dense").astype(int)

    #Leave out insufficiently covered days
    daily = daily[daily["n_readings"] >= min_readings_per_day]

    #Random sample of participants
    ids = daily[id_col].unique()
    rng = np.random.default_rng(seed)
    sample_ids = rng.choice(ids, size=min(n_participants, len(ids)), replace=False)
    sample = daily[daily[id_col].isin(sample_ids)]

    #Median of the participants' daily means, for each day
    source = daily if median_over_all else sample
    max_day = int(sample["day"].max())
    all_days = np.arange(1, max_day + 1)
    by_day = source.groupby("day")["daily_mean"]
    min_n = min_participants_for_median if median_over_all else 1
    median = by_day.median().where(by_day.count() >= min_n).reindex(all_days)

    fig, ax = plt.subplots(figsize=(12, 5))
    for _, p in sample.groupby(id_col):
        y = p.set_index("day")["daily_mean"].reindex(all_days)
        ax.plot(all_days, y.to_numpy(), color="tab:blue", linewidth=0.8, alpha=0.5,
                marker=".", markersize=3)

    who = f"all {daily[id_col].nunique()}" if median_over_all else f"the {len(sample_ids)} drawn"
    ax.plot(all_days, median.to_numpy(), color="black", linewidth=2.5,
            label=f"median of daily means ({who} participants)")

    step = 1 if max_day <= 30 else int(np.ceil(max_day / 30))
    ax.set_xticks(all_days[::step])
    ax.set_xlabel("recorded day")
    ax.set_ylabel("daily mean glucose (mmol/L)")
    ax.set_title(f"Daily mean glucose of {len(sample_ids)} random participants ")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

def plot_hour_day_heatmap(df, participant_id=None, id_col="user_id", date_col="date",
                          time_col="read_at", val_col="val", bin_minutes=60,
                          vmin=3, vmax=10, seed=42):
    """
    Heatmap of one participant's glucose: one row per recorded day, one column per
    time-of-day bin, colour = mean glucose in that bin.

    Args:
        df (pd.DataFrame): one row per reading, with a date column ("YYYY-MM-DD")
            and a time column ("HH:MM:SS").
        participant_id: participant to plot (None = one participant drawn at random).
        id_col (str): participant identifier column.
        date_col (str): date column.
        time_col (str): time-of-day column.
        val_col (str): glucose value column.
        bin_minutes (int): width of a time bin (60 = one column per hour, 15 = per reading).
        vmin, vmax (float): fixed colour scale in mmol/L, so that different
            participants can be compared.
        seed (int): random seed used when participant_id is None.

    """
    #Choose the participant
    if participant_id is None:
        rng = np.random.default_rng(seed)
        participant_id = rng.choice(df[id_col].dropna().unique())
    p = df.loc[df[id_col] == participant_id, [date_col, time_col, val_col]].copy()

    minutes = pd.to_timedelta(p[time_col]).dt.total_seconds() / 60
    p["bin"] = (minutes // bin_minutes) * bin_minutes / 60
    dates = pd.to_datetime(p[date_col])
    p["day"] = dates.rank(method="dense").astype(int)

    table = p.pivot_table(index="day", columns="bin", values=val_col, aggfunc="mean")
    all_bins = np.arange(0, 24, bin_minutes / 60)
    table = table.reindex(columns=all_bins)  

    row_labels = [f"day {d}" for d in table.index]

    #Heatmap
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")  # missing data
    fig, ax = plt.subplots(figsize=(13, 0.35 * len(table) + 2))
    im = ax.imshow(np.ma.masked_invalid(table.to_numpy()), aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=[0, 24, len(table) - 0.5, -0.5])

    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h}h" for h in range(0, 25, 2)])
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("time of day")
    ax.set_title(f"Glucose by time of day and recorded day — one participant "
                 f"({bin_minutes}-min bins, white = no data")
    fig.colorbar(im, ax=ax, label="mean glucose (mmol/L)")
    fig.tight_layout()