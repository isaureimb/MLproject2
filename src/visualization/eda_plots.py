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