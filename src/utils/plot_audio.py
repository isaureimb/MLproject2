from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_audit(cov_thresh: float = 0.75, block_days: int = 12) -> None:
    """
    Generate the three baseline audit plots from the cleaned Cohort B glucose dataset:
    1. Recording span per participant (total calendar days from first to last reading).
    2. Daily coverage distribution (% of expected 15-min bins observed per day, out of 96).
    3. Participant eligibility funnel (Cohort B -> >=12-day span -> >=12 consecutive valid days).

    Args:
        cov_thresh: Minimum daily fraction of observed 15-min bins to consider a day valid (default: 0.75).
        block_days: Required consecutive valid days for the main prediction block (default: 12).
    """
    project_root = Path(__file__).resolve().parents[3]
    df = pd.read_csv(project_root / "data" / "processed" / "glucose_cleaned.csv")
    df["local_date"] = pd.to_datetime(df["read_at"], format="ISO8601", utc=True).dt.tz_convert("Europe/Zurich").dt.date
    fig_dir = project_root / "MLproject2" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Daily coverage (96 expected 15-min bins/day)
    daily = df.groupby(["user_id", "local_date"]).size().reset_index(name="bins")
    daily["cov"] = (daily["bins"] / 96.0).clip(upper=1.0)
    daily["valid"] = daily["cov"] >= cov_thresh #>75%
    daily["date_dt"] = pd.to_datetime(daily["local_date"])

    # Span & max consecutive valid days per participant
    stats = []
    for uid, g in daily.groupby("user_id"):
        span = (g["date_dt"].max() - g["date_dt"].min()).days + 1
        v_dates = g.loc[g["valid"], "date_dt"].sort_values()
        consec = v_dates.groupby(v_dates.diff().dt.days.ne(1).cumsum()).size().max() if len(v_dates) else 0
        stats.append({"user_id": uid, "span": span, "consec": consec})
    stats_df = pd.DataFrame(stats)

    # Plot 3 audit panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(stats_df["span"], bins=range(0, 125, 2), edgecolor="white")
    axes[0].axvline(block_days, color="red", ls="--", label=f"{block_days}d target")
    axes[0].set(title="1. Recording Span", xlabel="Calendar days", ylabel="Participants", xlim=(0, 40))
    axes[0].legend()

    axes[1].hist(daily["cov"] * 100, bins=25, edgecolor="white")
    axes[1].axvline(cov_thresh * 100, color="red", ls="--", label=f"{int(cov_thresh*100)}% thresh")
    axes[1].set(title="2. Daily Coverage", xlabel="Observed bins (%)", ylabel="Participant-days")
    axes[1].legend()

    counts = [len(stats_df), (stats_df["span"] >= block_days).sum(), (stats_df["consec"] >= block_days).sum()]
    labels = ["Cohort B", f"Span >= {block_days}d", f"{block_days}d consec.\n(>= {int(cov_thresh*100)}%)"]
    bars = axes[2].bar(labels, counts)
    axes[2].set(title="3. Eligibility Funnel", ylabel="Participants", ylim=(0, max(counts) * 1.15))
    axes[2].bar_label(bars, padding=3)

    plt.tight_layout()
    plt.savefig(fig_dir / "cohort_b_audit_overview.png", dpi=300)
    plt.close()
    print("=== ELIGIBLE COUNTS ===")
    for label, count in zip(labels, counts):
        clean_label = label.replace("\n", " ")
        print(f"{clean_label:<25} : {int(count):>4} ({100 * count / counts[0]:.1f}%)")


if __name__ == "__main__":
    plot_audit()