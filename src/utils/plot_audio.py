from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.preprocess import extract_eligible_blocks



def plot_audit(cov_thresh: float = 0.75, block_days: int = 12) -> None:
    """
    Generate the three baseline audit plots using the exact eligibility metrics
    computed by `extract_eligible_blocks`.
    """
    project_root = Path(__file__).resolve().parents[3]
    cleaned_df = pd.read_csv(project_root / "data" / "processed" / "glucose_cleaned.csv")
    fig_dir = project_root / "MLproject2" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _, daily, stats_df = extract_eligible_blocks(
        cleaned_df, cov_thresh=cov_thresh, block_days=block_days
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    axes[0].hist(stats_df["span"], bins=range(0, 125, 2), edgecolor="white")
    axes[0].axvline(block_days, color="red", ls="--", label=f"{block_days}d target")
    axes[0].set(title="1. Recording Span", xlabel="Calendar days", ylabel="Participants", xlim=(0, 120))
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