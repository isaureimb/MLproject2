from pathlib import Path
import pandas as pd
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

