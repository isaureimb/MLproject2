from pathlib import Path
import pandas as pd


def build_cohort_b_dataset() -> None:
    """
    Audit ID linkage between glucose.csv and metadata.csv, filter for Cohort B,
    and save clean processed CSV files while keeping raw data untouched.
    """
    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "data"
    out_path = data_path / "processed"
    out_path.mkdir(parents=True, exist_ok=True)

    glucose_df = pd.read_csv(data_path / "glucose.csv")
    metadata_df = pd.read_csv(data_path / "metadata.csv")

    assert metadata_df["id"].is_unique, "Duplicate IDs found in metadata.csv"

    # Audit counts
    g_ids = set(glucose_df["user_id"])
    m_ids = set(metadata_df["id"])
    print(f"Glucose IDs: {len(g_ids)} | Metadata IDs: {len(m_ids)} | Matched: {len(g_ids & m_ids)}")
    print(f"Unmatched glucose IDs: {len(g_ids - m_ids)} | Unmatched metadata IDs: {len(m_ids - g_ids)}")

    # Filter Cohort B and keep exact glucose.csv columns
    cohort_b_ids = set(metadata_df.loc[metadata_df["cohort"] == "cohort_b", "id"])
    cohort_b_glucose = glucose_df[glucose_df["user_id"].isin(cohort_b_ids)]

    # Split read_at into date (YYYY-MM-DD) and time (HH:MM:SS), no timezone conversion
    timestamp = pd.to_datetime(cohort_b_glucose["read_at"], format="mixed")
    cohort_b_glucose.insert(0, "date", timestamp.dt.strftime("%Y-%m-%d"))
    cohort_b_glucose["read_at"] = timestamp.dt.strftime("%H:%M:%S")
    
    print(f"Cohort B participants: {cohort_b_glucose['user_id'].nunique()} | Rows: {len(cohort_b_glucose)}")

    cohort_b_glucose.to_csv(out_path / "cohort_b_glucose_data.csv", index=False)


if __name__ == "__main__":
    build_cohort_b_dataset()
    