"""
04_merge_features.py

Purpose:
- Merge behavioral_features.csv and imaging_features.csv by participant ID.
- Create model-ready datasets:
    final_behavioral_only.csv
    final_imaging_only.csv
    final_multimodal.csv

Run:
    python 04_merge_features.py
"""

import pandas as pd
from config import OUTPUT_DIR, SUBJECT_ID_COLUMN, TARGET_COLUMN


def normalize_id(x):
    if pd.isna(x):
        return x
    x = str(x)
    x = x.replace("sub-", "")
    x = x.replace("_", "")
    return x


def main():
    behavioral = pd.read_csv(OUTPUT_DIR / "behavioral_features.csv", low_memory=False)
    imaging = pd.read_csv(OUTPUT_DIR / "imaging_features.csv", low_memory=False)

    behavioral["merge_id"] = behavioral[SUBJECT_ID_COLUMN].apply(normalize_id)
    imaging["merge_id"] = imaging["participant_id"].apply(normalize_id)

    print("Behavioral rows:", len(behavioral))
    print("Imaging rows:", len(imaging))
    print("Behavioral unique IDs:", behavioral["merge_id"].nunique())
    print("Imaging unique IDs:", imaging["merge_id"].nunique())

    merged = behavioral.merge(imaging, on="merge_id", how="inner")
    merged = merged.dropna(subset=[TARGET_COLUMN])
    print("Merged participants:", len(merged))

    if len(merged) == 0:
        print("\nSample behavioral IDs:", behavioral["merge_id"].dropna().head(10).tolist())
        print("Sample imaging IDs:", imaging["merge_id"].dropna().head(10).tolist())
        raise ValueError("Merge produced 0 rows. Check ID formats.")

    imaging_cols = [c for c in imaging.columns if c not in ["participant_id", "merge_id"]]
    behavioral_predictor_cols = [
        c for c in behavioral.columns
        if c not in [SUBJECT_ID_COLUMN, TARGET_COLUMN, "merge_id"]
    ]

    id_target_cols = [SUBJECT_ID_COLUMN, TARGET_COLUMN]

    behavioral_only = merged[id_target_cols + behavioral_predictor_cols]
    imaging_only = merged[id_target_cols + imaging_cols]
    multimodal = merged[id_target_cols + behavioral_predictor_cols + imaging_cols]

    behavioral_only.to_csv(OUTPUT_DIR / "final_behavioral_only.csv", index=False)
    imaging_only.to_csv(OUTPUT_DIR / "final_imaging_only.csv", index=False)
    multimodal.to_csv(OUTPUT_DIR / "final_multimodal.csv", index=False)

    print("Saved:")
    print(" -", OUTPUT_DIR / "final_behavioral_only.csv", behavioral_only.shape)
    print(" -", OUTPUT_DIR / "final_imaging_only.csv", imaging_only.shape)
    print(" -", OUTPUT_DIR / "final_multimodal.csv", multimodal.shape)


if __name__ == "__main__":
    main()
