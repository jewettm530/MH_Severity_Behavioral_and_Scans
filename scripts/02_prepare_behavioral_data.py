"""
02_prepare_behavioral_data.py

Purpose:
- Build a reduced, more valid behavioral phenotype table.
- Load non-definition phenotype CSV/TSV files.
- Merge them into one participant-level table.
- Identify the target symptom severity column.
- Save behavioral_features.csv.
- Use qids01.csv only for the target qvtot.
- Remove non-valuable/admin/timing/QC columns.
- Remove item-level questionnaire variables when they are not totals/subscales.
- Keep likely psychiatric/behavioral summary scores and useful standalone variables.
- Drop columns with 25% or more missing values.

Before running:
- Edit TARGET_COLUMN and SUBJECT_ID_COLUMN in config.py.
- Optionally edit PHENOTYPE_FILES_TO_USE in config.py.

Run:
    python 02_prepare_behavioral_data.py
"""

"""
02_prepare_behavioral_data.py

Purpose:
- Build a reduced, more valid behavioral phenotype table.
- Load all non-definition phenotype CSV files.
- Use qids01.csv only for the target qvtot.
- Remove non-valuable/admin/timing/QC columns.
- Remove item-level questionnaire variables when they are not totals/subscales.
- Keep likely psychiatric/behavioral summary scores and useful standalone variables.
- Drop columns with 25% or more missing values.

Output:
    outputs/behavioral_features.csv

Run:
    python3 scripts/02_prepare_behavioral_data.py
"""

from pathlib import Path
import re
import pandas as pd

from config import (
    PHENOTYPE_DIR,
    OUTPUT_DIR,
    TARGET_FILE,
    TARGET_COLUMN,
    SUBJECT_ID_COLUMN,
    PHENOTYPE_FILES_TO_USE,
    EXCLUDE_PHENOTYPE_FILES,
    MAX_MISSING_FRACTION,
)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def normalize_id(x):
    if pd.isna(x):
        return x
    x = str(x)
    x = x.replace("sub-", "")
    x = x.replace("_", "")
    return x


def find_id_column(df: pd.DataFrame) -> str:
    for col in ["subjectkey", "src_subject_id", "participant_id", "subject_id"]:
        if col in df.columns:
            return col
    raise ValueError(f"No recognizable subject ID column found. Columns: {list(df.columns)[:25]}")


def standardize_subject_id(df: pd.DataFrame) -> pd.DataFrame:
    if SUBJECT_ID_COLUMN in df.columns:
        return df
    id_col = find_id_column(df)
    return df.rename(columns={id_col: SUBJECT_ID_COLUMN})


def list_predictor_files():
    if PHENOTYPE_FILES_TO_USE:
        files = [PHENOTYPE_DIR / f for f in PHENOTYPE_FILES_TO_USE]
    else:
        files = sorted([
            p for p in PHENOTYPE_DIR.iterdir()
            if p.suffix.lower() == ".csv"
            and "_definitions" not in p.name
            and not p.name.startswith(".")
        ])

    excluded = set(EXCLUDE_PHENOTYPE_FILES)
    return [p for p in files if p.name not in excluded]


def is_admin_or_nonvaluable_column(col: str) -> bool:
    c = col.lower()

    exact_drop = {
        "src_subject_id",
        "interview_date",
        "collection_id",
        "dataset_id",
        "interview_period",
        "respondent",
        "relationship",
        "rel_to_proband",
        "comments_misc",
        "notes",
    }

    if c in exact_drop:
        return True

    bad_fragments = [
        "time_first_click", "time_last_click", "time_page_submit", "time_click_count",
        "tot_timing", "_timing", "timing", "click", "page_submit", "first_click", "last_click",
    ]
    if any(fragment in c for fragment in bad_fragments):
        return True

    if c.startswith("qc_") or "_qc" in c:
        return True
    if "num_nonresp" in c or "num_items" in c:
        return True
    if c.endswith("_complete") or c.endswith("_completed_by"):
        return True
    if "note" in c or "comment" in c or "description" in c:
        return True

    return False


def looks_like_item_level_column(col: str, file_stem: str) -> bool:
    c = col.lower()
    stem = file_stem.lower().replace("01", "")

    # Examples: bapq_q1, bapq_q36
    if re.search(r"(^|_)q\d+[a-z]?$", c):
        return True

    # Examples: asi1, asi2, stai10, pss4, rrs22
    if re.fullmatch(r"[a-z]+[0-9]+[a-z]?", c):
        return True

    # Examples: cerq_1, cerq_2
    if re.search(r"_[0-9]+[a-z]?$", c):
        return True

    if stem and re.fullmatch(rf"{re.escape(stem)}_?\d+[a-z]?", c):
        return True

    return False


def looks_like_summary_or_trait_column(col: str) -> bool:
    c = col.lower()

    keep_fragments = [
        "total", "tot", "score", "sum", "subscale", "scale", "ave", "avg",
        "mean", "median", "std", "percent", "rate", "accuracy", "acc", "rt",
        "reaction", "error", "omission", "commission", "correct", "incorrect",
        "interference", "sensitivity", "severity", "positive", "negative", "general",
        "depression", "depress", "anxiety", "stress", "panic", "mania", "anhedonia",
        "rumination", "brooding", "reflection", "trauma", "avoidance", "attachment",
        "impuls", "reward", "inhibition", "activation", "support", "neuroticism",
        "extraversion", "openness", "agreeableness", "conscientiousness",
        "harm_avoidance", "novelty", "persistence", "pleasure", "risk",
    ]

    if any(fragment in c for fragment in keep_fragments):
        return True

    if c in {"age", "interview_age", "sex", "site", "group"}:
        return True

    return False


def should_keep_predictor_column(col: str, file_stem: str) -> bool:
    if col == SUBJECT_ID_COLUMN:
        return True

    if is_admin_or_nonvaluable_column(col):
        return False

    if col.lower() in {"age", "interview_age", "sex", "site", "group"}:
        return True

    if looks_like_item_level_column(col, file_stem) and not looks_like_summary_or_trait_column(col):
        return False

    if looks_like_summary_or_trait_column(col):
        return True

    return False


def clean_predictor_table(df: pd.DataFrame, file_stem: str) -> pd.DataFrame:
    df = standardize_subject_id(df)
    keep_cols = [col for col in df.columns if should_keep_predictor_column(col, file_stem)]
    df = df[keep_cols].copy()
    df = df.drop_duplicates(subset=[SUBJECT_ID_COLUMN])

    rename_map = {
        col: f"{file_stem}__{col}"
        for col in df.columns
        if col != SUBJECT_ID_COLUMN
    }
    return df.rename(columns=rename_map)


def load_target_table() -> pd.DataFrame:
    target_path = PHENOTYPE_DIR / TARGET_FILE
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    target_df = read_csv(target_path)
    target_df = standardize_subject_id(target_df)

    if TARGET_COLUMN not in target_df.columns:
        raise ValueError(f"Target column {TARGET_COLUMN} not found in {TARGET_FILE}")

    target_df = target_df[[SUBJECT_ID_COLUMN, TARGET_COLUMN]].copy()
    target_df[SUBJECT_ID_COLUMN] = target_df[SUBJECT_ID_COLUMN].apply(normalize_id)
    target_df = target_df.dropna(subset=[TARGET_COLUMN])
    target_df = target_df.drop_duplicates(subset=[SUBJECT_ID_COLUMN])

    print(f"Target file: {TARGET_FILE}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Rows with non-missing target: {len(target_df)}")

    return target_df


def drop_high_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    protected = {SUBJECT_ID_COLUMN, TARGET_COLUMN}
    missing_fraction = df.isna().mean()

    to_drop = [
        col for col, frac in missing_fraction.items()
        if col not in protected and frac >= MAX_MISSING_FRACTION
    ]

    if to_drop:
        print(f"\nDropping {len(to_drop)} columns with >= {MAX_MISSING_FRACTION:.0%} missing values.")
        for col in to_drop[:50]:
            print(f"  drop missing-heavy: {col} ({missing_fraction[col]:.1%} missing)")
        if len(to_drop) > 50:
            print(f"  ... and {len(to_drop) - 50} more")

    return df.drop(columns=to_drop, errors="ignore")


def drop_no_variance_columns(df: pd.DataFrame) -> pd.DataFrame:
    protected = {SUBJECT_ID_COLUMN, TARGET_COLUMN}
    to_drop = []

    for col in df.columns:
        if col in protected:
            continue
        if df[col].nunique(dropna=True) <= 1:
            to_drop.append(col)

    if to_drop:
        print(f"\nDropping {len(to_drop)} columns with no useful variation.")

    return df.drop(columns=to_drop, errors="ignore")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged = load_target_table()

    predictor_files = list_predictor_files()
    print(f"\nCandidate predictor phenotype files: {len(predictor_files)}")

    loaded = 0
    skipped = 0

    for path in predictor_files:
        try:
            raw_df = read_csv(path)
            cleaned = clean_predictor_table(raw_df, path.stem)
            predictor_cols = [c for c in cleaned.columns if c != SUBJECT_ID_COLUMN]

            if not predictor_cols:
                skipped += 1
                print(f"Skipping {path.name}: no summary/subscale/trait columns kept.")
                continue

            cleaned[SUBJECT_ID_COLUMN] = cleaned[SUBJECT_ID_COLUMN].apply(normalize_id)
            merged = merged.merge(cleaned, on=SUBJECT_ID_COLUMN, how="left")

            loaded += 1
            print(f"Loaded {path.name}: kept {len(predictor_cols)} columns")

        except Exception as e:
            skipped += 1
            print(f"WARNING: skipped {path.name}: {e}")

    print(f"\nLoaded predictor files: {loaded}")
    print(f"Skipped files: {skipped}")
    print(f"Shape before missingness filter: {merged.shape}")

    merged = drop_high_missing_columns(merged)
    merged = drop_no_variance_columns(merged)

    print(f"Final behavioral shape: {merged.shape}")

    out_path = OUTPUT_DIR / "behavioral_features.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    print("\nFirst 80 columns:")
    for col in merged.columns[:80]:
        print(" -", col)
    if merged.shape[1] > 80:
        print(f"... and {merged.shape[1] - 80} more")


if __name__ == "__main__":
    main()
