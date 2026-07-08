from pathlib import Path
from io import StringIO
import csv
import pandas as pd

phenotype_dir = Path("Dataset/phenotype")


def clean_quoted_line_file(path: Path) -> str:
    lines = path.read_text(encoding="latin1").splitlines()

    cleaned_lines = []
    for line in lines:
        line = line.strip()

        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]

        line = line.replace('""', '"')
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def find_real_header_row(raw: pd.DataFrame) -> int:
    max_rows_to_check = min(20, len(raw))

    for i in range(max_rows_to_check):
        row_values = raw.iloc[i].astype(str).str.strip().str.lower().tolist()

        if "subjectkey" in row_values:
            return i

        if "src_subject_id" in row_values:
            return i

        if "participant_id" in row_values:
            return i

    return 0


def repair_header_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    lower_cols = [str(c).strip().lower() for c in df.columns]

    if (
        "subjectkey" in lower_cols
        or "src_subject_id" in lower_cols
        or "participant_id" in lower_cols
    ):
        return df

    raw = df.reset_index(drop=True)
    header_row = find_real_header_row(raw)

    new_columns = raw.iloc[header_row].astype(str).str.strip().tolist()
    fixed = raw.iloc[header_row + 1:].copy()
    fixed.columns = new_columns

    return fixed


def drop_bad_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    keep_cols = [
        c for c in df.columns
        if not c.lower().startswith("unnamed")
        and c != ""
        and c.lower() != "nan"
    ]

    return df[keep_cols]


def read_messy_tsv(path: Path) -> pd.DataFrame:
    # Strategy 1: normal TSV
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            encoding="latin1",
            engine="python"
        )

        if df.shape[1] > 1:
            df = repair_header_if_needed(df)
            df = drop_bad_unnamed_columns(df)
            return df

    except Exception:
        pass

    # Strategy 2: quoted-line CSV-style repair
    text = clean_quoted_line_file(path)

    try:
        df = pd.read_csv(
            StringIO(text),
            sep=",",
            engine="python"
        )

        df = repair_header_if_needed(df)
        df = drop_bad_unnamed_columns(df)
        return df

    except Exception:
        pass

    # Strategy 3: read raw, manually promote real header
    raw = pd.read_csv(
        StringIO(text),
        sep=",",
        header=None,
        engine="python"
    )

    header_row = find_real_header_row(raw)
    new_columns = raw.iloc[header_row].astype(str).str.strip().tolist()

    df = raw.iloc[header_row + 1:].copy()
    df.columns = new_columns
    df = drop_bad_unnamed_columns(df)

    return df


def convert_all_tsvs():
    tsv_files = sorted(phenotype_dir.glob("*.tsv"))

    print(f"Found {len(tsv_files)} TSV files.")

    for tsv_path in tsv_files:
        try:
            df = read_messy_tsv(tsv_path)

            csv_path = tsv_path.with_suffix(".csv")

            df.to_csv(
                csv_path,
                index=False,
                quoting=csv.QUOTE_MINIMAL
            )

            print(f"Converted {tsv_path.name} -> {csv_path.name} | shape={df.shape}")

        except Exception as e:
            print(f"FAILED {tsv_path.name}: {e}")


if __name__ == "__main__":
    convert_all_tsvs()