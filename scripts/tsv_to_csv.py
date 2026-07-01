from pathlib import Path
import pandas as pd

# Change this path to your phenotype folder
PHENOTYPE_DIR = Path("/Users/maddiemac/AI4All/Final_Project/Dataset/phenotype")

# Output folder for CSVs
OUTPUT_DIR = PHENOTYPE_DIR / "csv_converted"
OUTPUT_DIR.mkdir(exist_ok=True)

tsv_files = list(PHENOTYPE_DIR.glob("*.tsv"))

print(f"Found {len(tsv_files)} TSV files.")

# Encodings to try in order
encodings_to_try = ["utf-8", "latin1", "cp1252"]

for tsv_file in tsv_files:
    print(f"Converting {tsv_file.name}...")

    df = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(tsv_file, sep="\t", encoding=enc)
            print(f"  -> read successfully with encoding: {enc}")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  -> failed with encoding {enc}: {e}")
            break

    if df is None:
        print(f"  -> SKIPPING {tsv_file.name}: could not decode with tried encodings")
        continue

    csv_name = tsv_file.with_suffix(".csv").name
    csv_path = OUTPUT_DIR / csv_name
    df.to_csv(csv_path, index=False)

print(f"\nDone. CSV files saved to: {OUTPUT_DIR}")