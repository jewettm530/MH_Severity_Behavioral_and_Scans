"""
01_explore_dataset.py

Purpose:
- Inspect the dataset folder structure.
- List phenotype files.
- List fMRI subject folders.
- Inspect one .h5 parcellated timeseries file so you can see its keys/shapes.

Run:
    python 01_explore_dataset.py
"""

from pathlib import Path
import h5py
from config import DATA_DIR, PHENOTYPE_DIR, FMRI_DIR, MOTION_DIR


def list_dir(path: Path, max_items: int = 30):
    print(f"\n--- {path} ---")
    if not path.exists():
        print("Path does not exist.")
        return []
    items = sorted(path.iterdir())
    for item in items[:max_items]:
        print(item.name + ("/" if item.is_dir() else ""))
    if len(items) > max_items:
        print(f"... and {len(items) - max_items} more")
    return items


def inspect_h5_file(path: Path):
    print(f"\nInspecting H5 file:\n{path}")
    if not path.exists():
        print("File does not exist.")
        return

    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET: {name}, shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"GROUP:   {name}")

    with h5py.File(path, "r") as f:
        print("Top-level keys:", list(f.keys()))
        f.visititems(walk)


def main():
    print("DATA_DIR:", DATA_DIR.resolve())

    list_dir(DATA_DIR)
    phenotype_files = list_dir(PHENOTYPE_DIR)
    fmri_subjects = [p for p in list_dir(FMRI_DIR) if p.is_dir()]
    list_dir(MOTION_DIR)

    print(f"\nPhenotype files found: {len([p for p in phenotype_files if p.suffix.lower() in ['.csv', '.tsv']])}")
    print(f"fMRI subject folders found: {len(fmri_subjects)}")

    # Try to inspect the first available rest .h5 file.
    for sub_dir in fmri_subjects[:10]:
        h5_files = sorted(sub_dir.glob("*rest*.h5"))
        if h5_files:
            inspect_h5_file(h5_files[0])
            break
    else:
        print("\nNo rest .h5 file found in the first few subject folders.")


if __name__ == "__main__":
    main()
