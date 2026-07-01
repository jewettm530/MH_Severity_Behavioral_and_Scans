"""
03_extract_imaging_features.py

Purpose:
- Load parcellated resting-state fMRI .h5 files.
- Convert each run into compact imaging features.
- Average features across available rest runs per subject.
- Save imaging_features.csv.

Important:
- This script tries to automatically find the largest numeric 2D dataset inside each .h5 file.
- After running 01_explore_dataset.py, you may update find_timeseries_array() if the H5 structure is unusual.

Run:
    python 03_extract_imaging_features.py
"""

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from config import FMRI_DIR, OUTPUT_DIR, REST_FILE_PATTERNS

def load_timeseries(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        if "dataset" not in f:
            raise ValueError(f"No dataset key in {h5_path}")
        arr = np.asarray(f["dataset"][()], dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    # TCP rest data are timepoints x parcels, commonly (434, 488). Do not transpose.
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def summarize_run(ts: np.ndarray) -> dict:
    corr = np.corrcoef(ts, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return {
        "signal_mean": float(np.mean(ts)),
        "signal_std": float(np.std(ts)),
        "signal_abs_mean": float(np.mean(np.abs(ts))),
        "conn_mean": float(np.mean(upper)),
        "conn_std": float(np.std(upper)),
        "conn_median": float(np.median(upper)),
        "conn_min": float(np.min(upper)),
        "conn_max": float(np.max(upper)),
        "conn_abs_mean": float(np.mean(np.abs(upper))),
        "conn_pos_mean": float(np.mean(upper[upper > 0])) if np.any(upper > 0) else 0.0,
        "conn_neg_mean": float(np.mean(upper[upper < 0])) if np.any(upper < 0) else 0.0,
        "conn_pos_fraction": float(np.mean(upper > 0)),
        "n_timepoints": float(ts.shape[0]),
        "n_parcels": float(ts.shape[1]),
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    subject_dirs = sorted(p for p in FMRI_DIR.iterdir() if p.is_dir())
    print(f"Found {len(subject_dirs)} subject folders in {FMRI_DIR}")
    for i, sub_dir in enumerate(subject_dirs, start=1):
        summaries = []
        for pattern in REST_FILE_PATTERNS:
            h5_path = sub_dir / pattern
            if h5_path.exists():
                try:
                    ts = load_timeseries(h5_path)
                    print(f"[{i}/{len(subject_dirs)}] {sub_dir.name}: {h5_path.name}, shape={ts.shape}")
                    summaries.append(summarize_run(ts))
                except Exception as e:
                    print(f"WARNING: failed {sub_dir.name} {h5_path.name}: {e}")
        if summaries:
            avg = pd.DataFrame(summaries).mean(numeric_only=True).to_dict()
            row = {"participant_id": sub_dir.name, "n_rest_runs_used": len(summaries)}
            row.update({f"rest_avg_{k}": v for k, v in avg.items()})
            rows.append(row)
    imaging = pd.DataFrame(rows)
    out = OUTPUT_DIR / "imaging_features.csv"
    imaging.to_csv(out, index=False)
    print("\nSaved:", out)
    print("Shape:", imaging.shape)
    print("Columns:", list(imaging.columns))

if __name__ == "__main__":
    main()
