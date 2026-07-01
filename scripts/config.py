"""
Shared configuration for the Transdiagnostic Connectome Project Random Forest workflow.

Updated version:
- Loads all phenotype CSV files
- Uses qids01.csv only for the target qvtot
- Keeps psychiatric / behavioral summary scores, subscales, demographics, and useful standalone values
- Drops timing/click/QC/admin columns
- Drops item-level questionnaire variables
- Drops variables with 25% or more missing values
"""

from pathlib import Path

DATA_DIR = Path("/Users/maddiemac/AI4All/Final_Project/Dataset")

PHENOTYPE_DIR = DATA_DIR / "phenotype"
FMRI_DIR = DATA_DIR / "fMRI_timeseries_clean_denoised_GSR_parcellated"
MOTION_DIR = DATA_DIR / "motion_FD"

OUTPUT_DIR = Path("outputs")
RESULTS_DIR = OUTPUT_DIR / "results"

RANDOM_STATE = 42

TARGET_FILE = "qids01.csv"
TARGET_COLUMN = "qvtot"
SUBJECT_ID_COLUMN = "subjectkey"

PHENOTYPE_FILES_TO_USE = []
EXCLUDE_PHENOTYPE_FILES = [TARGET_FILE]
MAX_MISSING_FRACTION = 0.25

REST_FILE_PATTERNS = [
    "task-restAP_run-01_bold_Atlas_hp2000_clean_GSR_parcellated.h5",
    "task-restAP_run-02_bold_Atlas_hp2000_clean_GSR_parcellated.h5",
    "task-restPA_run-01_bold_Atlas_hp2000_clean_GSR_parcellated.h5",
    "task-restPA_run-02_bold_Atlas_hp2000_clean_GSR_parcellated.h5",
]
