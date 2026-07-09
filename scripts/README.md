# Transdiagnostic Connectome Project: Random Forest Symptom Severity Prediction

This template is for a class project predicting mental health symptom severity using:

- behavioral/clinical data from `phenotype/`
- extracted/parcellated fMRI timeseries from `fMRI_timeseries_clean_denoised_GSR_parcellated/`
- Random Forest Regression models

## Required folders from the dataset

You likely need:

- `phenotype/`
- `fMRI_timeseries_clean_denoised_GSR_parcellated/`
- optional: `motion_FD/`

You likely do not need the raw `sub-*` folders if you are not using raw MRI images (which we dont plan to)

## Setup

Install packages:

```bash
pip install pandas numpy scikit-learn matplotlib h5py joblib
```

Edit `config.py`:

```python
DATA_DIR = Path("/path/to/ds005237-download")
TARGET_COLUMN = "your_symptom_severity_score"
SUBJECT_ID_COLUMN = "participant_id"
```

## Run order

```bash
python 01_explore_dataset.py
python 02_prepare_behavioral_data.py
python 03_extract_imaging_features.py
python 04_merge_features.py
python 05_train_random_forest.py
python 06_evaluate_results.py
```

## Outputs

Generated files are saved to `outputs/`:

- `behavioral_features.csv`
- `imaging_features.csv`
- `final_behavioral_only.csv`
- `final_imaging_only.csv`
- `final_multimodal.csv`
- `results/model_comparison_metrics.csv`
- feature importance CSVs and plots
