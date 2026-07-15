# Complete TCP Symptom-Severity Pipeline Update

This package starts from the uploaded project and adds both the earlier corrections and the new cross-category analysis.

## What changed

- Uses existing `outputs/imaging_features.csv`; imaging extraction does not need to be rerun.
- Removes repeated age, sex, site, and group columns from every questionnaire.
- Loads demographics once from `demos.csv`.
- Converts 999/9999 unavailable codes to missing values.
- Enforces valid target ranges for QIDS, STAI state, SHAPS, and PSS.
- Excludes each target questionnaire from its own predictors to prevent leakage.
- Uses the imaging-available cohort for every comparison so modalities are evaluated on the same participants.
- Uses repeated stratified 5-fold cross-validation and a fixed stratified 80/20 test split.
- Runs 28 modality/scope experiments:
  - 4 imaging-only models
  - 12 behavioral-only models: all/relevant/category × 4 targets
  - 12 multimodal models: all/relevant/category × 4 targets
- Runs 32 behavioral cross-category models: each target category alone plus every combination of the other three categories.

## Replace these scripts

Copy everything inside `scripts/` into your project's `scripts/` folder.

## Run order

From `/Users/maddiemac/AI4All/Final_Project`:

```bash
python3 scripts/02_prepare_behavioral_data.py
python3 scripts/experiments.py
python3 scripts/04_merge_features.py
python3 scripts/05_train_random_forest.py
python3 scripts/06_evaluate_results.py
python3 scripts/07_cross_category_analysis.py
```

Do **not** rerun `03_extract_imaging_features.py` because `outputs/imaging_features.csv` already exists.

## Main outputs

### Modality and scope comparison

- `outputs/results/model_metrics_comparison.csv`
- `outputs/results/model_metrics_comparison_sorted.csv`
- `outputs/results/best_model_by_target.csv`
- `outputs/results/modality_scope_summary.csv`

### Cross-category analysis

- `outputs/results/cross_category_metrics.csv`
- `outputs/results/cross_category_metric_improvements.html`
- `outputs/results/cross_category_summary.txt`

The HTML output is divided into QIDS, STAI, SHAPS, and PSS sections. The baseline row shows the target category's raw R². Added-category rows show both raw performance and the change from baseline.

## Before rerunning

Old results use the prior one-target structure. Move or delete generated experiment/result folders, but retain imaging features:

```bash
rm -rf outputs/behavioral_by_file
rm -rf outputs/experiment_datasets
rm -rf outputs/cross_category_datasets
rm -rf outputs/results
```

Keep:

```text
outputs/imaging_features.csv
```
