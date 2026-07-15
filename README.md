# Mental Health Severity Prediction Using Behavioral and Resting-State fMRI Features

## Overview

This project investigates whether behavioral measures, resting-state functional MRI (rs-fMRI) connectivity features, or a combination of both can predict mental health symptom severity. Multiple supervised machine learning models were trained and evaluated to predict four clinically relevant outcomes:

- **PSS** – Perceived Stress Scale
- **QIDS** – Quick Inventory of Depressive Symptomatology
- **SHAPS** – Snaith-Hamilton Pleasure Scale (Anhedonia)
- **STAI-State** – State Anxiety

The primary objective is to determine:

1. How accurately behavioral information predicts symptom severity.
2. Whether neuroimaging features provide additional predictive value.
3. Which behavioral feature selection strategy performs best.
4. Whether multimodal learning improves prediction compared to behavioral data alone.

---

# Research Questions

- Can behavioral measures accurately predict mental health severity?
- Can resting-state functional connectivity predict symptom severity independently?
- Does combining behavioral and imaging information improve prediction?
- Which behavioral variables are most informative for each mental health outcome?
- Which symptom domains are easiest and most difficult to predict?

---

# Dataset

The project uses data from the **OpenNeuro Transdiagnostic Connectome Dataset (ds005237)**.

Each participant contains:

- Demographic information
- Behavioral questionnaires
- Resting-state fMRI
- Clinical assessments

Only subjects with complete behavioral and imaging data are retained for modeling.

---

# Prediction Targets

| Target | Description |
|----------|-------------|
| PSS | Perceived Stress Scale |
| QIDS | Depression Severity |
| SHAPS | Anhedonia Severity |
| STAI | State Anxiety |

---

# Feature Types

## Behavioral Features

Behavioral variables include:

- Demographics
- Questionnaire responses
- Clinical assessments
- Cognitive measures
- Lifestyle variables

Three behavioral feature subsets are evaluated.

### All Features

Uses every available behavioral predictor.

Approximately 200 features.

---

### Relevant Features

A manually curated subset of variables considered theoretically relevant to each prediction target.

Approximately 20 features.

---

### Category Features

A very small subset representing broad behavioral domains.

Approximately 5–13 features depending on the target.

---

## Imaging Features

Resting-state fMRI time series are converted into functional connectivity matrices.

Connectivity matrices are summarized into global statistical descriptors.

Current imaging features include:

- Mean signal
- Signal standard deviation
- Absolute mean signal
- Mean connectivity
- Connectivity standard deviation
- Median connectivity
- Minimum connectivity
- Maximum connectivity
- Absolute connectivity mean
- Mean positive connectivity
- Mean negative connectivity
- Fraction positive connections
- Number of parcels
- Number of time points

These intentionally provide a lightweight baseline representation of the connectome.

---

# Experimental Conditions

Seven experiments are performed for every prediction target.

| Experiment | Description |
|------------|-------------|
| Behavioral Only (All) | All behavioral variables |
| Behavioral Only (Relevant) | Selected behavioral variables |
| Behavioral Only (Category) | Minimal behavioral subset |
| Imaging Only | Imaging summary features only |
| Multimodal (All) | Behavioral All + Imaging |
| Multimodal (Relevant) | Relevant Behavioral + Imaging |
| Multimodal (Category) | Category Behavioral + Imaging |

Total experiments:

4 targets × 7 feature sets = **28 trained models**

---

# Machine Learning Pipeline

Each experiment follows the same pipeline.

```
Load Data
      ↓
Merge Behavioral + Imaging
      ↓
Handle Missing Values
      ↓
Train/Test Split
      ↓
Cross Validation
      ↓
Train Final Model
      ↓
Evaluate Test Set
      ↓
Save Metrics
```

---

# Model

Current model:

Random Forest Regressor

Reasons for selection:

- Handles nonlinear relationships
- Robust to noisy variables
- Minimal preprocessing
- Handles mixed feature types
- Strong baseline for tabular biomedical data

---

# Validation Strategy

Each experiment uses:

- Train/Test Split
- 15-fold Cross Validation
- Independent Test Set

Reported metrics include:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R²
- Standard deviation across folds

Both cross-validation and held-out test performance are reported.

---

# Project Structure

```
Mental_Health_Severity/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── behavioral/
│   ├── imaging/
│   └── multimodal/
│
├── scripts/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── modeling/
│   ├── evaluation/
│   └── visualization/
│
├── outputs/
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   └── feature_importance/
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# Output Files

## best_model_by_target.csv

Contains the highest-performing model for each mental health target.

Columns include:

- Target
- Best experiment
- Performance metrics
- Number of features
- Dataset sizes

---

## model_metrics_comparison_sorted.csv

Complete comparison of every trained model.

Includes:

- Cross-validation metrics
- Test metrics
- Feature counts
- Experiment descriptions

---

## Predictions

Prediction files contain:

- Subject ID
- Actual score
- Predicted score
- Residual
- Absolute error

---

## Figures

Generated visualizations include:

- Predicted vs Actual
- Residual plots
- Feature importance
- Performance comparisons
- Cross-validation summaries

---

# Results

## Best Models

| Target | Best Model | Test R² |
|----------|------------|---------|
| PSS | Behavioral Relevant | **0.674** |
| QIDS | Behavioral All | **0.642** |
| SHAPS | Behavioral Category | **0.295** |
| STAI | Behavioral All | **0.719** |

---

## Main Findings

### Behavioral data consistently outperformed imaging.

Behavioral-only models achieved the highest predictive performance across every mental health outcome.

---

### Imaging-only models performed poorly.

Resting-state summary statistics alone produced negative test R² values for every target.

---

### Multimodal models did not improve prediction.

Adding imaging features to behavioral predictors resulted in little or no performance improvement.

---

### Anxiety was the easiest target.

STAI produced the strongest predictive performance.

---

### Anhedonia was the most difficult target.

SHAPS showed substantially lower predictive performance than the other symptom domains.

---

# Interpretation

These findings suggest that:

- Behavioral measures contain strong predictive information regarding mental health severity.
- The current resting-state connectivity summary features do not capture sufficient information to improve prediction.
- Richer neuroimaging feature engineering will likely be required before imaging contributes meaningful predictive value.

---

# Limitations

Current imaging features summarize the connectome using only global statistics.

Potential improvements include:

- Graph-theoretic metrics
- Network-level connectivity
- ROI-level connectivity
- Functional network strength
- Principal component analysis
- Independent component analysis
- Edge selection approaches
- Graph neural networks

---

# Future Work

Potential future directions include:

- More informative imaging representations
- Elastic Net regression
- Gradient Boosting
- XGBoost
- LightGBM
- CatBoost
- Support Vector Regression
- Deep learning approaches
- Explainable AI (SHAP)
- Nested cross-validation
- Hyperparameter optimization
- External dataset validation

---

# Reproducibility

To reproduce the project:

```bash
git clone <repository>

cd Mental_Health_Severity

pip install -r requirements.txt
```

Run the preprocessing pipeline.

Generate behavioral features.

Generate imaging features.

Merge datasets.

Train models.

Generate evaluation metrics and figures.

---

# Requirements

Major Python packages include:

```
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
joblib
nibabel
nilearn
```

Additional dependencies are listed in `requirements.txt`.

---

# Citation

If using this repository in academic work, please cite the original OpenNeuro dataset along with this repository.

---

# License

Specify the license used for this repository (e.g., MIT, BSD-3-Clause, Apache 2.0).

---

# Acknowledgements

- OpenNeuro
- Contributors to the ds005237 dataset
- scikit-learn
- Nilearn
- NiBabel
- NumPy
- Pandas