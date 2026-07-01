"""
05_train_random_forest.py

Purpose:
- Train Random Forest Regression models for:
    behavioral-only
    imaging-only
    multimodal
- Uses a held-out test set plus 5-fold cross-validation on training data.
- Saves metrics, predictions, and trained models.

Run:
    python 05_train_random_forest.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from config import OUTPUT_DIR, RESULTS_DIR, SUBJECT_ID_COLUMN, TARGET_COLUMN, RANDOM_STATE

def clean_X(df):
    X = df.drop(columns=[c for c in [TARGET_COLUMN, SUBJECT_ID_COLUMN, "participant_id"] if c in df.columns], errors="ignore")
    X = X.dropna(axis=1, how="all")
    object_cols = X.select_dtypes(include=["object"]).columns.tolist()
    too_unique = [c for c in object_cols if X[c].nunique(dropna=True) > max(20, 0.5 * len(X))]
    if too_unique:
        print("Dropping high-cardinality text/ID columns:", too_unique[:20], "..." if len(too_unique) > 20 else "")
        X = X.drop(columns=too_unique)
    return X

def build_pipeline(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ], remainder="drop")
    model = RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE, min_samples_leaf=5, max_features="sqrt", n_jobs=-1)
    return Pipeline([("preprocess", preprocess), ("model", model)])

def metrics(y_true, y_pred):
    return {"MAE": float(mean_absolute_error(y_true, y_pred)), "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))), "R2": float(r2_score(y_true, y_pred))}

def train_one(name):
    df = pd.read_csv(OUTPUT_DIR / f"final_{name}.csv", low_memory=False).dropna(subset=[TARGET_COLUMN])
    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    keep = y.notna()
    df, y = df.loc[keep].copy(), y.loc[keep]
    X = clean_X(df)
    print(f"\n=== {name} ===")
    print("Rows:", len(X), "Features:", X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
    pipe = build_pipeline(X_train)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring={"mae":"neg_mean_absolute_error", "rmse":"neg_root_mean_squared_error", "r2":"r2"}, n_jobs=-1)
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    test = metrics(y_test, pred)
    out = {
        "dataset": name, "n_total": int(len(df)), "n_train": int(len(X_train)), "n_test": int(len(X_test)), "n_features": int(X.shape[1]),
        "cv_MAE_mean": float(-cv_res["test_mae"].mean()), "cv_MAE_std": float(cv_res["test_mae"].std()),
        "cv_RMSE_mean": float(-cv_res["test_rmse"].mean()), "cv_RMSE_std": float(cv_res["test_rmse"].std()),
        "cv_R2_mean": float(cv_res["test_r2"].mean()), "cv_R2_std": float(cv_res["test_r2"].std()),
        "test_MAE": test["MAE"], "test_RMSE": test["RMSE"], "test_R2": test["R2"],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, RESULTS_DIR / f"{name}_random_forest.joblib")
    pd.DataFrame({"y_true": y_test.values, "y_pred": pred, "model": name}).to_csv(RESULTS_DIR / f"{name}_predictions.csv", index=False)
    with open(RESULTS_DIR / f"{name}_metrics.json", "w") as f: json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out

def main():
    all_metrics = [train_one(n) for n in ["behavioral_only", "imaging_only", "multimodal"]]
    pd.DataFrame(all_metrics).to_csv(RESULTS_DIR / "model_comparison_metrics.csv", index=False)
    print("\nSaved comparison:", RESULTS_DIR / "model_comparison_metrics.csv")

if __name__ == "__main__":
    main()
