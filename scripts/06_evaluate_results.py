"""
06_evaluate_results.py

Purpose:
- Summarize model performance.
- Plot model comparison.
- Extract Random Forest feature importance for presentation.

Run:
    python 06_evaluate_results.py
"""

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from config import OUTPUT_DIR, RESULTS_DIR, TARGET_COLUMN, SUBJECT_ID_COLUMN

def clean_X(df):
    X = df.drop(columns=[c for c in [TARGET_COLUMN, SUBJECT_ID_COLUMN, "participant_id"] if c in df.columns], errors="ignore")
    X = X.dropna(axis=1, how="all")
    object_cols = X.select_dtypes(include=["object"]).columns.tolist()
    too_unique = [c for c in object_cols if X[c].nunique(dropna=True) > max(20, 0.5 * len(X))]
    return X.drop(columns=too_unique, errors="ignore")

def feature_names(pipe, X):
    names = []
    for name, transformer, cols in pipe.named_steps["preprocess"].transformers_:
        if name == "num": names.extend(cols)
        elif name == "cat":
            try: names.extend(transformer.named_steps["onehot"].get_feature_names_out(cols))
            except Exception: names.extend(cols)
    return list(names)

def save_importance(name):
    model_path = RESULTS_DIR / f"{name}_random_forest.joblib"
    data_path = OUTPUT_DIR / f"final_{name}.csv"
    pipe = joblib.load(model_path)
    df = pd.read_csv(data_path, low_memory=False).dropna(subset=[TARGET_COLUMN])
    X = clean_X(df)
    names = feature_names(pipe, X)
    imps = pipe.named_steps["model"].feature_importances_
    n = min(len(names), len(imps))
    out = pd.DataFrame({"feature": names[:n], "importance": imps[:n]}).sort_values("importance", ascending=False)
    out.to_csv(RESULTS_DIR / f"{name}_feature_importance.csv", index=False)
    top = out.head(15)
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"][::-1], top["importance"][::-1])
    plt.xlabel("Random Forest feature importance")
    plt.title(f"Top 15 Features: {name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{name}_top15_feature_importance.png", dpi=200)
    plt.close()
    print("Saved feature importance for", name)

def main():
    metrics = pd.read_csv(RESULTS_DIR / "model_comparison_metrics.csv")
    print("\nModel comparison:")
    print(metrics[["dataset", "n_total", "n_features", "cv_MAE_mean", "cv_RMSE_mean", "cv_R2_mean", "test_MAE", "test_RMSE", "test_R2"]])
    plt.figure(figsize=(8,5))
    plt.bar(metrics["dataset"], metrics["test_R2"])
    plt.ylabel("Test R²")
    plt.xlabel("Model input type")
    plt.title("Random Forest Model Comparison")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison_test_R2.png", dpi=200)
    plt.close()
    for name in ["behavioral_only", "imaging_only", "multimodal"]: save_importance(name)

if __name__ == "__main__":
    main()
