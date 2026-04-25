"""
model_comparison.py
────────────────────────────────────────────────────────────────────────────────
A/B testing pipeline: Random Forest, XGBoost, Logistic Regression.
Runs on the processed feature-engineered dataset and outputs:
  - model_results.csv    ← metrics for all three models
  - best_model.joblib    ← serialised winning model (Random Forest)
  - stat_sig_report.txt  ← McNemar's test results

Usage:
    python src/model_comparison.py
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠  xgboost not installed — XGBoost skipped. pip install xgboost")


PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob, model_name: str) -> dict:
    return {
        "Model":     model_name,
        "Precision": round(precision_score(y_true, y_pred), 4),
        "Recall":    round(recall_score(y_true, y_pred), 4),
        "F1":        round(f1_score(y_true, y_pred), 4),
        "ROC_AUC":   round(roc_auc_score(y_true, y_prob), 4),
    }


def mcnemar_test(y_true, preds_a, preds_b, name_a: str, name_b: str) -> str:
    """McNemar's test comparing two classifiers on the same test set."""
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    b = int(( correct_a & ~correct_b).sum())  # A right, B wrong
    c = int((~correct_a &  correct_b).sum())  # A wrong, B right
    table = np.array([[0, b], [c, 0]])
    # Using mid-p McNemar (chi2 approximation for large samples)
    n_discordant = b + c
    if n_discordant == 0:
        return f"{name_a} vs {name_b}: identical predictions — p = 1.000"
    chi2 = (abs(b - c) - 1) ** 2 / n_discordant
    from scipy.stats import chi2 as chi2_dist
    p_value = chi2_dist.sf(chi2, df=1)
    sig = "✅ SIGNIFICANT" if p_value < 0.05 else "❌ NOT significant"
    return (
        f"{name_a} vs {name_b}:  b={b}, c={c}  |  "
        f"χ²={chi2:.3f}  p={p_value:.5f}  {sig} (α=0.05)"
    )


# ── Build models ─────────────────────────────────────────────────────────────
def build_models():
    models = {
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=0.5, max_iter=1000, class_weight="balanced", random_state=42
            )),
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = Pipeline([
            ("clf", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=1,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
            )),
        ])
    return models


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Load feature-engineered data
    feat_path = PROCESSED_DIR / "features.csv"
    if not feat_path.exists():
        print(
            f"⚠  {feat_path} not found.\n"
            "   Run notebooks/feature_engineering.ipynb first, or use the "
            "   synthetic dataset from visualization_app.py."
        )
        _run_on_synthetic()
        return

    df = pd.read_csv(feat_path)
    target_col = "is_churned"
    feature_cols = [c for c in df.columns
                    if c not in [target_col, "customer_id", "CustomerID"]]

    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target_col].astype(int)

    print(f"Dataset: {len(df):,} customers | {y.mean()*100:.1f}% churn rate")
    _run_comparison(X, y)


def _run_on_synthetic():
    """Fallback: generate synthetic RFM data and run comparison."""
    print("Running on synthetic RFM data …")
    np.random.seed(42)
    n = 4340
    recency   = np.random.exponential(60, n).astype(int) + 1
    frequency = np.random.poisson(8, n) + 1
    monetary  = np.abs(np.random.normal(450, 250, n))
    tenure    = np.random.randint(30, 730, n)

    churn_prob = (
        0.35 * (recency / recency.max())
        + 0.28 * (1 - frequency / frequency.max())
        + 0.18 * (1 - monetary / monetary.max())
        + 0.12 * (1 - tenure / tenure.max())
        + np.random.normal(0, 0.08, n)
    ).clip(0, 1)
    y = (churn_prob > 0.52).astype(int)
    X = pd.DataFrame({"recency": recency, "frequency": frequency,
                       "monetary": monetary, "tenure": tenure})

    print(f"Synthetic dataset: {n:,} customers | {y.mean()*100:.1f}% churn rate")
    _run_comparison(X, y)


def _run_comparison(X, y):
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mdls  = build_models()
    results, all_preds, all_probs = [], {}, {}

    for name, pipe in mdls.items():
        print(f"  Training {name} …")
        preds = cross_val_predict(pipe, X, y, cv=cv, method="predict")
        probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
        metrics = compute_metrics(y, preds, probs, name)
        results.append(metrics)
        all_preds[name] = preds
        all_probs[name] = probs
        print(f"    Precision={metrics['Precision']:.3f}  "
              f"Recall={metrics['Recall']:.3f}  "
              f"F1={metrics['F1']:.3f}  "
              f"AUC={metrics['ROC_AUC']:.3f}")

    results_df = pd.DataFrame(results).sort_values("Precision", ascending=False)
    results_df["Selected"] = results_df["Model"] == results_df.iloc[0]["Model"]
    results_df.to_csv(MODELS_DIR / "model_results.csv", index=False)
    print(f"\n✅ Model results saved → {MODELS_DIR / 'model_results.csv'}")

    # Statistical significance
    print("\n── McNemar's Test ─────────────────────────────────────────────────")
    sig_lines = []
    model_names = list(all_preds.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            line = mcnemar_test(
                y, all_preds[model_names[i]], all_preds[model_names[j]],
                model_names[i], model_names[j]
            )
            print(" ", line)
            sig_lines.append(line)

    (MODELS_DIR / "stat_sig_report.txt").write_text("\n".join(sig_lines))

    # Retrain winner on full data and save
    winner_name = results_df.iloc[0]["Model"]
    winner_pipe = mdls[winner_name]
    winner_pipe.fit(X, y)
    joblib.dump(winner_pipe, MODELS_DIR / "best_churn_model.joblib")
    print(f"\n✅ Winner: {winner_name} — model saved → {MODELS_DIR / 'best_churn_model.joblib'}")

    print("\n── Final Leaderboard ──────────────────────────────────────────────")
    print(results_df[["Model", "Precision", "Recall", "F1", "ROC_AUC"]].to_string(index=False))


if __name__ == "__main__":
    main()