"""
Yield prediction models for SECOM wafer fab data.

Trains Random Forest and LightGBM classifiers on pass/fail labels.
Class imbalance is handled via class_weight='balanced' (RF) and
scale_pos_weight (LightGBM) rather than SMOTE — this avoids synthetic
sample noise and is directly supported by both model APIs.

Primary metric: PR-AUC, because ROC-AUC can be misleadingly high when the
negative (pass) class dominates. PR-AUC focuses on the minority (fail) class.
"""

from __future__ import annotations

import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from src.data import load_and_prepare_data

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

MODELS_DIR = Path(__file__).parent.parent / "models"
RANDOM_STATE = 42


def _best_threshold(y_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Find probability threshold that maximises F1 on the fail class."""
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        score = f1_score(y_true, y_pred_t, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t


def _evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]

    # Tune threshold to maximise fail-class F1 (default 0.5 predicts all-pass
    # when imbalance is 14:1)
    threshold = _best_threshold(y_prob, y_test)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "model": name,
        "threshold": round(threshold, 4),
        "precision_fail": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall_fail": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_fail": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "pr_auc": round(average_precision_score(y_test, y_prob), 4),
    }

    print(f"\n{'='*50}")
    print(f"Model: {name}  (threshold={threshold:.3f})")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["pass", "fail"]))
    print(f"ROC-AUC : {metrics['roc_auc']}")
    print(f"PR-AUC  : {metrics['pr_auc']}  ← primary metric")
    return metrics


def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf


def train_lightgbm(X_train, y_train):
    neg, pos = np.bincount(y_train)
    spw = neg / pos
    lgbm = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=spw,
        min_child_samples=5,   # default 20 is too high for ~82 training fails
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    return lgbm


def train_and_save():
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data(
        save_preprocessor=True
    )
    MODELS_DIR.mkdir(exist_ok=True)

    results = []

    # Random Forest
    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train, y_train)
    results.append(_evaluate("RandomForest", rf, X_test, y_test))

    # LightGBM
    if HAS_LGBM:
        print("\nTraining LightGBM...")
        lgbm = train_lightgbm(X_train, y_train)
        results.append(_evaluate("LightGBM", lgbm, X_test, y_test))
    else:
        print("LightGBM not available — skipping.")

    # Pick winner by PR-AUC (best for imbalanced classification)
    best = max(results, key=lambda r: r["pr_auc"])
    winner_model = lgbm if (HAS_LGBM and best["model"] == "LightGBM") else rf
    print(f"\nWinner: {best['model']} (PR-AUC = {best['pr_auc']})")

    joblib.dump(winner_model, MODELS_DIR / "yield_model.joblib")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.joblib")
    joblib.dump(best["threshold"], MODELS_DIR / "yield_threshold.joblib")

    with open(MODELS_DIR / "model_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved: models/yield_model.joblib ({best['model']})")
    print(f"Saved: models/yield_threshold.joblib ({best['threshold']})")
    print("Saved: models/model_metrics.json")
    return winner_model, feature_names, best


if __name__ == "__main__":
    train_and_save()
